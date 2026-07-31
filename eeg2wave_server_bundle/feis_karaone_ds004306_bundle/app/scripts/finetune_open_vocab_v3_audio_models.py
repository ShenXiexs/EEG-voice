#!/usr/bin/env python3
"""Fine-tune every pretrained audio component used inside the v3 generator.

The primary ``fit`` scope is inductive and is the only checkpoint consumed by
validation/test.  ``all`` is an explicitly transductive audio-demo scope and
is written below a separate directory so it cannot silently contaminate the
scientific pipeline.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0724.audio_features import AudioPreparationConfig
from src.open_vocab_v3.audio_adaptation import (
    envelope_loss,
    file_sha256,
    module_parameter_change,
    module_state,
    multi_resolution_stft_loss,
    parameter_change,
    selected_audio_indices,
    tensor_state,
)
from src.open_vocab_v3.data import (
    _accepted_denoise_paths,
    _read_waveform,
    light_prepare_waveform,
    load_prepared,
)
from src.open_vocab_v3.runtime import (
    default_device,
    load_config,
    output_path,
    seed_everything,
    sha256_file,
    write_json,
)
from src.open_vocab_v3.vocoder import ADAPTER_FILE, KaraOneMelAdapter, model_manifest


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KaraOne audio-domain fine-tuning for v3")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=("fit", "all"), default="fit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deadline-epoch", type=float, default=0.0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args()


def _paths(config_path: Path, cfg: dict[str, Any], scope: str) -> dict[str, Path]:
    if scope == "fit":
        return {
            "vocoder": output_path(config_path, cfg, "vocoder_adapted_root"),
            "vocoder_manifest": output_path(config_path, cfg, "vocoder_manifest"),
            "speaker_checkpoint": output_path(config_path, cfg, "speaker_adapted_checkpoint"),
            "speaker_manifest": output_path(config_path, cfg, "speaker_adaptation_manifest"),
            "gate": output_path(config_path, cfg, "audio_adaptation_gate"),
        }
    root = output_path(config_path, cfg, "output_root") / "audio_adaptation" / "transductive_all"
    return {
        "vocoder": root / "speecht5_hifigan",
        "vocoder_manifest": root / "speecht5_hifigan_manifest.json",
        "speaker_checkpoint": root / "speechbrain_ecapa" / "adapted_backbone.pt",
        "speaker_manifest": root / "speechbrain_ecapa" / "adaptation_manifest.json",
        "gate": root / "A0_audio_domain_adaptation.json",
    }


def _audio_paths(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row["sample_key"]): str(row["audio_relpath"])
            for row in csv.DictReader(handle)
            if row.get("dataset") == "karaone"
        }


def _directory_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


class AudioDomainDataset(Dataset[dict[str, Any]]):
    def __init__(self, records, indices: list[int], *, config_path: Path, cfg: dict[str, Any]):
        self.records = records
        self.indices = list(indices)
        self.cfg = cfg
        self.audio_root = output_path(config_path, cfg, "audio_root")
        self.paths = _audio_paths(output_path(config_path, cfg, "unified_manifest"))
        self.denoised = _accepted_denoise_paths(config_path, cfg)
        self.preparation = AudioPreparationConfig(
            sample_rate=int(cfg["audio"]["sample_rate"]),
            max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
            target_rms=float(cfg["audio"]["target_rms"]),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def resolved_path(self, index: int) -> Path:
        key = str(self.records.arrays["sample_keys"][index])
        return self.denoised.get(key, self.audio_root / self.paths[key])

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        key = str(self.records.arrays["sample_keys"][index])
        waveform, rate = _read_waveform(self.resolved_path(index))
        prepared, _ = light_prepare_waveform(waveform, rate, self.preparation)
        return {
            "mel": torch.from_numpy(np.asarray(self.records.arrays["mel_raw"][index], dtype=np.float32)),
            "waveform": torch.from_numpy(np.asarray(prepared.waveform, dtype=np.float32)),
            "relative_length": float(prepared.valid_samples / max(len(prepared.waveform), 1)),
            "sample_key": key,
            "subject": str(self.records.arrays["subjects"][index]),
            "role": str(self.records.roles[index]),
        }

    def corpus_manifest(self) -> list[dict[str, Any]]:
        rows = []
        for index in tqdm(self.indices, desc="[v3 audio-adapt] hash WAV corpus", unit="wav", dynamic_ncols=True):
            path = self.resolved_path(index)
            rows.append(
                {
                    "sample_key": str(self.records.arrays["sample_keys"][index]),
                    "subject": str(self.records.arrays["subjects"][index]),
                    "label": str(self.records.arrays["labels"][index]),
                    "role": str(self.records.roles[index]),
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
        return rows


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mel": torch.stack([item["mel"] for item in items]),
        "waveform": torch.stack([item["waveform"] for item in items]),
        "relative_length": torch.tensor([item["relative_length"] for item in items], dtype=torch.float32),
        "sample_key": [str(item["sample_key"]) for item in items],
        "subject": [str(item["subject"]) for item in items],
        "role": [str(item["role"]) for item in items],
    }


def _deadline(args: argparse.Namespace) -> bool:
    return bool(args.deadline_epoch and time.time() >= float(args.deadline_epoch))


def _fit_vocoder(
    config_path: Path,
    cfg: dict[str, Any],
    dataset: AudioDomainDataset,
    destination: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import SpeechT5HifiGan

    base_root = output_path(config_path, cfg, "vocoder_root")
    model = SpeechT5HifiGan.from_pretrained(str(base_root), local_files_only=True).to(device)
    output_hop = int(np.prod(model.config.upsample_rates))
    adapter = KaraOneMelAdapter(
        bins=int(cfg["audio"]["mel_bins"]),
        input_hop_samples=int(cfg["audio_adaptation"]["input_mel_hop_samples"]),
        output_hop_samples=output_hop,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    before_model = tensor_state(model)
    before_adapter = tensor_state(adapter)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(adapter.parameters()),
        lr=float(cfg["audio_adaptation"]["vocoder_lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["audio_adaptation"]["vocoder_batch_size"]),
        shuffle=True, collate_fn=collate, num_workers=0,
    )
    first_loss = None
    best_loss = math.inf
    best_epoch = 0
    history = []
    for epoch in range(int(cfg["audio_adaptation"]["vocoder_epochs"])):
        if _deadline(args):
            break
        model.train(); adapter.train(); losses = []
        progress = tqdm(loader, desc=f"[v3 vocoder FT] epoch {epoch + 1}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(progress):
            if _deadline(args):
                break
            mel = batch["mel"].to(device)
            target = batch["waveform"].to(device)
            speech_t5_mel = adapter(mel)
            prediction = model(speech_t5_mel.transpose(1, 2))
            if target.shape[-1] < prediction.shape[-1]:
                target = F.pad(target, (0, prediction.shape[-1] - target.shape[-1]))
            target = target[..., : prediction.shape[-1]]
            spectral = multi_resolution_stft_loss(
                prediction, target,
                fft_sizes=cfg["audio_adaptation"]["stft_fft_sizes"],
                hop_sizes=cfg["audio_adaptation"]["stft_hop_sizes"],
            )
            waveform = F.l1_loss(prediction, target)
            envelope = envelope_loss(prediction, target)
            loss = (
                spectral
                + float(cfg["audio_adaptation"]["waveform_l1_weight"]) * waveform
                + float(cfg["audio_adaptation"]["envelope_weight"]) * envelope
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapter.parameters()), float(cfg["training"]["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
            if args.smoke_steps and step + 1 >= args.smoke_steps:
                break
        if not losses:
            break
        epoch_loss = float(np.mean(losses))
        first_loss = epoch_loss if first_loss is None else first_loss
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            destination.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(destination)
            torch.save(
                {
                    "schema_version": "openvoice-v3-karaone-mel-adapter-v1",
                    "bins": adapter.bins,
                    "input_hop_samples": adapter.input_hop_samples,
                    "output_hop_samples": adapter.output_hop_samples,
                    "state_dict": adapter.state_dict(),
                },
                destination / ADAPTER_FILE,
            )
        if args.smoke_steps:
            break
    if first_loss is None or not (destination / ADAPTER_FILE).is_file():
        raise RuntimeError("v3 vocoder fine-tuning completed no optimizer step")
    # Reload the selected checkpoint before measuring whether pretrained
    # parameters truly changed, rather than measuring a discarded final epoch.
    selected = SpeechT5HifiGan.from_pretrained(str(destination), local_files_only=True).to(device)
    adapter_payload = torch.load(destination / ADAPTER_FILE, map_location="cpu", weights_only=False)
    selected_adapter = KaraOneMelAdapter(
        bins=int(adapter_payload["bins"]),
        input_hop_samples=int(adapter_payload["input_hop_samples"]),
        output_hop_samples=int(adapter_payload["output_hop_samples"]),
    ).to(device)
    selected_adapter.load_state_dict(adapter_payload["state_dict"])
    change = parameter_change(before_model, selected)
    adapter_change = parameter_change(before_adapter, selected_adapter)
    relative_improvement = float((first_loss - best_loss) / max(abs(first_loss), 1.0e-8))
    return {
        "component": "SpeechT5HiFiGAN_generator_plus_KaraOneMelAdapter",
        "base_root": str(base_root),
        "base_manifest": str(output_path(config_path, cfg, "vocoder_base_manifest")),
        "base_manifest_sha256": sha256_file(output_path(config_path, cfg, "vocoder_base_manifest")),
        "adapted_root": str(destination),
        "all_pretrained_generator_parameters_trainable": True,
        "legacy_pretrained_weights_frozen": False,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "first_epoch_loss": first_loss,
        "best_loss": best_loss,
        "relative_loss_improvement": relative_improvement,
        "pretrained_parameter_change": change,
        "mel_adapter_parameter_change": adapter_change,
        "history": history,
        "manifest": model_manifest(destination, adapted=True),
    }


def _speaker_modules(classifier) -> list[tuple[str, torch.nn.Module]]:
    names = ("compute_features", "mean_var_norm", "embedding_model", "mean_var_norm_emb")
    modules = []
    for name in names:
        if hasattr(classifier.mods, name):
            module = getattr(classifier.mods, name)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            modules.append((name, module))
    if not any(name == "embedding_model" for name, _ in modules):
        raise RuntimeError("SpeechBrain ECAPA checkpoint lacks embedding_model")
    return modules


def _fit_speaker(
    config_path: Path,
    cfg: dict[str, Any],
    dataset: AudioDomainDataset,
    destination: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from speechbrain.inference.speaker import EncoderClassifier

    source = str(cfg["speaker"]["model_id"])
    base_root = output_path(config_path, cfg, "speaker_model_root")
    classifier = EncoderClassifier.from_hparams(
        source=source, savedir=str(base_root), run_opts={"device": str(device)}
    )
    modules = _speaker_modules(classifier)
    before = module_state(modules)
    subjects = sorted({str(dataset.records.arrays["subjects"][index]) for index in dataset.indices})
    subject_to_index = {subject: index for index, subject in enumerate(subjects)}
    probe = dataset[0]["waveform"].unsqueeze(0).to(device)
    with torch.no_grad():
        dimension = int(classifier.encode_batch(probe).reshape(1, -1).shape[-1])
    head = torch.nn.Linear(dimension, len(subjects)).to(device)
    parameters = [parameter for _, module in modules for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters + list(head.parameters()),
        lr=float(cfg["audio_adaptation"]["speaker_lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["audio_adaptation"]["speaker_batch_size"]),
        shuffle=True, collate_fn=collate, num_workers=0,
    )
    first_loss = None
    best_loss = math.inf
    best_epoch = 0
    best_accuracy = 0.0
    history = []
    for epoch in range(int(cfg["audio_adaptation"]["speaker_epochs"])):
        if _deadline(args):
            break
        for _, module in modules:
            module.train()
        head.train(); losses = []; correct = 0; count = 0
        progress = tqdm(loader, desc=f"[v3 ECAPA FT] epoch {epoch + 1}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(progress):
            if _deadline(args):
                break
            waveform = batch["waveform"].to(device)
            lengths = batch["relative_length"].to(device).clamp_min(1.0e-3)
            target = torch.tensor([subject_to_index[value] for value in batch["subject"]], device=device)
            embedding = classifier.encode_batch(waveform, lengths, normalize=False).reshape(len(target), -1)
            logits = head(F.normalize(embedding, dim=-1))
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters + list(head.parameters()), float(cfg["training"]["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(-1) == target).sum())
            count += len(target)
            progress.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{correct / max(count, 1):.3f}")
            if args.smoke_steps and step + 1 >= args.smoke_steps:
                break
        if not losses:
            break
        epoch_loss = float(np.mean(losses))
        accuracy = float(correct / max(count, 1))
        first_loss = epoch_loss if first_loss is None else first_loss
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "speaker_accuracy": accuracy})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            best_accuracy = accuracy
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": "openvoice-v3-ecapa-karaone-domain-adaptation-v1",
                    "source": source,
                    "adapted_module_names": [name for name, _ in modules],
                    "module_state_dicts": {name: module.state_dict() for name, module in modules},
                    "speaker_head_state_dict": head.state_dict(),
                    "subject_to_index": subject_to_index,
                    "embedding_dimension": dimension,
                },
                destination,
            )
        if args.smoke_steps:
            break
    if first_loss is None or not destination.is_file():
        raise RuntimeError("v3 ECAPA fine-tuning completed no optimizer step")
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    for name, module in modules:
        module.load_state_dict(payload["module_state_dicts"][name], strict=True)
    change = module_parameter_change(before, modules)
    relative_improvement = float((first_loss - best_loss) / max(abs(first_loss), 1.0e-8))
    return {
        "component": "speechbrain_ecapa_embedding_backbone",
        "source": source,
        "base_root": str(base_root),
        "base_files": _directory_manifest(base_root),
        "adapted_checkpoint": str(destination),
        "adapted_module_names": [name for name, _ in modules],
        "legacy_voxceleb_classifier_policy": "discarded; KaraOne subject head is training-only",
        "all_used_pretrained_backbone_parameters_trainable": True,
        "legacy_pretrained_weights_frozen": False,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "first_epoch_loss": first_loss,
        "best_loss": best_loss,
        "relative_loss_improvement": relative_improvement,
        "best_training_speaker_accuracy": best_accuracy,
        "pretrained_parameter_change": change,
        "checkpoint_sha256": sha256_file(destination),
        "history": history,
    }


def main() -> None:
    args = parse()
    config_path, cfg = load_config(args.config)
    seed_everything(int(cfg["training"]["seed"]))
    if args.scope == "fit" and str(cfg["audio_adaptation"]["primary_scope"]) != "fit":
        raise ValueError("primary v3 audio adaptation must use fit scope")
    records = load_prepared(output_path(config_path, cfg, "prepared_cache"))
    indices = selected_audio_indices(records, args.scope)
    dataset = AudioDomainDataset(records, indices, config_path=config_path, cfg=cfg)
    paths = _paths(config_path, cfg, args.scope)
    device = default_device(args.device)
    corpus = dataset.corpus_manifest()
    heldout = [row for row in corpus if row["role"] != "fit"]
    if args.scope == "fit" and heldout:
        raise RuntimeError("primary audio adaptation selected held-out WAVs")
    vocoder = _fit_vocoder(config_path, cfg, dataset, paths["vocoder"], device, args)
    write_json(paths["vocoder_manifest"], vocoder["manifest"] | {
        "adaptation_scope": args.scope,
        "corpus_size": len(corpus),
    })
    speaker = _fit_speaker(config_path, cfg, dataset, paths["speaker_checkpoint"], device, args)
    write_json(paths["speaker_manifest"], speaker | {
        "adaptation_scope": args.scope,
        "corpus_size": len(corpus),
    })
    threshold = cfg["audio_adaptation"]
    checks = {
        "corpus_nonempty": len(corpus) > 0,
        "primary_has_no_heldout_wav": args.scope != "fit" or not heldout,
        "vocoder_pretrained_parameters_changed": float(vocoder["pretrained_parameter_change"]["changed_parameter_fraction"]) >= float(threshold["min_changed_parameter_fraction"]),
        "vocoder_loss_improved": float(vocoder["relative_loss_improvement"]) >= float(threshold["min_relative_loss_improvement"]),
        "speaker_pretrained_parameters_changed": float(speaker["pretrained_parameter_change"]["changed_parameter_fraction"]) >= float(threshold["min_changed_parameter_fraction"]),
        "speaker_loss_improved": float(speaker["relative_loss_improvement"]) >= float(threshold["min_relative_loss_improvement"]),
    }
    gate = {
        "schema_version": "openvoice-v3-audio-domain-adaptation-gate-v1",
        "scope": args.scope,
        "protocol": "inductive_primary" if args.scope == "fit" else "transductive_audio_demo_only",
        "heldout_eeg_claims_allowed": args.scope == "fit",
        "device": str(device),
        "corpus": corpus,
        "corpus_size": len(corpus),
        "heldout_wav_count": len(heldout),
        "vocoder": vocoder,
        "speaker": speaker,
        "thresholds": {
            "min_relative_loss_improvement": float(threshold["min_relative_loss_improvement"]),
            "min_changed_parameter_fraction": float(threshold["min_changed_parameter_fraction"]),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "artifacts": {
            "vocoder_manifest": str(paths["vocoder_manifest"]),
            "vocoder_manifest_sha256": sha256_file(paths["vocoder_manifest"]),
            "speaker_manifest": str(paths["speaker_manifest"]),
            "speaker_manifest_sha256": sha256_file(paths["speaker_manifest"]),
        },
    }
    write_json(paths["gate"], gate)
    print(f"[v3 audio-adapt] scope={args.scope} corpus={len(corpus)} passed={gate['passed']}", flush=True)
    print(paths["gate"], flush=True)
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
