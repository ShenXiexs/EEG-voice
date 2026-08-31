#!/usr/bin/env python3
"""Fit the separate relative-MFCC -> native-duration SpeechT5-mel renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

APP = Path(__file__).resolve().parent; ROOT = APP.parent
sys.path.insert(0, str(APP / "src"))
from eeg2speech.data import JointManifestDataset, homogeneous_collate, phoneme_vocabulary_from_manifest, pilot_indices
from eeg2speech.model import DurationConditionedNativeRenderer


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def native_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = F.smooth_l1_loss(prediction, target, reduction="none").mean(1)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optimizer_to(optimizer: torch.optim.Optimizer, target: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value): state[key] = value.to(target)


def relative_native_template(dataset: JointManifestDataset, indices: list[int]) -> torch.Tensor:
    values = []
    for index in indices:
        mel = dataset[index]["native_speecht5_mel"].unsqueeze(0)
        values.append(F.interpolate(mel, size=161, mode="linear", align_corners=False).squeeze(0))
    if not values: raise RuntimeError("native renderer train template has no rows")
    return torch.stack(values).mean(0)


def evaluate(model: DurationConditionedNativeRenderer, loader: DataLoader, template: torch.Tensor,
             target_device: torch.device) -> dict[str, float]:
    model.eval(); model_errors=[]; baseline_errors=[]
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(target_device) if torch.is_tensor(value) else value for key, value in batch.items()}
            prediction, prediction_mask = model(batch["content_mfcc"], batch["audio_duration_frames"])
            if not torch.equal(prediction_mask, batch["native_audio_mask"]):
                raise RuntimeError("validation native duration/mask contract mismatch")
            baseline_rows=[]
            for frames in batch["audio_duration_frames"].tolist():
                baseline_rows.append(F.interpolate(template.unsqueeze(0), size=int(frames), mode="linear", align_corners=False))
            maximum=prediction.shape[-1]
            baseline=torch.cat([F.pad(value,(0,maximum-value.shape[-1])) for value in baseline_rows]).to(target_device)
            per_frame_model=(prediction-batch["native_speecht5_mel"]).abs().mean(1)
            per_frame_baseline=(baseline-batch["native_speecht5_mel"]).abs().mean(1)
            for index in range(len(prediction)):
                valid=prediction_mask[index]
                model_errors.append(float(per_frame_model[index,valid].mean()))
                baseline_errors.append(float(per_frame_baseline[index,valid].mean()))
    model.train()
    model_mae=float(sum(model_errors)/len(model_errors)); baseline_mae=float(sum(baseline_errors)/len(baseline_errors))
    return {"pairs":len(model_errors),"native_mel_mae":model_mae,"train_template_native_mel_mae":baseline_mae,
            "native_mel_improvement":float(1.0-model_mae/max(baseline_mae,1e-8))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ds004940_conditioned_v2.yaml")
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True); parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text())
    if args.max_steps < 1 or args.checkpoint_every < 1:
        raise ValueError("max-steps and checkpoint-every must be positive")
    artifact_hashes = {path.name: file_sha256(path) for path in
                       (args.config, args.manifest, args.split, args.targets, args.normalizer)}
    vocabulary = phoneme_vocabulary_from_manifest(args.manifest)
    dataset = JointManifestDataset(args.manifest, args.split, "train", "ds004940", args.targets, args.normalizer,
                                   float(cfg["loss"]["weak_content_weight"]), supervision_types={"paired_audio"},
                                   phoneme_vocabulary=vocabulary)
    indices = pilot_indices(dataset, cfg, "generalization", "train")
    validation_dataset = JointManifestDataset(args.manifest, args.split, "validation", "ds004940", args.targets, args.normalizer,
                                              float(cfg["loss"]["weak_content_weight"]), supervision_types={"paired_audio"},
                                              phoneme_vocabulary=vocabulary)
    validation_indices = pilot_indices(validation_dataset, cfg, "generalization", "validation")
    args.output.mkdir(parents=True, exist_ok=True)
    complete = args.output / "checkpoint.pt"; progress = args.output / "training_state.pt"
    if complete.exists():
        finished = torch.load(complete, map_location="cpu", weights_only=False)
        if finished.get("artifact_hashes") != artifact_hashes:
            raise RuntimeError("completed native renderer checkpoint artifact provenance changed")
        if not finished.get("gate") or not all(finished["gate"].values()):
            raise RuntimeError("completed native renderer has no passing validation-content gate")
        print(json.dumps({"status": "already_completed", "checkpoint": str(complete)})); dataset.close(); validation_dataset.close(); return 0
    torch.manual_seed(31)
    loader_generator = torch.Generator().manual_seed(31)
    loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=True, generator=loader_generator,
                        collate_fn=homogeneous_collate)
    validation_loader = DataLoader(Subset(validation_dataset, validation_indices), batch_size=8, shuffle=False,
                                   collate_fn=homogeneous_collate)
    train_template = relative_native_template(dataset, indices)
    target_device = device()
    model = DurationConditionedNativeRenderer().to(target_device); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    completed = 0; history = []
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        if int(saved.get("maximum_steps", -1)) != int(args.max_steps):
            raise RuntimeError("native renderer partial checkpoint has a different --max-steps")
        if saved.get("artifact_hashes") != artifact_hashes:
            raise RuntimeError("native renderer partial checkpoint artifact provenance changed")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); optimizer_to(optimizer, target_device)
        completed = int(saved["completed"])
        history = list(saved.get("history", [])); print(json.dumps({"status": "resumed", "completed_steps": completed}))
    iterator = iter(loader)
    for _ in range(completed):
        try: next(iterator)
        except StopIteration: iterator = iter(loader); next(iterator)
    if progress.exists() and saved.get("torch_rng_state") is not None:
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.backends.mps.is_available() and saved.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(saved["mps_rng_state"])
    try:
        for step in range(completed + 1, int(args.max_steps) + 1):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            batch = {key: value.to(target_device) if torch.is_tensor(value) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction, prediction_mask = model(batch["content_mfcc"], batch["audio_duration_frames"])
            if not torch.equal(prediction_mask, batch["native_audio_mask"]):
                raise RuntimeError("renderer duration contract differs from cached native audio mask")
            loss = native_loss(prediction, batch["native_speecht5_mel"], prediction_mask)
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite native renderer loss at step {step}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); completed = step
            if step == 1 or step % 100 == 0: history.append({"step": step, "native_mel_loss": float(loss.detach())})
            if step % int(args.checkpoint_every) == 0:
                atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed": completed,
                             "maximum_steps": int(args.max_steps), "history": history,
                             "artifact_hashes": artifact_hashes, "torch_rng_state": torch.get_rng_state(),
                             "mps_rng_state": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}, progress)
    except KeyboardInterrupt:
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed": completed,
                     "maximum_steps": int(args.max_steps), "history": history,
                     "artifact_hashes": artifact_hashes, "torch_rng_state": torch.get_rng_state(),
                     "mps_rng_state": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}, progress)
        dataset.close(); validation_dataset.close(); print(json.dumps({"status": "interrupted_resumable", "completed_steps": completed})); return 130
    validation = evaluate(model, validation_loader, train_template, target_device)
    gate = {"native_mel_improvement": validation["native_mel_improvement"] >=
            float(cfg["native_audio"]["renderer_mel_improvement_min"])}
    atomic_save({"model": model.state_dict(), "model_config": {}, "train_role": "train", "steps": int(args.max_steps),
                 "artifact_hashes": artifact_hashes,
                 "validation": validation, "gate": gate,
                 "warning": "audio-only renderer; no EEG checkpoint is included"}, complete)
    (args.output / "metrics.json").write_text(json.dumps({"history": history, "pairs": len(indices),
                                                           "validation":validation,"gate":gate}, indent=2) + "\n")
    dataset.close(); validation_dataset.close()
    print(json.dumps({"output": str(args.output), "pairs": len(indices), "history": history[-1],
                      "validation":validation,"gate":gate}, indent=2)); return 0 if all(gate.values()) else 2


if __name__ == "__main__": raise SystemExit(main())
