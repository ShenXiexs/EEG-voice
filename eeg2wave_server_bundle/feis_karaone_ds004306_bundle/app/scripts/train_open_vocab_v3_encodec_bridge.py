#!/usr/bin/env python3
"""Train only the frozen-renderer v3 EnCodec-bridge stages.

There is deliberately no validation/test phase in this file.  It can only
train E1, C1, M0, and M1 on fit data and writes auditable predictions for the
separate evaluation/export scripts.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_v3.data import V3Dataset, collate, load_prepared, time_shuffled_eeg, channel_shuffled_eeg
from src.open_vocab_v3.encodec_bridge import (
    PREPARATION_SCHEMA, SCHEMA, AudioCEncoder, ContinuousEnCodecBridge,
    EEGCEncoder, FrozenEnCodecRenderer, SharedContentMFCCDecoder,
    CState, envelope_loss, masked_token_infonce, multiresolution_stft_loss,
    temporal_delta, variance_covariance_loss,
)
from src.open_vocab_v3.runtime import checkpoint_schema, default_device, load_config, move_batch, output_path, seed_everything, sha256_file


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("bridge", "audio_c", "m0", "m1"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deadline-epoch", type=float, default=0.0)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def expired(args: argparse.Namespace) -> bool:
    return bool(args.deadline_epoch and time.time() >= args.deadline_epoch)


def save_checkpoint(path: Path, schema: str, modules: dict[str, nn.Module], **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": schema, "modules": {name: module.state_dict() for name, module in modules.items()}, "extra": extra}, path)


def load_checkpoint(path: Path, schema: str, modules: dict[str, nn.Module], device: torch.device) -> dict[str, Any]:
    raw = torch.load(path, map_location=device, weights_only=False)
    if raw.get("schema_version") != schema:
        raise RuntimeError(f"stale bridge-v2 checkpoint rejected: {path} has {raw.get('schema_version')!r}")
    for name, module in modules.items():
        module.load_state_dict(raw["modules"][name], strict=True)
    return raw


class TokenDataset(Dataset):
    def __init__(self, base: Dataset, cache: dict[str, np.ndarray], mapping: dict[int, int]):
        self.base, self.cache, self.mapping = base, cache, mapping

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict[str, Any]:
        result = dict(self.base[item])
        source = int(result["source_index"])
        if source not in self.mapping:
            raise RuntimeError(f"fit source index absent from frozen EnCodec bridge cache: {source}")
        cache_index = self.mapping[source]
        for name in ("encodec_codes", "encodec_mask", "target_latent", "waveform_16k", "waveform_mask", "waveform_samples"):
            result[name] = self.cache[name][cache_index]
        return result


def token_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    result = collate(items)
    for name in ("encodec_codes", "encodec_mask", "target_latent", "waveform_16k", "waveform_mask", "waveform_samples"):
        result[name] = torch.as_tensor(np.stack([item[name] for item in items]))
    result["encodec_codes"] = result["encodec_codes"].long()
    result["encodec_mask"] = result["encodec_mask"].bool()
    result["waveform_mask"] = result["waveform_mask"].bool()
    return result


def load_cache(cp: Path, cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    raw = np.load(output_path(cp, cfg, "encodec_cache"), allow_pickle=False)
    if str(raw["schema"].item()) != SCHEMA:
        raise RuntimeError("stale non-bridge EnCodec cache rejected")
    prepared = output_path(cp, cfg, "prepared_cache")
    if str(raw["prepared_cache_sha256"].item()) != sha256_file(prepared):
        raise RuntimeError("bridge EnCodec cache lineage differs from prepared cache")
    cache = {name: np.asarray(raw[name]) for name in raw.files}
    return cache, {int(source): position for position, source in enumerate(cache["source_indices"].tolist())}


def make_models(cfg: dict[str, Any], device: torch.device, subject_count: int) -> tuple[AudioCEncoder, SharedContentMFCCDecoder, EEGCEncoder, ContinuousEnCodecBridge]:
    model = cfg["model"]
    audio = AudioCEncoder(
        embedding_dimension=int(model["codebook_embedding_dimension"]), dimension=int(model["content_dimension"]),
        heads=int(model["heads"]), stem_layers=int(model["content_stem_layers"]), local_layers=int(model["content_branch_layers"]),
        dropout=float(model["dropout"]), speakers=subject_count, global_gradient_scale=float(model["global_gradient_scale"]),
    ).to(device)
    decoder = SharedContentMFCCDecoder(dimension=int(model["content_dimension"]), heads=int(model["heads"]), layers=int(model["decoder_layers"]), dropout=float(model["dropout"])).to(device)
    eeg = EEGCEncoder(dimension=int(model["eeg_dimension"]), heads=int(model["heads"]), layers=int(model["eeg_layers"]), local_layers=int(model["content_branch_layers"]), dropout=float(model["dropout"])).to(device)
    bridge = ContinuousEnCodecBridge(latent_dimension=int(model["encodec_latent_dimension"]), voice_dimension=int(cfg["speaker"]["embedding_dimension"]), dimension=int(model["bridge_dimension"]), blocks=int(model["bridge_blocks"])).to(device)
    return audio, decoder, eeg, bridge


def fit_indices(records, *, dev: bool | None = None) -> np.ndarray:
    selector = (records.roles == "fit") & records.arrays["fit_eligible"].astype(bool)
    if dev is not None:
        selector &= records.arrays["fit_internal_dev"].astype(bool) if dev else ~records.arrays["fit_internal_dev"].astype(bool)
    return np.flatnonzero(selector)


def base_subset(records, indices: np.ndarray) -> Dataset:
    base = V3Dataset(records, ("fit",), eligible_only=True)
    positions = {int(source): position for position, source in enumerate(base.indices)}
    return Subset(base, [positions[int(index)] for index in indices.tolist()])


def micro_indices(records, cfg: dict[str, Any]) -> np.ndarray:
    fit = fit_indices(records, dev=False)
    subject = str(cfg["micro_gate"]["subject"])
    per_label = int(cfg["micro_gate"]["per_label"])
    selected: list[int] = []
    for label in sorted(set(records.arrays["labels"][fit].astype(str).tolist())):
        candidates = [int(index) for index in fit if str(records.arrays["subjects"][index]) == subject and str(records.arrays["labels"][index]) == label]
        selected += sorted(candidates, key=lambda index: str(records.arrays["sample_keys"][index]))[:per_label]
    if len(selected) != 50:
        raise RuntimeError(f"M0/M1 requires exactly 50 MM05 pairs, found {len(selected)}")
    return np.asarray(selected, dtype=np.int32)


def micro_generalization_folds(indices: np.ndarray, sample_keys: np.ndarray,
                               labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Five deterministic one-held-trial-per-label folds for the 50-pair set."""
    values = np.asarray(indices, dtype=np.int32).tolist()
    if len(values) != 50:
        raise ValueError("M1 fold construction requires exactly 50 pairs")
    groups: dict[str, list[int]] = {}
    for index in values:
        groups.setdefault(str(labels[index]), []).append(index)
    if len(groups) != 10 or any(len(value) != 5 for value in groups.values()):
        raise ValueError("M1 requires five selected trials for each of ten labels")
    ordered = {label: sorted(value, key=lambda index: str(sample_keys[index])) for label, value in groups.items()}
    all_ordered = [index for label in sorted(ordered) for index in ordered[label]]
    result=[]
    for fold in range(5):
        held=np.asarray([ordered[label][fold] for label in sorted(ordered)],dtype=np.int32)
        held_set=set(held.tolist())
        train=np.asarray([index for index in all_ordered if index not in held_set],dtype=np.int32)
        result.append((train,held))
    return result


class LabelGroupedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Every audio-C batch has multiple labels and ≥2 trials where possible."""
    def __init__(self, dataset: Dataset, batch_size: int, seed: int):
        self.dataset, self.batch_size, self.seed = dataset, int(batch_size), int(seed)
        groups: dict[str, list[int]] = {}
        for position in range(len(dataset)):
            groups.setdefault(str(dataset[position]["label"]), []).append(position)
        self.groups = {key: sorted(value) for key, value in groups.items()}
        self.labels = sorted(self.groups)

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        cursors = {key: 0 for key in self.labels}
        ordered = {key: rng.permutation(value).tolist() for key, value in self.groups.items()}
        for step in range(len(self)):
            chosen = [self.labels[(step + offset) % len(self.labels)] for offset in range(max(2, self.batch_size // 2))]
            batch: list[int] = []
            for label in chosen:
                for _ in range(2):
                    values = ordered[label]
                    batch.append(values[cursors[label] % len(values)])
                    cursors[label] += 1
                    if len(batch) == self.batch_size:
                        break
                if len(batch) == self.batch_size:
                    break
            yield batch


def loader(dataset: Dataset, cfg: dict[str, Any], *, train: bool, grouped: bool = False) -> DataLoader:
    if train:
        batch = int(cfg["training"]["audio_batch_size"] if grouped else cfg["training"]["eeg_batch_size"])
        sampler = LabelGroupedBatchSampler(dataset, batch, int(cfg["training"]["seed"])) if grouped else None
        if sampler is not None:
            return DataLoader(dataset, batch_sampler=sampler, collate_fn=token_collate, num_workers=0)
        return DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=token_collate, num_workers=0)
    return DataLoader(dataset, batch_size=int(cfg["evaluation"]["batch_size"]), shuffle=False, collate_fn=token_collate, num_workers=0)


def _masked_latent_l1(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(left.dtype).unsqueeze(1)
    return (torch.abs(left - right) * weight).sum() / (weight.sum().clamp_min(1) * left.shape[1])


def _wave_shape(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] < target.shape[-1]:
        return F.pad(value, (0, target.shape[-1] - value.shape[-1]))
    return value[..., :target.shape[-1]]


def bridge_loss(bridge: ContinuousEnCodecBridge, renderer: FrozenEnCodecRenderer, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    latent = bridge(batch["content_mfcc"].float(), batch["p_base"].float(), batch["speaker_reference"].float(), batch["duration_fraction"].float())
    codes, quantized, straight = renderer.quantize_st(latent)
    target = batch["target_latent"].float()
    mask = batch["encodec_mask"]
    latent_l1 = _masked_latent_l1(latent, target, mask)
    cosine = 1 - F.cosine_similarity(latent * mask.unsqueeze(1), target * mask.unsqueeze(1), dim=1).mean()
    commitment = _masked_latent_l1(latent, quantized.detach(), mask)
    generated = _wave_shape(renderer.render_st(straight), batch["waveform_16k"].float())
    spectral = multiresolution_stft_loss(generated, batch["waveform_16k"].float(), batch["waveform_mask"])
    envelope = envelope_loss(generated, batch["waveform_16k"].float(), batch["waveform_mask"])
    predicted_hubert = F.interpolate(bridge.hubert_token(straight), size=161, mode="linear", align_corners=False).transpose(1, 2)
    target_hubert = F.interpolate(batch["hubert"].float().transpose(1, 2), size=161, mode="linear", align_corners=False).transpose(1, 2)
    hubert = 1 - F.cosine_similarity(predicted_hubert, target_hubert.detach(), dim=-1).mean()
    loss = .30 * latent_l1 + .10 * cosine + .10 * commitment + .20 * spectral + .15 * hubert + .15 * envelope
    return loss, {"latent_l1": float(latent_l1.detach()), "latent_cosine": float(cosine.detach()), "commitment": float(commitment.detach()), "mrstft": float(spectral.detach()), "hubert_distilled": float(hubert.detach()), "envelope": float(envelope.detach()), "codebook_count": float(codes.shape[1])}


def audio_c_loss(audio: AudioCEncoder, decoder: SharedContentMFCCDecoder, batch: dict[str, Any], subject_map: dict[str, int]) -> tuple[torch.Tensor, dict[str, float]]:
    state = audio(batch["encodec_codes"], batch["encodec_mask"])
    target_hubert = F.interpolate(batch["hubert"].float().transpose(1, 2), size=96, mode="linear", align_corners=False).transpose(1, 2)
    hubert_mask = F.interpolate(batch["hubert_mask"].float().unsqueeze(1), size=96, mode="nearest").squeeze(1).bool()
    token_hubert = audio.hubert_token(state.local)
    ot, _ = masked_token_infonce(token_hubert, target_hubert, state.token_mask, hubert_mask, batch["label"])
    content, _ = decoder(state.local, state.token_mask)
    mfcc = F.l1_loss(content, batch["content_mfcc"].float())
    delta = F.l1_loss(temporal_delta(content), temporal_delta(batch["content_mfcc"].float()))
    temporal = 1 - F.cosine_similarity(temporal_delta(content), temporal_delta(batch["content_mfcc"].float()), dim=1).mean()
    variance = variance_covariance_loss(content, batch["content_mfcc"].float())
    global_target = F.normalize(batch["hubert"].float().mean(1), dim=-1)
    global_pred = F.normalize(audio.hubert_global(state.global_embedding), dim=-1)
    logits = global_pred @ global_target.T / .07
    diagonal = torch.arange(len(logits), device=logits.device)
    global_loss = .5 * (F.cross_entropy(logits, diagonal) + F.cross_entropy(logits.T, diagonal))
    subjects = torch.tensor([subject_map[str(value)] for value in batch["subject"]], device=logits.device)
    speaker = F.cross_entropy(state.speaker_logits, subjects)
    loss = .40 * ot + .25 * mfcc + .15 * delta + .10 * temporal + .10 * variance + .15 * global_loss + .05 * speaker
    return loss, {"hubert_token_ot": float(ot.detach()), "mfcc": float(mfcc.detach()), "delta": float(delta.detach()), "temporal": float(temporal.detach()), "variance_covariance": float(variance.detach()), "hubert_global": float(global_loss.detach()), "speaker_adversarial": float(speaker.detach())}


def eeg_loss(eeg: EEGCEncoder, decoder: SharedContentMFCCDecoder, audio: AudioCEncoder, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    state = eeg(batch["eeg"].float(), batch["channel_xyz"].float(), batch["channel_mask"], batch["time_mask"])
    with torch.no_grad():
        teacher = audio(batch["encodec_codes"], batch["encodec_mask"])
    predicted, _ = decoder(state.local, state.token_mask)
    target = batch["content_mfcc"].float()
    mfcc = F.l1_loss(predicted, target)
    delta = F.l1_loss(temporal_delta(predicted), temporal_delta(target))
    token, _ = masked_token_infonce(state.local, teacher.local, state.token_mask, teacher.token_mask, batch["label"])
    global_logits = state.global_embedding @ teacher.global_embedding.T / .07
    diagonal = torch.arange(len(global_logits), device=global_logits.device)
    global_loss = .5 * (F.cross_entropy(global_logits, diagonal) + F.cross_entropy(global_logits.T, diagonal))
    return .55 * mfcc + .20 * delta + .15 * token + .10 * global_loss, predicted


def train_loop(*, model_modules: dict[str, nn.Module], optimizer: torch.optim.Optimizer, train_loader: DataLoader,
               dev_loader: DataLoader | None, loss_fn, epochs: int, patience: int, checkpoint: Path,
               schema: str, args: argparse.Namespace, device: torch.device, label: str) -> None:
    best, stale, history = math.inf, 0, []
    for epoch in range(int(epochs)):
        if expired(args): break
        for module in model_modules.values(): module.train()
        values: list[float] = []
        for step, batch in enumerate(train_loader):
            if expired(args): break
            batch = move_batch(batch, device); loss, parts = loss_fn(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {label} loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_([p for m in model_modules.values() for p in m.parameters() if p.requires_grad], 1.0); optimizer.step(); values.append(float(loss.detach()))
            if args.smoke_steps and step + 1 >= args.smoke_steps: break
        if not values: break
        if dev_loader is None:
            dev = float(np.mean(values))
        else:
            for module in model_modules.values(): module.eval()
            dev_values = []
            with torch.no_grad():
                for batch in dev_loader:
                    batch = move_batch(batch, device); value, _ = loss_fn(batch); dev_values.append(float(value))
            dev = float(np.mean(dev_values)) if dev_values else math.inf
        history.append({"epoch": epoch + 1, "train": float(np.mean(values)), "dev": dev, "parts": parts})
        if dev < best:
            best, stale = dev, 0
            save_checkpoint(checkpoint, schema, model_modules, history=history, best_dev=best)
        else:
            stale += 1
        print(f"[v3 bridge {label}] epoch={epoch+1}/{epochs} train={np.mean(values):.5f} dev={dev:.5f}", flush=True)
        if stale >= int(patience) or args.smoke_steps: break


def micro_metrics(prediction: np.ndarray, target: np.ndarray, labels: list[str]) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32); target = np.asarray(target, dtype=np.float32)
    distance = ((prediction[:, None] - target[None]) ** 2).mean((2, 3))
    label_set = sorted(set(labels)); label_distance = np.stack([distance[:, [i for i, label in enumerate(labels) if label == name]].mean(1) for name in label_set], 1)
    label_r1 = float(np.mean(np.asarray(label_set)[label_distance.argmin(1)] == np.asarray(labels)))
    within = []
    for row, label in enumerate(labels):
        candidates = [i for i, other in enumerate(labels) if other == label]
        within.append(int(candidates[int(distance[row, candidates].argmin())] == row))
    template = np.stack([target[[i for i, other in enumerate(labels) if other == label]].mean(0) for label in labels])
    return {"label_top1": label_r1, "paired_r1": float(np.mean(within)), "template_improvement": float(1 - np.mean((prediction-target)**2) / max(float(np.mean((template-target)**2)),1e-8)), "variance_ratio": float(prediction.var() / max(float(target.var()),1e-8))}


def write_predictions(path: Path, *, source_indices: list[int], labels: list[str], prediction: list[np.ndarray], target: list[np.ndarray], controls: dict[str, list[np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=np.asarray(SCHEMA), source_indices=np.asarray(source_indices, dtype=np.int32), labels=np.asarray(labels), prediction=np.stack(prediction).astype(np.float32), target=np.stack(target).astype(np.float32), **{name: np.stack(value).astype(np.float32) for name, value in controls.items()})


def train_bridge(cp, cfg, records, device, args) -> None:
    cache, mapping = load_cache(cp, cfg); train = TokenDataset(base_subset(records, fit_indices(records, dev=False)), cache, mapping); dev = TokenDataset(base_subset(records, fit_indices(records, dev=True)), cache, mapping)
    subjects = len(set(records.arrays["subjects"][fit_indices(records, dev=False)].astype(str).tolist()))
    _, _, _, bridge = make_models(cfg, device, subjects)
    renderer = FrozenEnCodecRenderer(output_path(cp, cfg, "encodec_root"), device=device, bandwidth=float(cfg["audio"]["encodec_bandwidth"]))
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=float(cfg["training"]["bridge_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    train_loop(model_modules={"bridge": bridge}, optimizer=optimizer, train_loader=loader(train,cfg,train=True), dev_loader=loader(dev,cfg,train=False), loss_fn=lambda b: bridge_loss(bridge,renderer,b), epochs=int(cfg["training"]["bridge_epochs"]), patience=int(cfg["training"]["bridge_patience"]), checkpoint=output_path(cp,cfg,"bridge_checkpoint"), schema=checkpoint_schema(cfg,"bridge"), args=args, device=device, label="E1 bridge")


def train_audio_c(cp, cfg, records, device, args) -> None:
    cache, mapping = load_cache(cp, cfg); train = TokenDataset(base_subset(records, fit_indices(records, dev=False)), cache, mapping); dev = TokenDataset(base_subset(records, fit_indices(records, dev=True)), cache, mapping)
    subject_names = sorted(set(records.arrays["subjects"][fit_indices(records, dev=False)].astype(str).tolist())); subject_map = {name: index for index, name in enumerate(subject_names)}
    audio, decoder, _, _ = make_models(cfg, device, len(subject_names))
    optimizer = torch.optim.AdamW(list(audio.parameters()) + list(decoder.parameters()), lr=float(cfg["training"]["audio_c_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    train_loop(model_modules={"audio": audio, "decoder": decoder}, optimizer=optimizer, train_loader=loader(train,cfg,train=True,grouped=True), dev_loader=loader(dev,cfg,train=False), loss_fn=lambda b: audio_c_loss(audio,decoder,b,subject_map), epochs=int(cfg["training"]["audio_c_epochs"]), patience=int(cfg["training"]["audio_c_patience"]), checkpoint=output_path(cp,cfg,"audio_c_checkpoint"), schema=checkpoint_schema(cfg,"audio_c"), args=args, device=device, label="C1 audio-C")


def frozen_audio_modules(cp, cfg, records, device):
    subjects = len(set(records.arrays["subjects"][fit_indices(records, dev=False)].astype(str).tolist()))
    audio, decoder, eeg, bridge = make_models(cfg, device, subjects)
    load_checkpoint(output_path(cp,cfg,"audio_c_checkpoint"), checkpoint_schema(cfg,"audio_c"), {"audio":audio,"decoder":decoder}, device)
    load_checkpoint(output_path(cp,cfg,"bridge_checkpoint"), checkpoint_schema(cfg,"bridge"), {"bridge":bridge}, device)
    for module in (audio, decoder, bridge):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    return audio, decoder, eeg, bridge


def _predict_eeg(eeg, decoder, batch, signal=None):
    signal = batch["eeg"].float() if signal is None else signal.float()
    state = eeg(signal, batch["channel_xyz"].float(), batch["channel_mask"], batch["time_mask"])
    return decoder(state.local, state.token_mask)[0]


def train_m0(cp, cfg, records, device, args) -> None:
    cache, mapping = load_cache(cp,cfg); selected = micro_indices(records,cfg); dataset = TokenDataset(base_subset(records, selected),cache,mapping)
    audio, decoder, eeg, _ = frozen_audio_modules(cp,cfg,records,device)
    optimizer=torch.optim.AdamW(eeg.parameters(),lr=float(cfg["training"]["eeg_micro_lr"]),weight_decay=float(cfg["training"]["weight_decay"]))
    def m0_loss(batch):
        loss, prediction = eeg_loss(eeg, decoder, audio, batch)
        return loss, {"mfcc": float(F.l1_loss(prediction, batch["content_mfcc"].float()).detach())}
    train_loop(model_modules={"eeg":eeg},optimizer=optimizer,train_loader=loader(dataset,cfg,train=True),dev_loader=None,loss_fn=m0_loss,epochs=int(cfg["training"]["micro_m0_epochs"]),patience=int(cfg["training"]["micro_m0_patience"]),checkpoint=output_path(cp,cfg,"micro_m0_checkpoint"),schema=checkpoint_schema(cfg,"micro_m0"),args=args,device=device,label="M0 EEG-C")
    load_checkpoint(output_path(cp,cfg,"micro_m0_checkpoint"),checkpoint_schema(cfg,"micro_m0"),{"eeg":eeg},device);eeg.eval();pred=[];target=[];indices=[];labels=[];controls={"zero":[],"time":[],"channel":[]}
    with torch.no_grad():
        for batch in loader(dataset,cfg,train=False):
            batch=move_batch(batch,device);pred.append(_predict_eeg(eeg,decoder,batch).cpu().numpy());target.append(batch["content_mfcc"].cpu().numpy());indices+=batch["source_index"].cpu().tolist();labels+=batch["label"]
            controls["zero"].append(_predict_eeg(eeg,decoder,batch,torch.zeros_like(batch["eeg"])).cpu().numpy());controls["time"].append(_predict_eeg(eeg,decoder,batch,time_shuffled_eeg(batch["eeg"],batch["time_mask"])).cpu().numpy());controls["channel"].append(_predict_eeg(eeg,decoder,batch,channel_shuffled_eeg(batch["eeg"],batch["channel_mask"])).cpu().numpy())
    write_predictions(output_path(cp,cfg,"micro_m0_predictions"),source_indices=indices,labels=labels,prediction=[x for value in pred for x in value],target=[x for value in target for x in value],controls={name:[x for value in values for x in value] for name,values in controls.items()})


def train_m1(cp, cfg, records, device, args) -> None:
    cache, mapping = load_cache(cp,cfg); selected=micro_indices(records,cfg);all_prediction=[];all_target=[];all_labels=[];all_indices=[];controls={"zero":[],"time":[],"channel":[]};fold_states=[]
    for fold,(train,held) in enumerate(micro_generalization_folds(selected,records.arrays["sample_keys"],records.arrays["labels"])):
        audio,decoder,eeg,_=frozen_audio_modules(cp,cfg,records,device);optimizer=torch.optim.AdamW(eeg.parameters(),lr=float(cfg["training"]["eeg_micro_lr"]),weight_decay=float(cfg["training"]["weight_decay"]))
        train_set=TokenDataset(base_subset(records,train),cache,mapping);held_set=TokenDataset(base_subset(records,held),cache,mapping)
        local_args=argparse.Namespace(**vars(args));local_args.smoke_steps=args.smoke_steps
        temporary=output_path(cp,cfg,"micro_m1_checkpoint").with_name(f"fold_{fold}.pt")
        def m1_loss(batch):
            loss, prediction = eeg_loss(eeg, decoder, audio, batch)
            return loss, {"mfcc": float(F.l1_loss(prediction, batch["content_mfcc"].float()).detach())}
        train_loop(model_modules={"eeg":eeg},optimizer=optimizer,train_loader=loader(train_set,cfg,train=True),dev_loader=None,loss_fn=m1_loss,epochs=int(cfg["training"]["micro_m1_epochs"]),patience=int(cfg["training"]["micro_m1_patience"]),checkpoint=temporary,schema=checkpoint_schema(cfg,"micro_m1"),args=local_args,device=device,label=f"M1 fold={fold}")
        load_checkpoint(temporary,checkpoint_schema(cfg,"micro_m1"),{"eeg":eeg},device);eeg.eval();fold_states.append(eeg.state_dict())
        with torch.no_grad():
            for batch in loader(held_set,cfg,train=False):
                batch=move_batch(batch,device);all_prediction += list(_predict_eeg(eeg,decoder,batch).cpu().numpy());all_target += list(batch["content_mfcc"].cpu().numpy());all_indices += batch["source_index"].cpu().tolist();all_labels += batch["label"]
                controls["zero"] += list(_predict_eeg(eeg,decoder,batch,torch.zeros_like(batch["eeg"])).cpu().numpy());controls["time"] += list(_predict_eeg(eeg,decoder,batch,time_shuffled_eeg(batch["eeg"],batch["time_mask"])).cpu().numpy());controls["channel"] += list(_predict_eeg(eeg,decoder,batch,channel_shuffled_eeg(batch["eeg"],batch["channel_mask"])).cpu().numpy())
    checkpoint=output_path(cp,cfg,"micro_m1_checkpoint");checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({"schema_version":checkpoint_schema(cfg,"micro_m1"),"fold_states":fold_states,"fold_protocol":"5x leave-one-trial-per-label; 40 EEG train / 10 held EEG per fold","audio_checkpoint_sha256":sha256_file(output_path(cp,cfg,"audio_c_checkpoint")),"bridge_checkpoint_sha256":sha256_file(output_path(cp,cfg,"bridge_checkpoint"))},checkpoint)
    write_predictions(output_path(cp,cfg,"micro_m1_predictions"),source_indices=all_indices,labels=all_labels,prediction=all_prediction,target=all_target,controls=controls)


def main() -> None:
    args=parse();cp,cfg=load_config(args.config);seed_everything(int(cfg["training"]["seed"]));device=default_device(args.device)
    records=load_prepared(output_path(cp,cfg,"prepared_cache"),expected_schema=PREPARATION_SCHEMA)
    if args.phase=="bridge":train_bridge(cp,cfg,records,device,args)
    elif args.phase=="audio_c":train_audio_c(cp,cfg,records,device,args)
    elif args.phase=="m0":train_m0(cp,cfg,records,device,args)
    else:train_m1(cp,cfg,records,device,args)


if __name__=="__main__":main()
