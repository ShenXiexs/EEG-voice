#!/usr/bin/env python3
"""Train the independent v0730 audio renderer and EEG C/P model without labels."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0730.data import CPDataset, collate, load_prepared, text_anchor
from src.open_vocab_0730.model import CPMelRenderer, ContentProsodyEEG
from src.open_vocab_0730.runtime import default_device, load_config, move_batch, resolve_config_path, seed_everything, write_json


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v0730 explicit C/P modules")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("renderer", "eeg", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument("--fresh", action="store_true", help="Remove only v0730 renderer/EEG run outputs before training")
    return parser.parse_args()


def loader(dataset: CPDataset, *, batch_size: int, train: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, collate_fn=collate, num_workers=0)


def checkpoint_path(config_path: Path, cfg: dict[str, Any], phase: str) -> Path:
    key = "renderer_checkpoint" if phase == "renderer" else "eeg_checkpoint"
    return resolve_config_path(config_path, cfg["paths"][key])


def reset_training_outputs(config_path: Path, cfg: dict[str, Any]) -> None:
    """Clear interrupted v0730 training state without touching cache, vocoder, or older versions."""
    root = resolve_config_path(config_path, cfg["paths"]["output_root"])
    targets = {
        checkpoint_path(config_path, cfg, "renderer").parents[1],
        checkpoint_path(config_path, cfg, "eeg").parents[1],
    }
    for target in targets:
        if target.parent != root or target.name not in {"renderer", "eeg_cp"}:
            raise ValueError(f"refusing unsafe v0730 reset target: {target}")
        if target.exists():
            shutil.rmtree(target)
    run_manifest = resolve_config_path(config_path, cfg["paths"]["run_manifest"])
    if run_manifest.is_file():
        run_manifest.unlink()


def save_checkpoint(path: Path, model: torch.nn.Module, *, epoch: int, score: float, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "openvoice-0730-checkpoint-v1", "epoch": epoch, "score": score, "state_dict": model.state_dict(), "extra": extra}, path)


def load_renderer(config_path: Path, cfg: dict[str, Any], device: torch.device) -> CPMelRenderer:
    model = CPMelRenderer(codebook_size=int(cfg["content"]["codebook_size"]), dimension=int(cfg["model"]["dimension"]), mel_frames=int(cfg["audio"]["mel_frames"]), mel_bins=int(cfg["audio"]["mel_bins"])).to(device)
    path = checkpoint_path(config_path, cfg, "renderer")
    raw = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.eval()


def prosody_statistics(dataset: CPDataset) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.stack([dataset[index]["prosody"] for index in range(len(dataset))])
    continuous = np.concatenate((values[:, :2], values[:, 34:]), axis=1)
    return torch.from_numpy(continuous.mean(0).astype(np.float32)), torch.from_numpy(np.maximum(continuous.std(0), 1e-4).astype(np.float32))


def p_loss(state: Any, target: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    duration_loudness = torch.stack((state.duration, state.loudness), dim=1)
    continuous_pred = torch.cat((duration_loudness, state.envelope), dim=1)
    continuous_target = torch.cat((target[:, :2], target[:, 34:]), dim=1)
    regression = F.smooth_l1_loss((continuous_pred - mean) / scale, (continuous_target - mean) / scale)
    activity = F.binary_cross_entropy_with_logits(state.activity_logits, target[:, 2:34])
    return regression + activity, {"p_regression": float(regression.detach()), "p_activity": float(activity.detach())}


def frozen_audio_tokens(hubert: torch.Tensor, hubert_mask: torch.Tensor, codebook: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen HuBERT audio tower, reduced by fit-only PCA and resampled to 16 tokens."""
    tokens = (hubert - codebook["pca_mean"]) @ codebook["pca_components"].T / codebook["pca_scale"]
    tokens = F.interpolate(tokens.transpose(1, 2), size=16, mode="linear", align_corners=False).transpose(1, 2)
    mask = F.interpolate(hubert_mask.float().unsqueeze(1), size=16, mode="nearest").squeeze(1).bool()
    return tokens, mask


def openai_style_token_clip(eeg_tokens: torch.Tensor, audio_tokens: torch.Tensor, mask: torch.Tensor, logit_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric CLIP logits for utterance and token pairs (diagonal positives)."""
    eeg_tokens = F.normalize(eeg_tokens, dim=-1)
    audio_tokens = F.normalize(audio_tokens, dim=-1)
    weights = mask.to(eeg_tokens.dtype).unsqueeze(-1)
    eeg_global = F.normalize((eeg_tokens * weights).sum(1) / weights.sum(1).clamp_min(1), dim=-1)
    audio_global = F.normalize((audio_tokens * weights).sum(1) / weights.sum(1).clamp_min(1), dim=-1)
    scale = logit_scale.clamp(max=math.log(100.0)).exp()
    global_logits = scale * eeg_global @ audio_global.T
    global_target = torch.arange(len(global_logits), device=global_logits.device)
    global_loss = 0.5 * (F.cross_entropy(global_logits, global_target) + F.cross_entropy(global_logits.T, global_target))
    selected_eeg, selected_audio = eeg_tokens[mask], audio_tokens[mask]
    local_logits = scale * selected_eeg @ selected_audio.T
    local_target = torch.arange(len(local_logits), device=local_logits.device)
    local_loss = 0.5 * (F.cross_entropy(local_logits, local_target) + F.cross_entropy(local_logits.T, local_target))
    return global_loss, local_loss


def multi_positive_text_clip(clip_tokens: torch.Tensor, labels: list[str], references: dict[str, str], *, temperature: float) -> torch.Tensor:
    anchors, available = text_anchor(labels, references, dimension=int(clip_tokens.shape[-1]))
    if not available.any():
        return torch.zeros((), device=clip_tokens.device)
    eeg = F.normalize(clip_tokens.mean(1), dim=-1)
    text = F.normalize(torch.from_numpy(anchors).to(clip_tokens.device), dim=-1)
    scores = eeg @ text.T / temperature
    normalized = [str(value).strip().strip("/").lower() for value in labels]
    mask = torch.tensor([[available[row] and available[column] and normalized[row] == normalized[column] for column in range(len(labels))] for row in range(len(labels))], device=clip_tokens.device, dtype=torch.bool)
    valid = mask.any(1)
    if not valid.any():
        return torch.zeros((), device=clip_tokens.device)
    return -(torch.logsumexp(scores.masked_fill(~mask, float("-inf")), dim=1)[valid] - torch.logsumexp(scores, dim=1)[valid]).mean()


def content_loss(state: Any, batch: dict[str, Any], codebook: dict[str, torch.Tensor], references: dict[str, str], logit_scale: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(state.content_logits.reshape(-1, state.content_logits.shape[-1]), batch["content_tokens"].reshape(-1))
    audio, mask = frozen_audio_tokens(batch["hubert"], batch["hubert_mask"], codebook)
    global_clip, local_clip = openai_style_token_clip(state.content_clip_tokens, audio, mask, logit_scale)
    text = multi_positive_text_clip(state.content_clip_tokens, batch["label"], references, temperature=0.08)
    total = 0.55 * global_clip + 0.25 * local_clip + 0.15 * ce + 0.05 * text
    return total, {"c_ce": float(ce.detach()), "c_global_clip": float(global_clip.detach()), "c_token_clip": float(local_clip.detach()), "c_text_clip": float(text.detach())}


def train_renderer(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, args: argparse.Namespace) -> None:
    train = CPDataset(records, ("fit",))
    valid = CPDataset(records, ("subject_holdout_seen",))
    model = CPMelRenderer(codebook_size=int(cfg["content"]["codebook_size"]), dimension=int(cfg["model"]["dimension"]), mel_frames=int(cfg["audio"]["mel_frames"]), mel_bins=int(cfg["audio"]["mel_bins"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["renderer_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best = math.inf
    stale = 0
    started = time.monotonic()
    history = checkpoint_path(config_path, cfg, "renderer").parent.parent / "metrics" / "training.jsonl"
    total = int(args.epochs or cfg["training"]["renderer_epochs"])
    for epoch in range(total):
        model.train(); values = []
        for step, batch in enumerate(loader(train, batch_size=int(cfg["training"]["renderer_batch_size"]), train=True)):
            batch = move_batch(batch, device)
            loss = F.smooth_l1_loss(model(batch["content_tokens"], batch["prosody"]), batch["mel"])
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"])); optimizer.step()
            values.append(float(loss.detach()))
            if args.smoke_steps and step + 1 >= args.smoke_steps:
                break
        model.eval(); validation = []
        with torch.no_grad():
            for step, batch in enumerate(loader(valid, batch_size=int(cfg["training"]["renderer_batch_size"]), train=False)):
                batch = move_batch(batch, device)
                validation.append(float(F.smooth_l1_loss(model(batch["content_tokens"], batch["prosody"]), batch["mel"]).detach()))
                if args.smoke_steps and step + 1 >= args.smoke_steps:
                    break
        score = float(np.mean(validation)); record = {"epoch": epoch + 1, "train_loss": float(np.mean(values)), "validation_loss": score, "elapsed_seconds": time.monotonic() - started}
        history.parent.mkdir(parents=True, exist_ok=True); history.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
        print(f"[0730 renderer] epoch={epoch + 1}/{total} train={record['train_loss']:.5f} valid={score:.5f} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        if score < best:
            best = score; stale = 0; save_checkpoint(checkpoint_path(config_path, cfg, "renderer"), model, epoch=epoch, score=score, extra={"fit_role": "fit", "validation_role": "subject_holdout_seen"})
        else:
            stale += 1
        if stale >= int(cfg["training"]["renderer_patience"]) or (args.wall_hours and time.monotonic() - started >= args.wall_hours * 3600):
            break


def train_eeg(config_path: Path, cfg: dict[str, Any], records: Any, device: torch.device, args: argparse.Namespace) -> None:
    train = CPDataset(records, ("fit",))
    valid = CPDataset(records, ("subject_holdout_seen",))
    mean, scale = prosody_statistics(train); mean, scale = mean.to(device), scale.to(device)
    codebook = {key: torch.from_numpy(value).to(device) for key, value in records.codebook.items() if key in {"pca_mean", "pca_components", "pca_scale"}}
    references = {str(key).strip().strip("/").lower(): str(value) for key, value in cfg.get("text_reference", {}).items()}
    model = ContentProsodyEEG(codebook_size=int(cfg["content"]["codebook_size"]), dimension=int(cfg["model"]["dimension"]), heads=int(cfg["model"]["heads"]), layers=int(cfg["model"]["layers"]), content_steps=int(cfg["content"]["steps"]), prosody_steps=32, dropout=float(cfg["model"]["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["eeg_lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best = math.inf; stale = 0; started = time.monotonic(); total = int(args.epochs or cfg["training"]["eeg_epochs"])
    history = checkpoint_path(config_path, cfg, "eeg").parent.parent / "metrics" / "training.jsonl"
    for epoch in range(total):
        model.train(); values = []
        for step, batch in enumerate(loader(train, batch_size=int(cfg["training"]["eeg_batch_size"]), train=True)):
            batch = move_batch(batch, device)
            state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
            c, c_metrics = content_loss(state, batch, codebook, references, model.clip_logit_scale)
            p, _ = p_loss(state, batch["prosody"], mean, scale)
            loss = 0.5 * (c + p)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"])); optimizer.step()
            values.append(float(loss.detach()))
            if args.smoke_steps and step + 1 >= args.smoke_steps:
                break
        model.eval(); validation = []
        with torch.no_grad():
            for step, batch in enumerate(loader(valid, batch_size=int(cfg["training"]["eeg_batch_size"]), train=False)):
                batch = move_batch(batch, device)
                state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
                c, _ = content_loss(state, batch, codebook, references, model.clip_logit_scale)
                p, _ = p_loss(state, batch["prosody"], mean, scale)
                validation.append(float(0.5 * (c + p)))
                if args.smoke_steps and step + 1 >= args.smoke_steps:
                    break
        score = float(np.mean(validation)); record = {"epoch": epoch + 1, "train_loss": float(np.mean(values)), "validation_loss": score, "elapsed_seconds": time.monotonic() - started}
        history.parent.mkdir(parents=True, exist_ok=True); history.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
        print(f"[0730 eeg] epoch={epoch + 1}/{total} train={record['train_loss']:.5f} valid={score:.5f} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
        if score < best:
            best = score; stale = 0; save_checkpoint(checkpoint_path(config_path, cfg, "eeg"), model, epoch=epoch, score=score, extra={"prosody_mean": mean.cpu(), "prosody_scale": scale.cpu(), "fit_role": "fit", "validation_role": "subject_holdout_seen", "content_loss": {"global_audio_clip": 0.55, "token_audio_clip": 0.25, "token_ce": 0.15, "text_anchor_clip": 0.05}, "labels_as_forward_input": False, "text_reference_labels": sorted(references)})
        else:
            stale += 1
        if stale >= int(cfg["training"]["eeg_patience"]) or (args.wall_hours and time.monotonic() - started >= args.wall_hours * 3600):
            break


def main() -> None:
    args = parse(); config_path, cfg = load_config(args.config); seed_everything(int(cfg["training"]["seed"])); device = default_device(args.device)
    if args.fresh:
        reset_training_outputs(config_path, cfg)
    records = load_prepared(resolve_config_path(config_path, cfg["paths"]["prepared_cache"]))
    configured_renderer_hours = float(cfg["training"]["renderer_wall_hours"])
    configured_eeg_hours = float(cfg["training"]["eeg_wall_hours"])
    if args.wall_hours is None:
        renderer_hours, eeg_hours = configured_renderer_hours, configured_eeg_hours
    else:
        # A single requested wall-clock budget is allocated once, never per phase.
        renderer_hours = min(configured_renderer_hours, float(args.wall_hours))
        eeg_hours = min(configured_eeg_hours, max(0.0, float(args.wall_hours) - renderer_hours))
    renderer_args, eeg_args = copy.copy(args), copy.copy(args)
    renderer_args.wall_hours, eeg_args.wall_hours = renderer_hours, eeg_hours
    if args.phase in {"renderer", "all"}:
        train_renderer(config_path, cfg, records, device, renderer_args)
    if args.phase in {"eeg", "all"}:
        if not checkpoint_path(config_path, cfg, "renderer").is_file():
            raise FileNotFoundError("EEG training requires a saved v0730 renderer checkpoint")
        train_eeg(config_path, cfg, records, device, eeg_args)
    write_json(resolve_config_path(config_path, cfg["paths"]["run_manifest"]), {"schema_version": "openvoice-0730-run-v1", "prepared_cache": str(resolve_config_path(config_path, cfg["paths"]["prepared_cache"])), "renderer_checkpoint": str(checkpoint_path(config_path, cfg, "renderer")), "eeg_checkpoint": str(checkpoint_path(config_path, cfg, "eeg")), "device": str(device), "seed": int(cfg["training"]["seed"]), "allocated_training_wall_hours": {"renderer": renderer_hours, "eeg": eeg_hours, "total": renderer_hours + eeg_hours}})


if __name__ == "__main__":
    main()
