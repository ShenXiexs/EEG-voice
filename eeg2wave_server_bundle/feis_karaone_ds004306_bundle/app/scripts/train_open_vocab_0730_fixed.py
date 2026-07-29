#!/usr/bin/env python3
"""Train the v0730-fixed renderer and spatial-temporal EEG C/P model."""
from __future__ import annotations

import argparse
import copy
import json
import math
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

from scripts.train_open_vocab_0730 import (
    checkpoint_path,
    frozen_audio_tokens,
    load_renderer,
    multi_positive_text_clip,
    reset_training_outputs,
    train_renderer,
)
from src.open_vocab_0730.data_fixed import CPDataset, collate, load_prepared
from src.open_vocab_0730.model_fixed import ContentProsodyEEGFixed
from src.open_vocab_0730.runtime import (
    default_device,
    load_config,
    move_batch,
    resolve_config_path,
    seed_everything,
    write_json,
)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v0730-fixed C/P modules")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("renderer", "eeg", "all"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def loader(dataset: CPDataset, *, batch_size: int, train: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, collate_fn=collate, num_workers=0
    )


def prosody_statistics(
    dataset: CPDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = np.stack([dataset[index]["prosody"] for index in range(len(dataset))])
    continuous = np.concatenate((values[:, :2], values[:, 34:]), axis=1)
    mean = continuous.mean(0).astype(np.float32)
    scale = np.maximum(continuous.std(0), 1e-4).astype(np.float32)
    activity = values[:, 2:34]
    positives = activity.sum(0)
    negatives = len(activity) - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 10.0).astype(np.float32)
    return torch.from_numpy(mean), torch.from_numpy(scale), torch.from_numpy(pos_weight)


def weighted_envelope_shape(
    prediction: torch.Tensor, target: torch.Tensor, activity: torch.Tensor
) -> torch.Tensor:
    weight = activity.clamp(0.0, 1.0)
    valid = weight.sum(1) >= 2.0
    if not valid.any():
        return prediction.new_zeros(())
    denom = weight.sum(1, keepdim=True).clamp_min(1.0)
    pred_centered = prediction - (prediction * weight).sum(1, keepdim=True) / denom
    target_centered = target - (target * weight).sum(1, keepdim=True) / denom
    numerator = (pred_centered * target_centered * weight).sum(1)
    scale = torch.sqrt(
        (pred_centered.square() * weight).sum(1)
        * (target_centered.square() * weight).sum(1)
    ).clamp_min(1e-6)
    return (1.0 - numerator[valid] / scale[valid]).mean()


def p_loss(
    state: Any,
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted_global = torch.stack((state.duration, state.loudness), dim=1)
    global_regression = F.smooth_l1_loss(
        (predicted_global - mean[:2]) / scale[:2],
        (target[:, :2] - mean[:2]) / scale[:2],
    )
    activity_target = target[:, 2:34]
    activity = F.binary_cross_entropy_with_logits(
        state.activity_logits, activity_target, pos_weight=pos_weight
    )
    envelope_target = target[:, 34:66]
    predicted_z = (state.envelope - mean[2:]) / scale[2:]
    target_z = (envelope_target - mean[2:]) / scale[2:]
    active_weight = activity_target.clamp(0.0, 1.0)
    inactive_weight = 1.0 - active_weight
    element = F.smooth_l1_loss(predicted_z, target_z, reduction="none")
    envelope_active = (element * active_weight).sum() / active_weight.sum().clamp_min(1.0)
    envelope_inactive = (element * inactive_weight).sum() / inactive_weight.sum().clamp_min(1.0)
    envelope_shape = weighted_envelope_shape(
        state.envelope, envelope_target, activity_target
    )
    total = (
        0.50 * global_regression
        + activity
        + envelope_active
        + 0.05 * envelope_inactive
        + 0.25 * envelope_shape
    )
    return total, {
        "p_global": float(global_regression.detach()),
        "p_activity": float(activity.detach()),
        "p_envelope_active": float(envelope_active.detach()),
        "p_envelope_inactive": float(envelope_inactive.detach()),
        "p_envelope_shape": float(envelope_shape.detach()),
    }


def multi_positive_loss(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    valid = positive.any(1)
    if not valid.any():
        return logits.new_zeros(())
    numerator = torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return -(numerator[valid] - denominator[valid]).mean()


def openai_multi_positive_token_clip(
    eeg_tokens: torch.Tensor,
    audio_tokens: torch.Tensor,
    audio_mask: torch.Tensor,
    labels: list[str],
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OpenAI-style dual-tower CLIP with same-label multi-positive pairs.

    The audio tower remains frozen.  Same-label trials are positives rather
    than false negatives; the local objective compares aligned token positions
    without constructing a flattened token-by-token false-negative matrix.
    """
    eeg = F.normalize(eeg_tokens, dim=-1)
    audio = F.normalize(audio_tokens, dim=-1)
    weights = audio_mask.to(eeg.dtype).unsqueeze(-1)
    eeg_global = F.normalize((eeg * weights).sum(1) / weights.sum(1).clamp_min(1.0), dim=-1)
    audio_global = F.normalize((audio * weights).sum(1) / weights.sum(1).clamp_min(1.0), dim=-1)
    scale = logit_scale.clamp(max=math.log(100.0)).exp()
    normalized = [str(value).strip().strip("/").lower() for value in labels]
    positive = torch.tensor(
        [[left == right for right in normalized] for left in normalized],
        dtype=torch.bool,
        device=eeg.device,
    )
    global_logits = scale * eeg_global @ audio_global.T
    global_loss = 0.5 * (
        multi_positive_loss(global_logits, positive)
        + multi_positive_loss(global_logits.T, positive.T)
    )
    token_similarity = torch.einsum("itd,jtd->ijt", eeg, audio)
    pair_weight = audio_mask.to(token_similarity.dtype).unsqueeze(0)
    local_logits = scale * (token_similarity * pair_weight).sum(-1) / pair_weight.sum(-1).clamp_min(1.0)
    local_loss = 0.5 * (
        multi_positive_loss(local_logits, positive)
        + multi_positive_loss(local_logits.T, positive.T)
    )
    return global_loss, local_loss


def content_loss(
    state: Any,
    batch: dict[str, Any],
    codebook: dict[str, torch.Tensor],
    references: dict[str, str],
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(
        state.content_logits.reshape(-1, state.content_logits.shape[-1]),
        batch["content_tokens"].reshape(-1),
    )
    audio, mask = frozen_audio_tokens(batch["hubert"], batch["hubert_mask"], codebook)
    global_clip, local_clip = openai_multi_positive_token_clip(
        state.content_clip_tokens, audio, mask, batch["label"], logit_scale
    )
    text = multi_positive_text_clip(
        state.content_clip_tokens, batch["label"], references, temperature=0.08
    )
    total = 0.50 * global_clip + 0.30 * local_clip + 0.15 * ce + 0.05 * text
    return total, {
        "c_global_clip": float(global_clip.detach()),
        "c_token_clip": float(local_clip.detach()),
        "c_ce": float(ce.detach()),
        "c_text_clip": float(text.detach()),
    }


def save_fixed_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    epoch: int,
    score: float,
    extra: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "openvoice-0730-fixed-checkpoint-v2",
            "epoch": epoch,
            "score": score,
            "state_dict": model.state_dict(),
            "extra": extra,
        },
        path,
    )


def load_eeg_fixed(
    config_path: Path, cfg: dict[str, Any], device: torch.device
) -> tuple[ContentProsodyEEGFixed, dict[str, Any]]:
    model = ContentProsodyEEGFixed(
        codebook_size=int(cfg["content"]["codebook_size"]),
        dimension=int(cfg["model"]["dimension"]),
        heads=int(cfg["model"]["heads"]),
        layers=int(cfg["model"]["layers"]),
        content_steps=int(cfg["content"]["steps"]),
        prosody_steps=32,
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    raw = torch.load(checkpoint_path(config_path, cfg, "eeg"), map_location=device, weights_only=False)
    if raw.get("schema_version") != "openvoice-0730-fixed-checkpoint-v2":
        raise ValueError(f"not a v0730-fixed checkpoint: {raw.get('schema_version')}")
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.eval(), raw["extra"]


def average_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([item[key] for item in values]))
        for key in values[0]
    } if values else {}


def train_eeg_fixed(
    config_path: Path,
    cfg: dict[str, Any],
    records: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    train = CPDataset(records, ("fit",))
    valid = CPDataset(records, ("subject_holdout_seen",))
    mean, scale, pos_weight = prosody_statistics(train)
    mean, scale, pos_weight = mean.to(device), scale.to(device), pos_weight.to(device)
    codebook = {
        key: torch.from_numpy(value).to(device)
        for key, value in records.codebook.items()
        if key in {"pca_mean", "pca_components", "pca_scale"}
    }
    references = {
        str(key).strip().strip("/").lower(): str(value)
        for key, value in cfg.get("text_reference", {}).items()
    }
    model = ContentProsodyEEGFixed(
        codebook_size=int(cfg["content"]["codebook_size"]),
        dimension=int(cfg["model"]["dimension"]),
        heads=int(cfg["model"]["heads"]),
        layers=int(cfg["model"]["layers"]),
        content_steps=int(cfg["content"]["steps"]),
        prosody_steps=32,
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["eeg_lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    best = math.inf
    stale = 0
    started = time.monotonic()
    total = int(args.epochs or cfg["training"]["eeg_epochs"])
    history = checkpoint_path(config_path, cfg, "eeg").parent.parent / "metrics" / "training.jsonl"
    for epoch in range(total):
        model.train()
        train_values: list[float] = []
        train_parts: list[dict[str, float]] = []
        for step, batch in enumerate(
            loader(train, batch_size=int(cfg["training"]["eeg_batch_size"]), train=True)
        ):
            batch = move_batch(batch, device)
            state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
            c, c_parts = content_loss(state, batch, codebook, references, model.clip_logit_scale)
            p, p_parts = p_loss(state, batch["prosody"], mean, scale, pos_weight)
            loss = 0.5 * (c + p)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            optimizer.step()
            train_values.append(float(loss.detach()))
            train_parts.append({**c_parts, **p_parts})
            if args.smoke_steps and step + 1 >= args.smoke_steps:
                break

        model.eval()
        validation: list[float] = []
        valid_parts: list[dict[str, float]] = []
        with torch.no_grad():
            for step, batch in enumerate(
                loader(valid, batch_size=int(cfg["training"]["eeg_batch_size"]), train=False)
            ):
                batch = move_batch(batch, device)
                state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
                c, c_parts = content_loss(state, batch, codebook, references, model.clip_logit_scale)
                p, p_parts = p_loss(state, batch["prosody"], mean, scale, pos_weight)
                validation.append(float((0.5 * (c + p)).detach()))
                valid_parts.append({**c_parts, **p_parts})
                if args.smoke_steps and step + 1 >= args.smoke_steps:
                    break
        score = float(np.mean(validation))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_values)),
            "validation_loss": score,
            "train_components": average_metrics(train_parts),
            "validation_components": average_metrics(valid_parts),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[0730-fixed eeg] epoch={epoch + 1}/{total} "
            f"train={record['train_loss']:.5f} valid={score:.5f} "
            f"elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if score < best:
            best = score
            stale = 0
            save_fixed_checkpoint(
                checkpoint_path(config_path, cfg, "eeg"),
                model,
                epoch=epoch,
                score=score,
                extra={
                    "prosody_mean": mean.cpu(),
                    "prosody_scale": scale.cpu(),
                    "activity_pos_weight": pos_weight.cpu(),
                    "fit_role": "fit",
                    "validation_role": "subject_holdout_seen",
                    "clip_positive_policy": "same_label_multi_positive_plus_diagonal",
                    "labels_as_forward_input": False,
                    "text_reference_labels": sorted(references),
                },
            )
        else:
            stale += 1
        if stale >= int(cfg["training"]["eeg_patience"]):
            break
        if args.wall_hours and time.monotonic() - started >= args.wall_hours * 3600:
            break


def main() -> None:
    args = parse()
    config_path, cfg = load_config(args.config)
    seed_everything(int(cfg["training"]["seed"]))
    device = default_device(args.device)
    if args.fresh:
        reset_training_outputs(config_path, cfg)
    records = load_prepared(resolve_config_path(config_path, cfg["paths"]["prepared_cache"]))
    requested = float(args.wall_hours) if args.wall_hours is not None else float(cfg["training"]["wall_hours"])
    renderer_hours = min(float(cfg["training"]["renderer_wall_hours"]), requested)
    eeg_hours = min(float(cfg["training"]["eeg_wall_hours"]), max(0.0, requested - renderer_hours))
    renderer_args, eeg_args = copy.copy(args), copy.copy(args)
    renderer_args.wall_hours, eeg_args.wall_hours = renderer_hours, eeg_hours
    if args.phase in {"renderer", "all"}:
        train_renderer(config_path, cfg, records, device, renderer_args)
    if args.phase in {"eeg", "all"}:
        train_eeg_fixed(config_path, cfg, records, device, eeg_args)
    write_json(
        resolve_config_path(config_path, cfg["paths"]["run_manifest"]),
        {
            "schema_version": "openvoice-0730-fixed-run-v2",
            "seed": int(cfg["training"]["seed"]),
            "device": str(device),
            "prepared_cache": str(resolve_config_path(config_path, cfg["paths"]["prepared_cache"])),
            "renderer_checkpoint": str(checkpoint_path(config_path, cfg, "renderer")),
            "eeg_checkpoint": str(checkpoint_path(config_path, cfg, "eeg")),
            "allocated_training_wall_hours": {
                "renderer": renderer_hours,
                "eeg": eeg_hours,
                "total": renderer_hours + eeg_hours,
            },
        },
    )


if __name__ == "__main__":
    main()
