#!/usr/bin/env python3
"""Evaluate C and P independently before treating generated audio as evidence."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0730.data import CPDataset, collate, load_prepared
from src.open_vocab_0730.metrics import activity_f1, bootstrap_subject_gain, envelope_correlation, envelope_from_mel, role_counts
from src.open_vocab_0730.model import CPMelRenderer, ContentProsodyEEG
from src.open_vocab_0730.runtime import default_device, load_config, move_batch, resolve_config_path, write_json
from scripts.train_open_vocab_0730 import checkpoint_path, load_renderer


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed v0730 C/P controls and gates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Per-role limit for a non-writing smoke run")
    return parser.parse_args()


def load_eeg(config_path: Path, cfg: dict[str, Any], device: torch.device) -> tuple[ContentProsodyEEG, dict[str, torch.Tensor]]:
    model = ContentProsodyEEG(codebook_size=int(cfg["content"]["codebook_size"]), dimension=int(cfg["model"]["dimension"]), heads=int(cfg["model"]["heads"]), layers=int(cfg["model"]["layers"]), content_steps=int(cfg["content"]["steps"]), prosody_steps=32, dropout=float(cfg["model"]["dropout"])).to(device)
    raw = torch.load(checkpoint_path(config_path, cfg, "eeg"), map_location=device, weights_only=False)
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.eval(), {"mean": raw["extra"]["prosody_mean"].to(device), "scale": raw["extra"]["prosody_scale"].to(device)}


def content_retrieval(logits: np.ndarray, targets: np.ndarray, labels: list[str]) -> dict[str, float]:
    probabilities = torch.from_numpy(logits).softmax(-1).numpy()
    scores = np.take_along_axis(probabilities[:, None, :, :], targets[None, :, :, None], axis=-1).squeeze(-1).mean(-1)
    nearest = scores.argmax(1)
    paired = float(np.mean(nearest == np.arange(len(nearest))))
    label_correct = float(np.mean([labels[nearest[index]] == labels[index] for index in range(len(nearest))]))
    return {"paired_top1": paired, "label_audit_top1": label_correct, "candidate_count": int(len(nearest))}


@torch.no_grad()
def evaluate_role(model: ContentProsodyEEG, renderer: CPMelRenderer, dataset: CPDataset, device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    batches = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, collate_fn=collate, num_workers=0)
    output: dict[str, list[Any]] = defaultdict(list)
    for batch in batches:
        batch = move_batch(batch, device)
        variants = {
            "correct": batch["eeg"],
            "zero": torch.zeros_like(batch["eeg"]),
            "time_shuffled": torch.flip(batch["eeg"], dims=(-1,)),
            "channel_shuffled": torch.flip(batch["eeg"], dims=(1,)),
        }
        for name, eeg in variants.items():
            state = model(eeg, batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
            output[f"{name}_logits"].append(state.content_logits.cpu().numpy())
            output[f"{name}_duration"].append(state.duration.cpu().numpy())
            output[f"{name}_activity"].append(torch.sigmoid(state.activity_logits).cpu().numpy())
            output[f"{name}_envelope"].append(state.envelope.cpu().numpy())
        output["target_tokens"].append(batch["content_tokens"].cpu().numpy())
        output["target_p"].append(batch["prosody"].cpu().numpy())
        output["subjects"].extend(batch["subject"]); output["labels"].extend(batch["label"])
        correct = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
        rendered = renderer(correct.content_logits, correct.prosody)
        output["renderer_mel_l1"].extend(np.abs(rendered.cpu().numpy() - batch["mel"].cpu().numpy()).mean(axis=(1, 2)).tolist())
    joined = {key: np.concatenate(value, axis=0) if isinstance(value, list) and value and isinstance(value[0], np.ndarray) else value for key, value in output.items()}
    target = joined["target_p"]
    report: dict[str, Any] = {"n": int(len(target)), "content": {}, "prosody": {}, "renderer": {"mean_mel_l1": float(np.mean(joined["renderer_mel_l1"]))}}
    for name in ("correct", "zero", "time_shuffled", "channel_shuffled"):
        report["content"][name] = content_retrieval(joined[f"{name}_logits"], joined["target_tokens"], joined["labels"])
        duration_mae = np.abs(joined[f"{name}_duration"] - target[:, 0])
        activity = np.asarray([activity_f1(pred, ref) for pred, ref in zip(joined[f"{name}_activity"], target[:, 2:34])])
        envelope = np.asarray([envelope_correlation(pred, ref) for pred, ref in zip(joined[f"{name}_envelope"], target[:, 34:66])])
        report["prosody"][name] = {"duration_mae": float(duration_mae.mean()), "activity_f1": float(activity.mean()), "envelope_correlation": float(envelope.mean())}
    gains = {
        "content_label_top1_over_zero": np.full(len(target), report["content"]["correct"]["label_audit_top1"] - report["content"]["zero"]["label_audit_top1"]),
        "prosody_duration_over_zero": np.abs(joined["zero_duration"] - target[:, 0]) - np.abs(joined["correct_duration"] - target[:, 0]),
        "prosody_activity_over_zero": np.asarray([activity_f1(a, b) - activity_f1(c, b) for a, c, b in zip(joined["correct_activity"], joined["zero_activity"], target[:, 2:34])]),
        "prosody_envelope_over_zero": np.asarray([envelope_correlation(a, b) - envelope_correlation(c, b) for a, c, b in zip(joined["correct_envelope"], joined["zero_envelope"], target[:, 34:66])]),
    }
    report["bootstrap"] = {name: bootstrap_subject_gain(joined["subjects"], gain, samples=int(cfg["evaluation"]["bootstrap_samples"]), seed=int(cfg["evaluation"]["bootstrap_seed"])) for name, gain in gains.items()}
    return report


@torch.no_grad()
def renderer_gate(renderer: CPMelRenderer, dataset: CPDataset, device: torch.device, cfg: dict[str, Any]) -> dict[str, Any]:
    values = []
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, collate_fn=collate, num_workers=0)
    for batch in loader:
        batch = move_batch(batch, device)
        oracle = renderer(batch["content_tokens"], batch["prosody"])
        c_swap = renderer(torch.roll(batch["content_tokens"], shifts=1, dims=0), batch["prosody"])
        p_swap = renderer(batch["content_tokens"], torch.roll(batch["prosody"], shifts=1, dims=0))
        values.append({"oracle_l1": np.abs(oracle.cpu().numpy() - batch["mel"].cpu().numpy()).mean(axis=(1, 2)), "c_swap_change": np.abs(c_swap.cpu().numpy() - oracle.cpu().numpy()).mean(axis=(1, 2)), "p_swap_change": np.abs(p_swap.cpu().numpy() - oracle.cpu().numpy()).mean(axis=(1, 2))})
    joined = {key: np.concatenate([item[key] for item in values]) for key in values[0]}
    report = {"oracle_mel_l1": float(joined["oracle_l1"].mean()), "content_swap_mel_change": float(joined["c_swap_change"].mean()), "prosody_swap_mel_change": float(joined["p_swap_change"].mean())}
    report["passed"] = bool(report["oracle_mel_l1"] <= float(cfg["evaluation"]["renderer_mel_l1_max_db"]) and min(report["content_swap_mel_change"], report["prosody_swap_mel_change"]) >= float(cfg["evaluation"]["swap_mel_change_minimum"]))
    return report


def main() -> None:
    args = parse(); config_path, cfg = load_config(args.config); device = default_device(args.device)
    records = load_prepared(resolve_config_path(config_path, cfg["paths"]["prepared_cache"])); model, _ = load_eeg(config_path, cfg, device); renderer = load_renderer(config_path, cfg, device)
    datasets = {role: CPDataset(records, (role,)) for role in ("subject_holdout_seen", "label_holdout_seen_subject", "subject_and_label_holdout")}
    if args.limit:
        datasets = {role: Subset(dataset, range(min(args.limit, len(dataset)))) for role, dataset in datasets.items()}
    reports = {role: evaluate_role(model, renderer, dataset, device, cfg) for role, dataset in datasets.items()}
    gate = renderer_gate(renderer, datasets["subject_holdout_seen"], device, cfg)
    if args.limit:
        print({role: value["n"] for role, value in reports.items()}, flush=True)
        return
    write_json(resolve_config_path(config_path, cfg["paths"]["renderer_gate"]), gate)
    write_json(resolve_config_path(config_path, cfg["paths"]["evaluation_report"]), {"schema_version": "openvoice-0730-evaluation-v1", "role_counts": role_counts(records.roles), "renderer_gate": gate, "results": reports, "waveform_interpretation": "Conditional generative approximation only; imagined EEG and later overt audio are weakly paired."})


if __name__ == "__main__":
    main()
