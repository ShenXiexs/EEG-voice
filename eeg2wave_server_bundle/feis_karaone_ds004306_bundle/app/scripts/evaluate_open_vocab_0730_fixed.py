#!/usr/bin/env python3
"""Evaluate v0730-fixed C/P, CLIP controls, and generated-Mel collapse gates."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from scripts.train_open_vocab_0730 import frozen_audio_tokens, load_renderer
from scripts.train_open_vocab_0730_fixed import load_eeg_fixed
from src.open_vocab_0730.data_fixed import CPDataset, collate, load_prepared
from src.open_vocab_0730.metrics import (
    activity_f1,
    bootstrap_subject_gain,
    envelope_correlation,
    role_counts,
)
from src.open_vocab_0730.runtime import (
    default_device,
    load_config,
    move_batch,
    resolve_config_path,
    write_json,
)


VARIANTS = ("correct", "zero", "time_shuffled", "channel_shuffled")


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v0730-fixed controls and gates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def clip_retrieval(
    eeg_tokens: np.ndarray,
    audio_tokens: np.ndarray,
    audio_mask: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    eeg = torch.from_numpy(eeg_tokens).float()
    audio = torch.from_numpy(audio_tokens).float()
    mask = torch.from_numpy(audio_mask).bool()
    eeg = F.normalize(eeg, dim=-1)
    audio = F.normalize(audio, dim=-1)
    weight = mask.float().unsqueeze(-1)
    eeg_global = F.normalize((eeg * weight).sum(1) / weight.sum(1).clamp_min(1.0), dim=-1)
    audio_global = F.normalize((audio * weight).sum(1) / weight.sum(1).clamp_min(1.0), dim=-1)
    global_similarity = eeg_global @ audio_global.T
    token_similarity = torch.einsum("itd,jtd->ijt", eeg, audio)
    pair_weight = mask.float().unsqueeze(0)
    local_similarity = (token_similarity * pair_weight).sum(-1) / pair_weight.sum(-1).clamp_min(1.0)
    scores = (0.65 * global_similarity + 0.35 * local_similarity).numpy()
    nearest = scores.argmax(1)
    paired = nearest == np.arange(len(nearest))
    label_correct = np.asarray(
        [labels[prediction] == labels[index] for index, prediction in enumerate(nearest)]
    )
    return (
        {
            "paired_top1": float(paired.mean()),
            "label_top1": float(label_correct.mean()),
            "candidate_count": int(len(nearest)),
        },
        {"paired": paired.astype(np.float32), "label_correct": label_correct.astype(np.float32)},
    )


def active_mel_l1(prediction: np.ndarray, target: np.ndarray, activity: np.ndarray) -> np.ndarray:
    mask = F.interpolate(
        torch.from_numpy(activity).float().unsqueeze(1), size=target.shape[-1], mode="nearest"
    ).squeeze(1).numpy() >= 0.5
    values = np.abs(prediction - target).mean(1)
    return np.asarray(
        [value[item].mean() if item.any() else value.mean() for value, item in zip(values, mask)]
    )


@torch.no_grad()
def evaluate_role(
    model: torch.nn.Module,
    renderer: torch.nn.Module,
    dataset: CPDataset,
    device: torch.device,
    cfg: dict[str, Any],
    codebook: dict[str, torch.Tensor],
    role: str,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    output: dict[str, list[Any]] = defaultdict(list)
    for batch in tqdm(
        loader,
        total=len(loader),
        desc=f"[0730-fixed eval] {role}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
    ):
        batch = move_batch(batch, device)
        inputs = {
            "correct": batch["eeg"],
            "zero": torch.zeros_like(batch["eeg"]),
            "time_shuffled": torch.flip(batch["eeg"], dims=(-1,)),
            "channel_shuffled": torch.flip(batch["eeg"], dims=(1,)),
        }
        states = {
            name: model(eeg, batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
            for name, eeg in inputs.items()
        }
        audio, audio_mask = frozen_audio_tokens(batch["hubert"], batch["hubert_mask"], codebook)
        output["audio_clip"].append(audio.cpu().numpy())
        output["audio_mask"].append(audio_mask.cpu().numpy())
        output["target_p"].append(batch["prosody"].cpu().numpy())
        output["target_mel"].append(batch["mel"].cpu().numpy())
        output["subjects"].extend(batch["subject"])
        output["labels"].extend(batch["label"])
        for name, state in states.items():
            output[f"{name}_clip"].append(state.content_clip_tokens.cpu().numpy())
            output[f"{name}_duration"].append(state.duration.cpu().numpy())
            output[f"{name}_activity"].append(torch.sigmoid(state.activity_logits).cpu().numpy())
            output[f"{name}_envelope"].append(state.envelope.cpu().numpy())
            rendered = renderer(state.content_logits, state.prosody)
            output[f"{name}_mel"].append(rendered.cpu().numpy())

    joined = {
        key: np.concatenate(value, axis=0)
        if value and isinstance(value[0], np.ndarray)
        else value
        for key, value in output.items()
    }
    target_p = joined["target_p"]
    target_mel = joined["target_mel"]
    report: dict[str, Any] = {"n": int(len(target_p)), "clip": {}, "prosody": {}, "generated_mel": {}}
    retrieval_rows: dict[str, dict[str, np.ndarray]] = {}
    per_sample: dict[str, dict[str, np.ndarray]] = {}
    for name in VARIANTS:
        report["clip"][name], retrieval_rows[name] = clip_retrieval(
            joined[f"{name}_clip"], joined["audio_clip"], joined["audio_mask"], joined["labels"]
        )
        duration_error = np.abs(joined[f"{name}_duration"] - target_p[:, 0])
        activity = np.asarray(
            [activity_f1(pred, ref) for pred, ref in zip(joined[f"{name}_activity"], target_p[:, 2:34])]
        )
        envelope = np.asarray(
            [
                envelope_correlation(pred, ref)
                for pred, ref in zip(joined[f"{name}_envelope"], target_p[:, 34:66])
            ]
        )
        mel_l1 = active_mel_l1(joined[f"{name}_mel"], target_mel, target_p[:, 2:34])
        report["prosody"][name] = {
            "duration_mae": float(duration_error.mean()),
            "activity_f1": float(activity.mean()),
            "envelope_correlation": float(envelope.mean()),
        }
        report["generated_mel"][name] = {
            "active_mel_l1": float(mel_l1.mean()),
            "fraction_above_minus_60_db": float((joined[f"{name}_mel"] > -60.0).mean()),
            "sample_mean_db_sd": float(joined[f"{name}_mel"].mean(axis=(1, 2)).std()),
        }
        per_sample[name] = {
            "duration_error": duration_error,
            "activity": activity,
            "envelope": envelope,
            "mel_l1": mel_l1,
        }

    target_variance = float(np.var(target_mel, axis=0).mean())
    generated_variance = float(np.var(joined["correct_mel"], axis=0).mean())
    report["generated_mel"]["correct"]["variance_ratio"] = generated_variance / max(target_variance, 1e-8)
    gains = {
        "clip_label_top1_over_zero": retrieval_rows["correct"]["label_correct"] - retrieval_rows["zero"]["label_correct"],
        "clip_label_top1_over_channel_shuffle": retrieval_rows["correct"]["label_correct"] - retrieval_rows["channel_shuffled"]["label_correct"],
        "duration_over_zero": per_sample["zero"]["duration_error"] - per_sample["correct"]["duration_error"],
        "activity_over_zero": per_sample["correct"]["activity"] - per_sample["zero"]["activity"],
        "envelope_over_zero": per_sample["correct"]["envelope"] - per_sample["zero"]["envelope"],
        "active_mel_l1_over_zero": per_sample["zero"]["mel_l1"] - per_sample["correct"]["mel_l1"],
    }
    report["bootstrap"] = {
        name: bootstrap_subject_gain(
            joined["subjects"],
            gain,
            samples=int(cfg["evaluation"]["bootstrap_samples"]),
            seed=int(cfg["evaluation"]["bootstrap_seed"]),
        )
        for name, gain in gains.items()
    }
    return report


@torch.no_grad()
def renderer_gate(
    renderer: torch.nn.Module,
    dataset: CPDataset,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    values: list[dict[str, np.ndarray]] = []
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    for batch in tqdm(loader, desc="[0730-fixed eval] renderer gate", unit="batch", dynamic_ncols=True):
        batch = move_batch(batch, device)
        oracle = renderer(batch["content_tokens"], batch["prosody"])
        c_swap = renderer(torch.roll(batch["content_tokens"], 1, 0), batch["prosody"])
        p_swap = renderer(batch["content_tokens"], torch.roll(batch["prosody"], 1, 0))
        values.append(
            {
                "oracle": np.abs(oracle.cpu().numpy() - batch["mel"].cpu().numpy()).mean((1, 2)),
                "content": np.abs(c_swap.cpu().numpy() - oracle.cpu().numpy()).mean((1, 2)),
                "prosody": np.abs(p_swap.cpu().numpy() - oracle.cpu().numpy()).mean((1, 2)),
            }
        )
    joined = {key: np.concatenate([item[key] for item in values]) for key in values[0]}
    report = {
        "oracle_mel_l1": float(joined["oracle"].mean()),
        "content_swap_mel_change": float(joined["content"].mean()),
        "prosody_swap_mel_change": float(joined["prosody"].mean()),
    }
    report["passed"] = bool(
        report["oracle_mel_l1"] <= float(cfg["evaluation"]["renderer_mel_l1_max_db"])
        and min(report["content_swap_mel_change"], report["prosody_swap_mel_change"])
        >= float(cfg["evaluation"]["swap_mel_change_minimum"])
    )
    return report


def generated_gate(reports: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    roles = list(cfg["evaluation"]["primary_gate_roles"])
    thresholds = cfg["evaluation"]["generated_gate"]
    pooled = {
        "clip_label_gain_over_zero": float(np.mean([reports[r]["clip"]["correct"]["label_top1"] - reports[r]["clip"]["zero"]["label_top1"] for r in roles])),
        "clip_label_gain_over_channel_shuffle": float(np.mean([reports[r]["clip"]["correct"]["label_top1"] - reports[r]["clip"]["channel_shuffled"]["label_top1"] for r in roles])),
        "activity_gain_over_zero": float(np.mean([reports[r]["prosody"]["correct"]["activity_f1"] - reports[r]["prosody"]["zero"]["activity_f1"] for r in roles])),
        "envelope_gain_over_zero": float(np.mean([reports[r]["prosody"]["correct"]["envelope_correlation"] - reports[r]["prosody"]["zero"]["envelope_correlation"] for r in roles])),
        "duration_gain_over_zero": float(np.mean([reports[r]["prosody"]["zero"]["duration_mae"] - reports[r]["prosody"]["correct"]["duration_mae"] for r in roles])),
        "active_mel_l1_gain_over_zero": float(np.mean([reports[r]["generated_mel"]["zero"]["active_mel_l1"] - reports[r]["generated_mel"]["correct"]["active_mel_l1"] for r in roles])),
        "fraction_above_minus_60_db": float(np.mean([reports[r]["generated_mel"]["correct"]["fraction_above_minus_60_db"] for r in roles])),
        "variance_ratio": float(np.mean([reports[r]["generated_mel"]["correct"]["variance_ratio"] for r in roles])),
    }
    checks = {
        "clip_over_zero": pooled["clip_label_gain_over_zero"] >= float(thresholds["minimum_clip_label_gain"]),
        "clip_over_channel_shuffle": pooled["clip_label_gain_over_channel_shuffle"] >= float(thresholds["minimum_clip_label_gain"]),
        "activity_over_zero": pooled["activity_gain_over_zero"] >= float(thresholds["minimum_activity_gain"]),
        "envelope_over_zero": pooled["envelope_gain_over_zero"] >= float(thresholds["minimum_envelope_gain"]),
        "duration_over_zero": pooled["duration_gain_over_zero"] >= float(thresholds["minimum_duration_gain_seconds"]),
        "mel_over_zero": pooled["active_mel_l1_gain_over_zero"] >= float(thresholds["minimum_active_mel_l1_gain_db"]),
        "non_silent": pooled["fraction_above_minus_60_db"] >= float(thresholds["minimum_fraction_above_minus_60_db"]),
        "non_collapsed_variance": pooled["variance_ratio"] >= float(thresholds["minimum_variance_ratio"]),
    }
    return {"passed": bool(all(checks.values())), "primary_roles": roles, "metrics": pooled, "checks": checks, "thresholds": thresholds}


def main() -> None:
    args = parse()
    config_path, cfg = load_config(args.config)
    device = default_device(args.device)
    report_path = resolve_config_path(config_path, cfg["paths"]["evaluation_report"])
    status_path = report_path.parent / "evaluation_status.json"
    if not args.limit:
        write_json(status_path, {"state": "running", "device": str(device), "schema_version": "openvoice-0730-fixed-evaluation-status-v2"})
    print(f"[0730-fixed eval] device={device}", flush=True)
    records = load_prepared(resolve_config_path(config_path, cfg["paths"]["prepared_cache"]))
    model, _ = load_eeg_fixed(config_path, cfg, device)
    renderer = load_renderer(config_path, cfg, device)
    codebook = {
        key: torch.from_numpy(value).to(device)
        for key, value in records.codebook.items()
        if key in {"pca_mean", "pca_components", "pca_scale"}
    }
    roles = list(cfg["evaluation"]["report_roles"])
    datasets: dict[str, Any] = {role: CPDataset(records, (role,)) for role in roles}
    if args.limit:
        datasets = {
            role: Subset(dataset, range(min(args.limit, len(dataset))))
            for role, dataset in datasets.items()
        }
    reports = {
        role: evaluate_role(model, renderer, dataset, device, cfg, codebook, role)
        for role, dataset in datasets.items()
    }
    gate_dataset = CPDataset(records, ("subject_holdout_seen",))
    if args.limit:
        gate_dataset = Subset(gate_dataset, range(min(args.limit, len(gate_dataset))))
    oracle_gate = renderer_gate(renderer, gate_dataset, device, cfg)
    if args.limit:
        print({role: report["n"] for role, report in reports.items()}, flush=True)
        return
    waveform_gate = generated_gate(reports, cfg)
    write_json(resolve_config_path(config_path, cfg["paths"]["renderer_gate"]), oracle_gate)
    write_json(resolve_config_path(config_path, cfg["paths"]["generated_gate"]), waveform_gate)
    write_json(
        report_path,
        {
            "schema_version": "openvoice-0730-fixed-evaluation-v2",
            "device": str(device),
            "role_counts": role_counts(records.roles),
            "renderer_gate": oracle_gate,
            "generated_gate": waveform_gate,
            "results": reports,
            "primary_claim_requires_generated_gate": True,
        },
    )
    write_json(status_path, {"state": "complete", "device": str(device), "report": str(report_path), "schema_version": "openvoice-0730-fixed-evaluation-status-v2"})
    print(f"[0730-fixed eval] complete report={report_path}", flush=True)


if __name__ == "__main__":
    main()
