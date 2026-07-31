#!/usr/bin/env python3
"""Run EnCodec/CLIP/MFCC v3 gates, with an explicitly labelled explore mode."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from scripts.train_open_vocab_v3_encodec_clip import (TokenDataset, attach_codes,
    micro_subset, token_collate)
from src.open_vocab_v3.data import V3Dataset, load_prepared
from src.open_vocab_v3.encodec_content import SCHEMA
from src.open_vocab_v3.full_evaluation import (ReferenceAudio, eeg_metrics,
    gate_t0, gate_t0b, gate_t1, gate_t1d, gate_t2_family, hubert_metrics,
    mfcc_prior_wavs, selected)
from src.open_vocab_v3.hubert import HubertMetric
from src.open_vocab_v3.metrics import paired_r_at_1_above_chance, same_label_template
from src.open_vocab_v3.runtime import (capture_lineage, default_device, load_config,
    move_batch, output_path, read_json, require_passed_gate, seed_everything,
    sha256_file, write_json)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--phase", choices=("t0", "t0b", "t1", "t1d", "t2", "t2v", "t3", "micro", "fit", "validation", "locked", "locked_unseen"), required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-fail", action="store_true")
    p.add_argument("--explore", action="store_true", help="bypass prerequisite and human-review gates; outputs are exploratory only")
    return p.parse_args()


def token_batches(dataset, cfg, device):
    for batch in DataLoader(dataset, batch_size=int(cfg["evaluation"]["batch_size"]), shuffle=False, collate_fn=token_collate, num_workers=0):
        yield move_batch(batch, device)


def save_gate(cp, cfg, key, payload, no_fail, artifacts=(), explore=False):
    payload.update({
        "schema_version": SCHEMA,
        "config_sha256": sha256_file(cp),
        "lineage": capture_lineage(cp, cfg, artifact_keys=tuple(artifacts)),
        "exploratory_gate_bypass": bool(explore),
    })
    payload["passed"] = bool(all(payload["checks"].values()))
    write_json(output_path(cp, cfg, key), payload)
    print(f"[v3 {payload['gate']}] passed={payload['passed']} explore={bool(explore)}", flush=True)
    if not payload["passed"] and not no_fail:
        raise SystemExit(2)


def eeg_gate(cp, cfg, records, device, stage, dataset):
    prediction, target, controls, labels, _keys, metric = eeg_metrics(cp, cfg, records, device, stage, dataset)
    template_ratio = float(np.mean(abs(prediction - target)) / max(float(np.mean(abs(same_label_template(target, labels) - target))), 1e-8))
    metric["template_error_ratio"] = template_ratio
    gate = cfg["gates"][stage]
    checks = {
        "label": metric["label_top1"] >= gate["label_top1_min"],
        "paired": metric["paired_r_at_1"] >= gate["paired_r_at_1_min"],
        "variance": metric["variance_ratio"] >= gate["variance_ratio_min"],
        **{f"{name}_win": value >= gate["paired_win_rate_min"] for name, value in metric["control_win_rates"].items()},
    }
    if stage == "micro":
        checks["template"] = template_ratio <= gate["template_ratio_max"]
    else:
        bootstrap = paired_r_at_1_above_chance(prediction, target, labels, samples=int(cfg["evaluation"]["bootstrap_samples"]), seed=int(cfg["training"]["seed"]))
        metric["paired_r1_bootstrap"] = bootstrap
        checks["paired_bootstrap"] = bootstrap["ci_low"] > 0
    return {"gate": "C" if stage == "micro" else "D", "n": len(labels), "metrics": metric, "thresholds": gate, "checks": checks}


def heldout(cp, cfg, records, device, phase, explore=False):
    review = output_path(cp, cfg, "training_review")
    expected = capture_lineage(cp, cfg, artifact_keys=("fit_checkpoint", "fit_gate", "fit_preview_manifest"))
    review_payload = read_json(review) if review.is_file() else {}
    if not explore and (not review_payload.get("passed", False) or review_payload.get("lineage") != expected):
        raise RuntimeError("held-out access refused before exact, non-stale training-WAV approval")
    roles = {"validation": ("subject_holdout_seen",), "locked": ("locked_test_seen_label",), "locked_unseen": ("locked_test_unseen_label",)}[phase]
    dataset = selected(records, roles, eligible=False)
    prediction, target, controls, labels, keys, metric = eeg_metrics(cp, cfg, records, device, "fit", dataset)
    references = [ReferenceAudio(cp, cfg)(key) for key in keys]
    teacher = HubertMetric(output_path(cp, cfg, "hubert_root"), layer=int(cfg["teachers"]["hubert_layer"]), device=device)
    generated = mfcc_prior_wavs(cp, cfg, records, device, prediction)
    metric["wav_content"] = hubert_metrics(generated, references, labels, teacher)
    metric["control_wav_content"] = {name: hubert_metrics(mfcc_prior_wavs(cp, cfg, records, device, value), references, labels, teacher) for name, value in controls.items()}
    metric["role"] = roles[0]
    metric["exploratory"] = phase == "locked_unseen" or bool(explore)
    metric["checkpoint_sha256"] = sha256_file(output_path(cp, cfg, "fit_checkpoint"))
    report_lineage_keys = ("fit_checkpoint", "fit_gate") if explore else ("fit_checkpoint", "fit_gate", "training_review")
    report = {
        "schema_version": SCHEMA,
        "phase": phase,
        "n": len(labels),
        "metrics": metric,
        "human_review_sha256": sha256_file(review) if review.is_file() else None,
        "human_review_bypassed": bool(explore),
        "exploratory_gate_bypass": bool(explore),
        "lineage": capture_lineage(cp, cfg, artifact_keys=report_lineage_keys),
    }
    key = {"validation": "validation_report", "locked": "locked_report", "locked_unseen": "locked_unseen_report"}[phase]
    write_json(output_path(cp, cfg, key), report)
    print(f"[v3 {phase}] n={len(labels)} explore={bool(explore)}", flush=True)


def require_if_primary(explore, cp, cfg, key, **kwargs):
    if not explore:
        require_passed_gate(cp, cfg, key, **kwargs)


def main():
    args = parse()
    cp, cfg = load_config(args.config)
    seed_everything(int(cfg["training"]["seed"]))
    records = load_prepared(output_path(cp, cfg, "prepared_cache"))
    device = default_device(args.device)
    if args.phase == "t0":
        save_gate(cp, cfg, "t0_gate", gate_t0(cp, cfg, records, device), args.no_fail, explore=args.explore)
        return
    if args.phase == "t0b":
        require_if_primary(args.explore, cp, cfg, "t0_gate")
        save_gate(cp, cfg, "t0b_gate", gate_t0b(cp, cfg, records, device), args.no_fail, explore=args.explore)
        return
    if args.phase in {"t1", "t1d"}:
        require_if_primary(args.explore, cp, cfg, "t0b_gate")
        cache, mapping = attach_codes(records, cp, cfg)
        dataset = TokenDataset(V3Dataset(records, ("fit",), eligible_only=True), cache, mapping)
        payload = gate_t1(cp, cfg, records, device, dataset, token_batches) if args.phase == "t1" else gate_t1d(cp, cfg, records, device, dataset, token_batches)
        save_gate(cp, cfg, "t1_gate" if args.phase == "t1" else "t1d_gate", payload, args.no_fail, explore=args.explore)
        return
    if args.phase in {"t2", "t2v", "t3"}:
        require_if_primary(args.explore, cp, cfg, "t1_gate")
        require_if_primary(args.explore, cp, cfg, "t1d_gate")
        if args.phase in {"t2v", "t3"}:
            require_if_primary(args.explore, cp, cfg, "t2_gate")
        if args.phase == "t3":
            require_if_primary(args.explore, cp, cfg, "t2v_gate")
        save_gate(cp, cfg, {"t2": "t2_gate", "t2v": "t2v_gate", "t3": "t3_gate"}[args.phase], gate_t2_family(cp, cfg, records, device, args.phase), args.no_fail, explore=args.explore)
        return
    if args.phase == "micro":
        require_if_primary(args.explore, cp, cfg, "t3_gate")
        save_gate(cp, cfg, "micro_gate", eeg_gate(cp, cfg, records, device, "micro", micro_subset(records, cfg)), args.no_fail, artifacts=("micro_checkpoint",), explore=args.explore)
        return
    if args.phase == "fit":
        require_if_primary(args.explore, cp, cfg, "micro_gate", lineage_artifact_keys=("micro_checkpoint",))
        save_gate(cp, cfg, "fit_gate", eeg_gate(cp, cfg, records, device, "fit", selected(records, ("fit",), eligible=True)), args.no_fail, artifacts=("fit_checkpoint", "micro_gate"), explore=args.explore)
        return
    heldout(cp, cfg, records, device, args.phase, args.explore)


if __name__ == "__main__":
    main()
