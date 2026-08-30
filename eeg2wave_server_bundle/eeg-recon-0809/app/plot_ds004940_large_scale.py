#!/usr/bin/env python3
"""Compare the 512-pair and large DS004940-only exploratory experiments.

Reads saved JSON only: this script never loads EEG, models, or checkpoints.
It produces reproducible seed-level figures and a machine-readable summary;
all labels retain the exploratory/not-registered interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

from paper_plot_style import COLORS, configure, panel_label, plt, save_figure

ROLES = ("validation", "test")
CONTROLS = ("zero", "time_shuffle", "channel_shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-root", type=Path, required=True)
    parser.add_argument("--small-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", default="31,47,73")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required evaluation is missing: {path}")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"evaluation is not a JSON object: {path}")
    return result


def bootstrap_mean(values: list[float], repetitions: int, seed: int) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        return {"n": int(len(array)), "mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "estimable": False}
    if len(array) == 1:
        return {"n": 1, "mean": float(array[0]), "ci_low": float(array[0]),
                "ci_high": float(array[0]), "estimable": False}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(int(repetitions), len(array)), replace=True).mean(1)
    return {"n": int(len(array)), "mean": float(array.mean()),
            "ci_low": float(np.quantile(samples, 0.025)), "ci_high": float(np.quantile(samples, 0.975)),
            "estimable": True}


def collect(root: Path, label: str, seeds: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    sources: list[dict] = []
    for seed in seeds:
        for role in ROLES:
            path = root / "generalization" / "ds004940" / f"seed-{seed}" / f"evaluation_ds004940_{role}.json"
            result = load_json(path)
            if result.get("dataset") != "ds004940" or result.get("role") != role:
                raise RuntimeError(f"evaluation identity mismatch: {path}")
            controls = {name: float(result["controls"][name]) for name in ("correct", *CONTROLS)}
            margins = {name: controls[name] - controls["correct"] for name in CONTROLS}
            records.append({
                "scale": label, "seed": seed, "role": role, "pairs": int(result["pairs"]),
                "mfcc_l1": float(result["mfcc_l1"]), "delta_l1": float(result["delta_l1"]),
                "retrieval_r1": float(result["retrieval"]["r1"]),
                "retrieval_mrr": float(result["retrieval"]["mrr"]),
                "chance_r1": float(result["retrieval"]["chance_r1"]),
                "hubert_global_r1": float(result["hubert_similarity"]["global_retrieval"]["r1"]),
                "hubert_global_mrr": float(result["hubert_similarity"]["global_retrieval"]["mrr"]),
                "template_improvement": float(result.get("templates", {}).get("dataset_mean_template_improvement", float("nan"))),
                "controls": controls, "control_margins": margins,
                "subject_mfcc_l1": result.get("subject_mfcc_l1", {}),
                "subject_control_mfcc_l1": result.get("subject_control_mfcc_l1", {}),
            })
            sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "kind": f"{label}_evaluation"})
    return records, sources


def plot_seed_paired(records: list[dict], output: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(8.1, 3.0), constrained_layout=True)
    for axis, role, letter in zip(axes, ROLES, "ab"):
        small = {row["seed"]: row for row in records if row["scale"] == "small_512" and row["role"] == role}
        large = {row["seed"]: row for row in records if row["scale"] == "large_3380" and row["role"] == role}
        for seed in sorted(set(small) & set(large)):
            axis.plot([0, 1], [small[seed]["mfcc_l1"], large[seed]["mfcc_l1"]], color="0.70", linewidth=0.9)
            axis.scatter([0, 1], [small[seed]["mfcc_l1"], large[seed]["mfcc_l1"]],
                         color=[COLORS["single"], COLORS["joint"]], s=26, zorder=2)
        axis.set_xticks([0, 1], ["512-pair", "3,380-pair"])
        axis.set_ylabel("Held-out MFCC L1 (lower is better)")
        axis.grid(axis="y", alpha=0.2)
        panel_label(axis, f"({letter}) {role}")
    return save_figure(figure, output, "scale_mfcc_l1", formats, dpi)


def plot_retrieval(records: list[dict], output: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 5.9), constrained_layout=True)
    for row_index, role in enumerate(ROLES):
        group = [row for row in records if row["role"] == role]
        for column, (metric, label) in enumerate((("retrieval_r1", "MFCC retrieval R@1"), ("retrieval_mrr", "MFCC retrieval MRR"))):
            axis = axes[row_index, column]
            for offset, scale in enumerate(("small_512", "large_3380")):
                values = [row[metric] for row in group if row["scale"] == scale]
                axis.scatter(np.full(len(values), offset), values,
                             color=COLORS["single"] if scale == "small_512" else COLORS["joint"], s=28)
                if values:
                    axis.plot([offset - 0.12, offset + 0.12], [np.mean(values)] * 2, color="black", linewidth=1.1)
            if metric == "retrieval_r1":
                chance = np.mean([row["chance_r1"] for row in group])
                axis.axhline(chance, color=COLORS["chance"], linestyle="--", linewidth=0.9, label="chance")
            axis.set_xticks([0, 1], ["512-pair", "3,380-pair"])
            axis.set_ylabel(label)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(axis="y", alpha=0.2)
            panel_label(axis, f"({chr(ord('a') + row_index * 2 + column)}) {role}")
    axes[0, 0].legend(frameon=False)
    return save_figure(figure, output, "scale_content_retrieval", formats, dpi)


def plot_controls(records: list[dict], output: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(8.1, 3.0), constrained_layout=True)
    for axis, role, letter in zip(axes, ROLES, "ab"):
        group = [row for row in records if row["scale"] == "large_3380" and row["role"] == role]
        positions = np.arange(len(CONTROLS))
        for index, control in enumerate(CONTROLS):
            values = [row["control_margins"][control] for row in group]
            axis.scatter(np.full(len(values), positions[index]), values, color=COLORS[control], s=28)
            if values:
                axis.plot([positions[index] - 0.15, positions[index] + 0.15], [np.mean(values)] * 2,
                          color="black", linewidth=1.1)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(positions, ["Zero", "Time", "Channel"])
        axis.set_ylabel("Control error − correct EEG error\n(positive favors correct EEG)")
        axis.grid(axis="y", alpha=0.2)
        panel_label(axis, f"({letter}) large scale — {role}")
    return save_figure(figure, output, "large_scale_control_margins", formats, dpi)


def plot_validation_history(large_root: Path, seeds: tuple[int, ...], output: Path,
                            formats: tuple[str, ...], dpi: int, sources: list[dict]) -> list[Path]:
    figure, axis = plt.subplots(1, 1, figsize=(5.4, 3.1), constrained_layout=True)
    for seed in seeds:
        path = large_root / "generalization" / "ds004940" / f"seed-{seed}" / "metrics.json"
        payload = load_json(path)
        history = payload.get("validation_history", [])
        if not history:
            raise RuntimeError(f"large-scale metrics has no validation history: {path}")
        epochs = [int(item["epoch"]) for item in history]
        mrr = [float(item["retrieval"]["mrr"]) for item in history]
        axis.plot(epochs, mrr, marker="o", markersize=3, linewidth=1.1, label=f"seed {seed}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "large_training_metrics"})
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation MFCC retrieval MRR")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    return save_figure(figure, output, "large_scale_validation_mrr", formats, dpi)


def subject_control_summary(records: list[dict], repetitions: int) -> list[dict]:
    result = []
    for role in ROLES:
        for control in CONTROLS:
            values = []
            for row in records:
                if row["scale"] != "large_3380" or row["role"] != role:
                    continue
                subject_correct = row["subject_mfcc_l1"]
                subject_controls = row["subject_control_mfcc_l1"].get(control, {})
                values.extend(float(subject_controls[subject]) - float(error)
                              for subject, error in subject_correct.items() if subject in subject_controls)
            result.append({"role": role, "control": control,
                           "correct_eeg_margin_bootstrap": bootstrap_mean(values, repetitions, 31047 + len(result))})
    return result


def main() -> int:
    args = parse_args()
    configure()
    large_root, small_root = args.large_root.resolve(), args.small_root.resolve()
    output = (args.output or large_root / "generalization" / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.replace(" ", ",").split(",") if value)
    formats = tuple(value.strip().lower() for value in args.formats.split(",") if value.strip())
    if not seeds or not formats or any(value not in {"png", "pdf", "svg"} for value in formats):
        raise ValueError("provide seeds and png/pdf/svg formats")
    small, small_sources = collect(small_root, "small_512", seeds)
    large, large_sources = collect(large_root, "large_3380", seeds)
    records, sources = small + large, small_sources + large_sources
    figures = []
    figures += plot_seed_paired(records, output, formats, int(args.dpi))
    figures += plot_retrieval(records, output, formats, int(args.dpi))
    figures += plot_controls(records, output, formats, int(args.dpi))
    figures += plot_validation_history(large_root, seeds, output, formats, int(args.dpi), sources)
    summary = {
        "schema_version": "eeg2speech-ds004940-scale-comparison-v1",
        "interpretation": "exploratory_only_not_registered",
        "scales": {"small_512": {"train_pairs": 512}, "large_3380": {"train_pairs": 3380}},
        "records": records,
        "large_scale_correct_eeg_control_margins": subject_control_summary(records, int(args.bootstrap_repetitions)),
    }
    (output / "scale_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {"schema_version": "eeg2speech-ds004940-scale-figure-manifest-v1",
                "interpretation": "exploratory_only_not_registered", "large_root": str(large_root),
                "small_root": str(small_root), "seeds": list(seeds),
                "figures": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in figures],
                "sources": sorted(sources, key=lambda value: (value["path"], value["kind"]))}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "pass", "output": str(output), "figures": [str(path) for path in figures],
                      "summary": str(output / "scale_comparison_summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
