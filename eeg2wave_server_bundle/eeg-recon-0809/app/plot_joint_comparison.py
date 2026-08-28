#!/usr/bin/env python3
"""Create reproducible single-dataset versus joint-training comparison figures.

The script reads evaluation/metrics JSON files only. It never loads a model or
changes checkpoints. Subject-bootstrap intervals are computed after averaging
the paired single-minus-joint gain across seeds for each held-out subject; seed
dispersion is reported separately and never presented as subject uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP))

from paper_plot_style import COLORS, configure, panel_label, plt, save_figure

DATASETS = ("ds004940", "ds006104")
ROLES = ("validation", "test")
CONTROLS = ("correct", "zero", "time_shuffle", "channel_shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="experiment root containing generalization/<mode>/seed-*/")
    parser.add_argument("--output", type=Path,
                        help="default: <input-root>/generalization/figures")
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


def finite_or_none(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required result is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return payload


def bootstrap_mean(values: list[float], repetitions: int, seed: int = 31047) -> dict[str, float | int | bool]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        return {"n": int(len(array)), "mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "estimable": False}
    if len(array) == 1:
        value = float(array[0])
        return {"n": 1, "mean": value, "ci_low": value, "ci_high": value, "estimable": False}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(int(repetitions), len(array)), replace=True).mean(axis=1)
    return {"n": int(len(array)), "mean": float(array.mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)), "estimable": True}


def collect_results(input_root: Path, seeds: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    generalization = input_root / "generalization"
    records: list[dict] = []
    sources: list[dict] = []
    for seed in seeds:
        for dataset in DATASETS:
            for role in ROLES:
                paths = {
                    "single": generalization / dataset / f"seed-{seed}" / f"evaluation_{dataset}_{role}.json",
                    "joint": generalization / "joint" / f"seed-{seed}" / f"evaluation_{dataset}_{role}.json",
                }
                values = {kind: load_json(path) for kind, path in paths.items()}
                for kind, path in paths.items():
                    sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "evaluation"})
                    if values[kind].get("dataset") != dataset or values[kind].get("role") != role:
                        raise RuntimeError(f"evaluation identity mismatch: {path}")
                single, joint = values["single"], values["joint"]
                single_subject = single.get("subject_mfcc_l1", {})
                joint_subject = joint.get("subject_mfcc_l1", {})
                common = sorted(set(single_subject) & set(joint_subject))
                if not common:
                    raise RuntimeError(f"no paired subjects for {dataset}/{role}/seed-{seed}")
                subject_gains = {subject: float(single_subject[subject]) - float(joint_subject[subject])
                                 for subject in common}
                records.append({
                    "seed": seed, "dataset": dataset, "role": role,
                    "pairs": int(single.get("pairs", 0)), "subjects": common,
                    "single_mfcc_l1": float(single["mfcc_l1"]),
                    "joint_mfcc_l1": float(joint["mfcc_l1"]),
                    "mfcc_gain": float(single["mfcc_l1"]) - float(joint["mfcc_l1"]),
                    "single_delta_l1": float(single["delta_l1"]),
                    "joint_delta_l1": float(joint["delta_l1"]),
                    "single_retrieval_r1": float(single["retrieval"]["r1"]),
                    "joint_retrieval_r1": float(joint["retrieval"]["r1"]),
                    "chance_r1": finite_or_none(single["retrieval"].get("chance_r1")),
                    "single_template_improvement": finite_or_none(single.get("templates", {}).get("dataset_mean_template_improvement")),
                    "joint_template_improvement": finite_or_none(joint.get("templates", {}).get("dataset_mean_template_improvement")),
                    "single_controls": {key: float(single["controls"][key]) for key in CONTROLS},
                    "joint_controls": {key: float(joint["controls"][key]) for key in CONTROLS},
                    "subject_gains": subject_gains,
                })
    return records, sources


def summarize(records: list[dict], repetitions: int) -> list[dict]:
    summary = []
    for dataset in DATASETS:
        for role in ROLES:
            group = [row for row in records if row["dataset"] == dataset and row["role"] == role]
            if not group:
                raise RuntimeError(f"no records for {dataset}/{role}")
            by_subject: dict[str, list[float]] = defaultdict(list)
            for row in group:
                for subject, gain in row["subject_gains"].items():
                    by_subject[subject].append(float(gain))
            subject_means = {subject: float(np.mean(values)) for subject, values in sorted(by_subject.items())}
            seed_gains = [float(row["mfcc_gain"]) for row in group]
            subject_interval = bootstrap_mean(list(subject_means.values()), repetitions)
            seed_values = {
                "n": len(seed_gains), "mean": float(np.mean(seed_gains)),
                "sd": float(np.std(seed_gains, ddof=1)) if len(seed_gains) > 1 else 0.0,
                "min": float(np.min(seed_gains)), "max": float(np.max(seed_gains)),
            }
            summary.append({
                "dataset": dataset, "role": role, "seeds": [int(row["seed"]) for row in group],
                "pairs_per_seed": sorted({int(row["pairs"]) for row in group}),
                "heldout_subjects": sorted(subject_means), "subject_gain_by_subject": subject_means,
                "single_mfcc_l1_mean": float(np.mean([row["single_mfcc_l1"] for row in group])),
                "joint_mfcc_l1_mean": float(np.mean([row["joint_mfcc_l1"] for row in group])),
                "seed_gain_dispersion": seed_values,
                "subject_bootstrap_gain": subject_interval,
                "positive_gain_all_seeds": all(value > 0 for value in seed_gains),
                "scientific_interpretation": "exploratory_only_not_registered",
            })
    return summary


def write_records(records: list[dict], path: Path) -> None:
    fields = ["seed", "dataset", "role", "pairs", "subjects", "single_mfcc_l1", "joint_mfcc_l1",
              "mfcc_gain", "single_delta_l1", "joint_delta_l1", "single_retrieval_r1",
              "joint_retrieval_r1", "chance_r1", "single_template_improvement", "joint_template_improvement"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: ";".join(row[key]) if key == "subjects" else row.get(key) for key in fields})


def _panel_axes():
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=True)
    return figure, axes.flatten()


def plot_mfcc(records: list[dict], output: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    figure, axes = _panel_axes()
    for axis, dataset, role, letter in zip(axes, DATASETS * 2, ("validation", "validation", "test", "test"), "abcd"):
        group = [row for row in records if row["dataset"] == dataset and row["role"] == role]
        for row in group:
            axis.plot([0, 1], [row["single_mfcc_l1"], row["joint_mfcc_l1"]], color="0.72", linewidth=0.8, zorder=1)
            axis.scatter([0, 1], [row["single_mfcc_l1"], row["joint_mfcc_l1"]],
                         color=[COLORS["single"], COLORS["joint"]], s=24, zorder=2)
        means = [np.mean([row["single_mfcc_l1"] for row in group]), np.mean([row["joint_mfcc_l1"] for row in group])]
        axis.plot([0, 1], means, color="black", marker="D", markersize=4, linewidth=1.3, label="seed mean")
        axis.set_xticks([0, 1], ["Single", "Joint"])
        axis.set_ylabel("MFCC L1 error (lower is better)")
        axis.grid(axis="y", alpha=0.18)
        panel_label(axis, f"({letter}) {dataset.upper()} — {role}")
    return save_figure(figure, output, "joint_vs_single_mfcc", formats, dpi)


def plot_controls(records: list[dict], output: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    figure, axes = _panel_axes()
    width = 0.18
    x = np.arange(2)
    labels = {"correct": "Correct", "zero": "Zero", "time_shuffle": "Time", "channel_shuffle": "Channel"}
    for axis, dataset, role, letter in zip(axes, DATASETS * 2, ("validation", "validation", "test", "test"), "abcd"):
        group = [row for row in records if row["dataset"] == dataset and row["role"] == role]
        for index, control in enumerate(CONTROLS):
            means = [np.mean([row[f"{kind}_controls"][control] for row in group]) for kind in ("single", "joint")]
            positions = x + (index - 1.5) * width
            axis.bar(positions, means, width, label=labels[control], color=COLORS[control])
            for method_index, kind in enumerate(("single", "joint")):
                seed_values = [row[f"{kind}_controls"][control] for row in group]
                axis.scatter(np.full(len(seed_values), positions[method_index]), seed_values,
                             color="black", s=8, alpha=0.55, linewidths=0, zorder=3)
        axis.set_xticks(x, ["Single", "Joint"])
        axis.set_ylabel("MFCC L1 error")
        axis.grid(axis="y", alpha=0.18)
        panel_label(axis, f"({letter}) {dataset.upper()} — {role}")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, ncol=4,
                  loc="lower center", bbox_to_anchor=(0.5, -0.025))
    return save_figure(figure, output, "eeg_counterfactual_controls", formats, dpi)


def plot_retrieval(records: list[dict], output: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    figure, axes = _panel_axes()
    for axis, dataset, role, letter in zip(axes, DATASETS * 2, ("validation", "validation", "test", "test"), "abcd"):
        group = [row for row in records if row["dataset"] == dataset and row["role"] == role]
        for row in group:
            axis.plot([0, 1], [row["single_retrieval_r1"], row["joint_retrieval_r1"]], color="0.72", linewidth=0.8)
            axis.scatter([0, 1], [row["single_retrieval_r1"], row["joint_retrieval_r1"]],
                         color=[COLORS["single"], COLORS["joint"]], s=24)
        chances = [row["chance_r1"] for row in group if row["chance_r1"] is not None]
        if chances:
            axis.axhline(float(np.mean(chances)), color=COLORS["chance"], linestyle="--", linewidth=0.9, label="chance")
        axis.set_xticks([0, 1], ["Single", "Joint"])
        axis.set_ylabel("Content retrieval R@1")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.18)
        panel_label(axis, f"({letter}) {dataset.upper()} — {role}")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, legend_labels, frameon=False)
    return save_figure(figure, output, "content_retrieval_r1", formats, dpi)


def training_curves(input_root: Path, seeds: tuple[int, ...], output: Path,
                    formats: tuple[str, ...], dpi: int, sources: list[dict]) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    for axis, dataset, letter in zip(axes, DATASETS, "ab"):
        for kind, mode in (("single", dataset), ("joint", "joint")):
            series: dict[int, list[float]] = defaultdict(list)
            for seed in seeds:
                path = input_root / "generalization" / mode / f"seed-{seed}" / "metrics.json"
                payload = load_json(path)
                sources.append({"path": str(path.resolve()), "sha256": sha256_file(path), "kind": "training_metrics"})
                for item in payload.get("history", []):
                    retrieval = item.get("full_content_retrieval_r1", {})
                    if dataset in retrieval:
                        series[int(item["step"])].append(float(retrieval[dataset]))
            if not series:
                raise RuntimeError(f"training metrics contain no retrieval history for {dataset}/{kind}")
            steps = sorted(series)
            means = np.asarray([np.mean(series[step]) for step in steps])
            lows = np.asarray([np.min(series[step]) for step in steps])
            highs = np.asarray([np.max(series[step]) for step in steps])
            color = COLORS[kind]
            axis.plot(steps, means, color=color, linewidth=1.4, label=kind.capitalize())
            axis.fill_between(steps, lows, highs, color=color, alpha=0.12, linewidth=0)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Train-fold content retrieval R@1")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.18)
        panel_label(axis, f"({letter}) {dataset.upper()}")
    axes[0].legend(frameon=False)
    return save_figure(figure, output, "training_retrieval_curves", formats, dpi)


def write_latex(output: Path) -> None:
    text = r"""% Auto-generated exploratory comparison figures.
% These plots are not registered evidence and require a caption noting the gate bypass.
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{joint_vs_single_mfcc.pdf}
  \caption{Exploratory paired-seed MFCC error comparison for single-dataset and joint training.}
  \label{fig:joint-mfcc-explore}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{eeg_counterfactual_controls.pdf}
  \caption{Exploratory EEG counterfactual controls; a valid EEG-driven model should outperform all controls.}
  \label{fig:eeg-controls-explore}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{content_retrieval_r1.pdf}
  \caption{Exploratory content retrieval under double out-of-distribution subject and content splits.}
  \label{fig:retrieval-explore}
\end{figure}
"""
    (output / "latex_includes.tex").write_text(text)


def main() -> int:
    args = parse_args()
    configure()
    input_root = args.input_root.resolve()
    output = (args.output or input_root / "generalization" / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.replace(" ", ",").split(",") if value)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    formats = tuple(value.strip().lower() for value in args.formats.split(",") if value.strip())
    if not formats or any(value not in {"png", "pdf", "svg"} for value in formats):
        raise ValueError("--formats must contain png, pdf, and/or svg")
    records, sources = collect_results(input_root, seeds)
    summary = summarize(records, int(args.bootstrap_repetitions))
    figure_paths = []
    figure_paths += plot_mfcc(records, output, formats, int(args.dpi))
    figure_paths += plot_controls(records, output, formats, int(args.dpi))
    figure_paths += plot_retrieval(records, output, formats, int(args.dpi))
    figure_paths += training_curves(input_root, seeds, output, formats, int(args.dpi), sources)
    write_records(records, output / "comparison_records.csv")
    (output / "comparison_summary.json").write_text(json.dumps({
        "schema_version": "eeg2speech-joint-comparison-v1",
        "interpretation": "exploratory_only_not_registered",
        "uncertainty": "subject bootstrap after averaging paired gain across seeds; seed spread reported separately",
        "groups": summary,
    }, indent=2) + "\n")
    write_latex(output)
    manifest = {
        "schema_version": "eeg2speech-joint-figure-manifest-v1",
        "input_root": str(input_root), "seeds": list(seeds),
        "interpretation": "exploratory_only_not_registered",
        "figures": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                    for path in figure_paths],
        "sources": sorted(sources, key=lambda item: (item["path"], item["kind"])),
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "pass", "output": str(output), "figures": [str(path) for path in figure_paths],
                      "summary": str(output / "comparison_summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
