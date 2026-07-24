#!/usr/bin/env python3
"""Render numerical reference-vs-v0724 reconstruction comparison figures.

The script reads the float ``.mel.npy`` tensors and WAVs emitted by
``synthesize_open_vocab_0724.py``.  PNGs are presentation-only outputs: no
metric is calculated from rendered pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openvoice_0724_pair_plots_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/openvoice_0724_pair_plots_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0722.audio_io import read_wav  # noqa: E402
from src.open_vocab_0724.metrics import log_mel, rms_envelope  # noqa: E402
from src.open_vocab_0724.runtime import (  # noqa: E402
    load_config,
    resolve_config_path,
    run_identifier,
    write_json,
)


DISPLAY_MODES = (
    "correct_content_correct_realization",
    "correct_content_wrong_realization",
    "wrong_content_correct_realization",
    "wrong_content_wrong_realization",
    "content_only",
    "realization_only",
    "shuffled_eeg",
    "zero_eeg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot v0724 reference-versus-reconstruction waveform, envelope, "
            "and numeric log-mel comparison panels"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("karaone", "feis"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--synthesis-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generalization", choices=("g1", "g2", "g3"), default="g1")
    parser.add_argument("--holdout-label", default=None)
    parser.add_argument("--loso-subject", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--modes",
        default=",".join(DISPLAY_MODES),
        help="Comma-separated generated modes to include in every comparison panel",
    )
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not modes:
        raise ValueError("--modes must contain at least one generation mode")
    return modes


def resolve_synthesis_root(args: argparse.Namespace) -> Path:
    if args.synthesis_root is not None:
        return args.synthesis_root.resolve()
    config_path, cfg = load_config(args.config)
    seed = int(cfg["training"]["seed"] if args.seed is None else args.seed)
    run_id = run_identifier(
        cfg,
        seed=seed,
        loso_subject=args.loso_subject,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
    )
    root = (
        resolve_config_path(config_path, cfg["paths"]["output_root"])
        / "synthesis"
        / str(args.dataset)
        / str(args.split)
    )
    return root / "runs" / run_id if run_id is not None else root


def unit_rms(waveform: np.ndarray) -> np.ndarray:
    value = np.asarray(waveform, dtype=np.float64).reshape(-1)
    rms = np.sqrt(np.mean(np.square(value)) + 1e-12)
    return (value / rms if rms > 1e-8 else value).astype(np.float32)


def time_axis(samples: int, sample_rate: int) -> np.ndarray:
    return np.arange(max(0, int(samples)), dtype=np.float64) / float(sample_rate)


def metric_text(metrics: dict[str, Any]) -> str:
    def value(name: str) -> float:
        raw = metrics.get(name, float("nan"))
        return float(raw) if raw is not None else float("nan")

    return (
        f"SSIM={value('morphology_ssim'):.3f}; "
        f"soft-DTW={value('soft_dtw_divergence'):.3f}; "
        f"mel-MAE={value('native_log_mel_mae_db'):.2f} dB; "
        f"duration error={value('predicted_duration_error_seconds'):.3f}s"
    )


def stacked_energy(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Energy maps must be two-dimensional [mel,time] tensors")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("Reference and candidate energy maps need the same mel bins")
    frames = max(reference.shape[1], candidate.shape[1])
    output = np.full((reference.shape[0] * 2, frames), -80.0, dtype=np.float32)
    output[: reference.shape[0], : reference.shape[1]] = reference
    output[reference.shape[0] :, : candidate.shape[1]] = candidate
    return output


def plot_record(
    root: Path,
    record: dict[str, Any],
    modes: Iterable[str],
    destination: Path,
    *,
    dpi: int,
) -> list[str]:
    stem = str(record["stem"])
    reference_wav, sample_rate = read_wav(root / "reference" / f"{stem}.wav")
    reference = unit_rms(reference_wav)
    reference_mel = np.asarray(
        np.load(root / "reference" / f"{stem}.mel.npy", allow_pickle=False),
        dtype=np.float32,
    )
    requested = tuple(modes)
    missing: list[str] = []
    for mode in requested:
        missing_parts: list[str] = []
        if not (root / mode / f"{stem}.wav").is_file():
            missing_parts.append("WAV")
        if not (root / mode / f"{stem}.mel.npy").is_file():
            missing_parts.append("mel")
        if missing_parts:
            missing.append(f"{mode} ({', '.join(missing_parts)})")
    if missing:
        raise FileNotFoundError(
            "Incomplete v0724 synthesis output for comparison figure "
            f"{stem}: {'; '.join(missing)}"
        )

    reference_time = time_axis(len(reference), sample_rate)
    reference_envelope = rms_envelope(reference, sample_rate)
    envelope_time = np.arange(len(reference_envelope), dtype=np.float64) * 0.01
    figure, axes = plt.subplots(
        len(requested),
        4,
        figsize=(23, max(5.0, len(requested) * 3.1)),
        constrained_layout=True,
        squeeze=False,
    )
    metrics_by_mode = record.get("metrics") or {}
    written_modes: list[str] = []
    for row, mode in enumerate(requested):
        candidate_wav, candidate_rate = read_wav(root / mode / f"{stem}.wav")
        if candidate_rate != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch for {stem}/{mode}: {candidate_rate} != {sample_rate}"
            )
        candidate = unit_rms(candidate_wav)
        candidate_time = time_axis(len(candidate), sample_rate)
        candidate_envelope = rms_envelope(candidate, sample_rate)
        candidate_envelope_time = (
            np.arange(len(candidate_envelope), dtype=np.float64) * 0.01
        )
        candidate_mel = np.asarray(
            np.load(root / mode / f"{stem}.mel.npy", allow_pickle=False),
            dtype=np.float32,
        )
        decoded_reference_mel = log_mel(reference_wav, sample_rate)
        decoded_candidate_mel = log_mel(candidate_wav, sample_rate)
        metrics = dict(metrics_by_mode.get(mode) or {})

        waveform_axis, envelope_axis, predicted_energy_axis, decoded_energy_axis = axes[
            row
        ]
        waveform_axis.plot(
            reference_time, reference, color="0.55", linewidth=0.6, label="reference"
        )
        waveform_axis.plot(
            candidate_time,
            candidate,
            color="#2563eb",
            linewidth=0.65,
            alpha=0.9,
            label=mode,
        )
        waveform_axis.set_title(f"{mode} | waveform", loc="left", fontsize=9)
        waveform_axis.grid(alpha=0.15)

        envelope_axis.plot(
            envelope_time,
            reference_envelope,
            color="0.45",
            linewidth=0.8,
            label="reference envelope",
        )
        envelope_axis.plot(
            candidate_envelope_time,
            candidate_envelope,
            color="#dc2626",
            linewidth=0.8,
            label=f"{mode} envelope",
        )
        envelope_axis.set_title(metric_text(metrics), loc="left", fontsize=8)
        envelope_axis.set_xlabel("time (s)")
        envelope_axis.grid(alpha=0.15)

        predicted_energy_axis.imshow(
            stacked_energy(reference_mel, candidate_mel),
            origin="lower",
            aspect="auto",
            vmin=-80.0,
            vmax=0.0,
            cmap="magma",
        )
        predicted_energy_axis.axhline(
            reference_mel.shape[0] - 0.5, color="white", linewidth=0.7
        )
        predicted_energy_axis.set_yticks(
            [reference_mel.shape[0] / 2, reference_mel.shape[0] * 1.5],
            ["reference", mode],
        )
        predicted_energy_axis.set_xlabel(
            "10-ms frame (native time; no frequency scaling)"
        )
        predicted_energy_axis.set_title(
            "explicit energy output: reference / predicted map", fontsize=8
        )

        decoded_energy_axis.imshow(
            stacked_energy(decoded_reference_mel, decoded_candidate_mel),
            origin="lower",
            aspect="auto",
            vmin=-80.0,
            vmax=0.0,
            cmap="magma",
        )
        decoded_energy_axis.axhline(
            decoded_reference_mel.shape[0] - 0.5, color="white", linewidth=0.7
        )
        decoded_energy_axis.set_yticks(
            [
                decoded_reference_mel.shape[0] / 2,
                decoded_reference_mel.shape[0] * 1.5,
            ],
            ["reference WAV", "decoded WAV"],
        )
        decoded_energy_axis.set_xlabel(
            "10-ms frame (native time; no frequency scaling)"
        )
        decoded_energy_axis.set_title(
            "waveform-derived log-mel: reference / decoded", fontsize=8
        )
        written_modes.append(mode)

    axes[0, 0].legend(loc="upper right", fontsize=7)
    axes[0, 1].legend(loc="upper right", fontsize=7)
    figure.suptitle(
        "v0724 reconstruction comparison | "
        f"sample={record.get('sample_key')} | label={record.get('label')}\n"
        "waveforms RMS-normalized for display; metrics come from numerical tensors",
        fontsize=12,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(dpi))
    plt.close(figure)
    return written_modes


def render_comparisons(
    root: Path,
    output: Path,
    *,
    modes: tuple[str, ...],
    limit: int,
    dpi: int,
) -> dict[str, Any]:
    manifest_path = root / "synthesis_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing v0724 synthesis manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(manifest.get("records") or [])
    if not records:
        raise ValueError("v0724 synthesis manifest has no records to plot")
    if limit < -1 or limit == 0:
        raise ValueError("--limit must be -1 or a positive number of records")
    if limit >= 0:
        records = records[:limit]
    output.mkdir(parents=True, exist_ok=True)
    plots: list[dict[str, Any]] = []
    for record in tqdm(records, desc="[0724 comparison plots]", unit="figure"):
        stem = str(record.get("stem") or safe_name(str(record.get("sample_key"))))
        figure = output / f"{safe_name(stem)}.png"
        modes_written = plot_record(root, record, modes, figure, dpi=dpi)
        if modes_written:
            plots.append(
                {
                    "sample_key": str(record.get("sample_key")),
                    "label": str(record.get("label")),
                    "figure": str(figure),
                    "modes": modes_written,
                }
            )
    if len(plots) != len(records):
        raise RuntimeError(
            "Every v0724 synthesis record must produce one complete comparison "
            f"figure; wrote {len(plots)} of {len(records)}"
        )
    summary = {
        "schema_version": "openvoice-0724-comparison-plots-v1",
        "source_manifest": str(manifest_path),
        "source_manifest_schema": manifest.get("schema_version"),
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split"),
        "plots_written": len(plots),
        "requested_modes": list(modes),
        "metrics_use_png_pixels": False,
        "frequency_axis_scaled": False,
        "energy_panels": {
            "explicit": "reference cache log-mel / predicted condition log-mel",
            "decoded": "reference WAV log-mel / reconstructed WAV log-mel",
        },
        "plots": plots,
    }
    write_json(output / "comparison_manifest.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    root = resolve_synthesis_root(args)
    output = args.output.resolve() if args.output else root / "comparison_pairs"
    summary = render_comparisons(
        root,
        output,
        modes=parse_modes(args.modes),
        limit=int(args.limit),
        dpi=int(args.dpi),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
