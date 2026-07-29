#!/usr/bin/env python3
"""Render reference-versus-v0730-fixed WAV comparison panels.

This is a presentation-only post-processing step.  It reads the already
exported WAV pairs listed in ``manifest.csv`` and never loads a model or
changes the WAV files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/open_vocab_0730_pair_plots_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/open_vocab_0730_pair_plots_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot existing v0730-fixed reference/reconstruction WAV pairs; no training or inference."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0, help="0 means all pairs")
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def read_wave(path: Path) -> tuple[np.ndarray, int]:
    sample_rate, waveform = wavfile.read(path)
    value = np.asarray(waveform)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float32) / float(np.iinfo(value.dtype).max)
    else:
        value = value.astype(np.float32)
    return value.reshape(-1), int(sample_rate)


def display_wave(waveform: np.ndarray, sample_rate: int, maximum: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    if len(waveform) <= maximum:
        indices = np.arange(len(waveform))
    else:
        indices = np.linspace(0, len(waveform) - 1, maximum, dtype=np.int64)
    return indices / float(sample_rate), waveform[indices]


def envelope(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, round(0.025 * sample_rate))
    hop = max(1, round(0.010 * sample_rate))
    if len(waveform) < frame:
        padded = np.pad(waveform, (0, frame - len(waveform)))
    else:
        padded = waveform
    starts = np.arange(0, max(1, len(padded) - frame + 1), hop)
    values = np.array([np.sqrt(np.mean(np.square(padded[start : start + frame])) + 1e-12) for start in starts])
    return (starts + frame / 2.0) / float(sample_rate), values


def log_spectrogram(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(512, max(32, len(waveform)))
    _, times, spectrum = stft(waveform, fs=sample_rate, nperseg=nperseg, noverlap=nperseg * 3 // 4, padded=False)
    db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-5))
    frequencies = np.linspace(0.0, sample_rate / 2.0, spectrum.shape[0])
    return times, frequencies, db


def plot_pair(row: dict[str, str], output: Path, dpi: int) -> None:
    reference, reference_rate = read_wave(Path(row["reference_wav"]))
    reconstruction, reconstruction_rate = read_wave(Path(row["reconstruction_wav"]))
    if reference_rate != reconstruction_rate:
        raise ValueError(f"Sample-rate mismatch for {row['sample_key']}: {reference_rate} != {reconstruction_rate}")
    sample_rate = reference_rate
    ref_time, ref_display = display_wave(reference, sample_rate)
    rec_time, rec_display = display_wave(reconstruction, sample_rate)
    ref_env_time, ref_env = envelope(reference, sample_rate)
    rec_env_time, rec_env = envelope(reconstruction, sample_rate)
    ref_spec_time, freqs, ref_spec = log_spectrogram(reference, sample_rate)
    rec_spec_time, _, rec_spec = log_spectrogram(reconstruction, sample_rate)
    color_max = max(float(np.max(ref_spec)), float(np.max(rec_spec)))
    color_min = color_max - 80.0

    figure, axes = plt.subplots(2, 2, figsize=(14, 7.2), constrained_layout=True)
    figure.suptitle(
        f"{row['sample_key']} | subject={row['subject']} | label={row['label']} | role={row['evaluation_role']}\n"
        f"reference={len(reference) / sample_rate:.3f}s; reconstruction={len(reconstruction) / sample_rate:.3f}s; "
        f"predicted={float(row['predicted_duration_seconds']):.3f}s",
        fontsize=10,
    )
    axes[0, 0].plot(ref_time, ref_display, color="0.38", linewidth=0.55, label="reference")
    axes[0, 0].plot(rec_time, rec_display, color="#2563eb", linewidth=0.55, alpha=0.85, label="reconstruction")
    axes[0, 0].set_title("Waveform (native time scale)", loc="left", fontsize=9)
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 0].grid(alpha=0.15)
    axes[0, 1].plot(ref_env_time, ref_env, color="0.38", linewidth=0.9, label="reference")
    axes[0, 1].plot(rec_env_time, rec_env, color="#dc2626", linewidth=0.9, label="reconstruction")
    axes[0, 1].set_title("25 ms RMS envelope (10 ms hop)", loc="left", fontsize=9)
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    axes[0, 1].grid(alpha=0.15)
    axes[1, 0].pcolormesh(ref_spec_time, freqs / 1000.0, ref_spec, shading="auto", cmap="magma", vmin=color_min, vmax=color_max)
    axes[1, 0].set_title("Reference log magnitude spectrogram", loc="left", fontsize=9)
    axes[1, 1].pcolormesh(rec_spec_time, freqs / 1000.0, rec_spec, shading="auto", cmap="magma", vmin=color_min, vmax=color_max)
    axes[1, 1].set_title("Reconstruction log magnitude spectrogram", loc="left", fontsize=9)
    for axis in axes[1, :]:
        axis.set_xlabel("time (s)")
        axis.set_ylabel("frequency (kHz)")
        axis.set_ylim(0.0, min(8.0, sample_rate / 2000.0))
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    output = (args.output or manifest.parent / "comparison_pairs").resolve()
    output.mkdir(parents=True, exist_ok=True)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No pair rows in {manifest}")
    if args.limit > 0:
        rows = rows[: args.limit]
    written: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="[0730-fixed pairs] PNG render", unit="pair", dynamic_ncols=True, mininterval=0.5):
        filename = safe_name(row["sample_key"]) + ".png"
        png = output / filename
        if not (args.resume_existing and png.is_file() and png.stat().st_size > 0):
            plot_pair(row, png, int(args.dpi))
        written.append({"sample_key": row["sample_key"], "png": str(png)})
    (output / "comparison_manifest.json").write_text(
        json.dumps({"schema_version": "openvoice-0730-fixed-comparison-plots-v1", "source_manifest": str(manifest), "plots": written}, indent=2),
        encoding="utf-8",
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
