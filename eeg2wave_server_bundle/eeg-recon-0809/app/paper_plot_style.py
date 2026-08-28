"""Shared publication-style settings for EEG-to-speech result figures."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eeg2speech_joint_figures_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/eeg2speech_joint_figures_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "single": "#0072B2",
    "joint": "#D55E00",
    "correct": "#009E73",
    "zero": "#CC79A7",
    "time_shuffle": "#E69F00",
    "channel_shuffle": "#56B4E9",
    "chance": "#666666",
}


def configure() -> None:
    matplotlib.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "mathtext.fontset": "stix",
    })


def panel_label(axis, text: str) -> None:
    axis.text(0.0, 1.02, text, transform=axis.transAxes, ha="left", va="bottom", fontweight="bold")


def save_figure(figure, output_dir: Path, stem: str, formats: tuple[str, ...], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        target = output_dir / f"{stem}.{extension}"
        figure.savefig(target, dpi=dpi if extension == "png" else None)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"figure was not written: {target}")
        written.append(target)
    plt.close(figure)
    return written
