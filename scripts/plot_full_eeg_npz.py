#!/usr/bin/env python3
"""Plot a short visual slice from a full EEG NPZ derivative.

The full derivative files are training arrays, not image files. This helper
loads one `*_full_eeg.npz` and writes a compact PNG/SVG for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_slice(path: Path, epoch: int, max_channels: int, start_sec: float, duration_sec: float):
    z = np.load(path, allow_pickle=True)
    eeg = z["eeg"]
    sfreq = float(z["sfreq"])
    ch_names = [str(x) for x in z["ch_names"]]
    eeg_kind = str(z["eeg_kind"])

    if eeg.ndim == 3:
        if epoch < 0 or epoch >= eeg.shape[0]:
            raise ValueError(f"epoch index {epoch} outside range [0, {eeg.shape[0] - 1}]")
        data = eeg[epoch]
        title_epoch = f", epoch={epoch}/{eeg.shape[0] - 1}"
    elif eeg.ndim == 2:
        data = eeg
        title_epoch = ""
    else:
        raise ValueError(f"Expected eeg with 2 or 3 dimensions, got shape {eeg.shape}")

    start = max(0, int(round(start_sec * sfreq)))
    stop = min(data.shape[-1], start + int(round(duration_sec * sfreq)))
    if stop <= start:
        raise ValueError("empty time slice")
    data = data[:max_channels, start:stop].astype("float64")
    names = ch_names[: data.shape[0]]
    times = np.arange(data.shape[1]) / sfreq + start / sfreq
    return data, times, names, sfreq, eeg_kind, title_epoch


def robust_scale(channel: np.ndarray) -> np.ndarray:
    channel = channel - np.nanmedian(channel)
    scale = np.nanpercentile(np.abs(channel), 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(channel)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return channel / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--max-channels", type=int, default=16)
    args = parser.parse_args()

    data, times, names, sfreq, eeg_kind, title_epoch = load_slice(
        args.npz,
        epoch=args.epoch,
        max_channels=args.max_channels,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )

    out = args.out
    if out is None:
        out = args.npz.with_suffix(f".slice_{int(args.start_sec)}s_{int(args.duration_sec)}s.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig_h = max(5.0, 0.36 * data.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)
    for idx, channel in enumerate(data):
        ax.plot(times, robust_scale(channel) + idx * 2.0, linewidth=0.8)
    ax.set_yticks([i * 2.0 for i in range(len(names))])
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{args.npz.name} | {eeg_kind}{title_epoch} | sfreq={sfreq:g} Hz")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(times[0], times[-1])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
