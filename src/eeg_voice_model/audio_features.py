"""Lightweight audio feature helpers for ds005345.

This module intentionally uses the Python standard library so voice embedding
metadata can be created before a full audio stack is installed. The vector is a
statistical placeholder for v0 retrieval; it can later be replaced by HuBERT,
wav2vec, ECAPA, or mel encoders without changing the model head interface.
"""

from __future__ import annotations

import csv
import math
import statistics
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VoiceStats:
    stream: str
    duration_sec: float
    sample_rate: int
    rms: float
    zero_crossing_rate: float
    f0_mean: float
    f0_median: float
    f0_std: float
    voiced_ratio: float
    intensity_mean: float
    intensity_median: float
    intensity_std: float

    def vector(self) -> list[float]:
        return [
            self.duration_sec,
            float(self.sample_rate),
            self.rms,
            self.zero_crossing_rate,
            self.f0_mean,
            self.f0_median,
            self.f0_std,
            self.voiced_ratio,
            self.intensity_mean,
            self.intensity_median,
            self.intensity_std,
        ]


def _read_wav_stats(path: Path, max_seconds: float = 60.0) -> tuple[float, int, float, float]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        total = wf.getnframes()
        frames = wf.readframes(min(total, int(sr * max_seconds)))
    if width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported in v0 stats: {path}")
    values = [int.from_bytes(frames[i : i + 2], "little", signed=True) / 32768.0 for i in range(0, len(frames), 2)]
    mono = []
    if channels > 1:
        for i in range(0, len(values), channels):
            mono.append(sum(values[i : i + channels]) / channels)
    else:
        mono = values
    rms = math.sqrt(sum(x * x for x in mono) / max(1, len(mono)))
    crossings = 0
    for prev, cur in zip(mono, mono[1:]):
        crossings += int((prev >= 0) != (cur >= 0))
    zcr = crossings / max(1, len(mono) - 1)
    return total / sr if sr else 0.0, sr, rms, zcr


def _read_acoustic_stats(path: Path) -> tuple[float, float, float, float, float, float, float]:
    f0: list[float] = []
    intensity: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try:
                f0_value = float(row.get("f0", "0"))
                intensity_value = float(row.get("intensity", "0"))
            except ValueError:
                continue
            if f0_value > 0:
                f0.append(f0_value)
            intensity.append(intensity_value)
    f0_mean = statistics.mean(f0) if f0 else 0.0
    f0_median = statistics.median(f0) if f0 else 0.0
    f0_std = statistics.pstdev(f0) if len(f0) > 1 else 0.0
    voiced_ratio = len(f0) / max(1, len(intensity))
    int_mean = statistics.mean(intensity) if intensity else 0.0
    int_median = statistics.median(intensity) if intensity else 0.0
    int_std = statistics.pstdev(intensity) if len(intensity) > 1 else 0.0
    return f0_mean, f0_median, f0_std, voiced_ratio, int_mean, int_median, int_std


def build_ds005345_voice_stats(root: Path) -> dict[str, VoiceStats]:
    """Build statistical stream embeddings for `single_female`, `single_male`, and `mix`."""
    streams = {
        "single_female": ("single_female.wav", "single_female_acoustic.csv"),
        "single_male": ("single_male.wav", "single_male_acoustic.csv"),
        "mix": ("mix.wav", "mix_acoustic.csv"),
    }
    out: dict[str, VoiceStats] = {}
    for stream, (wav_name, csv_name) in streams.items():
        duration, sr, rms, zcr = _read_wav_stats(root / "stimuli" / wav_name)
        f0_mean, f0_median, f0_std, voiced_ratio, int_mean, int_median, int_std = _read_acoustic_stats(
            root / "annotation" / csv_name
        )
        out[stream] = VoiceStats(
            stream=stream,
            duration_sec=duration,
            sample_rate=sr,
            rms=rms,
            zero_crossing_rate=zcr,
            f0_mean=f0_mean,
            f0_median=f0_median,
            f0_std=f0_std,
            voiced_ratio=voiced_ratio,
            intensity_mean=int_mean,
            intensity_median=int_median,
            intensity_std=int_std,
        )
    return out
