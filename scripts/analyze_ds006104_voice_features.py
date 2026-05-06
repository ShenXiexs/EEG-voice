#!/usr/bin/env python3
"""Analyze local ds006104 speech stimuli for timbre and pitch examples.

The script intentionally uses only the Python standard library so it can run in
the current project without installing a plotting or signal-processing stack.
It writes compact meeting artifacts under data/meeting_examples/ds006104/.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import wave
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path


ROOT = Path("data/meeting_examples/ds006104")
STIMULI = ROOT / "stimuli"


@dataclass
class VoiceFeature:
    path: str
    name: str
    token: str
    emotion: str
    is_control: bool
    unit_type: str
    sample_rate: int
    channels: int
    duration_sec: float
    rms_db: float
    zcr: float
    f0_median_hz: float | None
    f0_iqr_hz: float | None
    voiced_ratio: float
    centroid_proxy_hz: float
    brightness_ratio: float
    bandwidth_proxy_hz: float


def read_wav(path: Path) -> tuple[list[float], int, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    samples: list[float] = []
    if width == 2:
        vals = [int.from_bytes(frames[i : i + 2], "little", signed=True) / 32768.0 for i in range(0, len(frames), 2)]
    elif width == 4:
        vals = [
            int.from_bytes(frames[i : i + 4], "little", signed=True) / 2147483648.0
            for i in range(0, len(frames), 4)
        ]
    elif width == 1:
        vals = [(b - 128) / 128.0 for b in frames]
    else:
        raise ValueError(f"Unsupported sample width {width}: {path}")

    if channels == 1:
        samples = vals
    else:
        for i in range(0, len(vals), channels):
            samples.append(sum(vals[i : i + channels]) / channels)
    return samples, sr, channels


def parse_name(path: Path) -> tuple[str, str, bool, str]:
    stem = path.stem
    is_control = stem.endswith("_control")
    base = stem.removesuffix("_control")
    if "_happy" in base:
        token = base.split("_happy", 1)[0]
        emotion = "happy"
    elif "_angry" in base:
        token = base.split("_angry", 1)[0]
        emotion = "angry"
    else:
        token = base
        emotion = "unknown"

    if len(token) == 1:
        unit_type = "vowel"
    elif len(token) == 2 and token[0].isupper():
        unit_type = "CV"
    elif len(token) == 2 and token[1].isupper():
        unit_type = "VC"
    elif len(token) >= 3:
        unit_type = "word_or_cvc"
    else:
        unit_type = "other"
    return token, emotion, is_control, unit_type


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(x * x for x in samples) / len(samples))


def zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    changes = 0
    prev = samples[0] >= 0
    for x in samples[1:]:
        cur = x >= 0
        if cur != prev:
            changes += 1
        prev = cur
    return changes / (len(samples) - 1)


def downsample_for_pitch(samples: list[float], sr: int, target_sr: int = 8000) -> tuple[list[float], int]:
    step = max(1, round(sr / target_sr))
    return samples[::step], round(sr / step)


def estimate_pitch(samples: list[float], sr: int) -> tuple[float | None, float | None, float]:
    y, ds_sr = downsample_for_pitch(samples, sr)
    frame = max(64, round(ds_sr * 0.04))
    hop = max(16, round(ds_sr * 0.02))
    min_lag = max(1, round(ds_sr / 450.0))
    max_lag = min(frame - 2, round(ds_sr / 60.0))
    if len(y) < frame or max_lag <= min_lag:
        return None, None, 0.0

    f0s: list[float] = []
    frame_count = 0
    max_frames = 40
    for start in range(0, len(y) - frame + 1, hop):
        if frame_count >= max_frames:
            break
        chunk = y[start : start + frame]
        frame_count += 1
        mean = sum(chunk) / len(chunk)
        chunk = [x - mean for x in chunk]
        e = rms(chunk)
        if e < 0.01:
            continue
        base = sum(x * x for x in chunk)
        if base <= 1e-9:
            continue
        best_lag = None
        best_corr = -1.0
        for lag in range(min_lag, max_lag):
            corr = 0.0
            for i in range(0, frame - lag, 2):
                corr += chunk[i] * chunk[i + lag]
            corr = corr / base
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        if best_lag and best_corr > 0.22:
            f0s.append(ds_sr / best_lag)

    if not f0s:
        return None, None, 0.0
    f0s.sort()
    median = statistics.median(f0s)
    q1 = f0s[len(f0s) // 4]
    q3 = f0s[(len(f0s) * 3) // 4]
    return median, q3 - q1, len(f0s) / max(1, frame_count)


def timbre_proxies(samples: list[float], sr: int, zcr: float, signal_rms: float) -> tuple[float, float, float]:
    if len(samples) < 3 or signal_rms <= 1e-12:
        return 0.0, 0.0, 0.0
    diffs = [samples[i] - samples[i - 1] for i in range(1, len(samples))]
    second = [diffs[i] - diffs[i - 1] for i in range(1, len(diffs))]
    diff_rms = rms(diffs)
    second_rms = rms(second)
    centroid_proxy = min(sr / 2, zcr * sr / 2)
    brightness = diff_rms / signal_rms
    bandwidth_proxy = min(sr / 2, second_rms / signal_rms * sr / (2 * math.pi * 20))
    return centroid_proxy, brightness, bandwidth_proxy


def analyze_file(path: Path) -> VoiceFeature:
    samples, sr, channels = read_wav(path)
    token, emotion, is_control, unit_type = parse_name(path)
    signal_rms = rms(samples)
    zcr = zero_crossing_rate(samples)
    f0, f0_iqr, voiced_ratio = estimate_pitch(samples, sr)
    centroid, brightness, bandwidth = timbre_proxies(samples, sr, zcr, signal_rms)
    return VoiceFeature(
        path=str(path),
        name=path.name,
        token=token,
        emotion=emotion,
        is_control=is_control,
        unit_type=unit_type,
        sample_rate=sr,
        channels=channels,
        duration_sec=round(len(samples) / sr, 4),
        rms_db=round(20 * math.log10(signal_rms + 1e-12), 3),
        zcr=round(zcr, 5),
        f0_median_hz=round(f0, 3) if f0 is not None else None,
        f0_iqr_hz=round(f0_iqr, 3) if f0_iqr is not None else None,
        voiced_ratio=round(voiced_ratio, 3),
        centroid_proxy_hz=round(centroid, 3),
        brightness_ratio=round(brightness, 5),
        bandwidth_proxy_hz=round(bandwidth, 3),
    )


def choose_examples(features: list[VoiceFeature]) -> list[VoiceFeature]:
    by_name = {f.name: f for f in features if Path(f.path).parent == STIMULI}
    preferred = [
        "a_happy1.wav",
        "a_angry1.wav",
        "a_happy1_control.wav",
        "Ba_happy1.wav",
        "Ba_angry1.wav",
        "Bad_happy1.wav",
        "Bad_angry1.wav",
    ]
    selected = [by_name[name] for name in preferred if name in by_name]
    if len(selected) >= 4:
        return selected
    fallback = [f for f in features if f.emotion in {"happy", "angry"} and not f.is_control]
    return (selected + fallback)[:8]


def waveform_points(samples: list[float], x0: int, y0: int, width: int, height: int) -> str:
    if not samples:
        return ""
    bins = min(width, len(samples))
    step = max(1, len(samples) // bins)
    points = []
    mid = y0 + height / 2
    scale = height * 0.45
    for idx in range(0, len(samples), step):
        x = x0 + (idx / max(1, len(samples) - 1)) * width
        y = mid - max(-1.0, min(1.0, samples[idx])) * scale
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def write_svg(features: list[VoiceFeature]) -> Path:
    selected = choose_examples(features)
    fig_path = ROOT / "ds006104_voice_timbre_pitch_examples.svg"
    width = 1220
    row_h = 150
    height = 90 + row_h * len(selected)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="30" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#1f2d36">ds006104 local stimuli: timbre and pitch probes</text>',
        '<text x="30" y="64" font-family="Arial" font-size="13" fill="#53636d">Waveform plus pitch/timbre proxy bars. Use these targets before waveform reconstruction.</text>',
    ]
    for i, feature in enumerate(selected):
        samples, sr, _ = read_wav(Path(feature.path))
        y = 95 + i * row_h
        parts.append(f'<text x="30" y="{y}" font-family="Arial" font-size="15" font-weight="700" fill="#1f2d36">{escape(feature.name)}</text>')
        parts.append(f'<text x="30" y="{y + 20}" font-family="Arial" font-size="12" fill="#53636d">emotion={feature.emotion}, token={escape(feature.token)}, duration={feature.duration_sec}s, rms={feature.rms_db}dB</text>')
        parts.append(f'<rect x="30" y="{y + 34}" width="520" height="78" fill="#ffffff" stroke="#d7d2c8"/>')
        parts.append(f'<polyline points="{waveform_points(samples, 35, y + 39, 510, 68)}" fill="none" stroke="#2f4858" stroke-width="1"/>')
        f0 = feature.f0_median_hz or 0
        bars = [
            ("F0", min(f0 / 500, 1.0), f"{f0:.1f} Hz"),
            ("centroid proxy", min(feature.centroid_proxy_hz / 6000, 1.0), f"{feature.centroid_proxy_hz:.0f} Hz"),
            ("brightness", min(feature.brightness_ratio / 1.2, 1.0), f"{feature.brightness_ratio:.2f}"),
            ("bandwidth proxy", min(feature.bandwidth_proxy_hz / 6000, 1.0), f"{feature.bandwidth_proxy_hz:.0f} Hz"),
        ]
        for j, (label, frac, value) in enumerate(bars):
            by = y + 36 + j * 22
            parts.append(f'<text x="590" y="{by + 12}" font-family="Arial" font-size="12" fill="#34444d">{escape(label)}</text>')
            parts.append(f'<rect x="700" y="{by}" width="330" height="14" fill="#eee9df"/>')
            parts.append(f'<rect x="700" y="{by}" width="{330 * frac:.1f}" height="14" fill="#b45f3c"/>')
            parts.append(f'<text x="1045" y="{by + 12}" font-family="Arial" font-size="12" fill="#34444d">{escape(value)}</text>')
    parts.append("</svg>")
    fig_path.write_text("\n".join(parts), encoding="utf-8")
    return fig_path


def median_or_nan(vals: list[float]) -> float:
    return statistics.median(vals) if vals else float("nan")


def summarize(features: list[VoiceFeature], fig_path: Path) -> str:
    by_emotion: dict[str, list[VoiceFeature]] = {}
    for f in features:
        by_emotion.setdefault(f.emotion, []).append(f)
    lines = [
        "# ds006104 Voice Timbre/Pitch Local Summary",
        "",
        "这个目录用于当前项目的说话形象重构方向：EEG -> token -> 与声音内容/音色/音调表征对齐 -> 说话形象重构。",
        "",
        f"- stimuli root: `{STIMULI}`",
        f"- wav files: {len(features)}",
        f"- pitch-estimable files: {sum(1 for f in features if f.f0_median_hz is not None)}",
        f"- figure: `{fig_path}`",
        "",
        "## Condition medians",
        "",
        "| condition | n | median F0 Hz | centroid proxy Hz | brightness | duration s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for emotion in sorted(by_emotion):
        rows = by_emotion[emotion]
        f0 = median_or_nan([x.f0_median_hz for x in rows if x.f0_median_hz is not None])
        centroid = median_or_nan([x.centroid_proxy_hz for x in rows])
        brightness = median_or_nan([x.brightness_ratio for x in rows])
        duration = median_or_nan([x.duration_sec for x in rows])
        lines.append(f"| {emotion} | {len(rows)} | {f0:.2f} | {centroid:.2f} | {brightness:.3f} | {duration:.3f} |")
    lines.extend(
        [
            "",
            "## Recommended use",
            "",
            "- 先把音频侧特征作为对齐目标：content unit、log-F0、voicing、mel 或谱包络、brightness/centroid/bandwidth。",
            "- 再把 happy/angry/control 作为 probe，检查 EEG token 是否保留声音内容、音调和音色线索。",
            "- 训练 waveform vocoder 之前，先报告 content retrieval、F0 correlation、声学特征误差、timbre embedding cosine 和条件分类准确率。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not STIMULI.exists():
        raise SystemExit(f"Missing local stimuli directory: {STIMULI}")
    wavs = sorted(p for p in STIMULI.rglob("*.wav") if p.is_file())
    if not wavs:
        raise SystemExit(f"No wav files found under {STIMULI}")

    features = [analyze_file(path) for path in wavs]
    ROOT.mkdir(parents=True, exist_ok=True)
    json_path = ROOT / "voice_timbre_pitch_manifest.json"
    csv_path = ROOT / "voice_timbre_pitch_manifest.csv"
    fig_path = write_svg(features)
    summary_path = ROOT / "voice_timbre_pitch_summary.md"

    json_path.write_text(json.dumps([asdict(f) for f in features], indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(features[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in features)
    summary_path.write_text(summarize(features, fig_path), encoding="utf-8")

    print(json.dumps({
        "wav_files": len(features),
        "json": str(json_path),
        "csv": str(csv_path),
        "figure": str(fig_path),
        "summary": str(summary_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
