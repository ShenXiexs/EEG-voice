#!/usr/bin/env python3
"""Prepare small EEG/audio examples for a research meeting.

The script intentionally downloads only presentation-sized assets:
- short speech audio clips from the selected OpenNeuro datasets;
- small byte ranges from raw EEG files where possible;
- PNG figures that pair the audio waveform with EEG/event evidence.

Outputs are written under data/meeting_examples/ by default. This directory is
ignored by git in this project.
"""

from __future__ import annotations

import csv
import io
import json
import math
import struct
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.io import loadmat


S3_BASE = "https://s3.amazonaws.com/openneuro.org"
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
OUT = Path("data/meeting_examples")


@dataclass
class AudioClip:
    dataset: str
    label: str
    url: str
    start_sec: float
    duration_sec: float


@dataclass
class BrainVisionSpec:
    dataset: str
    label: str
    vhdr_url: str
    eeg_url: str
    start_sec: float
    duration_sec: float


def s3_url(key: str) -> str:
    return f"{S3_BASE}/{key}"


def get_bytes(url: str, byte_range: tuple[int, int] | None = None, timeout: int = 60) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last_exc: Exception | None = None
    for attempt in range(1, 5):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(0.75 * attempt)
    assert last_exc is not None
    raise last_exc


def write_bytes_once(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != len(blob):
        path.write_bytes(blob)


def download_file_once(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {path}")
    tmp = path.with_suffix(path.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, 5):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(path)
            return
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(1.0 * attempt)
    assert last_exc is not None
    raise last_exc


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    return data.reshape(-1, n_ch), sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def make_audio_clip(spec: AudioClip) -> dict[str, Any]:
    d = OUT / spec.dataset
    full_path = d / "raw" / Path(spec.url).name.replace("%20", "_")
    clip_path = d / f"{spec.dataset}_{spec.label}_audio_{int(spec.duration_sec)}s.wav"
    download_file_once(spec.url, full_path)
    audio, sr = read_wav(full_path)
    start = int(round(spec.start_sec * sr))
    dur = int(round(spec.duration_sec * sr))
    clip = audio[start : start + dur]
    write_wav(clip_path, clip, sr)
    actual_duration = round(len(clip) / sr, 3)
    return {
        "dataset": spec.dataset,
        "audio_label": spec.label,
        "source_url": spec.url,
        "source_path": str(full_path),
        "clip_path": str(clip_path),
        "sample_rate": sr,
        "channels": int(clip.shape[1]) if clip.ndim == 2 else 1,
        "start_sec": spec.start_sec,
        "duration_sec": spec.duration_sec,
        "actual_duration_sec": actual_duration,
    }


def make_local_audio_entry(dataset: str, label: str, path: Path) -> dict[str, Any]:
    audio, sr = read_wav(path)
    return {
        "dataset": dataset,
        "audio_label": label,
        "source_url": None,
        "source_path": str(path),
        "clip_path": str(path),
        "sample_rate": sr,
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "start_sec": 0,
        "duration_sec": round(len(audio) / sr, 3),
        "actual_duration_sec": round(len(audio) / sr, 3),
    }


def parse_brainvision_header(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    channels = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("NumberOfChannels="):
            out["n_channels"] = int(line.split("=", 1)[1])
        elif line.startswith("SamplingInterval="):
            out["sampling_interval_us"] = float(line.split("=", 1)[1])
        elif line.startswith("BinaryFormat="):
            out["binary_format"] = line.split("=", 1)[1].strip()
        elif line.startswith("Ch") and "=" in line:
            _, rhs = line.split("=", 1)
            parts = rhs.split(",")
            name = parts[0]
            resolution = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
            channels.append((name, resolution))
    out["sfreq"] = 1_000_000.0 / out["sampling_interval_us"]
    out["channels"] = channels
    return out


def read_brainvision_partial(spec: BrainVisionSpec) -> tuple[np.ndarray, float, list[str]]:
    vhdr = get_bytes(spec.vhdr_url).decode("utf-8", errors="replace")
    meta = parse_brainvision_header(vhdr)
    n_ch = int(meta["n_channels"])
    sfreq = float(meta["sfreq"])
    fmt = str(meta["binary_format"])
    if fmt == "IEEE_FLOAT_32":
        dtype = "<f4"
        sample_bytes = 4
    elif fmt == "INT_16":
        dtype = "<i2"
        sample_bytes = 2
    else:
        raise ValueError(f"Unsupported BrainVision binary format: {fmt}")
    start_sample = int(round(spec.start_sec * sfreq))
    n_samples = int(round(spec.duration_sec * sfreq))
    byte_start = start_sample * n_ch * sample_bytes
    byte_end = byte_start + n_samples * n_ch * sample_bytes - 1
    blob = get_bytes(spec.eeg_url, (byte_start, byte_end))
    raw = np.frombuffer(blob, dtype=dtype).reshape(-1, n_ch).astype(np.float32)
    resolutions = np.array([r for _, r in meta["channels"]], dtype=np.float32)
    raw *= resolutions
    names = [name for name, _ in meta["channels"]]
    return raw.T, sfreq, names


def parse_edf_header(blob: bytes) -> dict[str, Any]:
    header_bytes = int(blob[184:192].decode("ascii").strip())
    n_records = int(blob[236:244].decode("ascii").strip())
    record_duration = float(blob[244:252].decode("ascii").strip())
    n_signals = int(blob[252:256].decode("ascii").strip())
    pos = 256

    def field(width: int) -> list[str]:
        nonlocal pos
        vals = [
            blob[pos + i * width : pos + (i + 1) * width].decode("latin1").strip()
            for i in range(n_signals)
        ]
        pos += width * n_signals
        return vals

    labels = field(16)
    field(80)  # transducer
    field(8)  # physical dimension
    phys_min = np.array([float(x) for x in field(8)], dtype=np.float64)
    phys_max = np.array([float(x) for x in field(8)], dtype=np.float64)
    dig_min = np.array([float(x) for x in field(8)], dtype=np.float64)
    dig_max = np.array([float(x) for x in field(8)], dtype=np.float64)
    field(80)  # prefilter
    samples_per_record = np.array([int(x) for x in field(8)], dtype=np.int64)
    record_bytes = int(samples_per_record.sum() * 2)
    return {
        "header_bytes": header_bytes,
        "n_records": n_records,
        "record_duration": record_duration,
        "n_signals": n_signals,
        "labels": labels,
        "phys_min": phys_min,
        "phys_max": phys_max,
        "dig_min": dig_min,
        "dig_max": dig_max,
        "samples_per_record": samples_per_record,
        "record_bytes": record_bytes,
    }


def read_edf_partial(url: str, start_sec: float, duration_sec: float) -> tuple[np.ndarray, float, list[str]]:
    first_header = get_bytes(url, (0, 255))
    header_bytes = int(first_header[184:192].decode("ascii").strip())
    header_probe = get_bytes(url, (0, header_bytes - 1))
    h = parse_edf_header(header_probe)
    record_duration = float(h["record_duration"])
    start_record = int(math.floor(start_sec / record_duration))
    n_records = int(math.ceil(duration_sec / record_duration))
    byte_start = h["header_bytes"] + start_record * h["record_bytes"]
    byte_end = byte_start + n_records * h["record_bytes"] - 1
    blob = get_bytes(url, (byte_start, byte_end), timeout=90)
    ints = np.frombuffer(blob, dtype="<i2")
    spr = h["samples_per_record"]
    n_signals = int(h["n_signals"])
    per_record = int(spr.sum())
    records = ints[: n_records * per_record].reshape(n_records, per_record)
    n_use = min(8, n_signals)
    chans = []
    for ch in range(n_use):
        pieces = []
        offset = int(spr[:ch].sum())
        n = int(spr[ch])
        for rec in records:
            pieces.append(rec[offset : offset + n])
        digital = np.concatenate(pieces).astype(np.float64)
        scale = (h["phys_max"][ch] - h["phys_min"][ch]) / (h["dig_max"][ch] - h["dig_min"][ch])
        phys = (digital - h["dig_min"][ch]) * scale + h["phys_min"][ch]
        chans.append(phys.astype(np.float32))
    sfreq = float(spr[0] / record_duration)
    return np.vstack(chans), sfreq, h["labels"][:n_use]


def read_eeglab_fdt_partial(set_url: str, fdt_url: str, start_sec: float, duration_sec: float) -> tuple[np.ndarray, float, list[str]]:
    d = OUT / "ds004718" / "raw"
    set_path = d / Path(set_url).name
    download_file_once(set_url, set_path)
    mat = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    # EEGLAB .set files may be saved either as a single EEG struct or as a
    # flat MATLAB struct with EEG fields at the top level.
    eeg = mat.get("EEG")
    if eeg is None:
        n_ch = int(mat["nbchan"])
        sfreq = float(mat["srate"])
        chanlocs = np.atleast_1d(mat["chanlocs"])
    else:
        n_ch = int(eeg.nbchan)
        sfreq = float(eeg.srate)
        chanlocs = np.atleast_1d(eeg.chanlocs)
    names = [str(getattr(c, "labels", f"Ch{i+1}")) for i, c in enumerate(chanlocs)]
    start_sample = int(round(start_sec * sfreq))
    n_samples = int(round(duration_sec * sfreq))
    byte_start = start_sample * n_ch * 4
    byte_end = byte_start + n_samples * n_ch * 4 - 1
    blob = get_bytes(fdt_url, (byte_start, byte_end), timeout=90)
    # EEGLAB .fdt is channel-major float32 for continuous data.
    raw = np.frombuffer(blob, dtype="<f4").reshape(n_samples, n_ch).T
    return raw, sfreq, names


def parse_textgrid_words(path: Path, start_sec: float, duration_sec: float) -> list[tuple[float, float, str]]:
    text = path.read_text(errors="replace")
    rows = []
    intervals = text.split("intervals [")
    for chunk in intervals:
        try:
            xmin = float(chunk.split("xmin =", 1)[1].splitlines()[0].strip())
            xmax = float(chunk.split("xmax =", 1)[1].splitlines()[0].strip())
            label = chunk.split("text =", 1)[1].splitlines()[0].strip().strip('"')
        except Exception:
            continue
        if label and xmax >= start_sec and xmin <= start_sec + duration_sec:
            rows.append((xmin - start_sec, xmax - start_sec, label))
    return rows[:20]


def read_csv_rows(path: Path, limit: int = 2000) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return [row for _, row in zip(range(limit), csv.DictReader(f))]


def plot_pair(
    dataset: str,
    title: str,
    audio_info: dict[str, Any] | None,
    eeg: np.ndarray | None,
    sfreq: float | None,
    ch_names: list[str] | None,
    annotations: list[tuple[float, float, str]] | None,
    notes: list[str],
) -> str:
    fig_path = OUT / dataset / f"{dataset}_meeting_pair.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 3 if audio_info else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 7.2), constrained_layout=True)
    if rows == 2:
        axes = [None, axes[0], axes[1]]
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if audio_info:
        audio, sr = read_wav(Path(audio_info["clip_path"]))
        mono = audio.mean(axis=1)
        t = np.arange(len(mono)) / sr
        axes[0].plot(t, mono, color="#34495e", lw=0.8)
        axes[0].set_title(f"Audio clip: {Path(audio_info['clip_path']).name}")
        axes[0].set_ylabel("amplitude")
        axes[0].set_xlim(0, t[-1] if len(t) else 1)
        if annotations:
            ymax = max(0.2, float(np.nanmax(np.abs(mono))) if len(mono) else 1)
            for x0, x1, label in annotations:
                if x1 < 0 or x0 > t[-1]:
                    continue
                axes[0].axvspan(max(0, x0), min(t[-1], x1), color="#f39c12", alpha=0.12)
                if label.strip():
                    axes[0].text(max(0, x0), ymax * 0.92, label[:14], fontsize=8, rotation=25)

    if eeg is not None and sfreq is not None:
        use = min(8, eeg.shape[0])
        n = eeg.shape[1]
        tt = np.arange(n) / sfreq
        traces = eeg[:use].copy()
        traces -= np.nanmedian(traces, axis=1, keepdims=True)
        scale = np.nanpercentile(np.abs(traces), 95) or 1.0
        offset = np.arange(use)[::-1] * 3.0
        top_y = float(offset.max() + 1.6) if len(offset) else 1.0
        for i in range(use):
            normalized = np.clip(traces[i] / scale, -1.2, 1.2)
            axes[1].plot(tt, normalized + offset[i], lw=0.7)
        if annotations and not audio_info:
            for x0, _, label in annotations:
                if 0 <= x0 <= (tt[-1] if len(tt) else 0):
                    axes[1].axvline(x0, color="#c0392b", lw=0.9, alpha=0.7)
                    axes[1].text(x0, top_y, label[:8], rotation=30, fontsize=8, color="#922b21")
        axes[1].set_title("EEG snippet from matched/open dataset recording")
        axes[1].set_yticks(offset)
        axes[1].set_yticklabels((ch_names or [f"Ch{i+1}" for i in range(use)])[:use])
        axes[1].set_xlabel("seconds")
        axes[1].set_xlim(0, tt[-1] if len(tt) else 1)
    else:
        axes[1].axis("off")
        axes[1].text(0.02, 0.5, "No parseable EEG snippet generated.", fontsize=12)

    axes[2].axis("off")
    axes[2].text(0.02, 0.95, "\n".join(notes), va="top", fontsize=10, family="monospace")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return str(fig_path)


def prepare_ds004408() -> dict[str, Any]:
    ds = "ds004408"
    audio = make_audio_clip(
        AudioClip(ds, "audio01_english_natural_speech", s3_url("ds004408/stimuli/audio01.wav"), 20, 10)
    )
    tg_path = OUT / ds / "raw" / "audio01.TextGrid"
    write_bytes_once(tg_path, get_bytes(s3_url("ds004408/stimuli/audio01.TextGrid")))
    eeg, sfreq, ch_names = read_brainvision_partial(
        BrainVisionSpec(
            ds,
            "sub-001 run-01",
            s3_url("ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.vhdr"),
            s3_url("ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.eeg"),
            20,
            10,
        )
    )
    fig = plot_pair(
        ds,
        "ds004408: English naturalistic speech + 128ch EEG",
        audio,
        eeg,
        sfreq,
        ch_names,
        parse_textgrid_words(tg_path, 20, 10),
        [
            "Use: tokenizer pretraining; phoneme/onset supervision.",
            "Audio: actual audio01.wav, 20-30 s clip.",
            "EEG: sub-001 run-01 BrainVision byte-range, same 20-30 s window.",
            "Annotation: TextGrid labels overlaid on audio waveform.",
        ],
    )
    return {"dataset": ds, "audio": audio, "figure": fig, "eeg_sfreq": sfreq, "eeg_channels": len(ch_names)}


def prepare_ds005345() -> dict[str, Any]:
    ds = "ds005345"
    audio = make_audio_clip(
        AudioClip(ds, "single_female_mandarin", s3_url("ds005345/stimuli/single_female.wav"), 30, 10)
    )
    male_audio = make_audio_clip(
        AudioClip(ds, "single_male_mandarin", s3_url("ds005345/stimuli/single_male.wav"), 30, 10)
    )
    mix_audio = make_audio_clip(
        AudioClip(ds, "mix_two_talker_mandarin", s3_url("ds005345/stimuli/mix.wav"), 30, 10)
    )
    word_csv = OUT / ds / "raw" / "single_female_word_information.csv"
    write_bytes_once(word_csv, get_bytes(s3_url("ds005345/annotation/single_female_word_information.csv")))
    annotations = []
    for row in read_csv_rows(word_csv):
        keys = {k.lower(): k for k in row.keys()}
        onset_key = next((keys[k] for k in keys if "onset" in k or "start" in k), None)
        offset_key = next((keys[k] for k in keys if "offset" in k or "end" in k), None)
        word_key = next((keys[k] for k in keys if "word" in k), None)
        if not onset_key or not word_key:
            continue
        try:
            onset = float(row[onset_key])
            offset = float(row[offset_key]) if offset_key else onset + 0.25
        except Exception:
            continue
        if 30 <= offset and onset <= 40:
            annotations.append((onset - 30, offset - 30, row[word_key]))
    eeg, sfreq, ch_names = read_brainvision_partial(
        BrainVisionSpec(
            ds,
            "sub-01 multitalker",
            s3_url("ds005345/sub-01/eeg/sub-01_task-multitalker_eeg.vhdr"),
            s3_url("ds005345/sub-01/eeg/sub-01_task-multitalker_eeg.eeg"),
            30,
            10,
        )
    )
    fig = plot_pair(
        ds,
        "ds005345: Mandarin single-speaker condition + 64ch EEG",
        audio,
        eeg,
        sfreq,
        ch_names,
        None,
        [
            "Use: single/mixed speech comparison; attended-stream retrieval.",
            "Audio: actual single_female.wav, 30-40 s clip.",
            "EEG: sub-01 raw BrainVision byte-range, same 30-40 s window.",
            "Annotation: word-level CSV if onset/offset columns are available.",
        ],
    )
    return {
        "dataset": ds,
        "audio": audio,
        "extra_audio": [male_audio, mix_audio],
        "figure": fig,
        "eeg_sfreq": sfreq,
        "eeg_channels": len(ch_names),
    }


def prepare_ds004718() -> dict[str, Any]:
    ds = "ds004718"
    audio = make_audio_clip(
        AudioClip(
            ds,
            "cantonese_sentence_1_003",
            s3_url("ds004718/sourcedata/stimuli/audio_files_segmented_by_sentence/Part%201/1.003.wav"),
            0,
            5,
        )
    )
    eeg, sfreq, ch_names = read_eeglab_fdt_partial(
        s3_url("ds004718/derivatives/sub-HK001/eeg/sub-HK001_task-lppHK_eeg_preprocessed.set"),
        s3_url("ds004718/derivatives/sub-HK001/eeg/sub-HK001_task-lppHK_eeg_preprocessed.fdt"),
        20,
        10,
    )
    fig = plot_pair(
        ds,
        "ds004718 LPPHK: Cantonese sentence audio + preprocessed EEG",
        audio,
        eeg,
        sfreq,
        ch_names,
        None,
        [
            "Use: word/prosody alignment; Cantonese speech representation.",
            "Audio: actual sentence WAV 1.003, first 5 s.",
            "EEG: sub-HK001 preprocessed EEGLAB .fdt byte-range, 20-30 s snippet.",
            "Note: segmented sentence audio and continuous EEG are not guaranteed to share t=0.",
        ],
    )
    return {"dataset": ds, "audio": audio, "figure": fig, "eeg_sfreq": sfreq, "eeg_channels": len(ch_names)}


def prepare_ds006104() -> dict[str, Any]:
    ds = "ds006104"
    stimuli_dir = OUT / ds / "stimuli"
    preferred = [
        stimuli_dir / "Ba_happy1.wav",
        stimuli_dir / "a_happy1.wav",
        stimuli_dir / "Bad_happy1.wav",
    ]
    audio = None
    for path in preferred:
        if path.exists():
            audio = make_local_audio_entry(ds, path.stem, path)
            break
    if audio is None:
        local_wavs = sorted(stimuli_dir.rglob("*.wav")) if stimuli_dir.exists() else []
        if local_wavs:
            audio = make_local_audio_entry(ds, local_wavs[0].stem, local_wavs[0])

    events_path = OUT / ds / "raw" / "sub-P01_ses-01_task-phonemes_events.tsv"
    write_bytes_once(events_path, get_bytes(s3_url("ds006104/sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_events.tsv")))
    eeg, sfreq, ch_names = read_edf_partial(
        s3_url("ds006104/sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_eeg.edf"),
        7,
        10,
    )
    annotations = []
    with events_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                onset = float(row["onset"])
            except Exception:
                continue
            if 7 <= onset <= 17 and row.get("trial_type") == "stimulus":
                label = (row.get("phoneme1", "") + row.get("phoneme2", "")).replace("n/a", "")
                annotations.append((onset - 7, onset - 7 + 0.2, label or "stim"))
    fig = plot_pair(
        ds,
        "ds006104: controlled voice stimuli + EDF EEG",
        audio,
        eeg,
        sfreq,
        ch_names,
        annotations,
        [
            "Use: controlled pitch/timbre/style probe for EEG voice reconstruction.",
            "Audio: local ds006104 stimuli under data/meeting_examples/ds006104/stimuli.",
            "EEG: sub-P01 EDF byte-range around first stimulus events.",
            "Events: phoneme labels from BIDS events.tsv; align exact stimulus names before training.",
        ],
    )
    return {"dataset": ds, "audio": audio, "figure": fig, "eeg_sfreq": sfreq, "eeg_channels": len(ch_names)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [
        prepare_ds004408(),
        prepare_ds004718(),
        prepare_ds005345(),
        prepare_ds006104(),
    ]
    manifest = {
        "created_for": "0429 EEG->Voice discussion",
        "scope": ["ds004408", "ds004718", "ds005345", "ds006104"],
        "outputs_root": str(OUT),
        "results": results,
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    md = OUT / "README.md"
    lines = [
        "# Meeting EEG/Audio Examples",
        "",
        "这些文件只作为讨论会展示样例，不是完整训练数据。",
        "",
    ]
    for item in results:
        lines.append(f"## {item['dataset']}")
        if item.get("audio"):
            lines.append(
                f"- audio: `{item['audio']['clip_path']}` "
                f"({item['audio']['actual_duration_sec']} s)"
            )
        else:
            lines.append("- audio: no local/public meeting clip available")
        for extra in item.get("extra_audio", []):
            lines.append(f"- extra audio: `{extra['clip_path']}` ({extra['actual_duration_sec']} s)")
        lines.append(f"- figure: `{item['figure']}`")
        lines.append(f"- EEG: {item['eeg_channels']} channels shown/parsed, sfreq={item['eeg_sfreq']:.2f} Hz")
        lines.append("")
    md.write_text("\n".join(lines))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
