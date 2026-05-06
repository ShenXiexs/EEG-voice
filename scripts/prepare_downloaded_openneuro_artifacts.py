#!/usr/bin/env python3
"""Create lightweight figures and inventories for downloaded OpenNeuro subsets.

The script is intentionally useful before a full MNE stack is installed:
- EDF preview figures are generated with a small standard-library EDF reader.
- WAV and acoustic CSV figures are generated with the standard library.
- BrainVision/FIF EEG previews are reported and skipped unless MNE is available.

Outputs are written under outputs/downloaded_openneuro_artifacts/ by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DS006104 = Path("data/raw/openneuro/ds006104_datalad")
DEFAULT_DS005345 = Path("data/raw/openneuro/ds005345_datalad")
DEFAULT_OUT = Path("outputs/downloaded_openneuro_artifacts")


@dataclass
class Artifact:
    dataset: str
    kind: str
    source: str
    output: str
    note: str


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def html_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def is_available(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def safe_name(path: Path) -> str:
    return path.name.replace("/", "_").replace(" ", "_")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_tsv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not is_available(path):
        return rows
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def parse_csv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not is_available(path):
        return rows
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def basic_counts(root: Path) -> dict[str, int]:
    suffixes = [".edf", ".eeg", ".vhdr", ".vmrk", ".fif", ".wav", ".tsv", ".csv", ".json"]
    counts = {suffix: 0 for suffix in suffixes}
    counts["files_total"] = 0
    counts["available_files"] = 0
    counts["missing_annex_targets"] = 0
    if not root.exists():
        return counts
    for path in iter_dataset_paths(root):
        if not path.is_file() and not path.is_symlink():
            continue
        counts["files_total"] += 1
        suffix = "".join(path.suffixes[-2:]) if path.name.endswith(".fif.gz") else path.suffix
        if suffix in counts:
            counts[suffix] += 1
        if is_available(path):
            counts["available_files"] += 1
        else:
            counts["missing_annex_targets"] += 1
    return counts


def iter_dataset_paths(root: Path) -> Iterable[Path]:
    """Yield files/symlinks in a DataLad worktree without descending into .git."""
    if not root.exists():
        return
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur.name == ".git":
            continue
        try:
            if cur.is_dir() and not cur.is_symlink():
                stack.extend(cur.iterdir())
            elif cur.is_file() or cur.is_symlink():
                yield cur
        except OSError:
            continue


def read_edf_preview(path: Path, duration_sec: float, max_channels: int) -> dict:
    with path.open("rb") as f:
        fixed = f.read(256)
        if len(fixed) < 256:
            raise ValueError("EDF header is shorter than 256 bytes")
        header_bytes = int(fixed[184:192].decode("ascii", errors="ignore").strip())
        n_records_text = fixed[236:244].decode("ascii", errors="ignore").strip()
        n_records = int(float(n_records_text)) if n_records_text else 0
        record_duration = float(fixed[244:252].decode("ascii", errors="ignore").strip())
        n_signals = int(fixed[252:256].decode("ascii", errors="ignore").strip())
        signal_header = f.read(header_bytes - 256)

        def field(offset: int, width: int) -> list[str]:
            vals = []
            for i in range(n_signals):
                start = offset + i * width
                vals.append(signal_header[start : start + width].decode("ascii", errors="ignore").strip())
            return vals

        offset = 0
        labels = field(offset, 16)
        offset += 16 * n_signals
        offset += 80 * n_signals
        phys_dim = field(offset, 8)
        offset += 8 * n_signals
        phys_min = [float(x or 0) for x in field(offset, 8)]
        offset += 8 * n_signals
        phys_max = [float(x or 0) for x in field(offset, 8)]
        offset += 8 * n_signals
        dig_min = [float(x or -32768) for x in field(offset, 8)]
        offset += 8 * n_signals
        dig_max = [float(x or 32767) for x in field(offset, 8)]
        offset += 8 * n_signals
        offset += 80 * n_signals
        samples_per_record = [int(float(x or 0)) for x in field(offset, 8)]

        chosen = list(range(min(max_channels, n_signals)))
        data = [[] for _ in chosen]
        records_to_read = max(1, min(n_records, math.ceil(duration_sec / record_duration)))
        for _ in range(records_to_read):
            for sig_idx in range(n_signals):
                raw = f.read(samples_per_record[sig_idx] * 2)
                if sig_idx not in chosen:
                    continue
                target = data[chosen.index(sig_idx)]
                for j in range(0, len(raw), 2):
                    digital = int.from_bytes(raw[j : j + 2], "little", signed=True)
                    denom = dig_max[sig_idx] - dig_min[sig_idx]
                    if denom == 0:
                        physical = float(digital)
                    else:
                        physical = (digital - dig_min[sig_idx]) / denom
                        physical = physical * (phys_max[sig_idx] - phys_min[sig_idx]) + phys_min[sig_idx]
                    target.append(physical)

        sample_rates = [
            samples_per_record[i] / record_duration if record_duration > 0 else 0.0 for i in chosen
        ]
        return {
            "path": str(path),
            "labels": [labels[i] for i in chosen],
            "units": [phys_dim[i] for i in chosen],
            "sample_rates": sample_rates,
            "record_duration": record_duration,
            "records_read": records_to_read,
            "signals_total": n_signals,
            "data": data,
        }


def polyline(values: list[float], x: int, y: int, width: int, height: int) -> str:
    if not values:
        return ""
    max_points = 1000
    step = max(1, len(values) // max_points)
    sampled = values[::step]
    lo = min(sampled)
    hi = max(sampled)
    if math.isclose(lo, hi):
        lo -= 1
        hi += 1
    points = []
    for idx, value in enumerate(sampled):
        px = x + idx / max(1, len(sampled) - 1) * width
        py = y + height - (value - lo) / (hi - lo) * height
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def write_edf_svg(preview: dict, out_path: Path) -> None:
    rows = len(preview["data"])
    row_h = 72
    width = 1220
    height = 92 + rows * row_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="28" y="35" font-family="Arial" font-size="22" font-weight="700" fill="#1f2d36">{html_escape(Path(preview["path"]).name)}</text>',
        f'<text x="28" y="60" font-family="Arial" font-size="13" fill="#53636d">EDF preview, first {preview["records_read"]} records. Total signals: {preview["signals_total"]}.</text>',
    ]
    for i, values in enumerate(preview["data"]):
        y = 88 + i * row_h
        label = preview["labels"][i]
        unit = preview["units"][i]
        sr = preview["sample_rates"][i]
        parts.append(f'<text x="28" y="{y + 22}" font-family="Arial" font-size="13" fill="#21313a">{html_escape(label)} ({sr:.1f} Hz, {html_escape(unit)})</text>')
        parts.append(f'<line x1="190" x2="1190" y1="{y + 34}" y2="{y + 34}" stroke="#d8d1c7" stroke-width="1"/>')
        pts = polyline(values, 190, y + 6, 1000, 56)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#2f6f8f" stroke-width="1.2"/>')
    parts.append("</svg>")
    write_text(out_path, "\n".join(parts))


def read_wav_mono(path: Path, max_seconds: float | None = None) -> tuple[list[float], int, int, float]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        total_frames = wf.getnframes()
        frames_to_read = total_frames if max_seconds is None else min(total_frames, int(sr * max_seconds))
        blob = wf.readframes(frames_to_read)
    vals: list[float] = []
    if width == 2:
        vals = [int.from_bytes(blob[i : i + 2], "little", signed=True) / 32768.0 for i in range(0, len(blob), 2)]
    elif width == 4:
        vals = [int.from_bytes(blob[i : i + 4], "little", signed=True) / 2147483648.0 for i in range(0, len(blob), 4)]
    elif width == 1:
        vals = [(b - 128) / 128.0 for b in blob]
    else:
        vals = []
    mono = []
    if channels > 1:
        for i in range(0, len(vals), channels):
            mono.append(sum(vals[i : i + channels]) / channels)
    else:
        mono = vals
    return mono, sr, channels, total_frames / sr if sr else 0.0


def write_wav_svg(path: Path, out_path: Path, max_seconds: float = 30.0) -> dict:
    samples, sr, channels, duration = read_wav_mono(path, max_seconds=max_seconds)
    width = 1220
    height = 210
    pts = polyline(samples, 36, 78, 1148, 92)
    rms = math.sqrt(sum(x * x for x in samples) / max(1, len(samples)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="30" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#1f2d36">{html_escape(path.name)}</text>',
        f'<text x="30" y="60" font-family="Arial" font-size="13" fill="#53636d">sr={sr} Hz, channels={channels}, duration={duration:.2f}s, displayed first {min(duration, max_seconds):.1f}s, RMS={rms:.4f}</text>',
        '<line x1="36" x2="1184" y1="124" y2="124" stroke="#d8d1c7" stroke-width="1"/>',
        f'<polyline points="{pts}" fill="none" stroke="#7a4e9d" stroke-width="1.2"/>',
        "</svg>",
    ]
    write_text(out_path, "\n".join(parts))
    return {"sample_rate": sr, "channels": channels, "duration_sec": duration, "rms": rms}


def write_acoustic_svg(path: Path, out_path: Path, max_rows: int = 4000) -> dict:
    rows = parse_csv(path, limit=max_rows)
    times: list[float] = []
    f0: list[float] = []
    intensity: list[float] = []
    for row in rows:
        try:
            t = float(row.get("time", row.get("Time", "")))
        except ValueError:
            continue
        def val(name: str) -> float | None:
            text = row.get(name, "")
            try:
                return float(text)
            except ValueError:
                return None
        f0_val = val("f0")
        int_val = val("intensity")
        if f0_val is not None and int_val is not None:
            times.append(t)
            f0.append(f0_val)
            intensity.append(int_val)
    width = 1220
    height = 320
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="30" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#1f2d36">{html_escape(path.name)}</text>',
        f'<text x="30" y="60" font-family="Arial" font-size="13" fill="#53636d">First {len(times)} rows with f0/intensity values.</text>',
    ]
    parts.append('<text x="30" y="98" font-family="Arial" font-size="13" fill="#21313a">F0</text>')
    parts.append(f'<polyline points="{polyline(f0, 96, 78, 1088, 82)}" fill="none" stroke="#2f6f8f" stroke-width="1.3"/>')
    parts.append('<text x="30" y="216" font-family="Arial" font-size="13" fill="#21313a">Intensity</text>')
    parts.append(f'<polyline points="{polyline(intensity, 96, 196, 1088, 82)}" fill="none" stroke="#b05f3c" stroke-width="1.3"/>')
    parts.append("</svg>")
    write_text(out_path, "\n".join(parts))
    return {
        "rows_used": len(times),
        "f0_median": statistics.median(f0) if f0 else None,
        "intensity_median": statistics.median(intensity) if intensity else None,
    }


def write_event_summary(events_path: Path, out_path: Path) -> dict:
    rows = parse_tsv(events_path)
    trial_types: dict[str, int] = {}
    categories: dict[str, int] = {}
    tms_targets: dict[str, int] = {}
    for row in rows:
        trial_types[row.get("trial_type", "")] = trial_types.get(row.get("trial_type", ""), 0) + 1
        categories[row.get("category", "")] = categories.get(row.get("category", ""), 0) + 1
        tms_targets[row.get("tms_target", "")] = tms_targets.get(row.get("tms_target", ""), 0) + 1
    lines = [
        f"# Events Summary: `{events_path.name}`",
        "",
        f"- Rows: {len(rows)}",
        f"- Columns: {', '.join(rows[0].keys()) if rows else 'n/a'}",
        "",
        "## trial_type",
        "",
    ]
    for key, value in sorted(trial_types.items(), key=lambda x: (-x[1], x[0]))[:20]:
        lines.append(f"- `{key or 'blank'}`: {value}")
    lines += ["", "## category", ""]
    for key, value in sorted(categories.items(), key=lambda x: (-x[1], x[0]))[:20]:
        lines.append(f"- `{key or 'blank'}`: {value}")
    lines += ["", "## tms_target", ""]
    for key, value in sorted(tms_targets.items(), key=lambda x: (-x[1], x[0]))[:20]:
        lines.append(f"- `{key or 'blank'}`: {value}")
    write_text(out_path, "\n".join(lines) + "\n")
    return {"rows": len(rows), "trial_types": trial_types, "categories": categories, "tms_targets": tms_targets}


def first_paths(paths: Iterable[Path], limit: int) -> list[Path]:
    out = []
    for path in sorted(paths):
        if is_available(path):
            out.append(path)
        if len(out) >= limit:
            break
    return out


def process_ds006104(root: Path, out_dir: Path, duration_sec: float, max_eeg_files: int, max_channels: int) -> list[Artifact]:
    artifacts: list[Artifact] = []
    ds_out = out_dir / "ds006104"
    ds_out.mkdir(parents=True, exist_ok=True)
    counts = basic_counts(root)
    write_text(ds_out / "inventory.json", json.dumps(counts, indent=2, ensure_ascii=False))
    artifacts.append(Artifact("ds006104", "inventory", str(root), str(ds_out / "inventory.json"), "file counts"))

    event_paths = (p for p in iter_dataset_paths(root) if p.name.endswith("events.tsv"))
    for events_path in first_paths(event_paths, max_eeg_files * 2):
        out = ds_out / f"{safe_name(events_path)}.summary.md"
        write_event_summary(events_path, out)
        artifacts.append(Artifact("ds006104", "events_summary", str(events_path), str(out), "trial/event counts"))

    edf_paths = (p for p in iter_dataset_paths(root) if p.suffix == ".edf")
    for edf_path in first_paths(edf_paths, max_eeg_files):
        out = ds_out / f"{safe_name(edf_path)}.preview.svg"
        try:
            preview = read_edf_preview(edf_path, duration_sec=duration_sec, max_channels=max_channels)
            write_edf_svg(preview, out)
            note = f"first {duration_sec}s, {max_channels} channels"
        except Exception as exc:
            write_text(out.with_suffix(".error.txt"), f"{type(exc).__name__}: {exc}\n")
            out = out.with_suffix(".error.txt")
            note = "failed to render EDF preview"
        artifacts.append(Artifact("ds006104", "edf_preview", str(edf_path), str(out), note))
    return artifacts


def process_ds005345(root: Path, out_dir: Path, max_eeg_files: int) -> list[Artifact]:
    artifacts: list[Artifact] = []
    ds_out = out_dir / "ds005345"
    ds_out.mkdir(parents=True, exist_ok=True)
    counts = basic_counts(root)
    write_text(ds_out / "inventory.json", json.dumps(counts, indent=2, ensure_ascii=False))
    artifacts.append(Artifact("ds005345", "inventory", str(root), str(ds_out / "inventory.json"), "file counts"))

    for wav_path in first_paths((root / "stimuli").glob("*.wav"), 10):
        out = ds_out / f"{safe_name(wav_path)}.waveform.svg"
        info = write_wav_svg(wav_path, out)
        artifacts.append(Artifact("ds005345", "wav_waveform", str(wav_path), str(out), f"{info['duration_sec']:.2f}s audio"))

    for csv_path in first_paths((root / "annotation").glob("*acoustic.csv"), 10):
        out = ds_out / f"{safe_name(csv_path)}.acoustic.svg"
        info = write_acoustic_svg(csv_path, out)
        artifacts.append(Artifact("ds005345", "acoustic_plot", str(csv_path), str(out), f"{info['rows_used']} rows plotted"))

    eeg_candidates = list((root / "sub-01" / "eeg").glob("*.vhdr")) + list((root / "derivatives" / "sub-01" / "eeg").glob("*.fif"))
    for path in first_paths(eeg_candidates, max_eeg_files):
        out = ds_out / f"{safe_name(path)}.mne_needed.txt"
        write_text(
            out,
            "This EEG file requires MNE-Python for a signal preview.\n"
            "Install with: python -m pip install mne matplotlib numpy scipy\n"
            f"Source: {path}\n",
        )
        artifacts.append(Artifact("ds005345", "eeg_preview_placeholder", str(path), str(out), "MNE needed for BrainVision/FIF preview"))
    return artifacts


def write_manifest(artifacts: list[Artifact], out_dir: Path) -> None:
    rows = [artifact.__dict__ for artifact in artifacts]
    write_text(out_dir / "manifest.json", json.dumps(rows, indent=2, ensure_ascii=False))
    md = ["# Downloaded OpenNeuro Artifacts", ""]
    for artifact in artifacts:
        md.append(f"- **{artifact.dataset} / {artifact.kind}**: `{artifact.output}`")
        md.append(f"  - Source: `{artifact.source}`")
        md.append(f"  - Note: {artifact.note}")
    write_text(out_dir / "README.md", "\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds006104-root", type=Path, default=DEFAULT_DS006104)
    parser.add_argument("--ds005345-root", type=Path, default=DEFAULT_DS005345)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--max-eeg-files", type=int, default=4)
    parser.add_argument("--max-channels", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Artifact] = []
    if args.ds006104_root.exists():
        artifacts.extend(process_ds006104(args.ds006104_root, args.out_dir, args.duration_sec, args.max_eeg_files, args.max_channels))
    if args.ds005345_root.exists():
        artifacts.extend(process_ds005345(args.ds005345_root, args.out_dir, args.max_eeg_files))
    write_manifest(artifacts, args.out_dir)
    print(f"Wrote {len(artifacts)} artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
