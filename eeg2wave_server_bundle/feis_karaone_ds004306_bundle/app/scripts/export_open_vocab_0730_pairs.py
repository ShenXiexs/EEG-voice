#!/usr/bin/env python3
"""Export one reference/reconstruction WAV pair for every 1,341 v0730 records."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0730.data import CPDataset, collate, load_prepared
from src.open_vocab_0730.model import ContentProsodyEEG
from src.open_vocab_0730.runtime import default_device, load_config, move_batch, resolve_config_path, write_json
from src.open_vocab_0730.vocoder import SpeechT5HiFiGan, pcm16
from scripts.train_open_vocab_0730 import checkpoint_path, load_renderer


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all v0730 reference/reconstruction WAV pairs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def active_bounds(waveform: np.ndarray, cfg: dict[str, Any]) -> tuple[int, int]:
    audio = cfg["audio"]; sr = int(cfg["data"]["sample_rate"])
    window = int(sr * float(audio["active_window_ms"]) / 1000)
    hop = int(sr * float(audio["active_hop_ms"]) / 1000)
    if len(waveform) < window:
        return 0, len(waveform)
    rms = np.asarray([np.sqrt(np.mean(np.square(waveform[index:index + window])) + 1e-12) for index in range(0, len(waveform) - window + 1, hop)])
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    threshold = max(np.percentile(db, 10) + float(audio["active_noise_margin_db"]), float(db.max()) - float(audio["active_peak_margin_db"]))
    active = np.flatnonzero(db >= threshold)
    if not len(active):
        return 0, len(waveform)
    context = int(float(audio["active_context_ms"]) * sr / 1000)
    return max(0, active[0] * hop - context), min(len(waveform), active[-1] * hop + window + context)


def reference_waveform(path: Path, cfg: dict[str, Any]) -> np.ndarray:
    native_sr, waveform = wavfile.read(path)
    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform.astype(np.float32) / max(float(np.iinfo(waveform.dtype).max), 1.0)
    else:
        waveform = waveform.astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(-1)
    target_sr = int(cfg["data"]["sample_rate"])
    if native_sr != target_sr:
        waveform = resample_poly(waveform, target_sr, int(native_sr)).astype(np.float32)
    begin, end = active_bounds(waveform, cfg)
    waveform = waveform[begin:end][: int(float(cfg["audio"]["max_seconds"]) * target_sr)]
    gain = min(10.0, float(cfg["audio"]["target_rms"]) / float(np.sqrt(np.mean(waveform ** 2) + 1e-12)))
    return np.asarray(waveform * gain, dtype=np.float32)


def manifest_audio_paths(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        result = {str(row["sample_key"]): str(row["audio_relpath"]) for row in rows if row.get("dataset") == "karaone"}
    if not result:
        raise ValueError(f"no KaraOne audio paths in {path}")
    return result


def load_eeg(config_path: Path, cfg: dict[str, Any], device: torch.device) -> ContentProsodyEEG:
    model = ContentProsodyEEG(codebook_size=int(cfg["content"]["codebook_size"]), dimension=int(cfg["model"]["dimension"]), heads=int(cfg["model"]["heads"]), layers=int(cfg["model"]["layers"]), content_steps=int(cfg["content"]["steps"]), prosody_steps=32, dropout=float(cfg["model"]["dropout"])).to(device)
    raw = torch.load(checkpoint_path(config_path, cfg, "eeg"), map_location=device, weights_only=False)
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.eval()


@torch.no_grad()
def main() -> None:
    args = parse(); config_path, cfg = load_config(args.config); device = default_device(args.device)
    records = load_prepared(resolve_config_path(config_path, cfg["paths"]["prepared_cache"])); dataset = CPDataset(records, ("fit", "subject_holdout_seen", "label_holdout_seen_subject", "subject_and_label_holdout"))
    if len(dataset) != 1341:
        raise ValueError(f"all-pair export requires 1,341 records, found {len(dataset)}")
    model = load_eeg(config_path, cfg, device); renderer = load_renderer(config_path, cfg, device)
    backend = SpeechT5HiFiGan(resolve_config_path(config_path, cfg["paths"]["vocoder_root"]), device=device)
    final_destination = resolve_config_path(config_path, cfg["paths"]["pair_root"])
    destination = final_destination.parent / "smoke" if args.limit else final_destination
    destination.mkdir(parents=True, exist_ok=True)
    audio_paths = manifest_audio_paths(resolve_config_path(config_path, cfg["data"]["unified_manifest"]))
    audio_root = resolve_config_path(config_path, cfg["data"]["audio_root"])
    gate_path = resolve_config_path(config_path, cfg["paths"]["renderer_gate"])
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {"passed": False, "reason": "renderer gate has not been evaluated"}
    rows: list[dict[str, Any]] = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    total = min(len(dataset), args.limit) if args.limit else len(dataset)
    progress = tqdm(loader, total=total, desc="[0730 pairs] WAV export", unit="pair", dynamic_ncols=True, mininterval=0.5)
    for index, batch in enumerate(progress):
        batch = move_batch(batch, device); key = batch["sample_key"][0]; stem = destination / key
        reconstruction_path = stem.with_name(stem.name + "__reconstruction.wav")
        reference_path = stem.with_name(stem.name + "__reference.wav")
        metadata_path = stem.with_suffix(".json")
        if args.resume and metadata_path.is_file() and reconstruction_path.is_file() and reference_path.is_file():
            row = json.loads(metadata_path.read_text(encoding="utf-8"))
            # Refresh run-level gate metadata even when WAV synthesis is resumed.
            # Older JSON files may contain 0/1 because bool subclasses int.
            row["renderer_gate_passed"] = bool(gate.get("passed", False))
            row["interpretation"] = "conditional_generative_approximation" if gate.get("passed", False) else "diagnostic_waveform_only"
            write_json(metadata_path, row)
            rows.append(row)
            continue
        state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
        mel = renderer(state.content_logits, state.prosody)
        generated = backend.synthesize(mel)
        waveform = pcm16(generated[0] if generated.ndim > 1 else generated)
        predicted_duration = float(torch.clamp(state.duration[0], 0.10, float(cfg["audio"]["max_seconds"])).cpu())
        waveform = waveform[: max(1, int(predicted_duration * int(cfg["vocoder"]["sample_rate"])))].copy()
        reference = reference_waveform(audio_root / audio_paths[key], cfg)
        wavfile.write(reconstruction_path, int(cfg["vocoder"]["sample_rate"]), (waveform * 32767.0).astype(np.int16))
        wavfile.write(reference_path, int(cfg["data"]["sample_rate"]), (reference * 32767.0).astype(np.int16))
        row = {"sample_key": key, "audio_key": batch["audio_key"][0], "subject": batch["subject"][0], "label": batch["label"][0], "evaluation_role": batch["role"][0], "reference_wav": str(reference_path), "reconstruction_wav": str(reconstruction_path), "predicted_duration_seconds": predicted_duration, "renderer_gate_passed": bool(gate.get("passed", False)), "interpretation": "conditional_generative_approximation" if gate.get("passed", False) else "diagnostic_waveform_only"}
        write_json(metadata_path, row); rows.append(row)
        if args.limit and index + 1 >= args.limit:
            break
    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_key"])
        writer.writeheader(); writer.writerows(rows)
    write_json(destination / "export_manifest.json", {"schema_version": "openvoice-0730-pairs-v1", "expected_pairs": 1341, "exported_pairs": len(rows), "renderer_gate": gate, "records": rows})
    print(manifest)


if __name__ == "__main__":
    main()
