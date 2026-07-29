#!/usr/bin/env python3
"""Export the original 1,341 v0730 pairs using the fixed EEG C/P model."""
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from scripts.export_open_vocab_0730_pairs import manifest_audio_paths, reference_waveform
from scripts.train_open_vocab_0730 import load_renderer
from scripts.train_open_vocab_0730_fixed import load_eeg_fixed
from src.open_vocab_0730.data_fixed import CPDataset, PAIR_ROLES, collate, load_prepared
from src.open_vocab_0730.runtime import default_device, load_config, move_batch, resolve_config_path, sha256_file, write_json
from src.open_vocab_0730.vocoder_fixed import SpeechT5HiFiGanFixed, pcm16


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all 1,341 v0730-fixed WAV pairs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse()
    config_path, cfg = load_config(args.config)
    device = default_device(args.device)
    records = load_prepared(resolve_config_path(config_path, cfg["paths"]["prepared_cache"]))
    dataset = CPDataset(records, PAIR_ROLES)
    if len(dataset) != 1341:
        raise ValueError(f"all-pair export requires 1,341 records, found {len(dataset)}")
    model, _ = load_eeg_fixed(config_path, cfg, device)
    renderer = load_renderer(config_path, cfg, device)
    model_signature = (
        sha256_file(resolve_config_path(config_path, cfg["paths"]["eeg_checkpoint"]))
        + ":"
        + sha256_file(resolve_config_path(config_path, cfg["paths"]["renderer_checkpoint"]))
    )
    backend = SpeechT5HiFiGanFixed(
        resolve_config_path(config_path, cfg["paths"]["vocoder_root"]), device=device
    )
    final_destination = resolve_config_path(config_path, cfg["paths"]["pair_root"])
    destination = final_destination.parent / "smoke" if args.limit else final_destination
    destination.mkdir(parents=True, exist_ok=True)
    audio_paths = manifest_audio_paths(resolve_config_path(config_path, cfg["data"]["unified_manifest"]))
    audio_root = resolve_config_path(config_path, cfg["data"]["audio_root"])
    gate_path = resolve_config_path(config_path, cfg["paths"]["generated_gate"])
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {"passed": False, "reason": "generated gate has not been evaluated"}
    rows: list[dict[str, Any]] = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    total = min(len(dataset), args.limit) if args.limit else len(dataset)
    for index, batch in enumerate(
        tqdm(loader, total=total, desc="[0730-fixed pairs] WAV export", unit="pair", dynamic_ncols=True, mininterval=0.5)
    ):
        batch = move_batch(batch, device)
        key = batch["sample_key"][0]
        stem = destination / key
        reconstruction_path = stem.with_name(stem.name + "__reconstruction.wav")
        reference_path = stem.with_name(stem.name + "__reference.wav")
        metadata_path = stem.with_suffix(".json")
        interpretation = "conditional_generative_approximation" if gate.get("passed", False) else "diagnostic_waveform_only"
        if args.resume and metadata_path.is_file() and reconstruction_path.is_file() and reference_path.is_file():
            row = json.loads(metadata_path.read_text(encoding="utf-8"))
            if row.get("model_signature") == model_signature:
                row["generated_gate_passed"] = bool(gate.get("passed", False))
                row["interpretation"] = interpretation
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
        row = {
            "sample_key": key,
            "audio_key": batch["audio_key"][0],
            "subject": batch["subject"][0],
            "label": batch["label"][0],
            "evaluation_role": batch["role"][0],
            "reference_wav": str(reference_path),
            "reconstruction_wav": str(reconstruction_path),
            "predicted_duration_seconds": predicted_duration,
            "model_signature": model_signature,
            "generated_gate_passed": bool(gate.get("passed", False)),
            "interpretation": interpretation,
        }
        write_json(metadata_path, row)
        rows.append(row)
        if args.limit and index + 1 >= args.limit:
            break
    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_key"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        destination / "export_manifest.json",
        {
            "schema_version": "openvoice-0730-fixed-pairs-v2",
            "expected_pairs": 1341,
            "exported_pairs": len(rows),
            "generated_gate": gate,
            "records": rows,
        },
    )
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
