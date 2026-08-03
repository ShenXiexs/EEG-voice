#!/usr/bin/env python3
"""Build the fit-only frozen-EnCodec token cache for CP-temporal v3."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0724.audio_features import AudioPreparationConfig
from src.open_vocab_v3.cp_temporal import PREPARATION_SCHEMA, SCHEMA
from src.open_vocab_v3.data import (_accepted_denoise_paths, _read_waveform,
                                    light_prepare_waveform, load_prepared)
from src.open_vocab_v3.encodec_content import EnCodecGenerator
from src.open_vocab_v3.runtime import (default_device, load_config, output_path,
                                       sha256_file, write_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path, cfg = load_config(args.config)
    destination = output_path(config_path, cfg, "encodec_cache")
    manifest = output_path(config_path, cfg, "encodec_cache_manifest")
    if destination.exists() and not args.force:
        raise RuntimeError("CP-temporal EnCodec cache exists; pass --force for a deliberate rebuild")
    prepared_path = output_path(config_path, cfg, "prepared_cache")
    records = load_prepared(prepared_path, expected_schema=PREPARATION_SCHEMA)
    indices = np.flatnonzero((records.roles == "fit") & records.arrays["fit_eligible"].astype(bool))
    with output_path(config_path, cfg, "unified_manifest").open(newline="", encoding="utf-8") as handle:
        paths = {str(row["sample_key"]): str(row["audio_relpath"]) for row in csv.DictReader(handle)
                 if row.get("dataset") == "karaone"}
    root = output_path(config_path, cfg, "audio_root")
    denoised = _accepted_denoise_paths(config_path, cfg)
    preparation = AudioPreparationConfig(
        sample_rate=16000, max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
        target_rms=float(cfg["audio"]["target_rms"]),
    )
    codec = EnCodecGenerator(
        output_path(config_path, cfg, "encodec_root"), device=default_device(args.device),
        bandwidth=float(cfg["audio"]["encodec_bandwidth"]),
    )
    codes, masks = [], []
    for index in tqdm(indices.tolist(), desc="[v3 CP frozen EnCodec]", unit="trial", dynamic_ncols=True):
        key = str(records.arrays["sample_keys"][index])
        waveform, rate = _read_waveform(denoised.get(key, root / paths[key]))
        prepared, _ = light_prepare_waveform(waveform, rate, preparation)
        value, mask = codec.encode(torch.from_numpy(prepared.waveform[:prepared.valid_samples]).unsqueeze(0))
        value, mask = value[0].cpu().numpy(), mask[0].cpu().numpy()
        if value.shape[0] != 8 or value.shape[1] > 192:
            raise RuntimeError(f"unexpected frozen EnCodec shape {value.shape} for {key}")
        padded = np.zeros((8, 192), dtype=np.int16); padded[:, :value.shape[1]] = value
        padded_mask = np.zeros(192, dtype=bool); padded_mask[:mask.size] = mask
        codes.append(padded); masks.append(padded_mask)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, schema=np.asarray(SCHEMA), source_prepared_sha256=np.asarray(sha256_file(prepared_path)),
        source_indices=indices.astype(np.int32), sample_keys=records.arrays["sample_keys"][indices].astype(str),
        encodec_codes=np.stack(codes), encodec_mask=np.stack(masks), tokenizer=np.asarray("frozen_encodec_24khz_6kbps"),
    )
    write_json(manifest, {
        "schema_version": SCHEMA, "scope": "fit_only", "tokenizer": "frozen_encodec_24khz_6kbps",
        "n": len(indices), "cache": str(destination), "sha256": sha256_file(destination),
        "prepared_cache_sha256": sha256_file(prepared_path), "shape": {"codes": [len(indices), 8, 192], "mask": [len(indices), 192]},
    })
    print(destination, flush=True)


if __name__ == "__main__":
    main()
