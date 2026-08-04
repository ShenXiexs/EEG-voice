#!/usr/bin/env python3
"""Build fit-only frozen-EnCodec codes, latent targets, and crop masks."""
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
from src.open_vocab_v3.data import _accepted_denoise_paths, _read_waveform, light_prepare_waveform, load_prepared
from src.open_vocab_v3.encodec_bridge import PREPARATION_SCHEMA, SCHEMA, FrozenEnCodecRenderer
from src.open_vocab_v3.runtime import default_device, load_config, output_path, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path, cfg = load_config(args.config)
    destination = output_path(config_path, cfg, "encodec_cache")
    manifest_path = output_path(config_path, cfg, "encodec_cache_manifest")
    if destination.exists() and not args.force:
        raise RuntimeError(f"bridge EnCodec cache exists: {destination}; pass --force to rebuild")
    records = load_prepared(output_path(config_path, cfg, "prepared_cache"), expected_schema=PREPARATION_SCHEMA)
    selector = (records.roles == "fit") & records.arrays["fit_eligible"].astype(bool)
    indices = np.flatnonzero(selector)
    # The cache intentionally contains both fit-train and the deterministic
    # internal dev; train scripts select the role mask and never re-tokenize.
    root = output_path(config_path, cfg, "audio_root")
    with output_path(config_path, cfg, "unified_manifest").open(newline="", encoding="utf-8") as handle:
        paths = {str(row["sample_key"]): str(row["audio_relpath"]) for row in csv.DictReader(handle) if row.get("dataset") == "karaone"}
    preparation = AudioPreparationConfig(
        sample_rate=16000, max_active_seconds=float(cfg["audio"]["max_active_seconds"]), target_rms=float(cfg["audio"]["target_rms"])
    )
    renderer = FrozenEnCodecRenderer(output_path(config_path, cfg, "encodec_root"), device=default_device(args.device), bandwidth=float(cfg["audio"]["encodec_bandwidth"]))
    denoised = _accepted_denoise_paths(config_path, cfg)
    codes_all: list[np.ndarray] = []; masks_all: list[np.ndarray] = []; latent_all: list[np.ndarray] = []
    waveform_all: list[np.ndarray] = []; sample_masks: list[np.ndarray] = []; sample_counts: list[int] = []
    maximum_samples = round(float(cfg["audio"]["max_active_seconds"]) * 16000)
    for index in tqdm(indices.tolist(), desc="[v3 bridge frozen EnCodec cache]", unit="trial", dynamic_ncols=True):
        key = str(records.arrays["sample_keys"][index])
        waveform, rate = _read_waveform(denoised.get(key, root / paths[key]))
        prepared, _ = light_prepare_waveform(waveform, rate, preparation)
        active = torch.from_numpy(prepared.waveform[: prepared.valid_samples]).unsqueeze(0)
        code, code_mask = renderer.encode_16k(active)
        if code.shape[1] != 8 or code.shape[-1] > 192:
            raise RuntimeError(f"unexpected EnCodec contract for {key}: {tuple(code.shape)}")
        padded_code = torch.zeros((1, 8, 192), dtype=torch.long, device=code.device)
        padded_mask = torch.zeros((1, 192), dtype=torch.bool, device=code.device)
        padded_code[..., :code.shape[-1]] = code
        padded_mask[..., :code_mask.shape[-1]] = code_mask
        latent = renderer.target_latent(padded_code)
        output = np.zeros(maximum_samples, dtype=np.float32); output[: prepared.valid_samples] = prepared.waveform[: prepared.valid_samples]
        output_mask = np.zeros(maximum_samples, dtype=bool); output_mask[: prepared.valid_samples] = True
        codes_all.append(padded_code[0].cpu().numpy().astype(np.int16))
        masks_all.append(padded_mask[0].cpu().numpy())
        latent_all.append(latent[0].cpu().numpy().astype(np.float16))
        waveform_all.append(output); sample_masks.append(output_mask); sample_counts.append(int(prepared.valid_samples))
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared_path = output_path(config_path, cfg, "prepared_cache")
    np.savez_compressed(
        destination, schema=np.asarray(SCHEMA), prepared_cache_sha256=np.asarray(sha256_file(prepared_path)),
        source_indices=indices.astype(np.int32), sample_keys=records.arrays["sample_keys"][indices].astype(str),
        encodec_codes=np.stack(codes_all), encodec_mask=np.stack(masks_all), target_latent=np.stack(latent_all),
        waveform_16k=np.stack(waveform_all), waveform_mask=np.stack(sample_masks), waveform_samples=np.asarray(sample_counts, dtype=np.int32),
        tokenizer=np.asarray("frozen_encodec_24khz_6kbps_8x1024"), rvq_mode=np.asarray("sequential_frozen_rvq"),
    )
    write_json(manifest_path, {
        "schema_version": SCHEMA, "scope": "fit_only_including_internal_dev", "tokenizer": "frozen_encodec_24khz_6kbps_8x1024",
        "rvq": "sequential_frozen", "n": int(len(indices)), "cache": str(destination), "sha256": sha256_file(destination),
        "prepared_cache": str(prepared_path), "prepared_cache_sha256": sha256_file(prepared_path),
        "encodec_config_sha256": sha256_file(output_path(config_path, cfg, "encodec_root") / "config.json"),
        "shapes": {"codes": [len(indices), 8, 192], "latent": [len(indices), renderer.latent_dimension, 192], "waveform": [len(indices), maximum_samples]},
    })
    print(destination, flush=True)


if __name__ == "__main__":
    main()
