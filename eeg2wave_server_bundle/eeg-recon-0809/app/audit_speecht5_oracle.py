#!/usr/bin/env python3
"""Audit the native-duration SpeechT5 mel -> HiFi-GAN audio contract.

This is intentionally an audio-only gate.  It never touches EEG checkpoints
and writes source/oracle WAVs without independent peak normalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.io import wavfile
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from cache_speech_targets import active_crop, hubert_features, load_hubert, load_wave
from prepare_training_data import load_config
from eeg2speech.speecht5 import HOP_SAMPLES, SAMPLE_RATE, SpeechT5HiFiGan, model_manifest


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def _write(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, SAMPLE_RATE, np.round(np.clip(np.nan_to_num(value), -1, 1) * 32767).astype(np.int16))


def _retrieval(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-8)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-8)
    order = np.argsort(-(left @ right.T), axis=1)
    ranks = np.asarray([int(np.flatnonzero(order[index] == index)[0]) + 1 for index in range(len(left))])
    return {"r1": float(np.mean(ranks == 1)), "mrr": float(np.mean(1.0 / ranks)),
            "chance_r1": float(1.0 / len(left))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ds004940_conditioned_v2.yaml")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs" / "training_data_v4_ds004940_fixed.yaml")
    parser.add_argument("--hubert-local-path", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=20)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, _ = load_config(args.data_config)
    data_cfg["audio"]["content"]["hubert_local_path"] = str(args.hubert_local_path.resolve())
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, keep_default_na=False, low_memory=False)
    frame = frame[(frame.dataset == "ds004940") & (frame.build_status == "included") &
                  (frame.supervision_type == "paired_audio")].drop_duplicates("audio_sha256")
    frame = frame.sort_values("trial_id").head(int(args.max_pairs))
    if not len(frame): raise RuntimeError("native oracle has no DS004940 paired-audio rows")
    configured_model_root = Path(cfg["native_audio"]["local_hifigan_path"])
    model_root = (configured_model_root if configured_model_root.is_absolute()
                  else (ROOT / configured_model_root).resolve())
    vocoder = SpeechT5HiFiGan(model_root, device=_device())
    hubert_runtime = load_hubert(data_cfg, allow_download=False)
    rows = []; oracle_globals = []; target_globals = []
    with h5py.File(args.targets, "r") as targets:
        if str(targets.attrs.get("native_mel_contract", "")) != "speecht5_native_log_mel_v1":
            raise RuntimeError("target cache lacks the pinned native SpeechT5 mel contract")
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc="native SpeechT5 oracle", unit="pair"):
            audio_id = str(row.get("audio_id") or f"audio-{row.audio_sha256[:16]}-{row.audio_semantics}")
            group = targets[audio_id]
            mel = torch.from_numpy(group["native_speecht5_mel"][:]).unsqueeze(0).to(_device())
            generated = vocoder.synthesize(mel).squeeze().detach().cpu().numpy().astype(np.float32)
            source, _, _ = load_wave(ROOT / str(row.audio_path))
            generated_content, _ = active_crop(generated - float(generated.mean()),
                                                float(data_cfg["audio"]["content"]["vad_threshold_db_below_peak"]))
            generated_local, generated_global = hubert_features(generated_content, data_cfg, hubert_runtime)
            target_local = group["hubert_local"][:].astype(np.float32)
            target_global = group["hubert_global"][:].astype(np.float32)
            local_cosine = float(np.mean(np.sum(
                generated_local / np.maximum(np.linalg.norm(generated_local, axis=1, keepdims=True), 1e-8) *
                target_local / np.maximum(np.linalg.norm(target_local, axis=1, keepdims=True), 1e-8), axis=1)))
            oracle_globals.append(generated_global); target_globals.append(target_global)
            expected_samples = int(group.attrs["native_duration_frames"]) * HOP_SAMPLES
            name = str(row.trial_id)
            _write(output / name / "00_source.wav", source)
            _write(output / name / "01_native_mel_hifigan_oracle.wav", generated)
            rows.append({"trial_id": name, "audio_id": audio_id, "source_seconds": len(source) / SAMPLE_RATE,
                         "oracle_seconds": len(generated) / SAMPLE_RATE,
                         "expected_seconds": expected_samples / SAMPLE_RATE,
                         "duration_error_samples": abs(len(generated) - expected_samples),
                         "hubert_local_cosine": local_cosine,
                         "source_path": str(row.audio_path)})
    retrieval = _retrieval(np.stack(oracle_globals), np.stack(target_globals))
    mean_local = float(np.mean([value["hubert_local_cosine"] for value in rows]))
    checks = {"duration": bool(all(value["duration_error_samples"] <= HOP_SAMPLES for value in rows)),
              "hubert_local": mean_local >= float(cfg["native_audio"]["oracle_hubert_local_min"]),
              "content_retrieval": retrieval["r1"] >= float(cfg["native_audio"]["oracle_retrieval_r1_min"])}
    report = {"schema_version": "speecht5-oracle-v1", "pairs": rows,
              "metrics": {"mean_hubert_local_cosine": mean_local, "content_retrieval": retrieval},
              "gate": {"checks": checks, "passed": all(checks.values())},
              "vocoder": model_manifest(model_root),
              "warning": "Machine audio-only gate; the 20-pair human listening review remains required."}
    (output / "oracle_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "pairs": len(rows), "metrics": report["metrics"], "gate": report["gate"]}, indent=2))
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
