#!/usr/bin/env python3
"""Assemble ready-to-play v1/v2 audio-control folders without training."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import wavfile

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from scripts.train_open_vocab_0730 import load_renderer
from scripts.train_open_vocab_0730_fixed import load_eeg_fixed
from src.open_vocab_0730.data_fixed import CPDataset, PAIR_ROLES, collate, load_prepared
from src.open_vocab_0730.runtime import default_device, load_config, move_batch, resolve_config_path, write_json
from src.open_vocab_0730.vocoder_fixed import SpeechT5HiFiGanFixed, pcm16


# v1 examples use the fixed best/typical/failure selection from the v1 report.
# v2 matches labels only; it is not the same subject or trial as v1.
V1_EXAMPLES = (
    ("0092_karaone_P02_92", "m", "best_EEG_sensitive"),
    ("0145_karaone_P02_145", "pat", "best_realization_sensitive"),
    ("0032_karaone_P02_32", "uw", "typical"),
    ("0122_karaone_P02_122", "gnaw", "failure"),
)
V2_EXAMPLES = (
    ("karaone:MM05:5", "m"),
    ("karaone:MM05:108", "pat"),
    ("karaone:MM05:3", "uw"),
    ("karaone:MM05:105", "gnaw"),
)
V1_FILES = (
    ("00_reference.wav", "reference"),
    ("01_audio_oracle.wav", "audio_condition_oracle"),
    ("02_full_EEG.wav", "correct_content_correct_realization"),
    ("03_same_label_realization_shuffle.wav", "correct_content_wrong_realization"),
    ("04_content_only.wav", "content_only"),
    ("05_realization_only.wav", "realization_only"),
    ("06_shuffled_EEG.wav", "shuffled_eeg"),
    ("07_zero_EEG.wav", "zero_eeg"),
)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble v1/v2 presentation WAV folders without training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def write_wave(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    waveform = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    wavfile.write(path, sample_rate, (waveform * 32767.0).astype(np.int16))


def crop(waveform: np.ndarray, duration: float, sample_rate: int, maximum: float) -> np.ndarray:
    duration = float(np.clip(duration, 0.10, maximum))
    return waveform[: max(1, int(duration * sample_rate))].copy()


def copy_v1(output: Path) -> list[dict[str, Any]]:
    root = APP.parent / "artifacts" / "open_vocab_0724_factorized_v1_exploratory" / "synthesis" / "karaone" / "validation"
    assembled: list[dict[str, Any]] = []
    for stem, label, selection in V1_EXAMPLES:
        destination = output / "v1" / f"{stem}__{label}"
        destination.mkdir(parents=True, exist_ok=True)
        conditions: dict[str, str] = {}
        for filename, condition in V1_FILES:
            source = root / condition / f"{stem}.wav"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination / filename)
            conditions[filename] = str(source)
        metadata = {
            "version": "v1",
            "sample_key": f"karaone:P02:{int(stem.split('_')[-1])}",
            "label": label,
            "selection": selection,
            "source_split": "KaraOne validation",
            "conditions": conditions,
            "interpretation": "exploratory diagnostic audio; v1 formal audio gate did not pass",
        }
        write_json(destination / "metadata.json", metadata)
        assembled.append({"folder": str(destination), **metadata})
    return assembled


@torch.no_grad()
def assemble_v2(config_path: Path, output: Path, device: torch.device, resume: bool) -> list[dict[str, Any]]:
    config_path, cfg = load_config(config_path)
    records = load_prepared(resolve_config_path(config_path, cfg["paths"]["prepared_cache"]))
    dataset = CPDataset(records, PAIR_ROLES)
    indices = {dataset[item]["sample_key"]: item for item in range(len(dataset))}
    missing = [key for key, _ in V2_EXAMPLES if key not in indices]
    if missing:
        raise ValueError(f"presentation sample keys absent from v2 pair records: {missing}")
    manifest_path = resolve_config_path(config_path, cfg["paths"]["pair_root"]) / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = {row["sample_key"]: row for row in csv.DictReader(handle)}
    model, _ = load_eeg_fixed(config_path, cfg, device)
    renderer = load_renderer(config_path, cfg, device)
    backend = SpeechT5HiFiGanFixed(resolve_config_path(config_path, cfg["paths"]["vocoder_root"]), device=device)
    rate, maximum = int(cfg["vocoder"]["sample_rate"]), float(cfg["audio"]["max_seconds"])
    assembled: list[dict[str, Any]] = []
    for key, expected_label in V2_EXAMPLES:
        item = dataset[indices[key]]
        if item["label"] != expected_label:
            raise ValueError(f"label mismatch for {key}: {item['label']} != {expected_label}")
        batch = move_batch(collate([item]), device)
        destination = output / "v2" / f"{safe_name(key)}__{expected_label}"
        destination.mkdir(parents=True, exist_ok=True)
        source = manifest[key]
        shutil.copy2(source["reference_wav"], destination / "00_reference.wav")
        shutil.copy2(source["reconstruction_wav"], destination / "02_full_EEG.wav")
        variants = {
            "01_audio_oracle_C_plus_P.wav": ("audio_oracle", None),
            "03_zero_EEG.wav": ("zero", torch.zeros_like(batch["eeg"])),
            "04_time_shuffled_EEG.wav": ("time_shuffled", torch.flip(batch["eeg"], dims=(-1,))),
            "05_channel_shuffled_EEG.wav": ("channel_shuffled", torch.flip(batch["eeg"], dims=(1,))),
        }
        conditions: dict[str, str] = {
            "00_reference.wav": source["reference_wav"],
            "02_full_EEG.wav": source["reconstruction_wav"],
        }
        for filename, (condition, eeg) in variants.items():
            target = destination / filename
            if not (resume and target.is_file() and target.stat().st_size > 0):
                if condition == "audio_oracle":
                    mel = renderer(batch["content_tokens"], batch["prosody"])
                    duration = float(batch["prosody"][0, 0].cpu())
                else:
                    state = model(eeg, batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
                    mel = renderer(state.content_logits, state.prosody)
                    duration = float(state.duration[0].cpu())
                write_wave(target, crop(pcm16(backend.synthesize(mel)), duration, rate, maximum), rate)
            conditions[filename] = condition
        metadata = {
            "version": "v2",
            "sample_key": key,
            "label": item["label"],
            "subject": item["subject"],
            "evaluation_role": item["role"],
            "selection": "label-matched presentation example; not subject/trial-matched to v1",
            "conditions": conditions,
            "interpretation": "diagnostic audio only; v2 generated-speech scientific gate did not pass",
        }
        write_json(destination / "metadata.json", metadata)
        assembled.append({"folder": str(destination), **metadata})
    return assembled


def write_readme(output: Path, v1: list[dict[str, Any]], v2: list[dict[str, Any]]) -> None:
    lines = [
        "# v1 / v2 presentation audio trials",
        "",
        "This is a ready-to-play qualitative demonstration, not a quantitative benchmark.",
        "The v1 and v2 folders share four labels (`m`, `pat`, `uw`, `gnaw`), but they are not the same subject or trial.",
        "v1 WAVs retain their original 24 kHz sample rate; v2 WAVs retain their original 16 kHz sample rate.",
        "They are intentionally not resampled, so the source artifacts remain acoustically unchanged.",
        "",
        "## v1 play order",
        "",
        "1. `00_reference.wav` — paired overt reference",
        "2. `01_audio_oracle.wav` — audio-condition oracle",
        "3. `02_full_EEG.wav` — full EEG condition",
        "4. `03_same_label_realization_shuffle.wav` — realization control",
        "5. `04_content_only.wav` / `05_realization_only.wav` — branch ablations",
        "6. `06_shuffled_EEG.wav` / `07_zero_EEG.wav` — EEG controls",
        "",
        "## v2 play order",
        "",
        "1. `00_reference.wav` — paired overt reference",
        "2. `01_audio_oracle_C_plus_P.wav` — target audio C+P through renderer/vocoder",
        "3. `02_full_EEG.wav` — existing all-pair full EEG reconstruction",
        "4. `03_zero_EEG.wav` / `04_time_shuffled_EEG.wav` / `05_channel_shuffled_EEG.wav` — controls",
        "",
        "Neither version establishes correct-EEG superiority over its strong controls.",
        "The audio is assembled to make the qualitative behavior and those controls easy to demonstrate.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "index.json", {"v1": v1, "v2": v2})


def main() -> None:
    args = parse()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v1 = copy_v1(output)
    v2 = assemble_v2(args.config, output, default_device(args.device), args.resume)
    write_readme(output, v1, v2)
    print(output, flush=True)


if __name__ == "__main__":
    main()
