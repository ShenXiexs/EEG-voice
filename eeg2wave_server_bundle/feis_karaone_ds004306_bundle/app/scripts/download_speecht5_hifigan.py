#!/usr/bin/env python3
"""Download and pin the only external v0730 waveform backend."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0730.runtime import load_config, resolve_config_path, write_json
from src.open_vocab_0730.vocoder import model_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin microsoft/speecht5_hifigan under the v0730 output namespace")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path, cfg = load_config(args.config)
    destination = resolve_config_path(config_path, cfg["paths"]["vocoder_root"])
    if destination.exists() and not args.force:
        print(f"[0730 vocoder] already cached: {destination}")
    else:
        try:
            from transformers import SpeechT5HifiGan
        except ImportError as error:
            raise RuntimeError("install the app requirements before downloading SpeechT5 HiFi-GAN") from error
        SpeechT5HifiGan.from_pretrained(cfg["vocoder"]["repo_id"]).save_pretrained(destination)
    manifest = model_manifest(destination)
    manifest.update({"schema_version": "openvoice-0730-vocoder-v1", "repo_id": cfg["vocoder"]["repo_id"]})
    write_json(resolve_config_path(config_path, cfg["paths"]["vocoder_manifest"]), manifest)
    print(destination)


if __name__ == "__main__":
    main()
