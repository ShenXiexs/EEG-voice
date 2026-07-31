#!/usr/bin/env python3
"""Cache the frozen vocoder required by V0 before any EEG training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_v3.runtime import load_config, output_path, write_json
from src.open_vocab_v3.vocoder import model_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache v3 SpeechT5 HiFi-GAN")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path, cfg = load_config(args.config)
    root = output_path(config_path, cfg, "vocoder_root")
    if not root.is_dir() or not any(root.iterdir()):
        from transformers import SpeechT5HifiGan
        root.parent.mkdir(parents=True, exist_ok=True)
        SpeechT5HifiGan.from_pretrained(cfg["vocoder"]["repo_id"]).save_pretrained(root)
    write_json(output_path(config_path, cfg, "vocoder_base_manifest"), model_manifest(root, adapted=False))
    print(root, flush=True)


if __name__ == "__main__":
    main()
