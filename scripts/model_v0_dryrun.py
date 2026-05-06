#!/usr/bin/env python3
"""Dry-run entrypoint for BrainOmni-style EEG Voice Model v0.

This script does not train. It checks tensor shapes, configured dataset
adapters, and voice embedding construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eeg_voice_model.audio_features import build_ds005345_voice_stats
from src.eeg_voice_model.config import load_simple_yaml


def synthetic(config_path: Path) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for synthetic model dry-run. Install the CUDA stack first, "
            "then rerun: python3 scripts/model_v0_dryrun.py --mode synthetic"
        ) from exc

    from src.eeg_voice_model.builders import build_ds005345_retrieval_head, build_tokenizer_v0
    from src.eeg_voice_model.heads import ProbeHead
    from src.eeg_voice_model.tokenizer import BrainStyleEEGTokenizerConfig

    cfg = load_simple_yaml(config_path)["tokenizer"]
    model_cfg = BrainStyleEEGTokenizerConfig(
        sample_rate=int(cfg["sample_rate"]),
        window_sec=float(cfg["window_sec"]),
        dim=int(cfg["dim"]),
        latent_queries=int(cfg["latent_queries"]),
        codebook_size=int(cfg["codebook_size"]),
        num_quantizers=int(cfg["num_quantizers"]),
        encoder_channels=int(cfg["encoder_channels"]),
        downsample_rates=tuple(int(x) for x in cfg["downsample_rates"]),
        n_heads=int(cfg["n_heads"]),
        dropout=float(cfg["dropout"]),
    )
    model = build_tokenizer_v0(model_cfg)
    eeg = torch.randn(2, 64, model_cfg.window_samples)
    sensor_pos = torch.randn(2, 64, 3) * 0.01
    channel_mask = torch.ones(2, 64, dtype=torch.bool)
    out = model(eeg, sensor_pos, channel_mask)
    probe = ProbeHead(model_cfg.dim, num_classes=8)
    probe_out = probe(out["z_q"], torch.tensor([0, 1]))
    audio_head = build_ds005345_retrieval_head(model_cfg.dim, audio_dim=11)
    audio_out = audio_head(out["z_q"], torch.randn(2, 11))
    print("synthetic ok")
    print("z", tuple(out["z"].shape))
    print("tokens", tuple(out["tokens"].shape), int(out["tokens"].max()))
    print("x_rec", tuple(out["x_rec"].shape))
    print("tokenizer_loss", float(out["losses"]["loss"]))
    print("probe_loss", float(probe_out["loss"]))
    print("contrastive_loss", float(audio_out["loss"]))


def dataset_summary(config_path: Path) -> None:
    cfg = load_simple_yaml(config_path)
    ds005345_root = Path(cfg["data"]["ds005345_root"])
    voice_stats = build_ds005345_voice_stats(ds005345_root)
    print("ds005345 voice embeddings")
    for stream, stats in voice_stats.items():
        print(stream, stats.vector())
    print("run mapping")
    run_cfg = load_simple_yaml(Path(cfg["data"]["ds005345_run_config"]))
    for run, value in run_cfg["runs"].items():
        print(run, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/model_v0.yaml"))
    parser.add_argument("--mode", choices=["synthetic", "dataset-summary"], default="synthetic")
    args = parser.parse_args()
    if args.mode == "synthetic":
        synthetic(args.config)
    else:
        dataset_summary(args.config)


if __name__ == "__main__":
    main()
