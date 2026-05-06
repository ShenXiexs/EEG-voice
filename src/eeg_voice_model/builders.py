"""Factory functions for assembling v0 models in a presentation-friendly way."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_simple_yaml
from .heads import AudioContrastiveHead, ProbeHead, VoiceAttributeHead
from .tokenizer import BrainStyleEEGTokenizerConfig, BrainStyleEEGTokenizerV0


def tokenizer_config_from_dict(cfg: dict[str, Any]) -> BrainStyleEEGTokenizerConfig:
    """Convert the `tokenizer:` section of `configs/model_v0.yaml`."""
    return BrainStyleEEGTokenizerConfig(
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


def build_tokenizer_v0(config: BrainStyleEEGTokenizerConfig | dict[str, Any] | None = None) -> BrainStyleEEGTokenizerV0:
    """Build only the BrainOmni-style EEG tokenizer."""
    if config is None:
        return BrainStyleEEGTokenizerV0()
    if isinstance(config, dict):
        config = tokenizer_config_from_dict(config)
    return BrainStyleEEGTokenizerV0(config)


def build_ds006104_probe_heads(dim: int, label_sizes: dict[str, int]) -> dict[str, ProbeHead]:
    """Build classification probes for ds006104 labels."""
    return {name: ProbeHead(dim, num_classes=size) for name, size in label_sizes.items()}


def build_ds005345_retrieval_head(dim: int, audio_dim: int = 11, proj_dim: int = 256) -> AudioContrastiveHead:
    """Build the EEG/audio contrastive head for speaker stream retrieval."""
    return AudioContrastiveHead(eeg_dim=dim, audio_dim=audio_dim, proj_dim=proj_dim)


def build_voice_attribute_heads(dim: int) -> dict[str, VoiceAttributeHead]:
    """Build optional pitch/intensity/timbre attribute heads."""
    return {
        "f0_bin": VoiceAttributeHead(dim, output_dim=2, task="classification"),
        "intensity_bin": VoiceAttributeHead(dim, output_dim=2, task="classification"),
        "voice_stats": VoiceAttributeHead(dim, output_dim=11, task="regression"),
    }


def build_model_v0_bundle(config_path: str | Path = "configs/model_v0.yaml") -> dict[str, Any]:
    """Build tokenizer and default heads from a YAML config.

    The bundle is intentionally a plain dict so notebooks and demo scripts can
    show each component without a large training framework.
    """
    cfg = load_simple_yaml(Path(config_path))
    tokenizer_cfg = tokenizer_config_from_dict(cfg["tokenizer"])
    tokenizer = build_tokenizer_v0(tokenizer_cfg)
    return {
        "config": cfg,
        "tokenizer": tokenizer,
        "ds005345_retrieval": build_ds005345_retrieval_head(
            dim=tokenizer_cfg.dim,
            audio_dim=int(cfg["heads"]["audio_embedding_dim"]),
            proj_dim=int(cfg["heads"]["projection_dim"]),
        ),
        "voice_attributes": build_voice_attribute_heads(tokenizer_cfg.dim),
    }
