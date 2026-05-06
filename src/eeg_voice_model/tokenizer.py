"""BrainOmni-style EEG tokenizer v0.

The tokenizer itself is intentionally thin. The explainable model blocks live
in `modules.py`, matching the reference-code style where core blocks are
defined separately and assembled by the top-level model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .losses import tokenizer_reconstruction_loss
from .modules import (
    LatentQueryAggregator,
    ResidualVectorQuantizer,
    SensorEmbedding,
    TemporalDecoder,
    TemporalEncoder,
)


@dataclass
class BrainStyleEEGTokenizerConfig:
    sample_rate: int = 250
    window_sec: float = 2.0
    dim: int = 256
    latent_queries: int = 16
    codebook_size: int = 512
    num_quantizers: int = 4
    encoder_channels: int = 64
    downsample_rates: tuple[int, ...] = (2, 2, 2, 2)
    n_heads: int = 4
    dropout: float = 0.1

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_sec))


class BrainStyleEEGTokenizerV0(nn.Module):
    """Assemble sensor-aware encoder, latent queries, RVQ, and decoder."""

    def __init__(self, config: BrainStyleEEGTokenizerConfig | None = None):
        super().__init__()
        self.config = config or BrainStyleEEGTokenizerConfig()
        cfg = self.config
        self.sensor_embedding = SensorEmbedding(cfg.dim, cfg.dropout)
        self.encoder = TemporalEncoder(cfg.dim, cfg.encoder_channels, cfg.downsample_rates, cfg.dropout)
        self.aggregator = LatentQueryAggregator(cfg.dim, cfg.latent_queries, cfg.n_heads, cfg.dropout)
        self.quantizer = ResidualVectorQuantizer(cfg.dim, cfg.codebook_size, cfg.num_quantizers)
        self.decoder = TemporalDecoder(cfg.dim, cfg.encoder_channels, cfg.downsample_rates, cfg.n_heads, cfg.dropout)

    @staticmethod
    def normalize_eeg(eeg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        eeg = eeg.float()
        return (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + eps)

    def encode_continuous(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized target, sensor embedding, and continuous latent z."""
        target = self.normalize_eeg(eeg)
        sensor = self.sensor_embedding(sensor_pos, channel_mask)
        channel_features = self.encoder(target)
        z = self.aggregator(channel_features, sensor, channel_mask)
        return target, sensor, z

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latent z with RVQ."""
        return self.quantizer(z)

    def reconstruct(self, z_q: torch.Tensor, sensor_embedding: torch.Tensor, output_samples: int) -> torch.Tensor:
        """Decode quantized latent tokens back to EEG channels."""
        return self.decoder(z_q, sensor_embedding, output_samples=output_samples)

    def forward(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        compute_loss: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        target, sensor, z = self.encode_continuous(eeg, sensor_pos, channel_mask)
        z_q, tokens, commitment_loss = self.quantize(z)
        x_rec = self.reconstruct(z_q, sensor, output_samples=target.shape[-1])
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "z": z,
            "z_q": z_q,
            "tokens": tokens,
            "x_rec": x_rec,
            "target": target,
            "commitment_loss": commitment_loss,
        }
        if compute_loss:
            out["losses"] = tokenizer_reconstruction_loss(x_rec, target, commitment_loss)
        return out

    @torch.no_grad()
    def tokenize(self, eeg: torch.Tensor, sensor_pos: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(eeg, sensor_pos, channel_mask, compute_loss=False)["tokens"]
