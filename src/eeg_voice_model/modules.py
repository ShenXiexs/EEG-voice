"""Reusable model blocks for the BrainOmni-style EEG voice model.

The file mirrors the role of BrainOmni's `model_utils/module.py`: each class is
small enough to explain independently in a meeting, and `tokenizer.py` only
assembles these blocks into the full tokenizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorEmbedding(nn.Module):
    """Embed electrode coordinates and channel availability.

    Input:
        sensor_pos: `[B, C, 3]`
        channel_mask: `[B, C]`, True for valid channels
    Output:
        sensor_embedding: `[B, C, D]`
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )
        self.type_embedding = nn.Embedding(2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sensor_pos: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if channel_mask is None:
            channel_mask = torch.ones(sensor_pos.shape[:2], dtype=torch.bool, device=sensor_pos.device)
        sensor_type = channel_mask.long()
        x = self.pos_mlp(sensor_pos.float()) + self.type_embedding(sensor_type)
        return self.dropout(self.norm(x))


class ConvBlock(nn.Module):
    """Conv1D + GroupNorm + GELU block used by the temporal encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalEncoder(nn.Module):
    """SEANet-like temporal encoder applied independently to each channel.

    Input:
        eeg: `[B, C, T]`
    Output:
        channel_features: `[B, C, W, D]`
    """

    def __init__(self, dim: int, hidden: int, downsample_rates: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        cur = hidden
        for stride in downsample_rates:
            layers.append(ConvBlock(in_ch, cur, stride=stride, dropout=dropout))
            in_ch = cur
            cur = min(dim, cur * 2)
        layers.append(nn.Conv1d(in_ch, dim, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        batch, channels, time = eeg.shape
        x = eeg.reshape(batch * channels, 1, time)
        x = self.net(x)
        return x.reshape(batch, channels, x.shape[-2], x.shape[-1]).permute(0, 1, 3, 2)


class LatentQueryAggregator(nn.Module):
    """Compress channel features into latent neural queries using cross-attention.

    This is the v0 analogue of BrainOmni's latent neuro queries / backward
    solution.

    Input:
        channel_features: `[B, C, W, D]`
        sensor_embedding: `[B, C, D]`
    Output:
        z: `[B, Q, W, D]`
    """

    def __init__(self, dim: int, latent_queries: int, n_heads: int, dropout: float):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(latent_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(
        self,
        channel_features: torch.Tensor,
        sensor_embedding: torch.Tensor,
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, channels, windows, dim = channel_features.shape
        x = channel_features + sensor_embedding[:, :, None, :]
        x = x.permute(0, 2, 1, 3).reshape(batch * windows, channels, dim)
        q = self.latent_queries[None, :, :].expand(batch * windows, -1, -1)
        key_padding_mask = None
        if channel_mask is not None:
            key_padding_mask = (~channel_mask.bool())[:, None, :].expand(batch, windows, channels)
            key_padding_mask = key_padding_mask.reshape(batch * windows, channels)
        z, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        z = self.norm(z + self.ff(z))
        return z.reshape(batch, windows, z.shape[1], dim).permute(0, 2, 1, 3)


class ResidualVectorQuantizer(nn.Module):
    """Small residual vector quantizer with straight-through gradients.

    Input:
        z: `[B, Q, W, D]`
    Output:
        z_q: `[B, Q, W, D]`
        tokens: `[B, Q, W, num_quantizers]`
    """

    def __init__(self, dim: int, codebook_size: int, num_quantizers: int):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.codebooks = nn.Parameter(torch.randn(num_quantizers, codebook_size, dim) / dim**0.5)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized_total = torch.zeros_like(z)
        all_indices = []
        commit_loss = z.new_tensor(0.0)
        flat_shape = z.shape[:-1]
        for idx in range(self.num_quantizers):
            codebook = self.codebooks[idx]
            flat = residual.reshape(-1, self.dim)
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat @ codebook.T
                + codebook.pow(2).sum(dim=1).unsqueeze(0)
            )
            indices = torch.argmin(distances, dim=-1)
            codes = F.embedding(indices, codebook).reshape_as(residual)
            quantized_total = quantized_total + codes
            commit_loss = commit_loss + F.mse_loss(residual, codes.detach()) + 0.25 * F.mse_loss(codes, residual.detach())
            residual = residual - codes.detach()
            all_indices.append(indices.reshape(*flat_shape))
        quantized = z + (quantized_total - z).detach()
        tokens = torch.stack(all_indices, dim=-1)
        return quantized, tokens, commit_loss / self.num_quantizers


class TemporalDecoder(nn.Module):
    """Reconstruct channel EEG from latent query tokens.

    Input:
        z_q: `[B, Q, W, D]`
        sensor_embedding: `[B, C, D]`
    Output:
        x_rec: `[B, C, T]`
    """

    def __init__(self, dim: int, hidden: int, downsample_rates: tuple[int, ...], n_heads: int, dropout: float):
        super().__init__()
        self.sensor_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        layers: list[nn.Module] = []
        rates = tuple(reversed(downsample_rates))
        in_ch = dim
        cur = hidden
        for stride in rates:
            layers.extend(
                [
                    nn.ConvTranspose1d(in_ch, cur, kernel_size=2 * stride, stride=stride, padding=stride // 2),
                    nn.GroupNorm(1, cur),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = cur
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=7, padding=3))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, sensor_embedding: torch.Tensor, output_samples: int) -> torch.Tensor:
        batch, queries, windows, dim = z.shape
        channels = sensor_embedding.shape[1]
        keys = z.permute(0, 2, 1, 3).reshape(batch * windows, queries, dim)
        query = sensor_embedding[:, None, :, :].expand(batch, windows, channels, dim).reshape(batch * windows, channels, dim)
        channel_features, _ = self.sensor_attn(query, keys, keys, need_weights=False)
        channel_features = channel_features.reshape(batch, windows, channels, dim).permute(0, 2, 3, 1)
        x = channel_features.reshape(batch * channels, dim, windows)
        x = self.net(x).reshape(batch, channels, -1)
        if x.shape[-1] > output_samples:
            x = x[..., :output_samples]
        elif x.shape[-1] < output_samples:
            x = F.pad(x, (0, output_samples - x.shape[-1]))
        return x
