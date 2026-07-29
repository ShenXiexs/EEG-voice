from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CPState, CPMelRenderer


def _encoder(dimension: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dimension,
        nhead=heads,
        dim_feedforward=dimension * 4,
        activation="gelu",
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


def _sinusoidal_position(steps: int, dimension: int) -> torch.Tensor:
    position = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    value = torch.zeros(steps, dimension, dtype=torch.float32)
    value[:, 0::2] = torch.sin(position * divisor)
    value[:, 1::2] = torch.cos(position * divisor[: value[:, 1::2].shape[1]])
    return value


class ContentProsodyEEGFixed(nn.Module):
    """v0730-fixed EEG encoder with paired spatial fusion and temporal order.

    Unlike v1, electrode coordinates are fused with each matching channel
    before channel pooling.  A learned 32-step temporal position is also added
    before the Transformer, so channel/time controls are no longer invariant by
    construction.
    """

    def __init__(
        self,
        *,
        codebook_size: int = 128,
        dimension: int = 128,
        heads: int = 4,
        layers: int = 2,
        content_steps: int = 16,
        prosody_steps: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.content_steps = content_steps
        self.prosody_steps = prosody_steps
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 64, 15, padding=7),
            nn.GELU(),
            nn.Conv1d(64, dimension, 9, padding=4),
            nn.GELU(),
        )
        self.coordinate = nn.Sequential(
            nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.channel_fusion = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
            nn.LayerNorm(dimension),
        )
        self.temporal_position = nn.Parameter(_sinusoidal_position(32, dimension))
        self.trunk = _encoder(dimension, heads, layers, dropout)
        self.content_query = nn.Parameter(torch.randn(content_steps, dimension) * 0.02)
        self.content_attention = nn.MultiheadAttention(
            dimension, heads, batch_first=True, dropout=dropout
        )
        self.content_clip_projection = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, 64, bias=False)
        )
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.content_out = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, codebook_size))
        self.prosody_norm = nn.LayerNorm(dimension)
        self.duration_out = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.loudness_out = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.activity_out = nn.Conv1d(dimension, 1, 1)
        self.envelope_out = nn.Conv1d(dimension, 1, 1)
        # Calibrated starts for KaraOne's fit-only target support: about 0.8 s,
        # -25 dB loudness, 20% activity, and -50 dB inactive envelope.  These
        # are initial biases only; no label or evaluation data enters them.
        nn.init.constant_(self.duration_out[1].bias, -1.52)
        nn.init.constant_(self.loudness_out[1].bias, 0.79)
        nn.init.constant_(self.activity_out.bias, -1.39)
        nn.init.constant_(self.envelope_out.bias, -0.85)

    def forward(
        self,
        eeg: torch.Tensor,
        channel_xyz: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> CPState:
        if (
            eeg.ndim != 3
            or channel_xyz.shape != (*eeg.shape[:2], 3)
            or channel_mask.shape != eeg.shape[:2]
            or time_mask.shape != (eeg.shape[0], eeg.shape[2])
        ):
            raise ValueError("EEG API expects eeg[B,C,T], xyz[B,C,3], channel_mask[B,C], time_mask[B,T]")
        if not torch.isfinite(eeg).all() or not torch.isfinite(channel_xyz).all():
            raise ValueError("EEG and channel coordinates must be finite")

        batch, channels, samples = eeg.shape
        temporal = self.temporal(eeg.reshape(batch * channels, 1, samples))
        temporal = F.adaptive_avg_pool1d(temporal, 32).transpose(1, 2)
        temporal = temporal.reshape(batch, channels, 32, -1)
        coordinate = self.coordinate(channel_xyz).unsqueeze(2).expand(-1, -1, 32, -1)
        paired = self.channel_fusion(torch.cat((temporal, coordinate), dim=-1))
        channel_weights = channel_mask.to(paired.dtype).view(batch, channels, 1, 1)
        pooled = (paired * channel_weights).sum(1) / channel_weights.sum(1).clamp_min(1.0)

        pooled_time_mask = F.interpolate(
            time_mask.float().unsqueeze(1), size=32, mode="nearest"
        ).squeeze(1).bool()
        latent = pooled + self.temporal_position.unsqueeze(0)
        latent = self.trunk(latent, src_key_padding_mask=~pooled_time_mask)

        query = self.content_query.unsqueeze(0).expand(batch, -1, -1)
        content, _ = self.content_attention(
            query, latent, latent, key_padding_mask=~pooled_time_mask
        )
        prosody_tokens = self.prosody_norm(latent)
        valid = pooled_time_mask.to(prosody_tokens.dtype).unsqueeze(-1)
        global_token = (prosody_tokens * valid).sum(1) / valid.sum(1).clamp_min(1.0)

        # Bound all renderer-facing global/curve values to the audio training
        # support.  This prevents an EEG head from driving a frozen renderer far
        # outside the oracle C/P distribution.
        duration = 0.10 + 3.90 * torch.sigmoid(self.duration_out(global_token).squeeze(-1))
        loudness = -80.0 + 80.0 * torch.sigmoid(self.loudness_out(global_token).squeeze(-1))
        activity = self.activity_out(prosody_tokens.transpose(1, 2)).squeeze(1)
        envelope = -80.0 + 100.0 * torch.sigmoid(
            self.envelope_out(prosody_tokens.transpose(1, 2)).squeeze(1)
        )
        return CPState(
            content_features=content,
            content_clip_tokens=self.content_clip_projection(content),
            content_logits=self.content_out(content),
            duration=duration,
            loudness=loudness,
            activity_logits=activity,
            envelope=envelope,
        )


__all__ = ["CPMelRenderer", "ContentProsodyEEGFixed"]
