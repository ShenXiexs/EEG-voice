from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _position(steps: int, dimension: int) -> torch.Tensor:
    position = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10_000.0) / dimension))
    output = torch.zeros(steps, dimension, dtype=torch.float32)
    output[:, 0::2] = torch.sin(position * divisor)
    output[:, 1::2] = torch.cos(position * divisor[: output[:, 1::2].shape[1]])
    return output


class MFCCMelDecoder(nn.Module):
    """Small audio-only decoder used by V1/V2, never by the EEG encoder.

    The speaker condition is deliberately isolated from the EEG path.  In v3
    primary synthesis callers pass ``canonical_voice``; target voice is used
    only by the V2 audio oracle.
    """

    def __init__(self, *, mfcc_bins: int = 40, mel_bins: int = 80, dimension: int = 128, voice_dim: int = 192):
        super().__init__()
        self.mfcc_bins = mfcc_bins
        self.mel_bins = mel_bins
        self.voice_dim = voice_dim
        self.input = nn.Conv1d(mfcc_bins, dimension, 5, padding=2)
        self.voice = nn.Sequential(nn.Linear(voice_dim, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, dimension * 2))
        self.blocks = nn.Sequential(
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
        )
        self.output = nn.Conv1d(dimension, mel_bins, 1)

    def forward(self, mfcc: torch.Tensor, voice: torch.Tensor) -> torch.Tensor:
        if mfcc.ndim != 3 or mfcc.shape[1] != self.mfcc_bins:
            raise ValueError(f"MFCC decoder expects [B,{self.mfcc_bins},T]")
        if voice.shape != (mfcc.shape[0], self.voice_dim):
            raise ValueError(f"voice must be [B,{self.voice_dim}]")
        hidden = self.input(mfcc)
        scale, bias = self.voice(voice).chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(-1)) + 0.1 * bias.unsqueeze(-1)
        return -80.0 + 80.0 * torch.sigmoid(self.output(self.blocks(hidden)))


class EEGMFCCEncoder(nn.Module):
    """Spatial-temporal EEG encoder with direct canonical MFCC output.

    It has no label, text, speaker, duration, energy, F0, or prosody input.
    Coordinate/channel fusion happens before pooling so reversing signal
    channels genuinely invalidates the spatial correspondence.
    """

    def __init__(self, *, mfcc_bins: int = 40, dimension: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.10, token_steps: int = 16):
        super().__init__()
        self.mfcc_bins = mfcc_bins
        self.token_steps = token_steps
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 64, 15, padding=7), nn.GELU(),
            nn.Conv1d(64, dimension, 9, padding=4), nn.GELU(),
        )
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.fusion = nn.Sequential(
            nn.Linear(dimension * 2, dimension), nn.GELU(), nn.Linear(dimension, dimension), nn.LayerNorm(dimension)
        )
        self.position = nn.Parameter(_position(32, dimension))
        self.trunk = _encoder(dimension, heads, layers, dropout)
        self.mfcc_head = nn.Sequential(
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(), nn.Conv1d(dimension, mfcc_bins, 1)
        )
        self.token_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, mfcc_bins, bias=False))
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(
        self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("eeg must be [B,C,T] and xyz [B,C,3]")
        if channel_mask.shape != eeg.shape[:2] or time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("invalid channel/time mask")
        if not torch.isfinite(eeg).all() or not torch.isfinite(channel_xyz).all():
            raise ValueError("EEG and coordinates must be finite")
        batch, channels, samples = eeg.shape
        temporal = self.temporal(eeg.reshape(batch * channels, 1, samples))
        temporal = F.adaptive_avg_pool1d(temporal, 32).transpose(1, 2).reshape(batch, channels, 32, -1)
        coordinate = self.coordinate(channel_xyz).unsqueeze(2).expand(-1, -1, 32, -1)
        fused = self.fusion(torch.cat((temporal, coordinate), dim=-1))
        weight = channel_mask.to(fused.dtype).view(batch, channels, 1, 1)
        pooled = (fused * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        pooled_mask = F.interpolate(time_mask.float().unsqueeze(1), size=32, mode="nearest").squeeze(1).bool()
        latent = self.trunk(pooled + self.position.unsqueeze(0), src_key_padding_mask=~pooled_mask)
        mfcc = F.interpolate(self.mfcc_head(latent.transpose(1, 2)), size=256, mode="linear", align_corners=False)
        tokens = F.adaptive_avg_pool1d(latent.transpose(1, 2), self.token_steps).transpose(1, 2)
        return mfcc, self.token_head(tokens)
