from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _encoder(dimension: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d_model=dimension, nhead=heads, dim_feedforward=dimension * 4, activation="gelu", dropout=dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers)


def _resample(tokens: torch.Tensor, steps: int) -> torch.Tensor:
    return F.interpolate(tokens.transpose(1, 2), size=steps, mode="linear", align_corners=False).transpose(1, 2)


@dataclass(frozen=True)
class CPState:
    content_features: torch.Tensor
    content_clip_tokens: torch.Tensor
    content_logits: torch.Tensor
    duration: torch.Tensor
    loudness: torch.Tensor
    activity_logits: torch.Tensor
    envelope: torch.Tensor

    @property
    def content_tokens(self) -> torch.Tensor:
        return self.content_logits.argmax(-1)

    @property
    def prosody(self) -> torch.Tensor:
        return torch.cat((self.duration[:, None], self.loudness[:, None], torch.sigmoid(self.activity_logits), self.envelope), dim=-1)


@dataclass(frozen=True)
class CPGeneration:
    state: CPState
    log_mel: torch.Tensor


class ContentProsodyEEG(nn.Module):
    """Label-free EEG encoder with only explicit C and P output heads."""

    def __init__(self, *, codebook_size: int = 128, dimension: int = 128, heads: int = 4, layers: int = 2, content_steps: int = 16, prosody_steps: int = 32, dropout: float = 0.1):
        super().__init__()
        self.content_steps = content_steps
        self.prosody_steps = prosody_steps
        self.temporal = nn.Sequential(nn.Conv1d(1, 64, 15, padding=7), nn.GELU(), nn.Conv1d(64, dimension, 9, padding=4), nn.GELU())
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.trunk = _encoder(dimension, heads, layers, dropout)
        self.content_query = nn.Parameter(torch.randn(content_steps, dimension) * 0.02)
        self.content_attention = nn.MultiheadAttention(dimension, heads, batch_first=True, dropout=dropout)
        self.content_clip_projection = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 64, bias=False))
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.content_out = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, codebook_size))
        self.prosody_norm = nn.LayerNorm(dimension)
        self.duration_out = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.loudness_out = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.activity_out = nn.Conv1d(dimension, 1, 1)
        self.envelope_out = nn.Conv1d(dimension, 1, 1)

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor) -> CPState:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3) or channel_mask.shape != eeg.shape[:2] or time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("EEG API expects eeg[B,C,T], xyz[B,C,3], channel_mask[B,C], time_mask[B,T]")
        if not torch.isfinite(eeg).all() or not torch.isfinite(channel_xyz).all():
            raise ValueError("EEG and channel coordinates must be finite")
        batch, channels, samples = eeg.shape
        temporal = self.temporal(eeg.reshape(batch * channels, 1, samples))
        temporal = F.adaptive_avg_pool1d(temporal, 32).transpose(1, 2).reshape(batch, channels, 32, -1)
        weights = channel_mask.to(temporal.dtype).view(batch, channels, 1, 1)
        pooled = (temporal * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        coordinate = (self.coordinate(channel_xyz) * channel_mask.to(eeg.dtype).unsqueeze(-1)).sum(1) / channel_mask.to(eeg.dtype).sum(1, keepdim=True).clamp_min(1.0)
        latent = self.trunk(pooled + coordinate.unsqueeze(1))
        query = self.content_query.unsqueeze(0).expand(batch, -1, -1)
        content, _ = self.content_attention(query, latent, latent)
        prosody_tokens = self.prosody_norm(latent)
        global_token = prosody_tokens.mean(1)
        activity = self.activity_out(prosody_tokens.transpose(1, 2)).squeeze(1)
        envelope = self.envelope_out(prosody_tokens.transpose(1, 2)).squeeze(1)
        return CPState(content_features=content, content_clip_tokens=self.content_clip_projection(content), content_logits=self.content_out(content), duration=self.duration_out(global_token).squeeze(-1), loudness=self.loudness_out(global_token).squeeze(-1), activity_logits=activity, envelope=envelope)


class CPMelRenderer(nn.Module):
    """Audio-only explicit C/P-to-mel renderer with a single shared neutral voice."""

    def __init__(self, *, codebook_size: int = 128, dimension: int = 128, mel_frames: int = 400, mel_bins: int = 80):
        super().__init__()
        self.mel_frames = mel_frames
        self.content_embedding = nn.Embedding(codebook_size, dimension)
        self.prosody = nn.Sequential(nn.Linear(2, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.neutral_voice = nn.Parameter(torch.zeros(dimension))
        self.blocks = nn.Sequential(nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(), nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(), nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU())
        self.output = nn.Conv1d(dimension, mel_bins, 1)

    def forward(self, content: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        if content.ndim == 3:
            tokens = content.softmax(-1) @ self.content_embedding.weight
        elif content.ndim == 2:
            tokens = self.content_embedding(content.long())
        else:
            raise ValueError("content must be token IDs [B,16] or logits [B,16,K]")
        if prosody.ndim != 2 or prosody.shape[-1] != 66:
            raise ValueError("prosody must be [B,66]")
        sequence = _resample(tokens, 32)
        global_p = self.prosody(prosody[:, :2]).unsqueeze(1)
        activity, envelope = prosody[:, 2:34], prosody[:, 34:66]
        p_curve = torch.stack((activity, envelope), -1)
        p_curve = self.prosody(p_curve).view(content.shape[0], 32, -1)
        sequence = sequence + p_curve + global_p + self.neutral_voice.view(1, 1, -1)
        value = _resample(sequence, self.mel_frames).transpose(1, 2)
        return -80.0 + 80.0 * torch.sigmoid(self.output(self.blocks(value)))


class CPGenerator(nn.Module):
    def __init__(self, eeg: ContentProsodyEEG, renderer: CPMelRenderer):
        super().__init__()
        self.eeg = eeg
        self.renderer = renderer

    def encode(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor) -> CPState:
        return self.eeg(eeg, channel_xyz, channel_mask, time_mask)

    def render(self, state: CPState) -> CPGeneration:
        return CPGeneration(state=state, log_mel=self.renderer(state.content_logits, state.prosody))

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor) -> CPGeneration:
        return self.render(self.encode(eeg, channel_xyz, channel_mask, time_mask))
