from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _transformer(dimension: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d_model=dimension, nhead=heads, dim_feedforward=dimension * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers)


def _resample(tokens: torch.Tensor, steps: int) -> torch.Tensor:
    return F.interpolate(tokens.transpose(1, 2), size=steps, mode="linear", align_corners=False).transpose(1, 2)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)


@dataclass(frozen=True)
class DualLatentAudioState:
    linguistic_latent: torch.Tensor
    realization_latent: torch.Tensor
    linguistic_mask: torch.Tensor
    realization_mask: torch.Tensor
    coarse_log_mel: torch.Tensor
    coarse_activity_logits: torch.Tensor
    log_mel: torch.Tensor
    activity_logits: torch.Tensor


@dataclass(frozen=True)
class DualLatentEEGState:
    linguistic_latent: torch.Tensor
    realization_latent: torch.Tensor
    linguistic_mask: torch.Tensor
    realization_mask: torch.Tensor
    evidence_probability: torch.Tensor


@dataclass(frozen=True)
class DualLatentGeneration:
    linguistic_latent: torch.Tensor
    realization_latent: torch.Tensor
    log_mel: torch.Tensor
    activity_mask: torch.Tensor
    duration_seconds: torch.Tensor
    evidence_probability: torch.Tensor


class ContentProjector(nn.Module):
    def __init__(self, input_dimension: int = 768, dimension: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dimension), nn.Linear(input_dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.encoder = _transformer(dimension, heads, 2, dropout)

    def forward(self, hubert: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hubert.ndim != 3 or hubert.shape[-1] != 768 or mask.shape != hubert.shape[:2]: raise ValueError("HuBERT input must be [B,T,768] with mask [B,T]")
        if not torch.isfinite(hubert).all(): raise ValueError("HuBERT features must be finite")
        value = self.encoder(self.net(hubert), src_key_padding_mask=~mask.bool())
        return _resample(value, 50), torch.ones(hubert.shape[0], 50, device=hubert.device, dtype=torch.bool)


class MelDecoder(nn.Module):
    def __init__(self, linguistic_dimension: int = 128, realization_dimension: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.linguistic = nn.Linear(linguistic_dimension, 128)
        self.realization = nn.Linear(realization_dimension, 128)
        self.encoder = _transformer(128, heads, layers, 0.1)
        self.temporal = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.GELU(), nn.Conv1d(128, 128, 5, padding=2), nn.GELU())
        self.mel = nn.Conv1d(128, 80, 1)
        self.activity = nn.Conv1d(128, 1, 1)

    def forward(self, linguistic: torch.Tensor, realization: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.linguistic(linguistic)
        if realization is not None: value = value + self.realization(realization)
        value = self.encoder(value)
        value = F.interpolate(value.transpose(1, 2), size=400, mode="linear", align_corners=False)
        value = self.temporal(value)
        return torch.clamp(self.mel(value), -80.0, 0.0), self.activity(value).squeeze(1)


class ResidualEncoder(nn.Module):
    def __init__(self, dimension: int = 64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(81, 128, 5, padding=2), nn.GELU(), nn.Conv1d(128, 128, 5, padding=2), nn.GELU())
        self.out = nn.Linear(128, dimension)

    def forward(self, residual_mel: torch.Tensor, activity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_mel.shape[1:] != (80, 400): raise ValueError("residual mel must be [B,80,400]")
        source = torch.cat((residual_mel, activity.to(residual_mel.dtype).unsqueeze(1)), 1)
        value = self.conv(source)
        value = F.interpolate(value, size=50, mode="linear", align_corners=False).transpose(1, 2)
        return self.out(value), torch.ones(value.shape[:2], device=value.device, dtype=torch.bool)


class DualLatentAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = ContentProjector()
        self.content_decoder = MelDecoder(realization_dimension=64)
        self.residual = ResidualEncoder()
        self.decoder = MelDecoder(realization_dimension=64)

    def encode_content(self, hubert: torch.Tensor, hubert_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        linguistic, mask = self.content(hubert, hubert_mask)
        coarse, coarse_activity = self.content_decoder(linguistic, None)
        return linguistic, mask, coarse, coarse_activity

    def decode_content(self, linguistic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.content_decoder(linguistic, None)

    def decode(self, linguistic: torch.Tensor, realization: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(linguistic, realization)

    def forward(self, hubert: torch.Tensor, hubert_mask: torch.Tensor, mel: torch.Tensor, activity: torch.Tensor) -> DualLatentAudioState:
        linguistic, linguistic_mask, coarse, coarse_activity = self.encode_content(hubert, hubert_mask)
        realization, realization_mask = self.residual(mel - coarse.detach(), activity)
        generated, generated_activity = self.decode(linguistic, realization)
        return DualLatentAudioState(linguistic, realization, linguistic_mask, realization_mask, coarse, coarse_activity, generated, generated_activity)


class EEGEncoder(nn.Module):
    def __init__(self, *, dimension: int = 128, heads: int = 4, layers: int = 2, patch_size: int = 64, patch_hop: int = 32):
        super().__init__()
        self.patch_size = patch_size; self.patch_hop = patch_hop
        self.temporal = nn.Sequential(nn.Conv1d(1, 64, 15, padding=7), nn.GELU(), nn.Conv1d(64, dimension, 9, padding=4), nn.GELU())
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.trunk = _transformer(dimension, heads, layers, 0.1)
        self.linguistic_query = nn.Parameter(torch.randn(50, dimension) * 0.02)
        self.realization_query = nn.Parameter(torch.randn(50, dimension) * 0.02)
        self.linguistic_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.realization_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.realization_out = nn.Linear(dimension, 64)
        self.evidence = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor) -> DualLatentEEGState:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3) or channel_mask.shape != eeg.shape[:2] or time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("EEG API expects eeg[B,C,T], xyz[B,C,3], channel_mask[B,C], time_mask[B,T]")
        if not torch.isfinite(eeg).all() or not torch.isfinite(channel_xyz).all(): raise ValueError("EEG and channel coordinates must be finite")
        batch, channels, samples = eeg.shape
        source = self.temporal(eeg.reshape(batch * channels, 1, samples))
        patches = source.unfold(2, self.patch_size, self.patch_hop).mean(-1).transpose(1, 2).reshape(batch, channels, -1, source.shape[1])
        weights = channel_mask.to(patches.dtype).unsqueeze(-1).unsqueeze(-1)
        pooled = (patches * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        coordinates = (self.coordinate(channel_xyz) * channel_mask.to(eeg.dtype).unsqueeze(-1)).sum(1) / channel_mask.to(eeg.dtype).sum(1, keepdim=True).clamp_min(1.0)
        pooled = pooled + coordinates.unsqueeze(1)
        latent = self.trunk(pooled)
        linguistic_query = self.linguistic_query.unsqueeze(0).expand(batch, -1, -1)
        realization_query = self.realization_query.unsqueeze(0).expand(batch, -1, -1)
        linguistic, _ = self.linguistic_attention(linguistic_query, latent, latent)
        realization, _ = self.realization_attention(realization_query, latent, latent)
        evidence = torch.sigmoid(self.evidence(latent.mean(1)).squeeze(-1))
        mask = torch.ones(batch, 50, dtype=torch.bool, device=eeg.device)
        return DualLatentEEGState(linguistic, self.realization_out(realization), mask, mask, evidence)


class DualLatentEEGToSpeech(nn.Module):
    """Strictly label-free four-input inference facade."""
    def __init__(self, eeg_encoder: EEGEncoder, audio_model: DualLatentAudioModel):
        super().__init__(); self.eeg_encoder = eeg_encoder; self.audio_model = audio_model

    def encode(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor) -> DualLatentEEGState:
        return self.eeg_encoder(eeg, channel_xyz, channel_mask, time_mask)

    def generate(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor) -> DualLatentGeneration:
        state = self.encode(eeg, channel_xyz, channel_mask, time_mask)
        mel, activity_logits = self.audio_model.decode(state.linguistic_latent, state.realization_latent)
        evidence = state.evidence_probability.view(-1, 1, 1)
        gated = -80.0 + evidence * (mel + 80.0)
        activity = (torch.sigmoid(activity_logits) * state.evidence_probability.unsqueeze(-1)) >= 0.5
        duration = activity.sum(-1).to(torch.float32) * 0.01
        return DualLatentGeneration(state.linguistic_latent, state.realization_latent, gated, activity, duration, state.evidence_probability)

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor) -> DualLatentEEGState:
        return self.encode(eeg, channel_xyz, channel_mask, time_mask)
