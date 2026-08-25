"""Coordinate-aware variable-channel EEG content model.

The encoder is a focused migration of ``open_vocab_v3/cp_temporal.py`` from
the reference bundle.  Dataset identity is used only for an input affine and
post-fusion normalization; labels, audio, subjects and conditions are never
model inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)


class ConformerBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float, kernel: int = 31):
        super().__init__()
        hidden = dimension * 4
        self.ff1_norm = nn.LayerNorm(dimension)
        self.ff1 = nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, dimension))
        self.attn_norm = nn.LayerNorm(dimension)
        self.attn = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dimension)
        self.conv_in = nn.Conv1d(dimension, dimension * 2, 1)
        self.depthwise = nn.Conv1d(dimension, dimension, kernel, padding=kernel // 2, groups=dimension)
        self.conv_out = nn.Conv1d(dimension, dimension, 1)
        self.ff2_norm = nn.LayerNorm(dimension)
        self.ff2 = nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, dimension))
        self.out_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value + 0.5 * self.dropout(self.ff1(self.ff1_norm(value)))
        norm = self.attn_norm(value)
        attended = self.attn(norm, norm, norm, key_padding_mask=~mask, need_weights=False)[0]
        value = value + self.dropout(attended)
        conv = self.conv_norm(value).transpose(1, 2)
        left, gate = self.conv_in(conv).chunk(2, dim=1)
        conv = self.conv_out(F.silu(self.depthwise(left * torch.sigmoid(gate)))).transpose(1, 2)
        value = value + self.dropout(conv)
        value = value + 0.5 * self.dropout(self.ff2(self.ff2_norm(value)))
        return torch.where(mask.unsqueeze(-1), self.out_norm(value), torch.zeros_like(value))


class ConformerStack(nn.Module):
    def __init__(self, dimension: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([ConformerBlock(dimension, heads, dropout) for _ in range(layers)])

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, mask)
        return value


class DatasetInputAdapter(nn.Module):
    """Two scalar affine frontends; no dataset vector enters content tokens."""

    def __init__(self, datasets: int):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(datasets))
        self.bias = nn.Parameter(torch.zeros(datasets))

    def forward(self, eeg: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
        gain = self.log_gain[dataset_id].exp().view(-1, 1, 1)
        bias = self.bias[dataset_id].view(-1, 1, 1)
        return eeg * gain + bias


@dataclass
class JointState:
    local: torch.Tensor
    global_embedding: torch.Tensor
    mfcc: torch.Tensor
    phoneme_logits: torch.Tensor
    token_mask: torch.Tensor


class JointEEGContentModel(nn.Module):
    def __init__(self, *, dimension: int = 192, heads: int = 6, layers: int = 4,
                 local_layers: int = 2, dropout: float = 0.1, token_steps: int = 96,
                 target_frames: int = 161, mfcc_dimension: int = 39,
                 hubert_dimension: int = 768, phoneme_classes: int = 64,
                 datasets: int = 2):
        super().__init__()
        self.token_steps = int(token_steps)
        self.target_frames = int(target_frames)
        branch = dimension // 3
        widths = (branch, branch, dimension - 2 * branch)
        self.input_adapter = DatasetInputAdapter(datasets)
        self.temporal = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, width, kernel, padding=kernel // 2), nn.GELU(),
                          nn.Conv1d(width, width, 5, padding=2), nn.GELU())
            for width, kernel in zip(widths, (9, 33, 65))
        ])
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.fusion = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.GELU())
        self.dataset_norms = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(datasets)])
        self.channel_score = nn.Linear(dimension, 1)
        self.position = nn.Parameter(torch.zeros(1, token_steps, dimension))
        nn.init.normal_(self.position, std=0.02)
        self.stem = ConformerStack(dimension, heads, layers, dropout)
        self.local_head = ConformerStack(dimension, heads, local_layers, dropout)
        self.global_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.mfcc_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, mfcc_dimension))
        self.phoneme_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, phoneme_classes))
        self.audio_projection = nn.Sequential(nn.LayerNorm(hubert_dimension), nn.Linear(hubert_dimension, dimension))
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def _dataset_norm(self, value: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(value)
        for identifier, norm in enumerate(self.dataset_norms):
            selected = dataset_id == identifier
            if selected.any():
                output[selected] = norm(value[selected])
        return output

    def project_audio(self, hubert_local: torch.Tensor) -> torch.Tensor:
        return self.audio_projection(hubert_local)

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor,
                time_mask: torch.Tensor, dataset_id: torch.Tensor) -> JointState:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("eeg must be [B,C,T] and channel_xyz [B,C,3]")
        if channel_mask.shape != eeg.shape[:2] or time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("channel/time mask shape mismatch")
        if not channel_mask.any(1).all() or not time_mask.any(1).all():
            raise ValueError("each example needs at least one valid channel and time sample")
        batch, channels, samples = eeg.shape
        adapted = self.input_adapter(eeg, dataset_id)
        # Robust normalization maps padded zeros away from zero, and the
        # dataset adapter has a learnable bias.  Re-apply both masks before any
        # temporal convolution so neither source can leak padding/bad channels.
        adapted = adapted * time_mask[:, None, :].to(adapted.dtype)
        adapted = adapted * channel_mask[:, :, None].to(adapted.dtype)
        flat = adapted.reshape(batch * channels, 1, samples)
        temporal = torch.cat([branch(flat) for branch in self.temporal], dim=1)
        temporal = F.interpolate(temporal, size=self.token_steps, mode="linear", align_corners=False)
        temporal = temporal.transpose(1, 2).reshape(batch, channels, self.token_steps, -1)
        coordinates = self.coordinate(channel_xyz).unsqueeze(2).expand(-1, -1, self.token_steps, -1)
        fused = self.fusion(torch.cat((temporal, coordinates), dim=-1))
        fused_shape = fused.shape
        fused = self._dataset_norm(
            fused.reshape(batch * channels, self.token_steps, -1),
            dataset_id[:, None].expand(-1, channels).reshape(-1),
        ).reshape(fused_shape)
        score = self.channel_score(fused).squeeze(-1).masked_fill(~channel_mask.unsqueeze(-1), -1e4)
        value = (fused * torch.softmax(score, dim=1).unsqueeze(-1)).sum(1)
        token_mask = F.interpolate(time_mask.float().unsqueeze(1), size=self.token_steps, mode="nearest").squeeze(1).bool()
        stem = self.stem(value + self.position, token_mask)
        local = self.local_head(stem, token_mask)
        pooled = masked_mean(stem, token_mask)
        global_embedding = F.normalize(self.global_head(pooled), dim=-1)
        acoustic_tokens = F.interpolate(local.transpose(1, 2), size=self.target_frames, mode="linear", align_corners=False).transpose(1, 2)
        mfcc = self.mfcc_head(acoustic_tokens).transpose(1, 2)
        return JointState(local, global_embedding, mfcc, self.phoneme_head(pooled), token_mask)


@dataclass
class RendererState:
    log_mel: torch.Tensor
    rms: torch.Tensor
    activity_logits: torch.Tensor


class AudioMFCCRenderer(nn.Module):
    """Audio-only MFCC -> acoustic renderer used as a separately gated oracle."""

    def __init__(self, hidden_dimension: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv1d(39, hidden_dimension, 5, padding=2), nn.GELU()]
        for _ in range(max(1, int(layers))):
            blocks.extend([
                nn.Conv1d(hidden_dimension, hidden_dimension, 5, padding=2, groups=hidden_dimension),
                nn.Conv1d(hidden_dimension, hidden_dimension, 1), nn.GELU(), nn.Dropout(dropout),
            ])
        self.backbone = nn.Sequential(*blocks)
        self.log_mel_head = nn.Conv1d(hidden_dimension, 80, 1)
        self.rms_head = nn.Sequential(nn.Conv1d(hidden_dimension, 1, 1), nn.Softplus())
        self.activity_head = nn.Conv1d(hidden_dimension, 1, 1)

    def forward(self, mfcc: torch.Tensor) -> RendererState:
        if mfcc.ndim != 3 or mfcc.shape[1] != 39:
            raise ValueError("renderer input must be [B,39,T]")
        state = self.backbone(mfcc)
        return RendererState(self.log_mel_head(state), self.rms_head(state).squeeze(1),
                             self.activity_head(state).squeeze(1))
