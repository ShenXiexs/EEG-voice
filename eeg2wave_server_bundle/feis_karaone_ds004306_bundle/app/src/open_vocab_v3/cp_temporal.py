"""Large C/P-temporal v3 models and losses.

The module is intentionally independent of the abandoned 32-token v3 model.
It keeps local acoustic trajectories separate from the global label/CLIP
projection and places every acoustic target on SpeechT5's 161-frame grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = "openvoice-v3-cp-temporal-large-v1"
PREPARATION_SCHEMA = "openvoice-v3-cp-temporal-preparation-v1-161"


class _GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient * ctx.scale, None


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -gradient


def gradient_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientScale.apply(value, float(scale))


def reverse_gradient(value: torch.Tensor) -> torch.Tensor:
    return _ReverseGradient.apply(value)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, dimension: int = 1) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(dimension) / weight.sum(dimension).clamp_min(1.0)


class ConformerBlock(nn.Module):
    """Small batch-first Conformer block that is MPS safe."""

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
        self.output_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value + 0.5 * self.dropout(self.ff1(self.ff1_norm(value)))
        norm = self.attn_norm(value)
        attention, _ = self.attn(norm, norm, norm, key_padding_mask=~mask, need_weights=False)
        value = value + self.dropout(attention)
        conv = self.conv_norm(value).transpose(1, 2)
        left, gate = self.conv_in(conv).chunk(2, dim=1)
        conv = self.conv_out(F.silu(self.depthwise(left * torch.sigmoid(gate)))).transpose(1, 2)
        value = value + self.dropout(conv)
        value = value + 0.5 * self.dropout(self.ff2(self.ff2_norm(value)))
        return torch.where(mask.unsqueeze(-1), self.output_norm(value), torch.zeros_like(value))


class ConformerStack(nn.Module):
    def __init__(self, dimension: int, heads: int, layers: int, dropout: float, kernel: int = 31):
        super().__init__()
        self.layers = nn.ModuleList([ConformerBlock(dimension, heads, dropout, kernel) for _ in range(layers)])

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, mask)
        return value


class CrossAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.self_norm = nn.LayerNorm(dimension)
        self.self_attn = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dimension)
        self.cross_attn = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dimension)
        self.ff = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension * 4, dimension))

    def forward(self, query: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor,
                monotonic_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = self.self_norm(query)
        query = query + self.self_attn(norm, norm, norm, need_weights=False)[0]
        norm = self.cross_norm(query)
        batch = len(query)
        # Fold padding into the additive monotonic mask. This keeps attn_mask
        # and padding semantics in one floating-point tensor and avoids the
        # deprecated mixed float/bool MHA mask path on PyTorch 2.8/MPS.
        bias = monotonic_bias.masked_fill(~memory_mask.unsqueeze(1), float("-inf"))
        bias = bias.repeat_interleave(self.heads, dim=0)
        cross, weights = self.cross_attn(
            norm, memory, memory, attn_mask=bias,
            need_weights=True, average_attn_weights=False,
        )
        query = query + cross
        query = query + self.ff(self.ff_norm(query))
        return query, weights.mean(1)


class DurationAwareDecoder(nn.Module):
    """96 local tokens -> 161 explicitly timed acoustic frames."""

    def __init__(self, dimension: int = 256, heads: int = 8, layers: int = 4,
                 output_dimension: int = 39, output_frames: int = 161,
                 input_frames: int = 96, dropout: float = 0.1):
        super().__init__()
        self.output_frames = int(output_frames)
        self.input_frames = int(input_frames)
        self.time_projection = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.query = nn.Parameter(torch.zeros(1, output_frames, dimension))
        nn.init.normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(dimension, heads, dropout) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, output_dimension))

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor,
                duration_fraction: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = len(memory)
        qpos = torch.linspace(0, 1, self.output_frames, device=memory.device, dtype=memory.dtype)
        duration = duration_fraction.clamp(1.0 / self.output_frames, 1.0).view(batch, 1)
        valid = qpos.view(1, -1) <= duration
        normalized = (qpos.view(1, -1) / duration).clamp(0, 1)
        time_features = torch.stack((normalized, torch.sin(math.pi * normalized), torch.cos(math.pi * normalized)), dim=-1)
        query = self.query.expand(batch, -1, -1) + self.time_projection(time_features)
        kpos = torch.linspace(0, 1, memory.shape[1], device=memory.device, dtype=memory.dtype)
        bias = -6.0 * torch.abs(normalized.unsqueeze(-1) - kpos.view(1, 1, -1))
        all_weights = []
        for block in self.blocks:
            query, weights = block(query, memory, memory_mask, bias)
            all_weights.append(weights)
        values = self.output(query).transpose(1, 2)
        values = torch.where(valid.unsqueeze(1), values, torch.zeros_like(values))
        attention = torch.stack(all_weights).mean(0)
        coverage = (attention.sum(1) > (0.25 / max(memory.shape[1], 1))).to(values.dtype).mean(1)
        expected = (attention * kpos.view(1, 1, -1)).sum(-1)
        slope = (expected[:, -1] - expected[:, 0]) / max(self.output_frames - 1, 1)
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(-1).mean(1)
        return values, {"attention": attention, "coverage": coverage, "slope": slope, "entropy": entropy, "valid_mask": valid}


@dataclass
class CPState:
    stem: torch.Tensor
    local: torch.Tensor
    global_embedding: torch.Tensor
    p_base: torch.Tensor
    p_plus: torch.Tensor
    duration_fraction: torch.Tensor
    token_mask: torch.Tensor
    acoustic_mask: torch.Tensor


class AudioCPEncoder(nn.Module):
    def __init__(self, codebooks: int = 8, vocabulary: int = 1024, embedding_dimension: int = 128,
                 dimension: int = 256, heads: int = 8, stem_layers: int = 6,
                 branch_layers: int = 2, token_steps: int = 96, acoustic_frames: int = 161,
                 dropout: float = 0.1, global_gradient_scale: float = 0.25):
        super().__init__()
        self.codebooks, self.token_steps = int(codebooks), int(token_steps)
        self.embeddings = nn.ModuleList([nn.Embedding(vocabulary, embedding_dimension) for _ in range(codebooks)])
        joined = codebooks * embedding_dimension
        self.gate = nn.Sequential(nn.Linear(joined, codebooks), nn.Sigmoid())
        self.projection = nn.Linear(joined, dimension)
        self.downsample = nn.Conv1d(dimension, dimension, 5, stride=2, padding=2)
        self.position = nn.Parameter(torch.zeros(1, token_steps, dimension)); nn.init.normal_(self.position, std=0.02)
        self.stem = ConformerStack(dimension, heads, stem_layers, dropout)
        self.local_head = ConformerStack(dimension, heads, branch_layers, dropout)
        self.global_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.p_memory = ConformerStack(dimension, heads, branch_layers, dropout)
        self.p_decoder = DurationAwareDecoder(dimension, heads, 2, 5, acoustic_frames, token_steps, dropout)
        self.duration = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 1))
        self.global_gradient_scale = float(global_gradient_scale)

    def _embed(self, codes: torch.Tensor) -> torch.Tensor:
        embedded = [layer(codes[:, index]) for index, layer in enumerate(self.embeddings)]
        joined = torch.cat(embedded, dim=-1)
        gates = self.gate(joined).unsqueeze(-1)
        gated = torch.cat([value * gates[:, :, index] for index, value in enumerate(embedded)], dim=-1)
        return self.projection(gated)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> CPState:
        if codes.ndim != 3 or codes.shape[1] != self.codebooks or mask.shape != (codes.shape[0], codes.shape[2]):
            raise ValueError("codes/mask must be [B,8,192] and [B,192]")
        value = self.downsample(self._embed(codes).transpose(1, 2)).transpose(1, 2)
        token_mask = F.max_pool1d(mask.float().unsqueeze(1), 5, stride=2, padding=2).squeeze(1).bool()
        if value.shape[1] != self.token_steps:
            raise ValueError(f"stride contract produced {value.shape[1]} tokens, expected {self.token_steps}")
        stem = self.stem(value + self.position, token_mask)
        local = self.local_head(stem, token_mask)
        global_source = gradient_scale(_masked_mean(stem, token_mask), self.global_gradient_scale)
        global_embedding = F.normalize(self.global_head(global_source), dim=-1)
        duration = torch.sigmoid(self.duration(_masked_mean(stem, token_mask)).squeeze(-1))
        p_values, p_diag = self.p_decoder(self.p_memory(stem, token_mask), token_mask, duration)
        p_values = p_values.transpose(1, 2)
        p_base, p_plus = p_values[..., :3], p_values[..., 3:]
        return CPState(stem, local, global_embedding, p_base, p_plus, duration, token_mask, p_diag["valid_mask"])


class ContentMFCCDecoder(nn.Module):
    def __init__(self, dimension: int = 256, heads: int = 8, layers: int = 4,
                 token_steps: int = 96, frames: int = 161, dropout: float = 0.1):
        super().__init__()
        self.decoder = DurationAwareDecoder(dimension, heads, layers, 39, frames, token_steps, dropout)
        self.c0 = nn.Sequential(nn.Conv1d(3, 64, 5, padding=2), nn.GELU(), nn.Conv1d(64, 1, 5, padding=2))

    def forward(self, local: torch.Tensor, token_mask: torch.Tensor, p_base: torch.Tensor,
                duration_fraction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        content, diagnostics = self.decoder(local, token_mask, duration_fraction)
        c0 = self.c0(p_base.transpose(1, 2))
        full = torch.cat((c0, content), dim=1)
        return content, full, diagnostics


class EEGCPEncoder(nn.Module):
    """EEG C/P encoder; no label, text, voice, duration, or audio input."""

    def __init__(self, dimension: int = 256, heads: int = 8, layers: int = 6,
                 token_steps: int = 96, acoustic_frames: int = 161, dropout: float = 0.1):
        super().__init__()
        branch = dimension // 3
        widths = (branch, branch, dimension - 2 * branch)
        self.temporal = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, width, kernel, padding=kernel // 2), nn.GELU(),
                          nn.Conv1d(width, width, 5, padding=2), nn.GELU())
            for width, kernel in zip(widths, (9, 33, 65))
        ])
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.fusion = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.GELU(), nn.LayerNorm(dimension))
        self.channel_score = nn.Linear(dimension, 1)
        self.position = nn.Parameter(torch.zeros(1, token_steps, dimension)); nn.init.normal_(self.position, std=0.02)
        self.stem = ConformerStack(dimension, heads, layers, dropout)
        self.local_head = ConformerStack(dimension, heads, 2, dropout)
        self.global_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.p_decoder = DurationAwareDecoder(dimension, heads, 2, 3, acoustic_frames, token_steps, dropout)
        self.duration = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor,
                time_mask: torch.Tensor) -> CPState:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("eeg must be [B,C,T] and coordinates [B,C,3]")
        batch, channels, samples = eeg.shape
        flat = eeg.reshape(batch * channels, 1, samples)
        temporal = torch.cat([branch(flat) for branch in self.temporal], dim=1)
        temporal = F.interpolate(temporal, size=96, mode="linear", align_corners=False)
        temporal = temporal.transpose(1, 2).reshape(batch, channels, 96, -1)
        coordinates = self.coordinate(channel_xyz).unsqueeze(2).expand(-1, -1, 96, -1)
        fused = self.fusion(torch.cat((temporal, coordinates), dim=-1))
        scores = self.channel_score(fused).squeeze(-1).masked_fill(~channel_mask.unsqueeze(-1), -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        value = (fused * weights).sum(1)
        token_mask = F.interpolate(time_mask.float().unsqueeze(1), size=96, mode="nearest").squeeze(1).bool()
        stem = self.stem(value + self.position, token_mask)
        local = self.local_head(stem, token_mask)
        global_embedding = F.normalize(self.global_head(_masked_mean(stem, token_mask)), dim=-1)
        duration = torch.sigmoid(self.duration(_masked_mean(stem, token_mask)).squeeze(-1))
        p_base, p_diag = self.p_decoder(stem, token_mask, duration)
        p_base = p_base.transpose(1, 2)
        return CPState(stem, local, global_embedding, p_base, p_base.new_zeros(batch, 161, 2), duration, token_mask, p_diag["valid_mask"])


class DilatedResidualBlock(nn.Module):
    def __init__(self, dimension: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(8, dimension)
        self.conv = nn.Conv1d(dimension, dimension * 2, 5, padding=2 * dilation, dilation=dilation)
        self.out = nn.Conv1d(dimension, dimension, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, gate = self.conv(F.silu(self.norm(value))).chunk(2, dim=1)
        return value + self.dropout(self.out(left * torch.sigmoid(gate)))


class DeterministicAcousticBackbone(nn.Module):
    def __init__(self, voice_dimension: int = 192, dimension: int = 256,
                 blocks: int = 8, include_p_plus: bool = True, dropout: float = 0.1):
        super().__init__()
        self.include_p_plus = bool(include_p_plus)
        inputs = 39 + 3 + (2 if include_p_plus else 0)
        self.input = nn.Conv1d(inputs, dimension, 5, padding=2)
        self.blocks = nn.ModuleList([DilatedResidualBlock(dimension, 2 ** (index % 4), dropout) for index in range(blocks)])
        self.voice = nn.Sequential(nn.Linear(voice_dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension * 2))
        self.output = nn.Sequential(nn.GroupNorm(8, dimension), nn.SiLU(), nn.Conv1d(dimension, 80, 1))

    def forward(self, content_mfcc: torch.Tensor, p_base: torch.Tensor, voice: torch.Tensor,
                p_plus: torch.Tensor | None = None) -> torch.Tensor:
        if content_mfcc.shape[1:] != (39, 161) or p_base.shape[1:] != (161, 3):
            raise ValueError("deterministic acoustic contract is MFCC[39,161] plus P[161,3]")
        parts = [content_mfcc, p_base.transpose(1, 2)]
        if self.include_p_plus:
            parts.append(torch.zeros(len(content_mfcc), 2, 161, device=content_mfcc.device, dtype=content_mfcc.dtype)
                         if p_plus is None else p_plus.transpose(1, 2))
        value = self.input(torch.cat(parts, dim=1))
        scale, bias = self.voice(voice).chunk(2, dim=-1)
        value = value * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(-1)) + 0.1 * bias.unsqueeze(-1)
        for block in self.blocks:
            value = block(value)
        return self.output(value).clamp(-10.0, 4.0)


class MelContentTeacher(nn.Module):
    """Differentiable Mel-side student of frozen HuBERT frame content."""

    def __init__(self, dimension: int = 256, hubert_dimension: int = 768, token_steps: int = 96):
        super().__init__()
        self.token_steps = int(token_steps)
        self.mel = nn.Sequential(nn.Conv1d(80, dimension, 7, padding=3), nn.GELU(),
                                 nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU())
        self.hubert = nn.Sequential(nn.LayerNorm(hubert_dimension), nn.Linear(hubert_dimension, dimension))
        self.label_head = nn.Linear(dimension, 10)

    def encode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.mel(mel), size=self.token_steps, mode="linear", align_corners=False).transpose(1, 2)

    def project_hubert(self, hubert: torch.Tensor) -> torch.Tensor:
        value = self.hubert(hubert)
        return F.interpolate(value.transpose(1, 2), size=self.token_steps, mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.encode_mel(mel)
        return token, self.label_head(token.mean(1))


class ResidualCVAE(nn.Module):
    """Hierarchical residual CVAE that cannot overwrite deterministic content."""

    def __init__(self, backbone: DeterministicAcousticBackbone, dimension: int = 256,
                 global_latent: int = 128, local_latent: int = 64,
                 local_steps: int = 64, residual_limit: float = 0.8):
        super().__init__()
        self.backbone = backbone
        self.global_latent, self.local_latent, self.local_steps = global_latent, local_latent, local_steps
        self.residual_limit = float(residual_limit)
        self.condition = nn.Sequential(nn.Conv1d(42, dimension, 5, padding=2), nn.GELU(), nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU())
        self.prior_global = nn.Linear(dimension, global_latent * 2)
        self.prior_local = nn.Conv1d(dimension, local_latent * 2, 3, padding=1)
        self.posterior = nn.Sequential(nn.Conv1d(160, dimension, 5, padding=2), nn.GELU(), nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU())
        self.posterior_global = nn.Linear(dimension, global_latent * 2)
        self.posterior_local = nn.Conv1d(dimension, local_latent * 2, 3, padding=1)
        self.global_projection = nn.Linear(global_latent, dimension)
        self.local_projection = nn.Conv1d(local_latent, dimension, 1)
        self.decoder = nn.Sequential(
            DilatedResidualBlock(dimension, 1, 0.1), DilatedResidualBlock(dimension, 2, 0.1),
            DilatedResidualBlock(dimension, 4, 0.1), nn.GroupNorm(8, dimension), nn.SiLU(),
            nn.Conv1d(dimension, 80, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _split(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value.chunk(2, dim=1)

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor, stochastic: bool) -> torch.Tensor:
        return mean if not stochastic else mean + torch.randn_like(mean) * torch.exp(0.5 * logvar.clamp(-12, 8))

    def distributions(self, content: torch.Tensor, p_base: torch.Tensor, target: torch.Tensor | None = None,
                      deterministic: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        condition = self.condition(torch.cat((content, p_base.transpose(1, 2)), dim=1))
        pooled = condition.mean(-1)
        prior_g_mean, prior_g_logvar = self._split(self.prior_global(pooled))
        local_condition = F.interpolate(condition, size=self.local_steps, mode="linear", align_corners=False)
        prior_l_mean, prior_l_logvar = self._split(self.prior_local(local_condition))
        result = {"condition": condition, "prior_global_mean": prior_g_mean, "prior_global_logvar": prior_g_logvar,
                  "prior_local_mean": prior_l_mean, "prior_local_logvar": prior_l_logvar}
        if target is not None:
            if deterministic is None:
                deterministic = self.backbone(content, p_base, target.new_zeros(len(target), 192))
            posterior = self.posterior(torch.cat((target, deterministic), dim=1))
            post_g_mean, post_g_logvar = self._split(self.posterior_global(posterior.mean(-1)))
            post_l_mean, post_l_logvar = self._split(self.posterior_local(F.interpolate(posterior, size=self.local_steps, mode="linear", align_corners=False)))
            result.update({"posterior_global_mean": post_g_mean, "posterior_global_logvar": post_g_logvar,
                           "posterior_local_mean": post_l_mean, "posterior_local_logvar": post_l_logvar})
        return result

    def residual(self, condition: torch.Tensor, global_value: torch.Tensor, local_value: torch.Tensor) -> torch.Tensor:
        hidden = condition + self.global_projection(global_value).unsqueeze(-1)
        hidden = hidden + F.interpolate(self.local_projection(local_value), size=161, mode="linear", align_corners=False)
        return self.residual_limit * torch.tanh(self.decoder(hidden))

    def forward(self, content: torch.Tensor, p_base: torch.Tensor, voice: torch.Tensor,
                p_plus: torch.Tensor | None = None, target: torch.Tensor | None = None,
                stochastic: bool = False) -> dict[str, torch.Tensor]:
        deterministic = self.backbone(content, p_base, voice, p_plus)
        values = self.distributions(content, p_base, target, deterministic)
        prefix = "posterior" if target is not None else "prior"
        global_value = self._sample(values[f"{prefix}_global_mean"], values[f"{prefix}_global_logvar"], stochastic)
        local_value = self._sample(values[f"{prefix}_local_mean"], values[f"{prefix}_local_logvar"], stochastic)
        residual = self.residual(values["condition"], global_value, local_value)
        values.update({"deterministic": deterministic, "residual": residual,
                       "mel": (deterministic + residual).clamp(-10.0, 4.0)})
        return values


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    value = (prediction - target).abs()
    if mask is None:
        return value.mean()
    weight = mask.to(value.dtype).unsqueeze(1)
    return (value * weight).sum() / (weight.sum() * value.shape[1]).clamp_min(1.0)


def temporal_delta_loss(prediction: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor | None = None) -> torch.Tensor:
    value=F.smooth_l1_loss(prediction[...,1:]-prediction[...,:-1],target[...,1:]-target[...,:-1],reduction="none")
    if mask is None:return value.mean()
    weight=(mask[...,1:]&mask[...,:-1]).to(value.dtype).unsqueeze(1)
    return (value*weight).sum()/(weight.sum()*value.shape[1]).clamp_min(1.0)


def temporal_cosine_loss(prediction: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
    value=1.0-F.cosine_similarity(prediction,target,dim=1)
    if mask is None:return value.mean()
    weight=mask.to(value.dtype);return (value*weight).sum()/weight.sum().clamp_min(1.0)


def envelope_correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left = prediction - prediction.mean(-1, keepdim=True)
    right = target - target.mean(-1, keepdim=True)
    corr = (left * right).sum(-1) / (left.square().sum(-1).sqrt() * right.square().sum(-1).sqrt()).clamp_min(1e-6)
    return (1.0 - corr).mean()


def soft_ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ux, uy = F.avg_pool1d(prediction, 7, 1, 3), F.avg_pool1d(target, 7, 1, 3)
    vx = F.avg_pool1d(prediction.square(), 7, 1, 3) - ux.square()
    vy = F.avg_pool1d(target.square(), 7, 1, 3) - uy.square()
    vxy = F.avg_pool1d(prediction * target, 7, 1, 3) - ux * uy
    c1, c2 = 1e-4, 9e-4
    score = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux.square() + uy.square() + c1) * (vx + vy + c2) + 1e-8)
    return 1.0 - score.mean()


def _multi_positive(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    denominator = torch.logsumexp(logits, dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -1e4), dim=1)
    return (denominator - numerator).mean()


def global_clip_loss(left: torch.Tensor, right: torch.Tensor, labels: Iterable[str], scale: torch.Tensor) -> torch.Tensor:
    logits = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T * scale.exp().clamp(max=100)
    names = [str(value).strip().strip("/").lower() for value in labels]
    positive = torch.tensor([[a == b for b in names] for a in names], dtype=torch.bool, device=logits.device)
    return 0.5 * (_multi_positive(logits, positive) + _multi_positive(logits.T, positive.T))


def _sinkhorn_score(left: torch.Tensor, right: torch.Tensor, iterations: int = 8, time_weight: float = 0.25) -> torch.Tensor:
    similarity = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T
    x = torch.linspace(0, 1, left.shape[0], device=left.device, dtype=left.dtype)
    y = torch.linspace(0, 1, right.shape[0], device=right.device, dtype=right.dtype)
    logits = (similarity - time_weight * torch.abs(x[:, None] - y[None, :])) / 0.07
    transport = torch.exp(logits - logits.max()).clamp_min(1e-8)
    for _ in range(iterations):
        transport = transport / transport.sum(1, keepdim=True).clamp_min(1e-8)
        transport = transport / transport.sum(0, keepdim=True).clamp_min(1e-8)
    transport = transport / transport.sum().clamp_min(1e-8)
    return (transport * similarity).sum()


def local_ot_clip_loss(left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor,
                       left_mask: torch.Tensor | None = None,
                       right_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if left_mask is None:left_mask=torch.ones(left.shape[:2],dtype=torch.bool,device=left.device)
    if right_mask is None:right_mask=torch.ones(right.shape[:2],dtype=torch.bool,device=right.device)
    scores = torch.stack([torch.stack([_sinkhorn_score(one[left_mask[i]],two[right_mask[j]]) for j,two in enumerate(right)]) for i,one in enumerate(left)])
    logits = scores * scale.exp().clamp(max=100)
    target = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)), scores


def attention_regularization(diagnostics: dict[str, torch.Tensor]) -> torch.Tensor:
    coverage = F.relu(0.80 - diagnostics["coverage"]).mean()
    slope = F.relu(1e-4 - diagnostics["slope"]).mean()
    entropy = diagnostics["entropy"]
    entropy_penalty = F.relu(0.5 - entropy).mean() + F.relu(entropy - math.log(96.0) * 0.95).mean()
    return coverage + slope + 0.1 * entropy_penalty


def horizontal_diagnostics(mel: np.ndarray, target: np.ndarray, active: np.ndarray | None = None) -> dict[str, float]:
    prediction = np.asarray(mel, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if active is None:
        active_mask = np.ones((prediction.shape[0], prediction.shape[-1]), dtype=bool)
    else:
        active_mask = np.asarray(active, dtype=bool)
    pred_std=[];target_std=[]
    for index in range(len(prediction)):
        support=active_mask[index]
        if support.sum()<2:support=np.ones_like(support)
        pred_std.append(np.std(prediction[index,:,support],axis=-1).mean());target_std.append(np.std(truth[index,:,support],axis=-1).mean())
    temporal_ratio = float(np.mean(pred_std) / max(np.mean(target_std), 1e-8))
    change_mask=active_mask[:,1:]&active_mask[:,:-1];pred_delta=np.abs(np.diff(prediction,axis=-1)).mean(1);target_delta=np.abs(np.diff(truth,axis=-1)).mean(1)
    pred_change=float(pred_delta[change_mask].mean()) if change_mask.any() else float(pred_delta.mean());target_change=float(target_delta[change_mask].mean()) if change_mask.any() else float(target_delta.mean())
    change_ratio = pred_change / max(target_change, 1e-8)
    centered = prediction.reshape(-1, prediction.shape[-1]) - prediction.reshape(-1, prediction.shape[-1]).mean(-1, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    weight = singular / max(float(singular.sum()), 1e-8)
    rank = float(np.exp(-(weight * np.log(np.maximum(weight, 1e-12))).sum()))
    return {"temporal_std_ratio": temporal_ratio, "spectral_change_ratio": change_ratio,
            "effective_temporal_rank": rank, "collapsed": bool(temporal_ratio < 0.5 or change_ratio < 0.4 or rank < 8.0)}


def parameter_count(*modules: nn.Module) -> int:
    return int(sum(parameter.numel() for module in modules for parameter in module.parameters()))


def deterministic_internal_dev(records: Any) -> tuple[np.ndarray, np.ndarray]:
    fit = (records.roles == "fit") & records.arrays["fit_eligible"].astype(bool)
    dev = fit & records.arrays["fit_internal_dev"].astype(bool)
    return np.flatnonzero(fit & ~dev), np.flatnonzero(dev)
