"""Sequential-RVQ renderer and time-aligned content utilities for v3 repair.

This module deliberately has no dependency on the bridge-v2 continuous latent
renderer.  It preserves the exact HuggingFace EnCodec normalization contract,
uses *hard* sequential RVQ targets, and exposes the two token objectives used
by the repaired audio/EEG teachers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cp_temporal import ConformerStack, DurationAwareDecoder
from .encodec_content import _resample

SCHEMA = "openvoice-v3-mfcc-encodec-rvq-repair-v3"
PREPARATION_SCHEMA = "openvoice-v3-mfcc-encodec-rvq-repair-preparation-v3-161"


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def token_mask(mask: torch.Tensor, steps: int = 96) -> torch.Tensor:
    value = F.interpolate(mask.float().unsqueeze(1), size=steps, mode="nearest").squeeze(1)
    return value.bool()


def temporal_delta(value: torch.Tensor) -> torch.Tensor:
    return value[..., 1:] - value[..., :-1]


def masked_l1(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    error = (left - right).abs()
    if mask is None:
        return error.mean()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(1)
    return (error * mask.to(error.dtype)).sum() / (mask.sum().clamp_min(1) * error.shape[1])


def diagonal_band_infonce(
    left: torch.Tensor, right: torch.Tensor, left_mask: torch.Tensor, right_mask: torch.Tensor,
    *, band: int = 2, temperature: float = 0.07, labels: Iterable[str] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Token InfoNCE with normalized-time diagonal positives, never all TxS mean.

    Same-label *other* trials are masked from denominator negatives.  This is
    important because their phonetic trajectories are allowed to be similar;
    only the trial diagonal is a positive in this objective.
    """
    left, right = F.normalize(left, dim=-1), F.normalize(right, dim=-1)
    batch, steps, _ = left.shape
    if right.shape[:2] != (batch, steps):
        raise ValueError("diagonal token alignment requires matching [B,T] grids")
    scores = torch.einsum("btd,csd->bcts", left, right) / temperature
    positions = torch.arange(steps, device=left.device)
    diagonal = (positions[:, None] - positions[None, :]).abs() <= int(band)
    valid = left_mask[:, None, :, None] & right_mask[None, :, None, :]
    identity = torch.eye(batch, device=left.device, dtype=torch.bool)
    if labels is not None:
        values = list(map(str, labels))
        same = torch.as_tensor([[a == b for b in values] for a in values], device=left.device)
        # Other same-label trials are absent from token-level negatives.
        valid = valid & ~(same & ~identity)[:, :, None, None]
    positive = valid & identity[:, :, None, None] & diagonal[None, None]
    # Each valid left token is a query. Its denominator contains all allowed
    # trial/time candidates; its numerator contains only the paired trial's
    # normalized-time ±band. This is a real token objective, not a TxS mean.
    logits = scores.permute(0, 2, 1, 3).reshape(batch, steps, batch * steps)
    valid_flat = valid.permute(0, 2, 1, 3).reshape(batch, steps, batch * steps)
    positive_flat = positive.permute(0, 2, 1, 3).reshape(batch, steps, batch * steps)
    log_denom = torch.logsumexp(logits.masked_fill(~valid_flat, -1e4), dim=-1)
    log_num = torch.logsumexp(logits.masked_fill(~positive_flat, -1e4), dim=-1)
    active = left_mask & positive_flat.any(-1)
    log_mass = log_num - log_denom
    loss = -(log_mass * active.to(log_mass.dtype)).sum() / active.sum().clamp_min(1)
    diagonal_mass = (log_mass.exp() * active.to(log_mass.dtype)).sum() / active.sum().clamp_min(1)
    return loss, {"diagonal_mass": diagonal_mass, "pair_logits": scores.mean((2, 3))}


def soft_dtw_token_clip(
    left: torch.Tensor, right: torch.Tensor, left_mask: torch.Tensor, right_mask: torch.Tensor,
    *, window_fraction: float = .20, temperature: float = .07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized monotonic entropic OT within a Sakoe--Chiba band.

    The public name is retained for checkpoint/config compatibility.  The
    implementation is a monotonic-OT alternative allowed by the protocol and
    avoids thousands of scalar dynamic-programming dispatches on Apple MPS.
    """
    left, right = F.normalize(left, dim=-1), F.normalize(right, dim=-1)
    batch, steps, _ = left.shape
    radius = max(1, int(round(steps * float(window_fraction))))
    grid = torch.arange(steps, device=left.device)
    allowed = (grid[:, None] - grid[None, :]).abs() <= radius
    valid = allowed.unsqueeze(0) & left_mask[:, :, None] & right_mask[:, None, :]
    cost = 1.0 - torch.einsum("btd,bsd->bts", left, right)
    kernel = torch.exp(-cost / max(float(temperature), 1e-4)) * valid.to(cost.dtype)
    row_mass = left_mask.to(cost.dtype); row_mass = row_mass / row_mass.sum(1, keepdim=True).clamp_min(1)
    col_mass = right_mask.to(cost.dtype); col_mass = col_mass / col_mass.sum(1, keepdim=True).clamp_min(1)
    u = torch.ones_like(row_mass); v = torch.ones_like(col_mass)
    for _ in range(8):
        u = row_mass / torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        v = col_mass / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    transport = u[:, :, None] * kernel * v[:, None, :]
    transport = transport / transport.sum((1, 2), keepdim=True).clamp_min(1e-8)
    normalized = grid.to(cost.dtype) / max(steps - 1, 1)
    monotonic_distance = (normalized[:, None] - normalized[None, :]).abs().unsqueeze(0)
    loss = ((cost + 0.10 * monotonic_distance) * transport).sum((1, 2)).mean()
    diagonal_mass = torch.diagonal(transport, dim1=1, dim2=2).sum(1).mean()
    return loss, diagonal_mass


class FrozenEnCodecRVQ(nn.Module):
    """Exact frozen EnCodec encoder/RVQ/decoder contract with audio scales."""

    def __init__(self, root: Path, *, device: torch.device, bandwidth: float = 6.0):
        super().__init__()
        from transformers import EncodecModel
        self.model = EncodecModel.from_pretrained(str(root), local_files_only=True).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.bandwidth = float(bandwidth)
        self.sample_rate = int(self.model.config.sampling_rate)
        self.latent_dimension = int(self.model.config.codebook_dim)
        self.codebooks = int(self.model.quantizer.get_num_quantizers_for_bandwidth(self.bandwidth))
        self.normalize = bool(self.model.config.normalize)

    @torch.no_grad()
    def encode_16k(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio = _resample(waveform.to(next(self.model.parameters()).device), 16000, self.sample_rate).unsqueeze(1)
        mask = torch.ones_like(audio, dtype=torch.bool)
        result = self.model.encode(audio, padding_mask=mask, bandwidth=self.bandwidth)
        codes = result.audio_codes
        if codes.ndim == 4:
            # 24 kHz EnCodec has a single chunk; retain [B,Q,T].
            codes = codes[0]
        scales = result.audio_scales
        if scales and scales[0] is not None:
            scale = scales[0].reshape(len(audio), 1).float()
        else:
            scale = torch.ones((len(audio), 1), device=audio.device, dtype=audio.dtype)
        return codes.long(), torch.ones((len(audio), codes.shape[-1]), device=audio.device, dtype=torch.bool), scale

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor, scales: torch.Tensor, *, target_samples: int | None = None) -> torch.Tensor:
        # HF ``EncodecModel.decode`` expects [frames, batch, codebooks, steps].
        # ``codes`` is already [batch, codebooks, steps]; transposing here
        # would silently reinterpret codebooks as batch elements.
        encoded = codes.long().unsqueeze(0)
        actual_scales = [scales.reshape(len(codes), 1)] if self.normalize else [None]
        output = self.model.decode(encoded, actual_scales)[0][:, 0]
        output = _resample(output, self.sample_rate, 16000)
        return output[..., :target_samples] if target_samples is not None else output

    @torch.no_grad()
    def code_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.quantizer.decode(codes.long().transpose(0, 1))


class SequentialRVQBridge(nn.Module):
    """MFCC/P/V → conditional residual codebook logits q0…q7.

    Every q_i consumes the preceding (teacher-forced or sampled) codebook
    embeddings.  There is no independent parallel-codebook prediction path.
    """

    def __init__(self, *, voice_dimension: int = 192, dimension: int = 256, codebooks: int = 8, vocabulary: int = 1024):
        super().__init__()
        self.codebooks, self.vocabulary = int(codebooks), int(vocabulary)
        self.input = nn.Conv1d(39 + 3 + 2, dimension, 5, padding=2)
        self.voice = nn.Linear(voice_dimension, dimension * 2)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(dimension, dimension, 3, padding=2 ** (i % 4), dilation=2 ** (i % 4)), nn.GELU(), nn.Conv1d(dimension, dimension, 1)) for i in range(8)])
        self.previous = nn.ModuleList([nn.Embedding(vocabulary, dimension) for _ in range(codebooks)])
        self.heads = nn.ModuleList([nn.Conv1d(dimension, vocabulary, 1) for _ in range(codebooks)])

    def forward(self, content: torch.Tensor, p_base: torch.Tensor, voice: torch.Tensor, duration: torch.Tensor,
                *, targets: torch.Tensor | None = None, teacher_forcing: float = 0.0) -> torch.Tensor:
        content = F.interpolate(content, size=192, mode="linear", align_corners=False)
        prosody = F.interpolate(p_base.transpose(1, 2), size=192, mode="linear", align_corners=False)
        pos = torch.linspace(0, 1, 192, device=content.device, dtype=content.dtype).view(1, 1, -1).expand(len(content), -1, -1)
        duration_mask = (pos <= duration.view(-1, 1, 1).clamp_min(1 / 192)).to(content.dtype)
        hidden = self.input(torch.cat((content, prosody, pos, duration_mask), 1))
        scale, shift = self.voice(voice).chunk(2, -1)
        hidden = hidden * (1 + .1 * torch.tanh(scale).unsqueeze(-1)) + .1 * shift.unsqueeze(-1)
        logits = []
        for level, (block, embed, head) in enumerate(zip(self.blocks, self.previous, self.heads)):
            hidden = hidden + block(hidden)
            score = head(hidden)
            logits.append(score)
            predicted = score.argmax(1)
            if targets is not None and teacher_forcing > 0:
                use_teacher = torch.rand((len(content), 1), device=content.device) < teacher_forcing
                previous = torch.where(use_teacher, targets[:, level].long(), predicted)
            else:
                previous = predicted
            hidden = hidden + embed(previous).transpose(1, 2)
        return torch.stack(logits, 1)

    @staticmethod
    def hard_codes(logits: torch.Tensor, *, code_mask: torch.Tensor | None = None,
                   duration_fraction: torch.Tensor | None = None) -> torch.Tensor:
        codes = logits.argmax(2)
        if code_mask is None and duration_fraction is not None:
            steps = codes.shape[-1]
            position = torch.arange(steps, device=codes.device).view(1, -1)
            valid = torch.ceil(duration_fraction.clamp(1 / steps, 1) * steps).long().view(-1, 1)
            code_mask = position < valid
        if code_mask is not None:
            codes = codes.masked_fill(~code_mask.bool().unsqueeze(1), 0)
        return codes

    @staticmethod
    def sample_residual_codes(logits: torch.Tensor, *, temperature: float = 1.0,
                              code_mask: torch.Tensor | None = None,
                              generator: torch.Generator | None = None) -> torch.Tensor:
        """Keep q0–q3 deterministic; sample only residual q4–q7 for demos."""
        codes = logits.argmax(2)
        for level in range(4, logits.shape[1]):
            probabilities = torch.softmax(logits[:, level].transpose(1, 2) / max(float(temperature), 1e-4), dim=-1)
            sampled = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator)
            codes[:, level] = sampled.reshape(probabilities.shape[:2])
        if code_mask is not None:
            codes = codes.masked_fill(~code_mask.bool().unsqueeze(1), 0)
        return codes


@dataclass
class CState:
    local: torch.Tensor
    global_embedding: torch.Tensor
    token_mask: torch.Tensor


class AudioCTeacher(nn.Module):
    """Coarse EnCodec q0/q1 + HuBERT frame fusion, C-local/global separated."""

    def __init__(self, *, dimension: int = 256, heads: int = 8, layers: int = 4, dropout: float = .1):
        super().__init__()
        self.q0 = nn.Embedding(1024, 128); self.q1 = nn.Embedding(1024, 128)
        self.hubert = nn.Linear(768, 256); self.fuse = nn.Linear(512, dimension)
        self.position = nn.Parameter(torch.randn(1, 96, dimension) * .02)
        self.local = ConformerStack(dimension, heads, layers, dropout)
        self.global_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 768))
        self.hubert_token = nn.Linear(dimension, 768)

    def forward(self, codes: torch.Tensor, code_mask: torch.Tensor, hubert: torch.Tensor, hubert_mask: torch.Tensor) -> CState:
        coarse = torch.cat((self.q0(codes[:, 0]), self.q1(codes[:, 1])), -1).transpose(1, 2)
        coarse = F.interpolate(coarse, size=96, mode="linear", align_corners=False).transpose(1, 2)
        features = F.interpolate(hubert.transpose(1, 2), size=96, mode="linear", align_corners=False).transpose(1, 2)
        mask = token_mask(code_mask, 96) & token_mask(hubert_mask, 96)
        local = self.local(self.fuse(torch.cat((coarse, self.hubert(features)), dim=-1)) + self.position, mask)
        return CState(local=local, global_embedding=F.normalize(self.global_head(masked_mean(local, mask)), dim=-1), token_mask=mask)


class ContentMFCCDecoder(nn.Module):
    def __init__(self, *, dimension: int = 256, heads: int = 8, layers: int = 4, dropout: float = .1):
        super().__init__()
        self.decoder = DurationAwareDecoder(dimension=dimension, heads=heads, layers=layers, output_dimension=39, output_frames=161, input_frames=96, dropout=dropout)

    def forward(self, local: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        duration = torch.ones(len(local), device=local.device, dtype=local.dtype)
        return self.decoder(local, mask, duration)


class DirectEEGMFCC(nn.Module):
    """Direct sanity head. No audio, label, P, voice, or renderer input."""
    def __init__(self, *, dimension: int = 256, heads: int = 8, layers: int = 6, dropout: float = .1):
        super().__init__()
        branch = dimension // 3
        self.temporal = nn.ModuleList([
            nn.Conv1d(1, branch, kernel, padding=kernel // 2) for kernel in (9, 33, 65)
        ])
        self.channel = nn.Sequential(
            nn.Conv1d(branch * 3, dimension, 1), nn.GELU(),
            nn.Conv1d(dimension, dimension, 1), nn.GELU(),
        )
        self.coordinate = nn.Sequential(
            nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.channel_score = nn.Sequential(
            nn.Linear(dimension, dimension // 2), nn.GELU(), nn.Linear(dimension // 2, 1)
        )
        self.position = nn.Parameter(torch.randn(1, 96, dimension) * .02)
        self.encoder = ConformerStack(dimension, heads, layers, dropout)
        self.decoder = ContentMFCCDecoder(dimension=dimension, heads=heads, layers=4, dropout=dropout)

    def tokens(self, eeg: torch.Tensor, channel_xyz: torch.Tensor,
               channel_mask: torch.Tensor, time_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, samples = eeg.shape
        signal = eeg.reshape(batch * channels, 1, samples)
        hidden = self.channel(torch.cat([F.gelu(layer(signal)) for layer in self.temporal], dim=1))
        hidden = F.interpolate(hidden, size=96, mode="linear", align_corners=False).transpose(1, 2).reshape(batch, channels, 96, -1)
        hidden = hidden + self.coordinate(channel_xyz).unsqueeze(2)
        score = self.channel_score(hidden).squeeze(-1).masked_fill(~channel_mask.unsqueeze(-1), -1e4)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        hidden = (hidden * weight).sum(1)
        mask = token_mask(time_mask, 96)
        return self.encoder(hidden + self.position, mask), mask

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor,
                channel_mask: torch.Tensor, time_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local, mask = self.tokens(eeg, channel_xyz, channel_mask, time_mask)
        mfcc, _ = self.decoder(local, mask)
        return mfcc, local, mask
