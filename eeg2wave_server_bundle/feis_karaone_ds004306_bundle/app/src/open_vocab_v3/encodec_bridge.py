"""Frozen-EnCodec continuous-latent renderer for the v3 C-first bridge.

The renderer intentionally predicts a *continuous* EnCodec latent and lets
the original residual vector quantizer choose its sequential codebooks.  It
therefore never combines eight independently predicted code streams, which
would violate EnCodec's residual-quantization contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cp_temporal import ConformerStack, DurationAwareDecoder, gradient_scale, reverse_gradient
from .encodec_content import _resample


SCHEMA = "openvoice-v3-mfcc-encodec-bridge-v2"
PREPARATION_SCHEMA = "openvoice-v3-mfcc-encodec-bridge-preparation-v2-161"


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def _token_mask(mask: torch.Tensor, steps: int = 96) -> torch.Tensor:
    pooled = F.max_pool1d(mask.float().unsqueeze(1), 5, stride=2, padding=2).squeeze(1).bool()
    if pooled.shape[1] != steps:
        raise ValueError(f"expected {steps} code tokens, got {pooled.shape[1]}")
    return pooled


@dataclass
class CState:
    local: torch.Tensor
    global_embedding: torch.Tensor
    token_mask: torch.Tensor
    speaker_logits: torch.Tensor


class AudioCEncoder(nn.Module):
    """Frozen-code IDs → separate local and global content representations."""

    def __init__(self, *, codebooks: int = 8, vocabulary: int = 1024,
                 embedding_dimension: int = 128, dimension: int = 256,
                 heads: int = 8, stem_layers: int = 6, local_layers: int = 2,
                 dropout: float = 0.1, speakers: int = 14,
                 global_gradient_scale: float = 0.25):
        super().__init__()
        self.codebooks = int(codebooks)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocabulary, embedding_dimension) for _ in range(codebooks)]
        )
        joined = codebooks * embedding_dimension
        self.gate = nn.Sequential(nn.Linear(joined, codebooks), nn.Sigmoid())
        self.input_projection = nn.Linear(joined, dimension)
        self.downsample = nn.Conv1d(dimension, dimension, 5, stride=2, padding=2)
        self.position = nn.Parameter(torch.empty(1, 96, dimension))
        nn.init.normal_(self.position, std=0.02)
        self.stem = ConformerStack(dimension, heads, stem_layers, dropout)
        self.local_head = ConformerStack(dimension, heads, local_layers, dropout)
        self.global_head = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.hubert_token = nn.Linear(dimension, 768)
        self.hubert_global = nn.Linear(dimension, 768)
        self.speaker_head = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, speakers))
        self.global_gradient_scale = float(global_gradient_scale)

    def forward(self, codes: torch.Tensor, code_mask: torch.Tensor) -> CState:
        if codes.ndim != 3 or codes.shape[1] != self.codebooks:
            raise ValueError(f"codes must be [B,{self.codebooks},192]")
        if code_mask.shape != (codes.shape[0], codes.shape[2]):
            raise ValueError("code mask must be [B,192]")
        encoded = [layer(codes[:, index].long()) for index, layer in enumerate(self.embeddings)]
        joined = torch.cat(encoded, dim=-1)
        gate = self.gate(joined).unsqueeze(-1)
        joined = torch.cat([item * gate[:, :, index] for index, item in enumerate(encoded)], dim=-1)
        hidden = self.downsample(self.input_projection(joined).transpose(1, 2)).transpose(1, 2)
        mask = _token_mask(code_mask)
        hidden = self.stem(hidden + self.position, mask)
        local = self.local_head(hidden, mask)
        pooled = masked_mean(hidden, mask)
        # Classification/adversarial pressure is only allowed through this
        # global projection.  C_local never receives its direct gradient.
        global_source = gradient_scale(pooled, self.global_gradient_scale)
        global_embedding = F.normalize(self.global_head(global_source), dim=-1)
        speaker_logits = self.speaker_head(reverse_gradient(global_embedding))
        return CState(local=local, global_embedding=global_embedding, token_mask=mask, speaker_logits=speaker_logits)


class SharedContentMFCCDecoder(nn.Module):
    """The only shared C → MFCC decoder; c0 and P are absent by design."""

    def __init__(self, *, dimension: int = 256, heads: int = 8, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.decoder = DurationAwareDecoder(
            dimension=dimension, heads=heads, layers=layers, output_dimension=39,
            output_frames=161, input_frames=96, dropout=dropout,
        )

    def forward(self, local: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # C is normalized-time VAD-active content, hence all 161 query frames
        # are valid.  Duration belongs to P and must not erase C positions.
        duration = torch.ones(len(local), dtype=local.dtype, device=local.device)
        return self.decoder(local, token_mask, duration)


class EEGCEncoder(nn.Module):
    """EEG → C only.  Labels, P, voice, and target audio are not inputs."""

    def __init__(self, *, dimension: int = 256, heads: int = 8, layers: int = 6,
                 local_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        branch = dimension // 3
        self.temporal = nn.ModuleList([
            nn.Conv1d(1, branch, kernel, padding=kernel // 2) for kernel in (9, 33, 65)
        ])
        self.temporal_projection = nn.Sequential(
            nn.Conv1d(branch * 3, dimension, 1), nn.GELU(), nn.Conv1d(dimension, dimension, 1), nn.GELU()
        )
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.channel_score = nn.Sequential(nn.Linear(dimension, dimension // 2), nn.GELU(), nn.Linear(dimension // 2, 1))
        self.position = nn.Parameter(torch.empty(1, 96, dimension)); nn.init.normal_(self.position, std=0.02)
        self.stem = ConformerStack(dimension, heads, layers, dropout)
        self.local_head = ConformerStack(dimension, heads, local_layers, dropout)
        self.global_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor,
                channel_mask: torch.Tensor, time_mask: torch.Tensor) -> CState:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("EEG/coordinate shapes must be [B,C,T] and [B,C,3]")
        batch, channels, samples = eeg.shape
        signal = eeg.reshape(batch * channels, 1, samples)
        multi = torch.cat([F.gelu(layer(signal)) for layer in self.temporal], dim=1)
        hidden = self.temporal_projection(multi)
        # PyTorch/MPS does not implement adaptive_avg_pool1d when the input
        # length is not divisible by 96 (KaraOne EEG commonly is not).  Linear
        # normalized-time resampling preserves the public 96-token contract,
        # remains differentiable, and is supported identically on CPU/MPS.
        hidden = F.interpolate(hidden, size=96, mode="linear", align_corners=False)
        hidden = hidden.transpose(1, 2).reshape(batch, channels, 96, -1)
        hidden = hidden + self.coordinate(channel_xyz).unsqueeze(2)
        score = self.channel_score(hidden).squeeze(-1).masked_fill(~channel_mask.unsqueeze(-1), -1e4)
        weights = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (hidden * weights).sum(1)
        token_mask = F.interpolate(time_mask.float().unsqueeze(1), size=96, mode="nearest").squeeze(1).bool()
        stem = self.stem(pooled + self.position, token_mask)
        local = self.local_head(stem, token_mask)
        global_embedding = F.normalize(self.global_head(masked_mean(stem, token_mask)), dim=-1)
        # An empty, fixed-width speaker tensor keeps CState uniform while
        # making it impossible for EEG-C training to consume a speaker target.
        return CState(local=local, global_embedding=global_embedding, token_mask=token_mask,
                      speaker_logits=torch.empty((batch, 0), device=eeg.device))


class _BridgeBlock(nn.Module):
    def __init__(self, dimension: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, dimension)
        self.depthwise = nn.Conv1d(dimension, dimension, 3, dilation=dilation, padding=dilation, groups=dimension)
        self.pointwise = nn.Conv1d(dimension, dimension, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pointwise(F.gelu(self.depthwise(self.norm(value))))


class ContinuousEnCodecBridge(nn.Module):
    """C/P/V → continuous EnCodec latent, prior to frozen sequential RVQ."""

    def __init__(self, *, latent_dimension: int = 128, voice_dimension: int = 192,
                 dimension: int = 256, blocks: int = 8):
        super().__init__()
        self.input = nn.Conv1d(39 + 3 + 2, dimension, 5, padding=2)
        self.voice = nn.Sequential(nn.Linear(voice_dimension, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, dimension * 2))
        self.blocks = nn.ModuleList([_BridgeBlock(dimension, 2 ** (index % 4)) for index in range(blocks)])
        self.output = nn.Conv1d(dimension, latent_dimension, 1)
        self.hubert_token = nn.Conv1d(latent_dimension, 768, 1)

    def forward(self, content_mfcc: torch.Tensor, p_base: torch.Tensor,
                voice: torch.Tensor, duration_fraction: torch.Tensor) -> torch.Tensor:
        if content_mfcc.shape[1:] != (39, 161) or p_base.shape[1:] != (161, 3):
            raise ValueError("bridge requires C=[B,39,161] and P=[B,161,3]")
        content = F.interpolate(content_mfcc, size=192, mode="linear", align_corners=False)
        prosody = F.interpolate(p_base.transpose(1, 2), size=192, mode="linear", align_corners=False)
        pos = torch.linspace(0.0, 1.0, 192, device=content.device, dtype=content.dtype).view(1, 1, -1)
        duration = duration_fraction.clamp(1.0 / 192.0, 1.0).view(-1, 1, 1)
        timing = torch.cat((pos.expand(len(content), -1, -1), (pos <= duration).to(content.dtype)), dim=1)
        hidden = self.input(torch.cat((content, prosody, timing), dim=1))
        scale, shift = self.voice(voice).chunk(2, dim=-1)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(-1)) + 0.1 * shift.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class FrozenEnCodecRenderer(nn.Module):
    """Frozen EnCodec RVQ/decoder with a differentiable straight-through path."""

    def __init__(self, root: Path, *, device: torch.device, bandwidth: float = 6.0):
        super().__init__()
        from transformers import EncodecModel
        model = EncodecModel.from_pretrained(str(root), local_files_only=True).to(device).eval()
        model.config.normalize = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.bandwidth = float(bandwidth)
        self.sample_rate = int(model.config.sampling_rate)
        self.latent_dimension = int(model.config.codebook_dim)
        self.codebooks = int(model.quantizer.get_num_quantizers_for_bandwidth(self.bandwidth))

    @torch.no_grad()
    def encode_16k(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        audio = _resample(waveform.to(device), 16000, self.sample_rate).unsqueeze(1)
        mask = torch.ones(audio.shape[0], audio.shape[-1], dtype=torch.bool, device=audio.device)
        result = self.model.encode(audio, padding_mask=mask, bandwidth=self.bandwidth)
        codes = result.audio_codes
        if codes.ndim == 4:
            codes = codes[:, 0]
        return codes.long(), torch.ones(codes.shape[0], codes.shape[-1], dtype=torch.bool, device=audio.device)

    @torch.no_grad()
    def target_latent(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.quantizer.decode(codes.long().transpose(0, 1))

    def quantize_st(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code_qbt = self.model.quantizer.encode(latent, bandwidth=self.bandwidth)
        quantized = self.model.quantizer.decode(code_qbt)
        straight_through = latent + (quantized - latent).detach()
        return code_qbt.transpose(0, 1).long(), quantized, straight_through

    def render_st(self, latent: torch.Tensor) -> torch.Tensor:
        waveform = self.model.decoder(latent)[:, 0]
        return _resample(waveform, self.sample_rate, 16000)

    @torch.no_grad()
    def render_codes(self, codes: torch.Tensor, *, target_samples: int | None = None) -> torch.Tensor:
        latent = self.model.quantizer.decode(codes.long().transpose(0, 1))
        waveform = self.render_st(latent)
        return waveform[..., :target_samples] if target_samples is not None else waveform


def temporal_delta(value: torch.Tensor) -> torch.Tensor:
    return value[..., 1:] - value[..., :-1]


def variance_covariance_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.transpose(1, 2).reshape(-1, prediction.shape[1])
    truth = target.transpose(1, 2).reshape(-1, target.shape[1])
    pred = pred - pred.mean(0, keepdim=True); truth = truth - truth.mean(0, keepdim=True)
    scale = F.l1_loss(pred.std(0), truth.std(0))
    covariance = (pred.T @ pred) / max(len(pred) - 1, 1)
    diagonal = torch.diag(torch.diag(covariance))
    return scale + 0.01 * (covariance - diagonal).square().mean()


def masked_token_infonce(left: torch.Tensor, right: torch.Tensor, left_mask: torch.Tensor,
                         right_mask: torch.Tensor, labels: Iterable[str] | None = None,
                         temperature: float = 0.07) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonal token contrastive objective with same-label negatives masked."""
    left = F.normalize(left, dim=-1); right = F.normalize(right, dim=-1)
    scores = torch.einsum("btd,csd->bcts", left, right)
    valid = left_mask[:, None, :, None] & right_mask[None, :, None, :]
    scores = scores.masked_fill(~valid, 0.0)
    denom = valid.to(scores.dtype).sum((2, 3)).clamp_min(1.0)
    pair = scores.sum((2, 3)) / denom
    if labels is not None:
        labels = list(map(str, labels))
        same = torch.as_tensor([[a == b for b in labels] for a in labels], device=pair.device, dtype=torch.bool)
        pair = pair.masked_fill(same & ~torch.eye(len(labels), device=pair.device, dtype=torch.bool), -1e4)
    logits = pair / float(temperature)
    target = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)), pair


def multiresolution_stft_loss(prediction: torch.Tensor, target: torch.Tensor,
                              sample_mask: torch.Tensor) -> torch.Tensor:
    values = []
    for size in (256, 512, 1024):
        if prediction.shape[-1] < size:
            continue
        if prediction.device.type == "mps":
            # MPS builds without FFT kernels have historically aborted rather
            # than raised when backpropagating through ``torch.stft``.  A
            # fixed multi-resolution DCT filterbank is an explicitly recorded
            # differentiable surrogate for those builds; CPU/CUDA retain true
            # MR-STFT.  It preserves gradients and avoids a silent CPU detach.
            time = torch.arange(size, device=prediction.device, dtype=prediction.dtype)
            orders = torch.arange(8, device=prediction.device, dtype=prediction.dtype).view(-1, 1)
            basis = torch.cos(math.pi * (orders + 0.5) * (time.view(1, -1) + 0.5) / size)
            basis = basis / basis.norm(dim=1, keepdim=True).clamp_min(1e-6)
            kernel = basis.unsqueeze(1)
            left = F.conv1d(prediction.unsqueeze(1), kernel, stride=size // 4).abs()
            right = F.conv1d(target.unsqueeze(1), kernel, stride=size // 4).abs()
        else:
            window = torch.hann_window(size, device=prediction.device, dtype=prediction.dtype)
            left = torch.stft(prediction, size, hop_length=size // 4, window=window, return_complex=True).abs()
            right = torch.stft(target, size, hop_length=size // 4, window=window, return_complex=True).abs()
        values.append(F.l1_loss(torch.log1p(left), torch.log1p(right)))
    # The waveform is padded to 2.56 s, so its L1 term explicitly obeys the
    # crop mask even though the spectral terms are deliberately low-weight.
    wave = (torch.abs(prediction - target) * sample_mask.to(prediction.dtype)).sum() / sample_mask.sum().clamp_min(1)
    return wave if not values else wave + torch.stack(values).mean()


def envelope_loss(prediction: torch.Tensor, target: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 320, device=prediction.device, dtype=prediction.dtype) / 320.0
    left = F.conv1d(prediction.abs().unsqueeze(1), kernel, stride=160).squeeze(1)
    right = F.conv1d(target.abs().unsqueeze(1), kernel, stride=160).squeeze(1)
    mask = F.max_pool1d(sample_mask.float().unsqueeze(1), 320, stride=160).squeeze(1).bool()
    return (torch.abs(left - right) * mask.to(left.dtype)).sum() / mask.sum().clamp_min(1)
