"""v3 EnCodec-token content path.

This module deliberately separates the high-fidelity codec representation from
the lower-dimensional content representation used by EEG.  ``AudioContentEncoder``
sees EnCodec IDs; EEG never does and is therefore never asked to predict codec
code IDs, speaker residuals, or waveform detail.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = "openvoice-v3-encodec-clip-mfcc-v1"


def _transformer(dimension: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dimension, nhead=heads, dim_feedforward=dimension * 4,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class AudioContentEncoder(nn.Module):
    """8 independent EnCodec codebook embeddings → 32 content tokens."""
    def __init__(self, *, codebooks: int = 8, vocabulary: int = 1024, dimension: int = 128,
                 tokens: int = 32, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.codebooks, self.vocabulary, self.dimension, self.tokens = codebooks, vocabulary, dimension, tokens
        self.embeddings = nn.ModuleList([nn.Embedding(vocabulary, dimension) for _ in range(codebooks)])
        self.norm = nn.LayerNorm(dimension)
        self.encoder = _transformer(dimension, heads, layers, dropout)
        self.position = nn.Parameter(torch.zeros(1, tokens, dimension))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3 or codes.shape[1] != self.codebooks:
            raise ValueError(f"codes must be [B,{self.codebooks},S]")
        if mask.shape != (codes.shape[0], codes.shape[2]):
            raise ValueError("encodec mask must be [B,S]")
        if int(codes.min()) < 0 or int(codes.max()) >= self.vocabulary:
            raise ValueError("EnCodec IDs outside configured vocabulary")
        hidden = sum(embed(codes[:, index]) for index, embed in enumerate(self.embeddings)) / self.codebooks
        hidden = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.tokens).transpose(1, 2)
        pooled_mask = F.interpolate(mask.float().unsqueeze(1), size=self.tokens, mode="nearest").squeeze(1).bool()
        return self.encoder(self.norm(hidden) + self.position, src_key_padding_mask=~pooled_mask)


class SharedMFCCDecoder(nn.Module):
    """The shared decoder establishes the MFCC target as the alignment output."""
    def __init__(self, *, dimension: int = 128, mfcc_bins: int = 40, token_steps: int = 32, frames: int = 256):
        super().__init__()
        self.mfcc_bins, self.frames, self.token_steps = mfcc_bins, frames, token_steps
        self.net = nn.Sequential(
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, dimension, 5, padding=2), nn.GELU(),
            nn.Conv1d(dimension, mfcc_bins, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.token_steps:
            raise ValueError(f"content tokens must be [B,{self.token_steps},D]")
        values = self.net(tokens.transpose(1, 2))
        values = F.interpolate(values, size=self.frames, mode="linear", align_corners=False)
        # c0 is absolute energy; primary EEG is deliberately not trained to
        # synthesize it.  Keeping it exact also makes this testable in exports.
        return torch.cat((torch.zeros_like(values[:, :1]), values[:, 1:]), dim=1)


class EEGContentEncoder(nn.Module):
    """EEG → 32 content tokens; labels/text/voice are never forward inputs."""
    def __init__(self, *, dimension: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, tokens: int = 32):
        super().__init__()
        self.dimension, self.tokens = dimension, tokens
        self.temporal = nn.Sequential(nn.Conv1d(1, 64, 15, padding=7), nn.GELU(), nn.Conv1d(64, dimension, 9, padding=4), nn.GELU())
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.fusion = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.GELU(), nn.LayerNorm(dimension))
        self.position = nn.Parameter(torch.zeros(1, tokens, dimension)); nn.init.normal_(self.position, std=0.02)
        self.encoder = _transformer(dimension, heads, layers, dropout)
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("eeg must be [B,C,T] and xyz [B,C,3]")
        if channel_mask.shape != eeg.shape[:2] or time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("invalid EEG masks")
        batch, channels, samples = eeg.shape
        temporal = self.temporal(eeg.reshape(batch * channels, 1, samples))
        temporal = F.adaptive_avg_pool1d(temporal, self.tokens).transpose(1, 2).reshape(batch, channels, self.tokens, -1)
        coordinate = self.coordinate(channel_xyz).unsqueeze(2).expand(-1, -1, self.tokens, -1)
        fused = self.fusion(torch.cat((temporal, coordinate), dim=-1))
        weights = channel_mask.to(fused.dtype).view(batch, channels, 1, 1)
        pooled = (fused * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        mask = F.interpolate(time_mask.float().unsqueeze(1), size=self.tokens, mode="nearest").squeeze(1).bool()
        return self.encoder(pooled + self.position, src_key_padding_mask=~mask)


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if int(source_rate) == int(target_rate):
        return waveform
    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, int(source_rate), int(target_rate))
    except Exception:
        return F.interpolate(waveform.unsqueeze(1), size=round(waveform.shape[-1] * target_rate / source_rate), mode="linear", align_corners=False).squeeze(1)


class EnCodecGenerator:
    """Exact 16 kHz ↔ 24 kHz EnCodec contract with crop/mask handling."""
    def __init__(self, root: Path, *, device: torch.device, bandwidth: float = 6.0):
        from transformers import EncodecModel
        self.device, self.bandwidth = device, float(bandwidth)
        # This wrapper is used exclusively as a frozen tokenizer/round-trip
        # evaluator.  Keep it in inference mode so evaluation cannot retain a
        # graph and exhaust memory when a complete audio split is processed.
        self.model = EncodecModel.from_pretrained(str(root), local_files_only=True).to(device).eval()
        self.model.config.normalize = True

    @property
    def sample_rate(self) -> int:
        return int(self.model.config.sampling_rate)

    def encode(self, waveform_16k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wave = _resample(waveform_16k.to(self.device), 16000, self.sample_rate).unsqueeze(1)
        mask = torch.ones(wave.shape[0], wave.shape[-1], dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            encoded = self.model.encode(wave, padding_mask=mask, bandwidth=self.bandwidth)
        codes = encoded.audio_codes
        if codes.ndim == 4:  # [B,frames,Q,S], normal files have one frame.
            codes = codes[:, 0]
        if codes.shape[1] != 8:
            raise RuntimeError(f"expected 8 EnCodec codebooks at 6 kbps, got {codes.shape[1]}")
        code_mask = torch.ones(codes.shape[0], codes.shape[-1], dtype=torch.bool, device=codes.device)
        return codes.to(torch.int16), code_mask

    def decode(self, codes: torch.Tensor, *, target_samples_16k: int | None = None) -> torch.Tensor:
        # EnCodec decoder accepts codes with the frame axis restored.
        with torch.inference_mode():
            values = self.model.decode(codes.to(self.device).long().unsqueeze(1), audio_scales=[None])
            waveform = _resample(values.audio_values[:, 0], self.sample_rate, 16000)
        return waveform[..., :target_samples_16k] if target_samples_16k is not None else waveform


def checkpoint_payload(*, model: nn.Module, schema: str, **extra: Any) -> dict[str, Any]:
    return {"schema_version": schema, "state_dict": model.state_dict(), "extra": extra}
