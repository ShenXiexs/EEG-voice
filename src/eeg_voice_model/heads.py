"""Downstream heads for EEG probes and voice retrieval."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import info_nce_loss


def pool_tokens(z: torch.Tensor) -> torch.Tensor:
    """Pool `[B, Q, W, D]` tokens to `[B, D]`."""
    if z.ndim != 4:
        raise ValueError(f"Expected z with shape [B,Q,W,D], got {tuple(z.shape)}")
    return z.mean(dim=(1, 2))


class ProbeHead(nn.Module):
    """Classification probe for ds006104 labels."""

    def __init__(self, dim: int, num_classes: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        logits = self.net(pool_tokens(z))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels.long())
        return out


class AudioContrastiveHead(nn.Module):
    """InfoNCE alignment between EEG token embeddings and audio stream embeddings."""

    def __init__(self, eeg_dim: int, audio_dim: int, proj_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.eeg_proj = nn.Sequential(nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.audio_proj = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def forward(self, z: torch.Tensor, audio_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        eeg = self.eeg_proj(pool_tokens(z))
        audio = self.audio_proj(audio_embedding.float())
        loss, logits = info_nce_loss(eeg, audio, temperature=torch.exp(self.log_temperature))
        return {"loss": loss, "logits": logits, "eeg_embedding": eeg, "audio_embedding": audio}


class VoiceAttributeHead(nn.Module):
    """Small head for pitch/intensity/timbre attribute probes."""

    def __init__(self, dim: int, output_dim: int, task: str = "classification", dropout: float = 0.1):
        super().__init__()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, output_dim),
        )

    def forward(self, z: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pred = self.net(pool_tokens(z))
        out = {"pred": pred}
        if target is not None:
            if self.task == "classification":
                out["loss"] = F.cross_entropy(pred, target.long())
            else:
                out["loss"] = F.smooth_l1_loss(pred, target.float())
        return out
