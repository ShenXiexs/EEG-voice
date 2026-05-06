"""Losses for BrainOmni-style EEG tokenization and voice alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def time_l1_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss in the time domain."""
    return torch.mean(torch.abs(predicted - target))


def pearson_corr(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean Pearson correlation over all non-time dimensions."""
    x = predicted.flatten(0, -2)
    y = target.flatten(0, -2)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    numerator = torch.sum(x * y, dim=-1)
    denominator = torch.sqrt(torch.sum(x * x, dim=-1) * torch.sum(y * y, dim=-1) + eps)
    return torch.mean(numerator / (denominator + eps))


def frequency_domain_loss(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Amplitude and phase losses after a Hamming-windowed rFFT."""
    window = torch.hamming_window(target.shape[-1], device=target.device, dtype=target.dtype)
    predicted_fft = torch.fft.rfft(predicted * window, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(target * window, dim=-1, norm="ortho")
    amp_loss = F.l1_loss(torch.abs(predicted_fft), torch.abs(target_fft))
    phase_loss = F.l1_loss(torch.angle(predicted_fft), torch.angle(target_fft))
    return amp_loss, phase_loss


def tokenizer_reconstruction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    commitment_loss: torch.Tensor,
    phase_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """BrainOmni-style tokenizer objective used by v0.

    L = L_time + L_freq_amp + phase_weight * L_freq_phase + exp(-PCC) + L_commit
    """
    time_loss = time_l1_loss(predicted, target)
    amp_loss, phase_loss = frequency_domain_loss(predicted, target)
    pcc = pearson_corr(predicted, target)
    loss = time_loss + amp_loss + phase_weight * phase_loss + torch.exp(-pcc) + commitment_loss
    return {
        "loss": loss,
        "time_loss": time_loss.detach(),
        "freq_amp_loss": amp_loss.detach(),
        "freq_phase_loss": phase_loss.detach(),
        "pcc": pcc.detach(),
        "commitment_loss": commitment_loss.detach(),
    }


def info_nce_loss(
    eeg_embedding: torch.Tensor,
    audio_embedding: torch.Tensor,
    temperature: torch.Tensor | float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric batch InfoNCE for EEG/audio retrieval."""
    eeg = F.normalize(eeg_embedding, dim=-1)
    audio = F.normalize(audio_embedding, dim=-1)
    logits = eeg @ audio.T
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_eeg_to_audio = F.cross_entropy(logits, labels)
    loss_audio_to_eeg = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_eeg_to_audio + loss_audio_to_eeg), logits
