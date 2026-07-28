from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def multi_positive_clip(first: torch.Tensor, second: torch.Tensor, positive: torch.Tensor, *, temperature: float = 0.08) -> torch.Tensor:
    """Symmetric contrastive loss with a non-empty boolean positive matrix."""
    first = F.normalize(first, dim=-1); second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    positive = positive.bool()
    if not positive.any(1).all() or not positive.any(0).all(): raise ValueError("every contrastive row/column needs a positive")
    def direction(value: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        log_prob = value - torch.logsumexp(value, 1, keepdim=True)
        return -(log_prob * relation.to(value.dtype)).sum(1).div(relation.sum(1)).mean()
    return 0.5 * (direction(logits, positive) + direction(logits.T, positive.T))


def soft_dtw_divergence(first: torch.Tensor, second: torch.Tensor, *, gamma: float = 0.1, band_fraction: float = 0.25) -> torch.Tensor:
    """Non-negative soft-DTW divergence for one [T,D] pair, with safe bands."""
    if first.ndim != 2 or second.ndim != 2: raise ValueError("soft-DTW expects [T,D]")
    def cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0).pow(2)
        n, m = distances.shape
        band = max(abs(n - m), int(max(n, m) * band_fraction))
        table = torch.full((n + 1, m + 1), float("inf"), device=x.device, dtype=x.dtype); table[0, 0] = 0
        for i in range(1, n + 1):
            low, high = max(1, i - band), min(m, i + band)
            for j in range(low, high + 1):
                previous = torch.stack((table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]))
                table[i, j] = distances[i - 1, j - 1] - gamma * torch.logsumexp(-previous / gamma, 0)
        if not torch.isfinite(table[n, m]): raise ValueError("soft-DTW band cannot connect sequence lengths")
        return table[n, m]
    return (cost(first, second) - 0.5 * cost(first, first) - 0.5 * cost(second, second)).clamp_min(0.0)


def sequence_soft_dtw(first: torch.Tensor, second: torch.Tensor, *, gamma: float, band_fraction: float) -> torch.Tensor:
    return torch.stack([soft_dtw_divergence(a, b, gamma=gamma, band_fraction=band_fraction) for a, b in zip(first, second)]).mean()


def soft_iou(prediction: torch.Tensor, target: torch.Tensor, *, threshold_db: float = -55.0) -> torch.Tensor:
    pred = torch.sigmoid((prediction - threshold_db) / 4.0); actual = torch.sigmoid((target - threshold_db) / 4.0)
    return (pred * actual).sum((-2, -1)) / (pred.sum((-2, -1)) + actual.sum((-2, -1)) - (pred * actual).sum((-2, -1))).clamp_min(1e-6)


def foreground_msssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Stable foreground SSIM surrogate at three time scales; inputs [B,80,T]."""
    values = []
    for scale in (1, 2, 4):
        first, second = prediction, target
        if scale > 1:
            first = F.avg_pool1d(first, scale, scale); second = F.avg_pool1d(second, scale, scale)
        mu_x, mu_y = first.mean(-1), second.mean(-1)
        var_x = (first - mu_x.unsqueeze(-1)).pow(2).mean(-1); var_y = (second - mu_y.unsqueeze(-1)).pow(2).mean(-1)
        cov = ((first - mu_x.unsqueeze(-1)) * (second - mu_y.unsqueeze(-1))).mean(-1)
        score = ((2 * mu_x * mu_y + 1e-4) * (2 * cov + 9e-4)) / ((mu_x.square() + mu_y.square() + 1e-4) * (var_x + var_y + 9e-4))
        values.append(score.clamp(-1, 1).mean(1))
    return torch.stack(values).mean(0)


def evidence_loss(probability: torch.Tensor) -> torch.Tensor:
    """Balanced real-vs-zero/noise BCE. Caller provides real samples only."""
    zeros = torch.zeros_like(probability); noise = torch.zeros_like(probability)
    real = F.binary_cross_entropy(probability, torch.ones_like(probability))
    return real + 0.5 * (F.binary_cross_entropy(zeros + 1e-6, zeros) + F.binary_cross_entropy(noise + 1e-6, zeros))


def channel_consistency(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    # Multi-head attention and query slicing can return non-contiguous views;
    # materialize both operands before the MPS smooth-L1 kernel flattens them.
    return F.smooth_l1_loss(first.contiguous(), second.contiguous())
