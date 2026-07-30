from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def mfcc_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def temporal_cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left = F.normalize(prediction.transpose(1, 2), dim=-1)
    right = F.normalize(target.transpose(1, 2), dim=-1)
    return (1.0 - (left * right).sum(-1)).mean()


def delta_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction[:, :, 1:] - prediction[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def overfit_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = mfcc_l1(prediction, target)
    cosine = temporal_cosine_loss(prediction, target)
    delta = delta_l1(prediction, target)
    total = 0.7 * l1 + 0.2 * cosine + 0.1 * delta
    return total, {"mfcc_l1": float(l1.detach()), "temporal_cosine": float(cosine.detach()), "delta_l1": float(delta.detach())}


def fixed_audio_tokens(mfcc: torch.Tensor, steps: int = 16) -> torch.Tensor:
    """Deterministic MFCC tokens; no learned audio tower or label is injected."""
    return F.adaptive_avg_pool1d(mfcc, steps).transpose(1, 2)


def _multi_positive(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    valid = positive.any(1)
    if not valid.any():
        return logits.new_zeros(())
    numerator = torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return -(numerator[valid] - denominator[valid]).mean()


def clip_losses(eeg_tokens: torch.Tensor, mfcc: torch.Tensor, labels: list[str], scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    audio_tokens = fixed_audio_tokens(mfcc, eeg_tokens.shape[1])
    eeg = F.normalize(eeg_tokens, dim=-1)
    audio = F.normalize(audio_tokens, dim=-1)
    eeg_global = F.normalize(eeg.mean(1), dim=-1)
    audio_global = F.normalize(audio.mean(1), dim=-1)
    multiplier = scale.clamp(max=np.log(100.0)).exp()
    canonical = [str(value).strip().strip("/").lower() for value in labels]
    positive = torch.tensor([[a == b for b in canonical] for a in canonical], dtype=torch.bool, device=eeg.device)
    global_logits = multiplier * eeg_global @ audio_global.T
    token_logits = multiplier * torch.einsum("itd,jtd->ij", eeg, audio) / eeg.shape[1]
    global_loss = 0.5 * (_multi_positive(global_logits, positive) + _multi_positive(global_logits.T, positive.T))
    token_loss = 0.5 * (_multi_positive(token_logits, positive) + _multi_positive(token_logits.T, positive.T))
    return global_loss, token_loss


def fit_loss(prediction: torch.Tensor, target: torch.Tensor, tokens: torch.Tensor, labels: list[str], scale: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = mfcc_l1(prediction, target)
    delta = delta_l1(prediction, target)
    token, global_ = clip_losses(tokens, target, labels, scale)
    total = 0.50 * l1 + 0.20 * delta + 0.15 * token + 0.15 * global_
    return total, {"mfcc_l1": float(l1.detach()), "delta_l1": float(delta.detach()), "token_clip": float(token.detach()), "global_clip": float(global_.detach())}


def mfcc_distance(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.asarray(prediction) - np.asarray(target)), axis=(1, 2))


def retrieval(predictions: np.ndarray, targets: np.ndarray, labels: Iterable[str], keys: Iterable[str]) -> dict[str, object]:
    """Label retrieval plus strict within-label target-trial R@1."""
    prediction = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    labels = [str(value).strip().strip("/").lower() for value in labels]
    keys = [str(value) for value in keys]
    distances = np.mean(np.abs(prediction[:, None] - target[None]), axis=(2, 3))
    nearest = distances.argmin(1)
    label_top1 = float(np.mean([labels[index] == labels[candidate] for index, candidate in enumerate(nearest)]))
    ranks, hit = [], []
    for index, label in enumerate(labels):
        candidates = np.flatnonzero(np.asarray(labels) == label)
        order = candidates[np.argsort(distances[index, candidates], kind="stable")]
        rank = int(np.flatnonzero(order == index)[0]) + 1
        ranks.append(rank)
        hit.append(rank == 1)
    return {
        "label_top1": label_top1,
        "paired_r_at_1": float(np.mean(hit)),
        "paired_rank_mean": float(np.mean(ranks)),
        "paired_rank_per_trial": ranks,
        "chance_within_label": float(np.mean([1.0 / labels.count(label) for label in labels])),
        "nearest_sample_keys": [keys[index] for index in nearest.tolist()],
    }


def same_label_template(targets: np.ndarray, labels: Iterable[str]) -> np.ndarray:
    labels = [str(value).strip().strip("/").lower() for value in labels]
    values = np.asarray(targets, dtype=np.float32)
    result = np.empty_like(values)
    for label in sorted(set(labels)):
        indices = np.flatnonzero(np.asarray(labels) == label)
        result[indices] = values[indices].mean(0, keepdims=True)
    return result


def variance_ratio(prediction: np.ndarray, target: np.ndarray, labels: Iterable[str]) -> float:
    """Ratio of same-label *between-trial* variance, excluding within-trial structure.

    Taking ``np.var(values[indices])`` over every axis would allow a single
    time-frequency template repeated for all trials to pass this anti-collapse
    check.  Here variance is taken along the trial axis first and then averaged
    over MFCC coefficient/time cells.
    """
    labels = [str(value).strip().strip("/").lower() for value in labels]
    ratios = []
    for label in sorted(set(labels)):
        indices = np.flatnonzero(np.asarray(labels) == label)
        if len(indices) < 2:
            continue
        numerator = float(np.var(np.asarray(prediction)[indices], axis=0).mean())
        denominator = max(float(np.var(np.asarray(target)[indices], axis=0).mean()), 1.0e-8)
        ratios.append(numerator / denominator)
    return float(np.mean(ratios)) if ratios else 0.0


def paired_win_rate(correct: np.ndarray, control: np.ndarray, target: np.ndarray) -> float:
    correct_error = mfcc_distance(correct, target)
    control_error = mfcc_distance(control, target)
    return float(np.mean(correct_error < control_error))


def paired_r_at_1_above_chance(
    predictions: np.ndarray,
    targets: np.ndarray,
    labels: Iterable[str],
    *, samples: int, seed: int,
) -> dict[str, float]:
    """Bootstrap strict one-to-one, within-label R@1 against its exact chance.

    The chance term is computed separately for each query because label group
    sizes need not be identical outside the fixed 50-pair sanity subset.
    """
    prediction = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    canonical = [str(value).strip().strip("/").lower() for value in labels]
    distances = np.mean(np.abs(prediction[:, None] - target[None]), axis=(2, 3))
    gains: list[float] = []
    for index, label in enumerate(canonical):
        candidates = np.flatnonzero(np.asarray(canonical) == label)
        order = candidates[np.argsort(distances[index, candidates], kind="stable")]
        hit = float(order[0] == index)
        gains.append(hit - 1.0 / len(candidates))
    values = np.asarray(gains, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [values[rng.integers(0, len(values), len(values))].mean() for _ in range(samples)]
    )
    return {
        "mean_gain_over_chance": float(values.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def bootstrap_mean_gain(correct: np.ndarray, control: np.ndarray, target: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    gains = mfcc_distance(control, target) - mfcc_distance(correct, target)
    rng = np.random.default_rng(seed)
    values = np.asarray([gains[rng.integers(0, len(gains), len(gains))].mean() for _ in range(samples)])
    return {"mean_gain": float(gains.mean()), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975))}
