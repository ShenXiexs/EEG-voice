from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def cvae_audio_loss(
    posterior_mel: torch.Tensor,
    prior_mel: torch.Tensor,
    analytic_mel: torch.Tensor,
    target_mel: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    *,
    kl_beta: float,
    free_bits: float,
    prior_weight: float,
    analytic_consistency_weight: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Conditional VAE objective with an explicitly usable audio-free prior."""
    def masked_smooth(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        value=F.smooth_l1_loss(left,right,reduction="none")
        if mask is None:return value.mean()
        weight=mask.to(value.dtype).unsqueeze(1).expand_as(value)
        return (value*weight).sum()/weight.sum().clamp_min(1.0)
    posterior_reconstruction = masked_smooth(posterior_mel, target_mel)
    prior_reconstruction = masked_smooth(prior_mel, target_mel)
    variance_ratio = torch.exp(posterior_logvar - prior_logvar)
    mean_term = (posterior_mean - prior_mean).square() * torch.exp(-prior_logvar)
    kl_per_dimension = 0.5 * (
        prior_logvar - posterior_logvar + variance_ratio + mean_term - 1.0
    )
    kl = kl_per_dimension.clamp_min(float(free_bits)).sum(-1).mean()
    residual_penalty = torch.mean(torch.abs(posterior_mel - analytic_mel)) / 80.0
    total = (
        posterior_reconstruction
        + float(prior_weight) * prior_reconstruction
        + float(kl_beta) * kl
        + float(analytic_consistency_weight) * residual_penalty
    )
    return total, {
        "posterior_mel": float(posterior_reconstruction.detach()),
        "prior_mel": float(prior_reconstruction.detach()),
        "kl": float(kl.detach()),
        "kl_beta": float(kl_beta),
        "analytic_residual": float(residual_penalty.detach()),
    }


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


def pairwise_mfcc_l1(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    query_chunk: int = 16,
    target_chunk: int = 64,
    feature_chunk: int = 2048,
) -> np.ndarray:
    """Exact pairwise MFCC L1 without an ``N×N×F×T`` allocation.

    The former broadcast for 1,016 trials materialized roughly 42 GB of
    float32 differences and was killed by macOS.  This computes the identical
    mean absolute distance in bounded blocks (about 8 MB with the defaults),
    retaining only the small ``N×M`` distance matrix.
    """
    prediction = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    if prediction.ndim < 2 or target.ndim < 2:
        raise ValueError("pairwise MFCC inputs require a batch plus feature dimensions")
    if prediction.shape[1:] != target.shape[1:]:
        raise ValueError(f"pairwise MFCC feature shapes differ: {prediction.shape[1:]} != {target.shape[1:]}")
    left = np.ascontiguousarray(prediction.reshape(len(prediction), -1))
    right = np.ascontiguousarray(target.reshape(len(target), -1))
    dimensions = int(left.shape[1])
    if dimensions == 0:
        raise ValueError("pairwise MFCC inputs have zero feature dimensions")
    result = np.empty((len(left), len(right)), dtype=np.float32)
    for q0 in range(0, len(left), int(query_chunk)):
        q1 = min(q0 + int(query_chunk), len(left))
        for t0 in range(0, len(right), int(target_chunk)):
            t1 = min(t0 + int(target_chunk), len(right))
            totals = np.zeros((q1 - q0, t1 - t0), dtype=np.float64)
            for f0 in range(0, dimensions, int(feature_chunk)):
                f1 = min(f0 + int(feature_chunk), dimensions)
                difference = np.abs(left[q0:q1, None, f0:f1] - right[None, t0:t1, f0:f1])
                totals += difference.sum(axis=2, dtype=np.float64)
            result[q0:q1, t0:t1] = (totals / dimensions).astype(np.float32)
    return result


def retrieval(predictions: np.ndarray, targets: np.ndarray, labels: Iterable[str], keys: Iterable[str]) -> dict[str, object]:
    """Label retrieval plus strict within-label target-trial R@1."""
    prediction = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    labels = [str(value).strip().strip("/").lower() for value in labels]
    keys = [str(value) for value in keys]
    distances = pairwise_mfcc_l1(prediction, target)
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


def clip_token_global_losses(eeg_tokens: torch.Tensor, audio_tokens: torch.Tensor, labels: list[str], scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """OpenAI-CLIP-style alignment with the preregistered positive masks.

    Token alignment is strict trial diagonal.  Global alignment may use the
    same-label multi-positive relation, but it is never used for paired R@1.
    """
    if eeg_tokens.shape != audio_tokens.shape:
        raise ValueError("EEG and audio content token tensors must have equal shape")
    eeg, audio = F.normalize(eeg_tokens, dim=-1), F.normalize(audio_tokens, dim=-1)
    multiplier = scale.clamp(max=np.log(100.0)).exp()
    token_logits = multiplier * torch.einsum("itd,jtd->ij", eeg, audio) / eeg.shape[1]
    diagonal = torch.arange(len(eeg), device=eeg.device)
    token = 0.5 * (F.cross_entropy(token_logits, diagonal) + F.cross_entropy(token_logits.T, diagonal))
    global_logits = multiplier * F.normalize(eeg.mean(1), dim=-1) @ F.normalize(audio.mean(1), dim=-1).T
    canonical = [str(value).strip().strip("/").lower() for value in labels]
    positive = torch.tensor([[a == b for b in canonical] for a in canonical], dtype=torch.bool, device=eeg.device)
    global_ = 0.5 * (_multi_positive(global_logits, positive) + _multi_positive(global_logits.T, positive.T))
    return token, global_


def audio_content_loss(prediction: torch.Tensor, target: torch.Tensor, tokens: torch.Tensor, text_target: torch.Tensor, speaker_logits: torch.Tensor, speaker_target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    l1, delta, cosine = mfcc_l1(prediction,target), delta_l1(prediction,target), temporal_cosine_loss(prediction,target)
    # deterministic text anchor: mean token direction is trained weakly, but
    # text is never an input to the model or used at inference.
    text = F.mse_loss(F.normalize(tokens.mean(1),dim=-1),F.normalize(text_target,dim=-1))
    adversarial = F.cross_entropy(speaker_logits, speaker_target)
    total=.60*l1+.20*delta+.10*cosine+.05*text+.05*adversarial
    return total,{'mfcc_l1':float(l1.detach()),'delta_l1':float(delta.detach()),'temporal_cosine':float(cosine.detach()),'text':float(text.detach()),'speaker_adversarial':float(adversarial.detach())}


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
    distances = pairwise_mfcc_l1(prediction, target)
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
