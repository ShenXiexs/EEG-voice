"""Losses and supervision routing for the factorized v0724 model.

The module deliberately keeps label/subject metadata on the *loss* side of the
training boundary.  None of the helpers construct conditioning embeddings from
metadata, so they can be used without weakening the label-free inference API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SupervisionRouting:
    """Per-sample eligibility for the three heterogeneous EEG datasets."""

    content: torch.Tensor
    exact_realization: torch.Tensor
    weak_timbre: torch.Tensor
    local_realization: torch.Tensor
    energy: torch.Tensor
    codec: torch.Tensor
    eeg_self_supervised: torch.Tensor
    audio_generation_eligible: torch.Tensor

    # Compatibility names used by the v0722 training vocabulary.
    @property
    def exact_acoustic(self) -> torch.Tensor:
        return self.exact_realization

    @property
    def weak_semantic(self) -> torch.Tensor:
        return self.content

    @property
    def timbre(self) -> torch.Tensor:
        """Samples eligible for either exact or weak global timbre loss."""

        return self.exact_realization | self.weak_timbre

    @property
    def realization(self) -> torch.Tensor:
        return self.exact_realization


# Backward-compatible type name for callers sharing generic training utilities.
LossEligibility = SupervisionRouting


def supervision_routing(
    datasets: Sequence[str],
    pairing_scopes: Sequence[str],
    *,
    duration_seconds: torch.Tensor | Sequence[float] | None = None,
    max_realization_seconds: float = 4.0,
    device: torch.device | str | None = None,
) -> SupervisionRouting:
    """Route supervision without treating weak audio links as exact pairs.

    KaraOne's same-trial overt recording is eligible for local realization,
    energy, and codec objectives.  FEIS contributes content supervision and a
    weak, global subject-label timbre prototype only.  ds004306 contributes EEG
    self-supervision/domain robustness only.  Audio longer than the configured
    generation horizon remains content-eligible but is removed from exact
    realization reconstruction.
    """

    if len(datasets) != len(pairing_scopes):
        raise ValueError("datasets and pairing_scopes must have the same length")
    if max_realization_seconds <= 0:
        raise ValueError("max_realization_seconds must be positive")

    normalized_datasets = [str(value).strip().lower() for value in datasets]
    normalized_pairings = [str(value).strip().lower() for value in pairing_scopes]
    karaone_content = torch.tensor(
        [
            dataset == "karaone"
            and pairing in {"karaone_same_trial_overt", "same_trial_overt", "exact"}
            for dataset, pairing in zip(normalized_datasets, normalized_pairings)
        ],
        dtype=torch.bool,
        device=device,
    )
    feis_weak = torch.tensor(
        [
            dataset == "feis"
            and pairing in {"feis_subject_label", "subject_label", "weak_subject_label"}
            for dataset, pairing in zip(normalized_datasets, normalized_pairings)
        ],
        dtype=torch.bool,
        device=device,
    )

    karaone_exact = karaone_content.clone()
    if duration_seconds is not None:
        duration = torch.as_tensor(
            duration_seconds, dtype=torch.float32, device=karaone_exact.device
        )
        if duration.shape != karaone_exact.shape:
            raise ValueError("duration_seconds must have one value per sample")
        within_horizon = (
            torch.isfinite(duration)
            & (duration > 0)
            & (duration <= float(max_realization_seconds))
        )
        karaone_exact = karaone_exact & within_horizon

    content = karaone_content | feis_weak
    self_supervised = torch.ones(
        len(datasets), dtype=torch.bool, device=karaone_exact.device
    )
    return SupervisionRouting(
        content=content,
        exact_realization=karaone_exact,
        weak_timbre=feis_weak,
        local_realization=karaone_exact,
        energy=karaone_exact,
        codec=karaone_exact,
        eeg_self_supervised=self_supervised,
        audio_generation_eligible=karaone_exact,
    )


def loss_eligibility(
    datasets: Sequence[str],
    pairing_confidence: Sequence[str],
    *,
    device: torch.device | str | None = None,
) -> SupervisionRouting:
    """Compatibility wrapper around :func:`supervision_routing`."""

    return supervision_routing(datasets, pairing_confidence, device=device)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _validate_embeddings(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError("embedding tensors must share shape [B,D]")


def _weighted_clip_direction(
    logits: torch.Tensor,
    positive_weights: torch.Tensor,
    allowed: torch.Tensor,
    row_eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_weights = positive_weights.to(dtype=logits.dtype, device=logits.device)
    allowed = allowed.to(dtype=torch.bool, device=logits.device)
    row_eligible = row_eligible.to(dtype=torch.bool, device=logits.device)
    positive_weights = positive_weights * allowed.to(logits.dtype)
    active_rows = row_eligible & (positive_weights.sum(dim=1) > 0) & allowed.any(dim=1)
    if not active_rows.any():
        zero = _zero(logits)
        return zero, active_rows

    selected_logits = logits[active_rows]
    selected_allowed = allowed[active_rows]
    selected_positive = positive_weights[active_rows]
    targets = selected_positive / selected_positive.sum(dim=1, keepdim=True).clamp_min(
        1e-12
    )
    # A large finite floor avoids the undefined 0 * -inf operation for entries
    # which are neither positives nor permitted negatives.
    floor = -torch.finfo(selected_logits.dtype).max / 4.0
    log_probability = F.log_softmax(
        selected_logits.masked_fill(~selected_allowed, floor), dim=1
    )
    cross_entropy = -torch.where(
        targets > 0, targets * log_probability, torch.zeros_like(log_probability)
    ).sum(dim=1)
    return cross_entropy.mean(), active_rows


def masked_symmetric_multi_positive_clip_loss(
    eeg: torch.Tensor,
    audio: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    eeg_eligible: torch.Tensor | None = None,
    audio_eligible: torch.Tensor | None = None,
    allowed: torch.Tensor | None = None,
    temperature: float = 0.08,
) -> dict[str, torch.Tensor]:
    """Symmetric CLIP loss with weighted multi-positives and eligibility masks."""

    _validate_embeddings(eeg, audio)
    batch = eeg.shape[0]
    if positive_weights.shape != (batch, batch):
        raise ValueError("positive_weights must be [B,B]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if eeg_eligible is None:
        eeg_eligible = torch.ones(batch, dtype=torch.bool, device=eeg.device)
    if audio_eligible is None:
        audio_eligible = torch.ones(batch, dtype=torch.bool, device=eeg.device)
    if eeg_eligible.shape != (batch,) or audio_eligible.shape != (batch,):
        raise ValueError("eligibility masks must be [B]")

    pair_allowed = eeg_eligible[:, None].bool() & audio_eligible[None, :].bool()
    if allowed is not None:
        if allowed.shape != (batch, batch):
            raise ValueError("allowed must be [B,B]")
        pair_allowed = pair_allowed & allowed.to(device=eeg.device, dtype=torch.bool)

    eeg_normalized = F.normalize(eeg, dim=-1)
    audio_normalized = F.normalize(audio, dim=-1)
    logits = eeg_normalized @ audio_normalized.T / float(temperature)
    forward, forward_rows = _weighted_clip_direction(
        logits, positive_weights, pair_allowed, eeg_eligible
    )
    backward, backward_rows = _weighted_clip_direction(
        logits.T,
        positive_weights.T,
        pair_allowed.T,
        audio_eligible,
    )
    total = 0.5 * (forward + backward)
    return {
        "total": total,
        "eeg_to_audio": forward.detach(),
        "audio_to_eeg": backward.detach(),
        "active_eeg_rows": forward_rows.sum().detach(),
        "active_audio_rows": backward_rows.sum().detach(),
    }


def symmetric_contrastive_loss(
    eeg: torch.Tensor,
    audio: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    allowed: torch.Tensor | None = None,
    temperature: float = 0.08,
) -> dict[str, torch.Tensor]:
    return masked_symmetric_multi_positive_clip_loss(
        eeg,
        audio,
        positive_weights,
        allowed=allowed,
        temperature=temperature,
    )


def exact_realization_clip_loss(
    eeg: torch.Tensor,
    audio: torch.Tensor,
    eligible: torch.Tensor,
    *,
    temperature: float = 0.08,
) -> dict[str, torch.Tensor]:
    """Exact-pair CLIP in which all other eligible utterances are negatives.

    Consequently, a different utterance with the same content label remains a
    hard negative and the realization branch cannot solve the objective using
    content alone.
    """

    _validate_embeddings(eeg, audio)
    if eligible.shape != (len(eeg),):
        raise ValueError("eligible must be [B]")
    eligible = eligible.to(device=eeg.device, dtype=torch.bool)
    positives = torch.diag(eligible.to(dtype=eeg.dtype))
    return masked_symmetric_multi_positive_clip_loss(
        eeg,
        audio,
        positives,
        eeg_eligible=eligible,
        audio_eligible=eligible,
        temperature=temperature,
    )


exact_pair_contrastive_loss = exact_realization_clip_loss


def content_positive_weights(
    labels: torch.Tensor,
    content_eligible: torch.Tensor,
    *,
    exact_pair_mask: torch.Tensor | None = None,
    weak_positive_weight: float = 1.0,
    exact_positive_weight: float = 1.0,
) -> torch.Tensor:
    """Build same-content positives without exposing labels to the encoder."""

    if labels.ndim != 1 or content_eligible.shape != labels.shape:
        raise ValueError("labels and content_eligible must be [B]")
    if weak_positive_weight < 0 or exact_positive_weight < 0:
        raise ValueError("positive weights must be nonnegative")
    eligible_pairs = content_eligible[:, None].bool() & content_eligible[None, :].bool()
    weights = ((labels[:, None] == labels[None, :]) & eligible_pairs).to(torch.float32)
    weights = weights * float(weak_positive_weight)
    if exact_pair_mask is not None:
        if exact_pair_mask.shape != labels.shape:
            raise ValueError("exact_pair_mask must be [B]")
        diagonal = torch.arange(len(labels), device=labels.device)
        weights[diagonal, diagonal] = torch.where(
            exact_pair_mask.bool(),
            torch.full_like(weights[diagonal, diagonal], float(exact_positive_weight)),
            weights[diagonal, diagonal],
        )
    return weights


def semantic_positive_weights(
    labels: torch.Tensor,
    exact_mask: torch.Tensor,
    semantic_mask: torch.Tensor,
    *,
    weak_weight: float = 0.15,
) -> torch.Tensor:
    return content_positive_weights(
        labels,
        semantic_mask,
        exact_pair_mask=exact_mask,
        weak_positive_weight=weak_weight,
        exact_positive_weight=1.0,
    )


def monotonic_local_alignment_loss(
    eeg_tokens: torch.Tensor,
    audio_tokens: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    eeg_token_mask: torch.Tensor | None = None,
    audio_token_mask: torch.Tensor | None = None,
    positional_sigma: float = 0.20,
    temperature: float = 0.08,
) -> dict[str, torch.Tensor]:
    """Bidirectional local alignment with a soft monotonic position prior."""

    if (
        eeg_tokens.ndim != 3
        or audio_tokens.ndim != 3
        or eeg_tokens.shape[0] != audio_tokens.shape[0]
        or eeg_tokens.shape[2] != audio_tokens.shape[2]
    ):
        raise ValueError("local tokens must be [B,T,D] with common B and D")
    batch, eeg_steps, _ = eeg_tokens.shape
    audio_steps = audio_tokens.shape[1]
    if sample_mask.shape != (batch,):
        raise ValueError("sample_mask must be [B]")
    if positional_sigma <= 0 or temperature <= 0:
        raise ValueError("positional_sigma and temperature must be positive")
    if eeg_token_mask is None:
        eeg_token_mask = torch.ones(
            (batch, eeg_steps), dtype=torch.bool, device=eeg_tokens.device
        )
    if audio_token_mask is None:
        audio_token_mask = torch.ones(
            (batch, audio_steps), dtype=torch.bool, device=audio_tokens.device
        )
    if eeg_token_mask.shape != (batch, eeg_steps) or audio_token_mask.shape != (
        batch,
        audio_steps,
    ):
        raise ValueError("token masks must match their [B,T] sequences")

    similarities: list[torch.Tensor] = []
    for index in torch.nonzero(sample_mask.bool(), as_tuple=False).flatten().tolist():
        eeg = eeg_tokens[index, eeg_token_mask[index].bool()]
        audio = audio_tokens[index, audio_token_mask[index].bool()]
        if len(eeg) == 0 or len(audio) == 0:
            continue
        eeg = F.normalize(eeg, dim=-1)
        audio = F.normalize(audio, dim=-1)
        eeg_position = torch.linspace(
            0.0, 1.0, len(eeg), dtype=eeg.dtype, device=eeg.device
        )
        audio_position = torch.linspace(
            0.0, 1.0, len(audio), dtype=audio.dtype, device=audio.device
        )
        bias = (
            -(eeg_position[:, None] - audio_position[None, :]).square()
            / float(positional_sigma) ** 2
        )
        similarity = eeg @ audio.T / float(temperature)
        forward_weights = torch.softmax(similarity + bias, dim=-1)
        backward_weights = torch.softmax(similarity.T + bias.T, dim=-1)
        aligned_audio = forward_weights @ audio
        aligned_eeg = backward_weights @ eeg
        score = 0.5 * (
            F.cosine_similarity(eeg, aligned_audio, dim=-1).mean()
            + F.cosine_similarity(audio, aligned_eeg, dim=-1).mean()
        )
        similarities.append(score)
    if not similarities:
        zero = _zero(eeg_tokens)
        return {
            "total": zero,
            "cosine": zero.detach(),
            "active_samples": torch.zeros((), device=eeg_tokens.device),
        }
    cosine = torch.stack(similarities).mean()
    return {
        "total": 1.0 - cosine,
        "cosine": cosine.detach(),
        "active_samples": torch.tensor(len(similarities), device=eeg_tokens.device),
    }


def _as_batched_sequence(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if value.ndim == 1:
        return value[None, :, None], True
    if value.ndim == 2:
        return value[None, :, :], True
    if value.ndim == 3:
        return value, False
    raise ValueError("soft-DTW inputs must be [T], [T,D], or [B,T,D]")


def _soft_dtw_cost(
    cost: torch.Tensor, gamma: float, band_ratio: float | None
) -> torch.Tensor:
    if not torch.isfinite(cost).all():
        raise FloatingPointError(
            "soft-DTW cost contains non-finite values; check the prediction and "
            "target sequences"
        )
    rows, columns = cost.shape
    # Dynamic programming by anti-diagonal reduces Python work from O(T^2)
    # scalar operations to O(T) vector operations and avoids in-place mutation
    # of values retained by autograd.
    # Do not use ``+inf`` as the blocked-path sentinel.  On MPS the backward
    # pass through logsumexp([-inf, -inf, -inf]) can produce NaN gradients for
    # cells outside a Sakoe--Chiba band.  Those cells are not part of the valid
    # alignment, but their NaNs can still corrupt the optimizer several epochs
    # later.  A detached finite barrier has the same forward semantics for the
    # bounded, normalized mel costs while keeping the recurrence differentiable.
    # It is at least twice the largest possible observed path cost and remains
    # comfortably below the dtype limit.
    dtype_limit = torch.finfo(cost.dtype).max / 32.0
    barrier_floor = min(10_000.0, dtype_limit / 2.0)
    barrier = (
        cost.detach().abs().amax() * float(rows + columns + 1) + 1.0
    ).clamp(min=barrier_floor, max=dtype_limit)
    diagonal_minus_two = cost.new_ones((rows + 1,)) * barrier
    diagonal_minus_two[0] = 0.0  # d = 0 contains only R[0,0]
    diagonal_minus_one = cost.new_ones((rows + 1,)) * barrier  # d = 1 boundaries
    tolerance = 0.0 if band_ratio is None else float(band_ratio)
    resolution = max(1.0 / max(rows, 1), 1.0 / max(columns, 1))
    for diagonal in range(2, rows + columns + 1):
        lower = max(1, diagonal - columns)
        upper = min(rows, diagonal - 1)
        row_indices = torch.arange(lower, upper + 1, device=cost.device)
        column_indices = diagonal - row_indices
        if band_ratio is not None:
            row_positions = (row_indices - 1).to(cost.dtype) / max(rows - 1, 1)
            column_positions = (column_indices - 1).to(cost.dtype) / max(columns - 1, 1)
            inside = (row_positions - column_positions).abs() <= tolerance + resolution
            row_indices = row_indices[inside]
            column_indices = column_indices[inside]
        current = cost.new_ones((rows + 1,)) * barrier
        if len(row_indices):
            previous = torch.stack(
                (
                    diagonal_minus_one[row_indices - 1],
                    diagonal_minus_one[row_indices],
                    diagonal_minus_two[row_indices - 1],
                ),
                dim=1,
            )
            if gamma == 0:
                soft_minimum = previous.min(dim=1).values
            else:
                soft_minimum = -float(gamma) * torch.logsumexp(
                    -previous / float(gamma), dim=1
                )
            values = cost[row_indices - 1, column_indices - 1] + soft_minimum
            current = current.scatter(0, row_indices, values)
        diagonal_minus_two, diagonal_minus_one = diagonal_minus_one, current
    result = diagonal_minus_one[rows]
    if not torch.isfinite(result):
        raise FloatingPointError(
            "soft-DTW dynamic program produced a non-finite result despite "
            "finite inputs"
        )
    if result >= barrier * 0.5:
        raise ValueError(
            "Sakoe-Chiba band admits no finite alignment path for the supplied "
            "sequence lengths"
        )
    return result


def soft_dtw_divergence_torch(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_mask: torch.Tensor | None = None,
    second_mask: torch.Tensor | None = None,
    gamma: float = 0.05,
    band_ratio: float | None = 0.25,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Differentiable, self-cost-corrected and explicitly nonnegative soft-DTW."""

    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if band_ratio is not None and not 0 <= band_ratio <= 1:
        raise ValueError("band_ratio must be between zero and one")
    first_batch, first_was_single = _as_batched_sequence(first)
    second_batch, second_was_single = _as_batched_sequence(second)
    if (
        first_batch.shape[0] != second_batch.shape[0]
        or first_batch.shape[2] != second_batch.shape[2]
    ):
        raise ValueError(
            "soft-DTW inputs must have common batch and feature dimensions"
        )
    batch, first_steps, _ = first_batch.shape
    second_steps = second_batch.shape[1]
    if first_mask is None:
        first_mask = torch.ones(
            (batch, first_steps), dtype=torch.bool, device=first.device
        )
    elif first_was_single and first_mask.ndim == 1:
        first_mask = first_mask[None, :]
    if second_mask is None:
        second_mask = torch.ones(
            (batch, second_steps), dtype=torch.bool, device=second.device
        )
    elif second_was_single and second_mask.ndim == 1:
        second_mask = second_mask[None, :]
    if first_mask.shape != (batch, first_steps) or second_mask.shape != (
        batch,
        second_steps,
    ):
        raise ValueError("soft-DTW masks must match [B,T]")

    values: list[torch.Tensor] = []
    for index in range(batch):
        x = first_batch[index, first_mask[index].bool()]
        y = second_batch[index, second_mask[index].bool()]
        if len(x) == 0 or len(y) == 0:
            values.append(_zero(first_batch[index]) + _zero(second_batch[index]))
            continue
        xy_cost = (x[:, None, :] - y[None, :, :]).square().mean(dim=-1)
        xx_cost = (x[:, None, :] - x[None, :, :]).square().mean(dim=-1)
        yy_cost = (y[:, None, :] - y[None, :, :]).square().mean(dim=-1)
        divergence = _soft_dtw_cost(xy_cost, gamma, band_ratio)
        divergence = divergence - 0.5 * (
            _soft_dtw_cost(xx_cost, gamma, band_ratio)
            + _soft_dtw_cost(yy_cost, gamma, band_ratio)
        )
        values.append(divergence.clamp_min(0.0) / float(len(x) + len(y)))
    result = torch.stack(values)
    if reduction == "none":
        return result
    if reduction == "sum":
        return result.sum()
    if reduction == "mean":
        return result.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def cross_covariance_loss(
    content: torch.Tensor,
    timbre: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize linear dependence between global content and timbre spaces."""

    _validate_embeddings(content, timbre)
    if sample_mask is None:
        sample_mask = torch.ones(len(content), dtype=torch.bool, device=content.device)
    if sample_mask.shape != (len(content),):
        raise ValueError("sample_mask must be [B]")
    selected_content = content[sample_mask.bool()]
    selected_timbre = timbre[sample_mask.bool()]
    if len(selected_content) < 2:
        return _zero(content) + _zero(timbre)
    selected_content = selected_content - selected_content.mean(dim=0, keepdim=True)
    selected_timbre = selected_timbre - selected_timbre.mean(dim=0, keepdim=True)
    covariance = selected_content.T @ selected_timbre / float(len(selected_content) - 1)
    return covariance.square().mean()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype, device=values.device)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1.0)


def _weighted_global_ssim_torch(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    weights = weights.to(prediction).clamp_min(0.0)
    denominator = weights.sum().clamp_min(1e-8)
    mean_prediction = (prediction * weights).sum() / denominator
    mean_target = (target * weights).sum() / denominator
    centered_prediction = prediction - mean_prediction
    centered_target = target - mean_target
    variance_prediction = (weights * centered_prediction.square()).sum() / denominator
    variance_target = (weights * centered_target.square()).sum() / denominator
    covariance = (weights * centered_prediction * centered_target).sum() / denominator
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)) / (
        (mean_prediction.square() + mean_target.square() + c1)
        * (variance_prediction + variance_target + c2)
    )


def energy_structure_loss(
    prediction_log_mel: torch.Tensor,
    target_log_mel: torch.Tensor,
    frame_mask: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    *,
    target_activity: torch.Tensor | None = None,
    l1_weight: float = 0.5,
    soft_dtw_weight: float = 0.5,
    ssim_weight: float = 0.0,
    gamma: float = 0.05,
    band_ratio: float | None = 0.25,
    soft_dtw_max_frames: int | None = None,
) -> dict[str, torch.Tensor]:
    """Composite numerical log-mel loss; inputs are expected in dB [-80,0]."""

    if prediction_log_mel.ndim != 3 or prediction_log_mel.shape != target_log_mel.shape:
        raise ValueError("log-mel tensors must share [B,M,T]")
    batch, _, frames = prediction_log_mel.shape
    if frame_mask.shape != (batch, frames):
        raise ValueError("frame_mask must be [B,T]")
    if sample_mask is None:
        sample_mask = torch.ones(
            batch, dtype=torch.bool, device=prediction_log_mel.device
        )
    if sample_mask.shape != (batch,):
        raise ValueError("sample_mask must be [B]")
    active_frame_mask = frame_mask.bool() & sample_mask[:, None].bool()
    if not active_frame_mask.any():
        zero = _zero(prediction_log_mel)
        return {
            "total": zero,
            "log_mel_l1": zero.detach(),
            "soft_dtw": zero.detach(),
            "ssim": zero.detach(),
        }

    log_mel_l1 = _masked_mean(
        (prediction_log_mel - target_log_mel).abs(), active_frame_mask
    )
    prediction_normalized = ((prediction_log_mel + 80.0) / 80.0).clamp(0.0, 1.0)
    target_normalized = ((target_log_mel + 80.0) / 80.0).clamp(0.0, 1.0)
    selected = sample_mask.bool()
    dtw_values: list[torch.Tensor] = []
    for index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
        valid = frame_mask[index].bool()
        prediction_sequence = prediction_normalized[index, :, valid].transpose(0, 1)
        target_sequence = target_normalized[index, :, valid].transpose(0, 1)
        if soft_dtw_max_frames is not None:
            if soft_dtw_max_frames < 2:
                raise ValueError("soft_dtw_max_frames must be at least two")
            if len(prediction_sequence) > soft_dtw_max_frames:
                prediction_sequence = (
                    F.interpolate(
                        prediction_sequence.transpose(0, 1).unsqueeze(0),
                        size=soft_dtw_max_frames,
                        mode="linear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
                target_sequence = (
                    F.interpolate(
                        target_sequence.transpose(0, 1).unsqueeze(0),
                        size=soft_dtw_max_frames,
                        mode="linear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
        dtw_values.append(
            soft_dtw_divergence_torch(
                prediction_sequence,
                target_sequence,
                gamma=gamma,
                band_ratio=band_ratio,
            )
        )
    soft_dtw = torch.stack(dtw_values).mean()

    if target_activity is not None:
        if target_activity.shape != (batch, frames):
            raise ValueError("target_activity must be [B,T]")
        foreground = target_activity[:, None, :].to(prediction_normalized)
    else:
        foreground = target_normalized.detach().clamp_min(0.0)
    foreground = foreground * active_frame_mask[:, None, :].to(prediction_normalized)
    ssim_values = [
        _weighted_global_ssim_torch(
            prediction_normalized[index],
            target_normalized[index],
            foreground[index].expand_as(target_normalized[index]),
        )
        for index in torch.nonzero(selected, as_tuple=False).flatten().tolist()
        if foreground[index].sum() > 0
    ]
    ssim = torch.stack(ssim_values).mean() if ssim_values else _zero(prediction_log_mel)
    total = (
        float(l1_weight) * log_mel_l1
        + float(soft_dtw_weight) * soft_dtw
        + float(ssim_weight) * (1.0 - ssim)
    )
    return {
        "total": total,
        "log_mel_l1": log_mel_l1.detach(),
        "soft_dtw": soft_dtw.detach(),
        "ssim": ssim.detach(),
    }


def prosody_activity_duration_loss(
    prediction_log_f0: torch.Tensor,
    target_log_f0: torch.Tensor,
    prediction_voicing_logits: torch.Tensor,
    target_voicing: torch.Tensor,
    prediction_log_rms: torch.Tensor,
    target_log_rms: torch.Tensor,
    prediction_activity_logits: torch.Tensor,
    target_activity: torch.Tensor,
    prediction_duration: torch.Tensor,
    target_duration: torch.Tensor,
    frame_mask: torch.Tensor,
    sample_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Masked auxiliary loss for local prosody, activity, and duration."""

    sequence_tensors = (
        prediction_log_f0,
        target_log_f0,
        prediction_voicing_logits,
        target_voicing,
        prediction_log_rms,
        target_log_rms,
        prediction_activity_logits,
        target_activity,
    )
    if any(value.ndim != 2 for value in sequence_tensors) or any(
        value.shape != sequence_tensors[0].shape for value in sequence_tensors
    ):
        raise ValueError("all prosody/activity tensors must share [B,T]")
    batch, frames = sequence_tensors[0].shape
    if frame_mask.shape != (batch, frames) or sample_mask.shape != (batch,):
        raise ValueError("frame_mask/sample_mask must be [B,T] and [B]")
    if prediction_duration.shape != (batch,) or target_duration.shape != (batch,):
        raise ValueError("duration tensors must be [B]")
    valid = frame_mask.bool() & sample_mask[:, None].bool()
    if not valid.any():
        zero = _zero(prediction_activity_logits)
        return {
            key: zero
            for key in ("total", "log_f0", "voicing", "log_rms", "activity", "duration")
        }

    voiced = valid & target_voicing.bool()
    log_f0 = (
        F.smooth_l1_loss(prediction_log_f0[voiced], target_log_f0[voiced])
        if voiced.any()
        else _zero(prediction_log_f0)
    )
    voicing = F.binary_cross_entropy_with_logits(
        prediction_voicing_logits[valid],
        target_voicing[valid].to(prediction_voicing_logits),
    )
    log_rms = F.smooth_l1_loss(prediction_log_rms[valid], target_log_rms[valid])
    activity = F.binary_cross_entropy_with_logits(
        prediction_activity_logits[valid],
        target_activity[valid].to(prediction_activity_logits),
    )
    duration = F.smooth_l1_loss(
        prediction_duration[sample_mask.bool()], target_duration[sample_mask.bool()]
    )
    total = (log_f0 + voicing + log_rms + activity + duration) / 5.0
    return {
        "total": total,
        "log_f0": log_f0.detach(),
        "voicing": voicing.detach(),
        "log_rms": log_rms.detach(),
        "activity": activity.detach(),
        "duration": duration.detach(),
    }


def code_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    codebook_weights: torch.Tensor | None = None,
    *,
    sample_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Masked EnCodec CE, suitable for the low-weight KaraOne-only auxiliary."""

    if (
        logits.ndim != 4
        or target.shape != logits.shape[:3]
        or mask.shape != target.shape
    ):
        raise ValueError("logits/target/mask must be [B,Q,T,V], [B,Q,T], [B,Q,T]")
    active_mask = mask.bool()
    if sample_mask is not None:
        if sample_mask.shape != (len(logits),):
            raise ValueError("sample_mask must be [B]")
        active_mask = active_mask & sample_mask[:, None, None].bool()
    per_token = F.cross_entropy(
        logits.permute(0, 3, 1, 2), target.long(), reduction="none"
    )
    weights = (
        torch.ones(logits.shape[1], dtype=logits.dtype, device=logits.device)
        if codebook_weights is None
        else codebook_weights.to(logits)
    )
    combined = active_mask.to(logits.dtype) * weights.view(1, -1, 1)
    total = (per_token * combined).sum() / combined.sum().clamp_min(1.0)
    prediction = logits.argmax(dim=-1)
    output = {"total": total}
    for index in range(logits.shape[1]):
        selected = active_mask[:, index]
        output[f"q{index}_accuracy"] = (
            (prediction[:, index][selected] == target[:, index][selected])
            .float()
            .mean()
            .detach()
            if selected.any()
            else total.detach() * 0.0
        )
    return output


def masked_patch_reconstruction_loss(
    reconstruction: torch.Tensor, target: torch.Tensor, patch_mask: torch.Tensor
) -> torch.Tensor:
    if reconstruction.shape != target.shape or patch_mask.shape != target.shape[:3]:
        raise ValueError("patch reconstruction tensors are inconsistent")
    return (
        F.smooth_l1_loss(reconstruction[patch_mask], target[patch_mask])
        if patch_mask.any()
        else _zero(reconstruction)
    )


def condition_consistency_loss(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("condition views must have the same shape")
    return (1.0 - F.cosine_similarity(first, second, dim=-1)).mean()


def moe_regularization(
    router: dict[str, torch.Tensor], *, z_weight: float = 0.1
) -> torch.Tensor:
    return router["balance_loss"] + float(z_weight) * router["z_loss"]


def router_collapse_flags(
    mass: torch.Tensor,
    *,
    dying_threshold: float = 0.05,
    collapse_threshold: float = 0.60,
) -> dict[str, bool]:
    value = mass.detach().float().cpu()
    return {
        "expert_dying": bool((value < float(dying_threshold)).any()),
        "routing_collapse": bool((value > float(collapse_threshold)).any()),
    }


__all__ = [
    "LossEligibility",
    "SupervisionRouting",
    "code_cross_entropy",
    "condition_consistency_loss",
    "content_positive_weights",
    "cross_covariance_loss",
    "energy_structure_loss",
    "exact_pair_contrastive_loss",
    "exact_realization_clip_loss",
    "loss_eligibility",
    "masked_patch_reconstruction_loss",
    "masked_symmetric_multi_positive_clip_loss",
    "moe_regularization",
    "monotonic_local_alignment_loss",
    "prosody_activity_duration_loss",
    "router_collapse_flags",
    "semantic_positive_weights",
    "soft_dtw_divergence_torch",
    "supervision_routing",
    "symmetric_contrastive_loss",
]
