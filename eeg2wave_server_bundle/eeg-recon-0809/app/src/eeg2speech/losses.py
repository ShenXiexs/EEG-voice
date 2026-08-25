"""Supervision-strength-aware content losses and counterfactual controls."""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from .model import JointState


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Normalize by the number of supervised examples, not by sum(weight).
    # Otherwise a homogeneous batch of 0.35-weight weak pairs is accidentally
    # promoted back to unit-strength supervision.
    return (values * weight).sum() / (weight > 0).sum().clamp_min(1)


def masked_mfcc_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     sample_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    element = F.smooth_l1_loss(prediction, target, reduction="none").mean(1)
    per_sample = (element * mask.to(element.dtype)).sum(1) / mask.sum(1).clamp_min(1)
    mfcc = weighted_mean(per_sample, sample_weight)
    delta_prediction = prediction[..., 1:] - prediction[..., :-1]
    delta_target = target[..., 1:] - target[..., :-1]
    delta_mask = mask[:, 1:] & mask[:, :-1]
    delta_element = F.smooth_l1_loss(delta_prediction, delta_target, reduction="none").mean(1)
    delta_per_sample = (delta_element * delta_mask.to(element.dtype)).sum(1) / delta_mask.sum(1).clamp_min(1)
    return mfcc, weighted_mean(delta_per_sample, sample_weight)


def soft_dtw_token_loss(left: torch.Tensor, right: torch.Tensor, left_mask: torch.Tensor,
                        right_mask: torch.Tensor, window_fraction: float = 0.20,
                        temperature: float = 0.07) -> torch.Tensor:
    left, right = F.normalize(left, dim=-1), F.normalize(right, dim=-1)
    if right.shape[1] != left.shape[1]:
        right = F.interpolate(right.transpose(1, 2), size=left.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        right_mask = F.interpolate(right_mask.float().unsqueeze(1), size=left.shape[1], mode="nearest").squeeze(1).bool()
    _, steps, _ = left.shape
    radius = max(1, int(round(steps * window_fraction)))
    grid = torch.arange(steps, device=left.device)
    allowed = (grid[:, None] - grid[None, :]).abs() <= radius
    valid = allowed.unsqueeze(0) & left_mask[:, :, None] & right_mask[:, None, :]
    cost = 1.0 - torch.einsum("btd,bsd->bts", left, right)
    kernel = torch.exp(-cost / max(temperature, 1e-4)) * valid.to(cost.dtype)
    row_mass = left_mask.to(cost.dtype); row_mass = row_mass / row_mass.sum(1, keepdim=True).clamp_min(1)
    col_mass = right_mask.to(cost.dtype); col_mass = col_mass / col_mass.sum(1, keepdim=True).clamp_min(1)
    u, v = torch.ones_like(row_mass), torch.ones_like(col_mass)
    for _ in range(8):
        u = row_mass / torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
        v = col_mass / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    transport = u[:, :, None] * kernel * v[:, None, :]
    transport = transport / transport.sum((1, 2), keepdim=True).clamp_min(1e-8)
    relative = grid.to(cost.dtype) / max(steps - 1, 1)
    monotonic = (relative[:, None] - relative[None, :]).abs().unsqueeze(0)
    return ((cost + 0.10 * monotonic) * transport).sum((1, 2))


def _multi_positive(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    denominator = torch.logsumexp(logits, dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -1e4), dim=1)
    return denominator - numerator


def global_clip_loss(left: torch.Tensor, right: torch.Tensor, labels: Iterable[str], scale: torch.Tensor,
                     sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    logits = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).T * scale.exp().clamp(max=100)
    names = [str(value).strip().lower() for value in labels]
    positive = torch.tensor([[a == b for b in names] for a in names], dtype=torch.bool, device=logits.device)
    weight = torch.ones(len(names), device=logits.device, dtype=logits.dtype) if sample_weight is None else sample_weight.to(logits.dtype)
    return 0.5 * (weighted_mean(_multi_positive(logits, positive), weight) +
                  weighted_mean(_multi_positive(logits.T, positive.T), weight))


def joint_content_loss(state: JointState, batch: dict, model, weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
    exact = batch["pairing_weight"].to(state.mfcc.dtype)
    mfcc, delta = masked_mfcc_loss(state.mfcc, batch["content_mfcc"], batch["content_mask"], exact)
    eligible = exact > 0
    zero = state.mfcc.new_zeros(())
    local = zero
    global_ = zero
    hubert_eligible = eligible & batch["hubert_mask"].any(1) if "hubert_mask" in batch else torch.zeros_like(eligible)
    if hubert_eligible.any() and "hubert_local" in batch:
        audio = model.project_audio(batch["hubert_local"][hubert_eligible])
        local_per_sample = soft_dtw_token_loss(state.local[hubert_eligible], audio, state.token_mask[hubert_eligible], batch["hubert_mask"][hubert_eligible])
        local = weighted_mean(local_per_sample, exact[hubert_eligible])
        audio_global = F.normalize(audio.mean(1), dim=-1)
        global_ = global_clip_loss(state.global_embedding[hubert_eligible], audio_global,
                                   [batch["linguistic_content_id"][i] for i in hubert_eligible.nonzero(as_tuple=False).flatten().tolist()],
                                   model.clip_logit_scale, exact[hubert_eligible])
    label_mask = batch["phoneme_index"] >= 0
    phoneme = F.cross_entropy(state.phoneme_logits[label_mask], batch["phoneme_index"][label_mask]) if label_mask.any() else zero
    total = (float(weights["mfcc"]) * mfcc + float(weights["delta"]) * delta +
             float(weights["local_alignment"]) * local + float(weights["global_clip"]) * global_ +
             float(weights["phoneme_auxiliary"]) * phoneme)
    metrics = {"total": float(total.detach()), "mfcc": float(mfcc.detach()), "delta": float(delta.detach()),
               "local_alignment": float(local.detach()), "global_clip": float(global_.detach()),
               "phoneme_auxiliary": float(phoneme.detach())}
    return total, metrics


def counterfactual_eeg(eeg: torch.Tensor, control: str, generator: torch.Generator | None = None,
                       time_mask: torch.Tensor | None = None,
                       channel_mask: torch.Tensor | None = None) -> torch.Tensor:
    if control == "zero":
        return torch.zeros_like(eeg)
    if control == "time_shuffle":
        output = eeg.clone()
        time_mask = torch.ones(eeg.shape[0], eeg.shape[-1], dtype=torch.bool, device=eeg.device) if time_mask is None else time_mask
        for batch in range(eeg.shape[0]):
            valid = time_mask[batch].nonzero(as_tuple=False).flatten()
            if generator is None:
                stride = next((value for value in range(2, len(valid)) if math.gcd(value, len(valid)) == 1), 1)
                permutation = (torch.arange(len(valid), device=eeg.device) * stride + 1) % max(len(valid), 1)
            else:
                permutation = torch.randperm(len(valid), generator=generator, device=eeg.device)
            order = valid[permutation]
            output[batch, :, valid] = eeg[batch, :, order]
        return output
    if control == "channel_shuffle":
        output = eeg.clone()
        channel_mask = torch.ones(eeg.shape[0], eeg.shape[1], dtype=torch.bool, device=eeg.device) if channel_mask is None else channel_mask
        for batch in range(eeg.shape[0]):
            valid = channel_mask[batch].nonzero(as_tuple=False).flatten()
            if generator is None:
                stride = next((value for value in range(2, len(valid)) if math.gcd(value, len(valid)) == 1), 1)
                permutation = (torch.arange(len(valid), device=eeg.device) * stride + 1) % max(len(valid), 1)
            else:
                permutation = torch.randperm(len(valid), generator=generator, device=eeg.device)
            order = valid[permutation]
            output[batch, valid] = eeg[batch, order]
        return output
    raise ValueError(f"unknown counterfactual control {control}")
