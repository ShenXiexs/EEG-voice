#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0722.data import stochastic_channel_view  # noqa: E402
from src.open_vocab_0724.audio_gate import require_frozen_audio_checkpoint  # noqa: E402
from src.open_vocab_0724.data import (  # noqa: E402
    FactorizedAudioDataset,
    FactorizedEEGDataset,
    TeacherCacheV2,
    collate_factorized,
    load_context,
    normalize_label,
)
from src.open_vocab_0724.lineage import (  # noqa: E402
    authorize_locked_test,
    authorize_locked_test_metadata,
    build_lineage,
    claim_locked_test_access,
    checkpoint_payload,
    file_sha256,
    validate_checkpoint,
)
from src.open_vocab_0724.losses import (  # noqa: E402
    code_cross_entropy,
    condition_consistency_loss,
    content_positive_weights,
    cross_covariance_loss,
    energy_structure_loss,
    exact_realization_clip_loss,
    masked_patch_reconstruction_loss,
    masked_symmetric_multi_positive_clip_loss,
    moe_regularization,
    monotonic_local_alignment_loss,
    prosody_activity_duration_loss,
)
from src.open_vocab_0724.model import (  # noqa: E402
    FactorizedAudioModel,
    FactorizedEEGEncoder,
    grad_reverse,
    random_code_mask,
    random_patch_mask,
)
from src.open_vocab_0724.runtime import (  # noqa: E402
    audio_model_config,
    default_device,
    eeg_model_config,
    load_config,
    move_batch,
    resolve_config_path,
    resolve_evaluation_output,
    resolve_run_checkpoint,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate OpenVoice-EEG v0724")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("audio", "eeg-pretrain", "eeg", "evaluate"), required=True
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--generalization", choices=("g1", "g2", "g3"), default="g1")
    parser.add_argument("--holdout-label", default=None)
    parser.add_argument(
        "--loso-subject",
        default=None,
        help="Development-only LOSO using one subject from the locked training split",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--test-access-id", default=None)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Limit batches per epoch; diagnostic only",
    )
    return parser.parse_args()


class PairAwareBatchSampler(Sampler[list[int]]):
    """Ensure audio batches contain distinct utterances sharing content IDs."""

    def __init__(self, labels: Sequence[str], batch_size: int, seed: int):
        if batch_size < 2:
            raise ValueError("pair-aware batches require batch_size >= 2")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.groups[str(label)].append(index)
        self.pair_groups = [value for value in self.groups.values() if len(value) >= 2]
        if not self.pair_groups:
            raise ValueError(
                "audio training needs at least one content ID with two utterances"
            )
        self.size = len(labels)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, math.ceil(self.size / self.batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        all_indices = list(range(self.size))
        for _ in range(len(self)):
            batch: list[int] = []
            while len(batch) + 1 < self.batch_size:
                group = rng.choice(self.pair_groups)
                pair = rng.sample(group, 2)
                batch.extend(pair)
            while len(batch) < self.batch_size:
                batch.append(rng.choice(all_indices))
            rng.shuffle(batch)
            yield batch


class DeterministicPairBatchSampler(Sampler[list[int]]):
    """Cover validation audio once while keeping same-content pairs together."""

    def __init__(self, labels: Sequence[str], batch_size: int):
        if batch_size < 2:
            raise ValueError("pair-aware batches require batch_size >= 2")
        self.labels = tuple(map(str, labels))
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.labels) / self.batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        remaining = list(range(len(self.labels)))
        while remaining:
            anchor = remaining.pop(0)
            batch = [anchor]
            partner = next(
                (
                    position
                    for position, candidate in enumerate(remaining)
                    if self.labels[candidate] == self.labels[anchor]
                ),
                None,
            )
            if partner is not None:
                batch.append(remaining.pop(partner))
            take = min(self.batch_size - len(batch), len(remaining))
            batch.extend(remaining[:take])
            del remaining[:take]
            yield batch


class PairAwareEEGBatchSampler(Sampler[list[int]]):
    """Dataset-balanced batches with KaraOne realization hard negatives.

    Every available KaraOne anchor is paired with a different utterance of the
    same content label.  The remaining positions use dataset-balanced weights;
    FEIS mass is balanced over unique subject-label audio rather than repeated
    EEG rows.
    """

    def __init__(self, rows: Sequence[dict[str, str]], batch_size: int, seed: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.rows = tuple(rows)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.size = len(self.rows)
        counts: defaultdict[str, int] = defaultdict(int)
        reuse: defaultdict[str, int] = defaultdict(int)
        for row in self.rows:
            dataset = str(row["dataset"])
            counts[dataset] += 1
            if dataset == "feis":
                reuse[str(row["audio_key"])] += 1
        unique_feis = max(1, len(reuse))
        self.weights: list[float] = []
        by_label_audio: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, row in enumerate(self.rows):
            dataset = str(row["dataset"])
            if dataset == "feis":
                weight = 1.0 / (
                    float(unique_feis) * max(1, reuse[str(row["audio_key"])])
                )
            else:
                weight = 1.0 / max(1, counts[dataset])
            self.weights.append(weight)
            if dataset == "karaone":
                by_label_audio[normalize_label(str(row["label"]))][
                    str(row["audio_key"])
                ].append(index)
        self.hard_negative_groups = [
            audio_groups
            for audio_groups in by_label_audio.values()
            if len(audio_groups) >= 2
        ]

    def __len__(self) -> int:
        return max(1, math.ceil(self.size / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        population = list(range(self.size))
        for _ in range(len(self)):
            batch: list[int] = []
            if self.batch_size >= 2 and self.hard_negative_groups:
                audio_groups = rng.choice(self.hard_negative_groups)
                first_key, second_key = rng.sample(list(audio_groups), 2)
                batch.extend(
                    (
                        rng.choice(audio_groups[first_key]),
                        rng.choice(audio_groups[second_key]),
                    )
                )
            if len(batch) < self.batch_size:
                batch.extend(
                    rng.choices(
                        population,
                        weights=self.weights,
                        k=self.batch_size - len(batch),
                    )
                )
            rng.shuffle(batch)
            yield batch


class AudioFactorAdversaries(nn.Module):
    """Training-only probes; their metadata never enters the generator."""

    def __init__(self, dimension: int, speakers: int, labels: int):
        super().__init__()
        self.content_speaker = nn.Linear(dimension, max(1, speakers))
        self.timbre_label = nn.Linear(dimension, max(1, labels))

    def forward(
        self, content: torch.Tensor, timbre: torch.Tensor, strength: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.content_speaker(grad_reverse(content, strength)),
            self.timbre_label(grad_reverse(timbre, strength)),
        )


def safe_classification(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    selected = mask.bool() & (target >= 0) & (target < logits.shape[-1])
    return (
        F.cross_entropy(logits[selected], target[selected])
        if selected.any()
        else logits.sum() * 0.0
    )


def masked_cosine(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not mask.any():
        return first.sum() * 0.0
    loss = 1.0 - F.cosine_similarity(first[mask], second[mask], dim=-1)
    if weights is not None:
        selected_weights = weights[mask].to(loss).clamp_min(0.0)
        return (loss * selected_weights).sum() / selected_weights.sum().clamp_min(1e-8)
    return loss.mean()


def string_ids(
    values: Sequence[str],
    lookup: dict[str, int] | None = None,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    if lookup is None:
        unique = {
            value: index for index, value in enumerate(sorted(set(map(str, values))))
        }
    else:
        unique = lookup
    keys = [normalize_label(value) if normalize else str(value) for value in values]
    return torch.tensor([unique.get(key, -1) for key in keys], dtype=torch.long)


def aggregate_update(
    total: defaultdict[str, float], metrics: dict[str, Any], batch_size: int
) -> None:
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            number = float(value.detach().cpu())
        elif isinstance(value, (float, int)):
            number = float(value)
        else:
            continue
        if np.isfinite(number):
            total[key] += number * batch_size


def run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "generalization": str(args.generalization),
        "holdout_label": args.holdout_label,
        "loso_subject": args.loso_subject,
    }


def validate_run_metadata(
    payload: dict[str, Any], args: argparse.Namespace, source: str
) -> None:
    expected = run_metadata(args)
    observed = payload.get("run")
    if observed != expected:
        raise ValueError(
            f"{source} run metadata mismatch: observed={observed!r}, expected={expected!r}"
        )


def ablation_switches(cfg: dict[str, Any]) -> dict[str, float]:
    mode = str(cfg.get("experiment", {}).get("ablation", "full_v0724"))
    supported = {
        "full_v0724",
        "full_contentvec",
        "dual_token_no_structure",
        "dual_token_no_disentanglement",
        "content_only",
        "realization_only",
    }
    if mode not in supported:
        raise ValueError(f"Unsupported v0724 ablation: {mode}")
    return {
        "content": 0.0 if mode == "realization_only" else 1.0,
        "realization": 0.0 if mode == "content_only" else 1.0,
        "structure": 0.0 if mode == "dual_token_no_structure" else 1.0,
        "disentanglement": (0.0 if mode == "dual_token_no_disentanglement" else 1.0),
    }


def safe_code_valid(mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool().clone()
    if valid.ndim == 2:
        missing = ~valid.any(dim=1)
        valid[missing, 0] = True
    elif valid.ndim == 3:
        missing = ~valid.any(dim=(1, 2))
        valid[missing, :, 0] = True
    else:
        raise ValueError("code_valid_mask must be [B,T] or [B,Q,T]")
    return valid


def expand_code_valid(mask: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Normalize either supported cache mask shape to ``[B,Q,T]``."""

    valid = safe_code_valid(mask)
    if valid.ndim == 2:
        valid = valid[:, None, :].expand_as(codes)
    if valid.shape != codes.shape:
        raise ValueError("code_valid_mask is incompatible with codes")
    return valid


def deterministic_patch_mask(valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """Stable validation mask that never depends on global RNG state."""

    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must be [B,C,P]")
    output = torch.zeros_like(valid_mask, dtype=torch.bool)
    for item in range(len(valid_mask)):
        available = torch.nonzero(
            valid_mask[item].reshape(-1), as_tuple=False
        ).flatten()
        if not len(available):
            continue
        count = max(1, min(len(available), round(len(available) * float(ratio))))
        positions = (
            torch.linspace(0, len(available) - 1, count, device=available.device)
            .round()
            .long()
        )
        output[item].view(-1)[available[positions]] = True
    return output


def audio_objective(
    model: FactorizedAudioModel,
    adversaries: AudioFactorAdversaries,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    speaker_lookup: dict[str, int],
    label_lookup: dict[str, int],
    adversary_strength: float,
    stochastic_mask: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    train_cfg, weights = cfg["training"], cfg["loss"]
    switches = ablation_switches(cfg)
    code_valid = expand_code_valid(batch["code_valid_mask"], batch["codes"])
    code_mask = (
        random_code_mask(
            batch["codes"],
            min_ratio=float(train_cfg["mask_ratio_min"]),
            max_ratio=float(train_cfg["mask_ratio_max"]),
            full_mask_probability=float(train_cfg["full_mask_probability"]),
            code_valid_mask=code_valid,
        )
        if stochastic_mask
        else code_valid.clone()
    )
    output = model(
        batch["content_tokens"],
        batch["content_token_mask"],
        batch["realization_features"],
        batch["realization_frame_mask"],
        batch["timbre_global"],
        batch["codes"],
        code_mask,
        code_valid,
    )
    state = output.state
    reconstruction = batch["code_supervision"].bool() & batch["has_codec"].bool()
    code = code_cross_entropy(
        output.code_logits,
        batch["codes"],
        code_mask & code_valid,
        torch.tensor(weights["codebook_weights"], device=output.code_logits.device),
        sample_mask=reconstruction,
    )
    if switches["structure"]:
        energy = energy_structure_loss(
            state.log_mel_energy,
            batch["log_mel_energy"],
            batch["realization_frame_mask"],
            batch["energy_supervision"].bool(),
            target_activity=batch["activity_mask"],
            l1_weight=float(weights["mel_l1"]),
            soft_dtw_weight=float(weights["mel_soft_dtw"]),
            gamma=float(weights["soft_dtw_gamma"]),
            band_ratio=float(weights["soft_dtw_band_fraction"]),
            soft_dtw_max_frames=int(weights["soft_dtw_train_frames"]),
        )
    else:
        zero_energy = state.log_mel_energy.sum() * 0.0
        energy = {
            "total": zero_energy,
            "log_mel_l1": zero_energy.detach(),
            "soft_dtw": zero_energy.detach(),
            "ssim": zero_energy.detach(),
        }
    prosody = prosody_activity_duration_loss(
        state.log_f0_hz,
        batch["f0_log_hz"],
        state.voicing_logits,
        batch["voicing"],
        state.log_rms_dbfs,
        batch["log_rms_dbfs"],
        state.activity_logits,
        batch["activity_mask"],
        state.duration_seconds,
        batch["duration_seconds"],
        batch["realization_frame_mask"],
        batch["realization_supervision"].bool(),
    )
    label_ids = string_ids(batch["content_id"], label_lookup, normalize=True).to(
        state.content_global.device
    )
    content_mask = batch["content_supervision"].bool()
    positives = content_positive_weights(
        label_ids, content_mask, weak_positive_weight=1.0
    )
    positives.fill_diagonal_(0.0)
    allowed = ~torch.eye(len(positives), dtype=torch.bool, device=positives.device)
    content_clip = masked_symmetric_multi_positive_clip_loss(
        state.content_global,
        state.content_global,
        positives,
        eeg_eligible=content_mask,
        audio_eligible=content_mask,
        allowed=allowed,
        temperature=float(weights["contrastive_temperature"]),
    )["total"]
    speaker_ids = string_ids(batch["audio_speaker_id"], speaker_lookup).to(
        state.content_global.device
    )
    speaker_logits, timbre_label_logits = adversaries(
        state.content_global, state.timbre_global, adversary_strength
    )
    speaker_loss = safe_classification(speaker_logits, speaker_ids, speaker_ids >= 0)
    label_loss = safe_classification(timbre_label_logits, label_ids, content_mask)
    covariance = cross_covariance_loss(
        state.content_global, state.timbre_global, content_mask
    )
    total = (
        float(weights["audio_code"]) * code["total"]
        + switches["structure"] * float(weights["audio_mel"]) * energy["total"]
        + float(weights["audio_prosody"]) * prosody["total"]
        + switches["content"] * float(weights["content_clip"]) * content_clip
        + switches["disentanglement"]
        * switches["content"]
        * switches["realization"]
        * float(weights["cross_covariance"])
        * covariance
        + switches["disentanglement"]
        * switches["content"]
        * float(weights["subject_adversarial"])
        * speaker_loss
        + switches["disentanglement"]
        * switches["realization"]
        * float(weights["timbre_label_adversarial"])
        * label_loss
    )
    return total, {
        "loss": float(total.detach()),
        "code": float(code["total"].detach()),
        "mel": float(energy["log_mel_l1"]),
        "mel_soft_dtw": float(energy["soft_dtw"]),
        "prosody": float(prosody["total"].detach()),
        "content_clip": float(content_clip.detach()),
        "cross_covariance": float(covariance.detach()),
        "speaker_adversary": float(speaker_loss.detach()),
        "timbre_label_adversary": float(label_loss.detach()),
        "q0_accuracy": float(code["q0_accuracy"]),
    }


@torch.no_grad()
def encode_audio_targets(
    model: FactorizedAudioModel, batch: dict[str, Any]
) -> dict[str, torch.Tensor]:
    routed = (
        batch["content_supervision"].bool()
        | batch["realization_supervision"].bool()
        | batch["timbre_supervision"].bool()
        | batch["energy_supervision"].bool()
        | batch["code_supervision"].bool()
    )
    eligible = (
        batch["has_audio_teacher"].bool()
        & batch["content_token_mask"].any(dim=1)
        & routed
    )
    batch_size = len(eligible)
    cfg = model.cfg
    device = batch["content_tokens"].device
    output = {
        "content_tokens": torch.zeros(
            batch_size, cfg.condition_steps, cfg.d_model, device=device
        ),
        "realization_tokens": torch.zeros(
            batch_size, cfg.condition_steps, cfg.d_model, device=device
        ),
        "content_global": torch.zeros(batch_size, cfg.d_model, device=device),
        "realization_global": torch.zeros(batch_size, cfg.d_model, device=device),
        "timbre_global": torch.zeros(batch_size, cfg.d_model, device=device),
    }
    if eligible.any():
        index = torch.nonzero(eligible, as_tuple=False).flatten()
        state = model.encode(
            batch["content_tokens"][index],
            batch["content_token_mask"][index],
            batch["realization_features"][index],
            batch["realization_frame_mask"][index],
            batch["timbre_global"][index],
        )
        for name in output:
            output[name].index_copy_(0, index, getattr(state, name))
    output["eligible"] = eligible
    return output


def eeg_objective(
    eeg: FactorizedEEGEncoder,
    audio: FactorizedAudioModel | None,
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    epoch: int,
    adversary_strength: float,
    pretrain: bool,
    augment: bool,
    stochastic_mask: bool = True,
) -> tuple[torch.Tensor, dict[str, float], Any, dict[str, torch.Tensor] | None]:
    weights = cfg["loss"]
    switches = ablation_switches(cfg)
    with torch.no_grad():
        _, patch_valid = eeg._patches(
            batch["eeg"], batch["channel_mask"], batch["time_mask"]
        )
    if stochastic_mask:
        patch_mask: torch.Tensor | None = random_patch_mask(
            patch_valid, float(cfg["training"]["patch_mask_ratio"])
        )
    elif pretrain:
        patch_mask = deterministic_patch_mask(
            patch_valid, float(cfg["training"]["patch_mask_ratio"])
        )
    else:
        # Paired validation/retrieval must use the clean EEG representation.
        patch_mask = None
    state = eeg(
        batch["eeg"],
        batch["channel_xyz"],
        batch["channel_mask"],
        batch["time_mask"],
        epoch=epoch,
        patch_mask=patch_mask,
        adversary_strength=adversary_strength,
    )
    patch_loss = masked_patch_reconstruction_loss(
        state.patch_reconstruction, state.patch_target, state.patch_mask
    )
    subject = safe_classification(
        state.subject_logits, batch["subject_idx"], batch["subject_idx"] >= 0
    )
    dataset = F.cross_entropy(state.dataset_logits, batch["dataset_idx"])
    router_dataset = F.cross_entropy(state.router_dataset_logits, batch["dataset_idx"])
    timbre_label = safe_classification(
        state.timbre_label_logits,
        batch["label_idx"],
        batch["timbre_supervision"].bool() & batch["has_timbre"].bool(),
    )
    moe = moe_regularization(dict(state.router))
    consistency = state.content_global.sum() * 0.0
    if augment:
        probability = random.choice(list(cfg["training"]["channel_drop_probabilities"]))
        view = stochastic_channel_view(
            batch,
            drop_probability=float(probability),
            coordinate_noise_std=float(cfg["training"]["coordinate_noise_std"]),
            noise_std=float(cfg["training"]["signal_noise_std"]),
        )
        second = eeg(
            view["eeg"],
            view["channel_xyz"],
            view["channel_mask"],
            view["time_mask"],
            epoch=epoch,
            adversary_strength=adversary_strength,
        )
        consistency = 0.5 * (
            condition_consistency_loss(state.content_global, second.content_global)
            + condition_consistency_loss(
                state.realization_global, second.realization_global
            )
        )

    base_total = (
        float(weights["eeg_masked_pretraining"]) * patch_loss
        + float(weights["channel_consistency"]) * consistency
        + switches["disentanglement"]
        * switches["content"]
        * float(weights["subject_adversarial"])
        * subject
        + float(weights["dataset_adversarial"]) * 0.5 * (dataset + router_dataset)
        + switches["disentanglement"]
        * switches["realization"]
        * float(weights["timbre_label_adversarial"])
        * timbre_label
        + float(weights["moe"]) * moe
    )
    metrics: dict[str, float] = {
        "loss": float(base_total.detach()),
        "patch": float(patch_loss.detach()),
        "channel_consistency": float(consistency.detach()),
        "subject_adversary": float(subject.detach()),
        "dataset_adversary": float(dataset.detach()),
        "timbre_label_adversary": float(timbre_label.detach()),
        "moe": float(moe.detach()),
    }
    if pretrain:
        return base_total, metrics, state, None
    if audio is None:
        raise ValueError("paired EEG training requires the frozen audio model")
    targets = encode_audio_targets(audio, batch)
    content_mask = batch["content_supervision"].bool() & targets["eligible"]
    exact_mask = batch["exact_pair_supervision"].bool() & targets["eligible"]
    positives = content_positive_weights(
        batch["label_idx"], content_mask, exact_pair_mask=exact_mask
    )
    content_clip = masked_symmetric_multi_positive_clip_loss(
        state.content_global,
        targets["content_global"],
        positives,
        eeg_eligible=content_mask,
        audio_eligible=content_mask,
        temperature=float(weights["contrastive_temperature"]),
    )["total"]
    realization_clip = exact_realization_clip_loss(
        state.realization_global,
        targets["realization_global"],
        batch["realization_supervision"].bool() & targets["eligible"],
        temperature=float(weights["contrastive_temperature"]),
    )["total"]
    exact_timbre_mask = (
        batch["timbre_supervision"].bool()
        & batch["exact_pair_supervision"].bool()
        & batch["has_timbre"].bool()
        & targets["eligible"]
    )
    exact_timbre = masked_cosine(
        state.timbre_global, targets["timbre_global"], exact_timbre_mask
    )
    content_local = monotonic_local_alignment_loss(
        state.content_tokens,
        targets["content_tokens"],
        content_mask,
        temperature=float(weights["contrastive_temperature"]),
    )["total"]
    realization_local = monotonic_local_alignment_loss(
        state.realization_tokens,
        targets["realization_tokens"],
        batch["realization_supervision"].bool(),
        temperature=float(weights["contrastive_temperature"]),
    )["total"]
    if switches["structure"]:
        energy = energy_structure_loss(
            state.log_mel_energy,
            batch["log_mel_energy"],
            batch["realization_frame_mask"],
            batch["energy_supervision"].bool(),
            target_activity=batch["activity_mask"],
            l1_weight=float(weights["mel_l1"]),
            soft_dtw_weight=float(weights["mel_soft_dtw"]),
            gamma=float(weights["soft_dtw_gamma"]),
            band_ratio=float(weights["soft_dtw_band_fraction"]),
            soft_dtw_max_frames=int(weights["soft_dtw_train_frames"]),
        )
    else:
        zero_energy = state.log_mel_energy.sum() * 0.0
        energy = {
            "total": zero_energy,
            "log_mel_l1": zero_energy.detach(),
            "soft_dtw": zero_energy.detach(),
            "ssim": zero_energy.detach(),
        }
    prosody = prosody_activity_duration_loss(
        state.log_f0_hz,
        batch["f0_log_hz"],
        state.voicing_logits,
        batch["voicing"],
        state.log_rms_dbfs,
        batch["log_rms_dbfs"],
        state.activity_logits,
        batch["activity_mask"],
        state.duration_seconds,
        batch["duration_seconds"],
        batch["realization_frame_mask"],
        batch["realization_supervision"].bool(),
    )
    safe_valid = expand_code_valid(batch["code_valid_mask"], batch["codes"])
    code_sample_mask = batch["code_supervision"].bool() & batch["has_codec"].bool()
    if code_sample_mask.any():
        code_mask = (
            random_code_mask(
                batch["codes"],
                min_ratio=float(cfg["training"]["mask_ratio_min"]),
                max_ratio=float(cfg["training"]["mask_ratio_max"]),
                full_mask_probability=float(cfg["training"]["full_mask_probability"]),
                code_valid_mask=safe_valid,
            )
            if stochastic_mask
            else safe_valid.clone()
        )
        code_logits = audio.decoder(
            batch["codes"], code_mask, state.fused_condition, safe_valid
        )
        code = code_cross_entropy(
            code_logits,
            batch["codes"],
            code_mask & safe_valid,
            torch.tensor(weights["codebook_weights"], device=code_logits.device),
            sample_mask=code_sample_mask,
        )
    else:
        zero = state.fused_condition.sum() * 0.0
        code = {"total": zero, "q0_accuracy": zero.detach()}
    feis = batch["feis_prototype_supervision"].bool() & batch["has_timbre"].bool()
    timbre_prototype = masked_cosine(
        state.timbre_global,
        targets["timbre_global"],
        feis,
        None if stochastic_mask else batch["feis_audio_weight"],
    )
    covariance = cross_covariance_loss(
        state.content_global, state.timbre_global, content_mask
    )
    total = (
        base_total
        + switches["content"] * float(weights["content_clip"]) * content_clip
        + switches["realization"]
        * float(weights["realization_clip"])
        * 0.5
        * (realization_clip + exact_timbre)
        + float(weights["realization_local"])
        * 0.5
        * (
            switches["content"] * content_local
            + switches["realization"] * realization_local
        )
        + switches["structure"] * energy["total"]
        + float(weights["activity_duration_prosody"]) * prosody["total"]
        + float(weights["eeg_code"]) * code["total"]
        + switches["realization"]
        * float(weights["feis_timbre_prototype"])
        * timbre_prototype
        + switches["disentanglement"]
        * switches["content"]
        * switches["realization"]
        * float(weights["cross_covariance"])
        * covariance
    )
    metrics.update(
        {
            "loss": float(total.detach()),
            "content_clip": float(content_clip.detach()),
            "realization_clip": float(realization_clip.detach()),
            "content_local": float(content_local.detach()),
            "exact_timbre": float(exact_timbre.detach()),
            "realization_local": float(realization_local.detach()),
            "mel": float(energy["log_mel_l1"]),
            "mel_soft_dtw": float(energy["soft_dtw"]),
            "prosody": float(prosody["total"].detach()),
            "code": float(code["total"].detach()),
            "q0_accuracy": float(code["q0_accuracy"]),
            "feis_timbre": float(timbre_prototype.detach()),
            "cross_covariance": float(covariance.detach()),
        }
    )
    return total, metrics, state, targets


def audio_loader(
    dataset: FactorizedAudioDataset,
    batch_size: int,
    workers: int,
    seed: int,
    *,
    train: bool,
) -> tuple[DataLoader[Any], PairAwareBatchSampler | None]:
    labels = [
        normalize_label(str(dataset.teachers.metadata(key)["content_id"]))
        for key in dataset.records
    ]
    if train:
        sampler = PairAwareBatchSampler(labels, batch_size, seed)
        return (
            DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate_factorized,
                num_workers=workers,
            ),
            sampler,
        )
    validation_sampler = DeterministicPairBatchSampler(labels, batch_size)
    return (
        DataLoader(
            dataset,
            batch_sampler=validation_sampler,
            collate_fn=collate_factorized,
            num_workers=workers,
        ),
        None,
    )


def eeg_loader(
    dataset: FactorizedEEGDataset,
    batch_size: int,
    workers: int,
    *,
    train: bool,
    seed: int = 0,
) -> tuple[DataLoader[Any], PairAwareEEGBatchSampler | None]:
    if not train:
        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_factorized,
                num_workers=workers,
            ),
            None,
        )
    sampler = PairAwareEEGBatchSampler(dataset.rows, batch_size, seed)
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_factorized,
            num_workers=workers,
        ),
        sampler,
    )


@torch.no_grad()
def validate_audio(
    model: FactorizedAudioModel,
    adversaries: AudioFactorAdversaries,
    loader: DataLoader[Any],
    cfg: dict[str, Any],
    device: torch.device,
    speaker_lookup: dict[str, int],
    label_lookup: dict[str, int],
    smoke_steps: int,
) -> dict[str, float]:
    model.eval()
    adversaries.eval()
    total: defaultdict[str, float] = defaultdict(float)
    count = 0
    for step, raw in enumerate(loader):
        if smoke_steps and step >= smoke_steps:
            break
        batch = move_batch(raw, device)
        _, metrics = audio_objective(
            model,
            adversaries,
            batch,
            cfg,
            speaker_lookup=speaker_lookup,
            label_lookup=label_lookup,
            adversary_strength=0.0,
            stochastic_mask=False,
        )
        aggregate_update(total, metrics, len(batch["codes"]))
        count += len(batch["codes"])
    return {key: value / max(count, 1) for key, value in total.items()}


@torch.no_grad()
def validate_eeg(
    eeg: FactorizedEEGEncoder,
    audio: FactorizedAudioModel | None,
    loader: DataLoader[Any],
    cfg: dict[str, Any],
    device: torch.device,
    *,
    epoch: int,
    pretrain: bool,
    smoke_steps: int,
) -> dict[str, float]:
    eeg.eval()
    if audio is not None:
        audio.eval()
    total: defaultdict[str, float] = defaultdict(float)
    count = 0
    eeg_content, audio_content, labels = [], [], []
    for step, raw in enumerate(loader):
        if smoke_steps and step >= smoke_steps:
            break
        batch = move_batch(raw, device)
        _, metrics, state, targets = eeg_objective(
            eeg,
            audio,
            batch,
            cfg,
            epoch=epoch,
            adversary_strength=0.0,
            pretrain=pretrain,
            augment=False,
            stochastic_mask=False,
        )
        aggregate_update(total, metrics, len(batch["eeg"]))
        count += len(batch["eeg"])
        if targets is not None:
            selected = batch["content_supervision"].bool() & targets["eligible"]
            if selected.any():
                eeg_content.append(state.content_global[selected].cpu())
                audio_content.append(targets["content_global"][selected].cpu())
                labels.append(batch["label_idx"][selected].cpu())
    result = {key: value / max(count, 1) for key, value in total.items()}
    if eeg_content:
        first = F.normalize(torch.cat(eeg_content), dim=-1)
        second = F.normalize(torch.cat(audio_content), dim=-1)
        target = torch.cat(labels)
        retrieved = target[(first @ second.T).argmax(dim=1)]
        per_label = [
            (retrieved[target == label] == label).float().mean()
            for label in target.unique()
        ]
        result["content_retrieval_macro_top1"] = float(torch.stack(per_label).mean())
        result["content_retrieval_balanced_chance"] = 1.0 / max(
            1, int(target.unique().numel())
        )
    return result


def load_audio_model(
    path: Path, cfg: dict[str, Any], lineage: dict[str, Any], device: torch.device
) -> tuple[FactorizedAudioModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, phase="audio", lineage=lineage, source=str(path))
    model = FactorizedAudioModel(audio_model_config(cfg))
    model.load_state_dict(payload["model_state"])
    return model.to(device), payload


def train_audio(
    args: argparse.Namespace,
    context: Any,
    teachers: TeacherCacheV2,
    lineage: dict[str, Any],
    device: torch.device,
) -> None:
    cfg = context.config
    train_cfg = cfg["training"]
    train_set = FactorizedAudioDataset(context, teachers, split="train")
    validation_set = FactorizedAudioDataset(context, teachers, split="validation")
    train_loader, pair_sampler = audio_loader(
        train_set,
        int(train_cfg["audio_batch_size"]),
        int(train_cfg["num_workers"]),
        int(args.seed),
        train=True,
    )
    validation_loader, _ = audio_loader(
        validation_set,
        int(train_cfg["audio_batch_size"]),
        int(train_cfg["num_workers"]),
        int(args.seed),
        train=False,
    )
    model = FactorizedAudioModel(audio_model_config(cfg)).to(device)
    adversaries = AudioFactorAdversaries(
        model.cfg.d_model, len(teachers.speaker_to_index), len(context.label_to_index)
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(adversaries.parameters()),
        lr=float(train_cfg["audio_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    start = 0
    best = float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint(
            payload, phase="audio", lineage=lineage, source=str(args.resume)
        )
        validate_run_metadata(payload, args, str(args.resume))
        model.load_state_dict(payload["model_state"])
        adversaries.load_state_dict(payload["audio_adversaries_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start = int(payload["epoch"]) + 1
    checkpoint = resolve_config_path(
        context.config_path, cfg["paths"]["audio_checkpoint"]
    )
    metrics_path = checkpoint.parent.parent / "metrics" / "training.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs or train_cfg["audio_epochs"])
    for epoch in range(start, epochs):
        model.train()
        adversaries.train()
        assert pair_sampler is not None
        pair_sampler.set_epoch(epoch)
        running: defaultdict[str, float] = defaultdict(float)
        count = 0
        progress = tqdm(
            train_loader, desc=f"[0724 audio] epoch {epoch + 1}/{epochs}", unit="batch"
        )
        for step, raw in enumerate(progress):
            if args.smoke_steps and step >= args.smoke_steps:
                break
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            strength = float(train_cfg["adversary_strength_max"]) * min(
                1.0, (epoch + 1) / max(1, epochs // 3)
            )
            loss, metrics = audio_objective(
                model,
                adversaries,
                batch,
                cfg,
                speaker_lookup=teachers.speaker_to_index,
                label_lookup=context.label_to_index,
                adversary_strength=strength,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adversaries.parameters()),
                float(train_cfg["grad_clip"]),
            )
            optimizer.step()
            aggregate_update(running, metrics, len(batch["codes"]))
            count += len(batch["codes"])
            progress.set_postfix(loss=f"{metrics['loss']:.3f}")
        validation = validate_audio(
            model,
            adversaries,
            validation_loader,
            cfg,
            device,
            teachers.speaker_to_index,
            context.label_to_index,
            args.smoke_steps,
        )
        summary = {
            "epoch": epoch,
            "train": {k: v / max(count, 1) for k, v in running.items()},
            "validation": validation,
            "smoke_steps": args.smoke_steps,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        if validation.get("loss", float("inf")) < best:
            best = validation["loss"]
            payload = checkpoint_payload(
                phase="audio",
                lineage=lineage,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                metrics=summary,
            )
            payload["audio_adversaries_state"] = adversaries.state_dict()
            payload["diagnostic_smoke"] = bool(args.smoke_steps)
            payload["run"] = run_metadata(args)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint)


def train_eeg(
    args: argparse.Namespace,
    context: Any,
    teachers: TeacherCacheV2,
    lineage: dict[str, Any],
    device: torch.device,
    *,
    pretrain: bool,
) -> None:
    cfg = context.config
    train_cfg = cfg["training"]
    audio_path = resolve_config_path(
        context.config_path, cfg["paths"]["audio_checkpoint"]
    )
    require_frozen_audio_checkpoint(context.config_path, cfg, lineage, audio_path)
    audio_checkpoint_sha256 = file_sha256(audio_path)
    audio = None
    dependencies: dict[str, str] = {"audio_checkpoint_sha256": audio_checkpoint_sha256}
    if not pretrain:
        audio, _ = load_audio_model(audio_path, cfg, lineage, device)
        audio.eval()
        for parameter in audio.parameters():
            parameter.requires_grad_(False)
    eeg = FactorizedEEGEncoder(
        eeg_model_config(
            cfg,
            num_train_subjects=len(context.subject_to_index),
            num_content_labels=len(context.label_to_index),
        )
    ).to(device)
    pretrain_path = resolve_run_checkpoint(
        context.config_path,
        cfg,
        "eeg_pretrain_checkpoint",
        seed=int(args.seed),
        loso_subject=args.loso_subject,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
    )
    if not pretrain:
        if not pretrain_path.is_file():
            raise FileNotFoundError(
                "Run the v0724 EEG masked-pretraining phase before paired EEG training: "
                f"{pretrain_path}"
            )
        pretrained = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        validate_checkpoint(
            pretrained,
            phase="eeg-pretrain",
            lineage=lineage,
            dependencies={
                "audio_checkpoint_sha256": audio_checkpoint_sha256,
            },
            source=str(pretrain_path),
        )
        validate_run_metadata(pretrained, args, str(pretrain_path))
        if not args.resume:
            eeg.load_state_dict(pretrained["model_state"])
        dependencies["eeg_pretrain_checkpoint_sha256"] = file_sha256(pretrain_path)
    optimizer = torch.optim.AdamW(
        eeg.parameters(),
        lr=float(train_cfg["eeg_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    phase = "eeg-pretrain" if pretrain else "eeg"
    checkpoint = (
        pretrain_path
        if pretrain
        else resolve_run_checkpoint(
            context.config_path,
            cfg,
            "eeg_checkpoint",
            seed=int(args.seed),
            loso_subject=args.loso_subject,
            generalization=args.generalization,
            holdout_label=args.holdout_label,
        )
    )
    start = 0
    best = float("inf")
    patience = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint(
            payload,
            phase=phase,
            lineage=lineage,
            dependencies=dependencies,
            source=str(args.resume),
        )
        validate_run_metadata(payload, args, str(args.resume))
        eeg.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start = int(payload["epoch"]) + 1
    train_set = FactorizedEEGDataset(
        context,
        teachers,
        split="train",
        generalization=args.generalization,
        holdout_label=args.holdout_label,
        loso_subject=args.loso_subject,
    )
    validation_set = FactorizedEEGDataset(
        context,
        teachers,
        split="validation",
        generalization=args.generalization,
        holdout_label=args.holdout_label,
        loso_subject=args.loso_subject,
    )
    train_loader, train_sampler = eeg_loader(
        train_set,
        int(train_cfg["eeg_batch_size"]),
        int(train_cfg["num_workers"]),
        train=True,
        seed=int(args.seed),
    )
    validation_loader, _ = eeg_loader(
        validation_set,
        int(train_cfg["eeg_batch_size"]),
        int(train_cfg["num_workers"]),
        train=False,
    )
    epochs = int(
        args.epochs
        or (train_cfg["eeg_pretrain_epochs"] if pretrain else train_cfg["eeg_epochs"])
    )
    metrics_path = checkpoint.parent.parent / "metrics" / "training.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, epochs):
        eeg.train()
        running: defaultdict[str, float] = defaultdict(float)
        count = 0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        strength = float(train_cfg["adversary_strength_max"]) * min(
            1.0, (epoch + 1) / max(1, epochs // 3)
        )
        progress = tqdm(
            train_loader,
            desc=f"[0724 {phase}] epoch {epoch + 1}/{epochs}",
            unit="batch",
        )
        for step, raw in enumerate(progress):
            if args.smoke_steps and step >= args.smoke_steps:
                break
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, _, _ = eeg_objective(
                eeg,
                audio,
                batch,
                cfg,
                epoch=epoch,
                adversary_strength=strength,
                pretrain=pretrain,
                augment=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                eeg.parameters(), float(train_cfg["grad_clip"])
            )
            optimizer.step()
            aggregate_update(running, metrics, len(batch["eeg"]))
            count += len(batch["eeg"])
            progress.set_postfix(loss=f"{metrics['loss']:.3f}")
        validation = validate_eeg(
            eeg,
            audio,
            validation_loader,
            cfg,
            device,
            epoch=epoch,
            pretrain=pretrain,
            smoke_steps=args.smoke_steps,
        )
        summary = {
            "epoch": epoch,
            "phase": phase,
            "train": {k: v / max(count, 1) for k, v in running.items()},
            "validation": validation,
            "smoke_steps": args.smoke_steps,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        score = validation.get("loss", float("inf"))
        if score < best:
            best = score
            patience = 0
            payload = checkpoint_payload(
                phase=phase,
                lineage=lineage,
                model_state=eeg.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                metrics=summary,
                dependencies=dependencies,
            )
            payload["diagnostic_smoke"] = bool(args.smoke_steps)
            payload["run"] = run_metadata(args)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint)
        else:
            patience += 1
            if patience >= int(train_cfg["early_stopping_patience"]):
                break


def expected_eeg_dependencies(
    payload: dict[str, Any],
    cfg: dict[str, Any],
    config_path: Path,
    *,
    seed: int,
    loso_subject: str | None,
    generalization: str,
    holdout_label: str | None,
) -> dict[str, str]:
    expected: dict[str, str] = {}
    declared = payload.get("dependencies") or {}
    if "audio_checkpoint_sha256" in declared:
        expected["audio_checkpoint_sha256"] = file_sha256(
            resolve_config_path(config_path, cfg["paths"]["audio_checkpoint"])
        )
    if "eeg_pretrain_checkpoint_sha256" in declared:
        expected["eeg_pretrain_checkpoint_sha256"] = file_sha256(
            resolve_run_checkpoint(
                config_path,
                cfg,
                "eeg_pretrain_checkpoint",
                seed=seed,
                loso_subject=loso_subject,
                generalization=generalization,
                holdout_label=holdout_label,
            )
        )
    return expected


def evaluate(
    args: argparse.Namespace,
    context: Any,
    teachers: TeacherCacheV2,
    lineage: dict[str, Any],
    device: torch.device,
) -> None:
    cfg = context.config
    audio_path = resolve_config_path(
        context.config_path, cfg["paths"]["audio_checkpoint"]
    )
    eeg_path = resolve_run_checkpoint(
        context.config_path,
        cfg,
        "eeg_checkpoint",
        seed=int(args.seed),
        loso_subject=args.loso_subject,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
    )
    require_frozen_audio_checkpoint(context.config_path, cfg, lineage, audio_path)
    if args.split == "test":
        authorize_locked_test(
            resolve_config_path(context.config_path, cfg["paths"]["validation_gate"]),
            lineage=lineage,
            audio_checkpoint=audio_path,
            eeg_checkpoint=eeg_path,
        )
    audio, _ = load_audio_model(audio_path, cfg, lineage, device)
    audio.eval()
    payload = torch.load(eeg_path, map_location="cpu", weights_only=False)
    validate_run_metadata(payload, args, str(eeg_path))
    dependencies = expected_eeg_dependencies(
        payload,
        cfg,
        context.config_path,
        seed=int(args.seed),
        loso_subject=args.loso_subject,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
    )
    validate_checkpoint(
        payload,
        phase="eeg",
        lineage=lineage,
        dependencies=dependencies,
        source=str(eeg_path),
    )
    eeg = FactorizedEEGEncoder(
        eeg_model_config(
            cfg,
            num_train_subjects=len(context.subject_to_index),
            num_content_labels=len(context.label_to_index),
        )
    ).to(device)
    eeg.load_state_dict(payload["model_state"])
    eeg.eval()
    dataset = FactorizedEEGDataset(
        context,
        teachers,
        split=args.split,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
        loso_subject=args.loso_subject,
        allow_locked_test=args.split == "test",
    )
    loader, _ = eeg_loader(
        dataset,
        int(cfg["training"]["eeg_batch_size"]),
        int(cfg["training"]["num_workers"]),
        train=False,
    )
    metrics = validate_eeg(
        eeg,
        audio,
        loader,
        cfg,
        device,
        epoch=int(payload["epoch"]),
        pretrain=False,
        smoke_steps=args.smoke_steps,
    )
    output = resolve_evaluation_output(
        context.config_path,
        cfg,
        split=args.split,
        seed=int(args.seed),
        loso_subject=args.loso_subject,
        generalization=args.generalization,
        holdout_label=args.holdout_label,
    )
    write_json(
        output,
        {
            "schema_version": "openvoice-0724-latent-evaluation-v1",
            "split": args.split,
            "generalization": args.generalization,
            "holdout_label": args.holdout_label,
            "loso_subject": args.loso_subject,
            "metrics": metrics,
            "lineage": lineage,
            "audio_checkpoint_sha256": file_sha256(audio_path),
            "eeg_checkpoint_sha256": file_sha256(eeg_path),
            "diagnostic_smoke": bool(args.smoke_steps),
            "run": run_metadata(args),
            "test_accessed": args.split == "test",
        },
    )
    print(
        json.dumps(
            {"output": str(output), "metrics": metrics}, indent=2, sort_keys=True
        )
    )


def main() -> None:
    args = parse_args()
    config_path, raw_cfg = load_config(args.config)
    if args.phase == "audio" and args.loso_subject is not None:
        raise ValueError("--loso-subject applies only to EEG phases")
    if (
        args.phase == "audio"
        and raw_cfg.get("experiment", {}).get("audio_prior") == "shared_primary_frozen"
    ):
        raise ValueError(
            "This same-parameter-count ablation reuses the passed primary "
            "audio oracle; do not overwrite it with an ablation run"
        )
    requested_seed = int(
        raw_cfg["training"]["seed"] if args.seed is None else args.seed
    )
    if args.phase == "audio" and requested_seed != int(raw_cfg["training"]["seed"]):
        raise ValueError(
            "The frozen audio prior is trained once with the primary seed; "
            "secondary seeds apply to EEG phases"
        )
    if args.phase == "evaluate" and args.split == "test":
        if args.smoke_steps:
            raise PermissionError("The one-shot locked test cannot use --smoke-steps")
        if (
            args.loso_subject is not None
            or requested_seed != int(raw_cfg["training"]["seed"])
            or args.generalization != "g1"
            or args.holdout_label is not None
        ):
            raise PermissionError(
                "Locked test is only available to the preregistered primary seed; "
                "LOSO, held-label, and secondary-seed runs are development-only"
            )
        if not args.allow_final_test:
            raise PermissionError("Locked test requires --allow-final-test")
        authorize_locked_test_metadata(
            resolve_config_path(config_path, raw_cfg["paths"]["validation_gate"]),
            config_path=config_path,
            audio_checkpoint=resolve_config_path(
                config_path, raw_cfg["paths"]["audio_checkpoint"]
            ),
            eeg_checkpoint=resolve_config_path(
                config_path, raw_cfg["paths"]["eeg_checkpoint"]
            ),
        )
        claim_locked_test_access(
            resolve_config_path(config_path, raw_cfg["paths"]["validation_gate"]),
            purpose="latent_evaluation",
            access_id=args.test_access_id,
        )
    context = load_context(config_path)
    seed = int(
        args.seed if args.seed is not None else context.config["training"]["seed"]
    )
    args.seed = seed
    seed_everything(seed)
    device = torch.device(args.device) if args.device else default_device()
    teachers = TeacherCacheV2(
        resolve_config_path(config_path, context.config["paths"]["teacher_cache"])
    )
    lineage = build_lineage(context)
    print(
        f"[openvoice-0724] phase={args.phase}; device={device}; seed={seed}; cache={len(teachers)}"
    )
    if args.phase == "audio":
        train_audio(args, context, teachers, lineage, device)
    elif args.phase == "eeg-pretrain":
        train_eeg(args, context, teachers, lineage, device, pretrain=True)
    elif args.phase == "eeg":
        train_eeg(args, context, teachers, lineage, device, pretrain=False)
    else:
        evaluate(args, context, teachers, lineage, device)


if __name__ == "__main__":
    main()
