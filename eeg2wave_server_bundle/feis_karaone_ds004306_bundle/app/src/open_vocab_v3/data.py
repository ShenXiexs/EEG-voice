from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.fft import dct
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.open_vocab_0724.audio_features import (
    AcousticFeatureConfig,
    AudioPreparationConfig,
    prepare_waveform_segment,
    extract_acoustic_features,
)


SOURCE_SPLITS = ("train", "validation", "locked_test", "diagnostic")
PREPARATION_SCHEMA = "openvoice-v3-mfcc-preparation-v4-cvae-denoise-cmvn-256"
PAIR_ROLES = (
    "fit",
    "subject_holdout_seen",
    "label_holdout_seen_subject",
    "subject_and_label_holdout",
)
FINAL_TEST_ROLES = (
    "locked_test_seen_label",
    "locked_test_unseen_label",
    "diagnostic_subject_seen_label",
    "diagnostic_subject_unseen_label",
)


def normalize_label(value: str) -> str:
    return str(value).strip().strip("/").lower()


def _load_source(root: Path, split: str) -> dict[str, np.ndarray]:
    path = root / f"records_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.load(path, allow_pickle=False)
    required = {
        "eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask",
        "mel", "activity", "duration", "sample_keys", "audio_keys", "labels", "subjects",
    }
    missing = required - set(raw.files)
    if missing:
        raise ValueError(f"source cache {path} lacks {sorted(missing)}")
    return {key: np.asarray(raw[key]) for key in required}


def merge_source(root: Path) -> dict[str, np.ndarray]:
    parts = [_load_source(root, split) for split in SOURCE_SPLITS]
    values = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    values["source_split"] = np.concatenate(
        [np.full(len(part["sample_keys"]), split) for split, part in zip(SOURCE_SPLITS, parts)]
    )
    keys = values["sample_keys"].astype(str)
    if len(keys) != 1913 or len(set(keys.tolist())) != len(keys):
        raise ValueError(f"expected 1,913 unique source records, found {len(keys)}")
    return values


def assign_roles(
    source_split: Sequence[str], subjects: Sequence[str], labels: Sequence[str], *,
    subject_holdout: Sequence[str], unseen_label: str,
) -> np.ndarray:
    held = set(map(str, subject_holdout))
    unseen = normalize_label(unseen_label)
    roles: list[str] = []
    for split, subject, label in zip(source_split, subjects, labels):
        label_unseen = normalize_label(str(label)) == unseen
        if split == "locked_test":
            roles.append("locked_test_unseen_label" if label_unseen else "locked_test_seen_label")
        elif split == "diagnostic":
            roles.append("diagnostic_subject_unseen_label" if label_unseen else "diagnostic_subject_seen_label")
        elif str(subject) in held and label_unseen:
            roles.append("subject_and_label_holdout")
        elif str(subject) in held:
            roles.append("subject_holdout_seen")
        elif label_unseen:
            roles.append("label_holdout_seen_subject")
        else:
            roles.append("fit")
    return np.asarray(roles)


def _manifest_audio_paths(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        result = {
            str(row["sample_key"]): str(row["audio_relpath"])
            for row in rows
            if row.get("dataset") == "karaone"
        }
    if len(result) != 1913:
        raise ValueError(f"expected 1,913 KaraOne manifest rows, found {len(result)}")
    return result


def _read_waveform(path: Path) -> tuple[np.ndarray, int]:
    waveform, rate = sf.read(path, always_2d=False, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if not len(waveform):
        raise ValueError(f"empty audio: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"non-finite audio: {path}")
    return np.asarray(waveform, dtype=np.float32), int(rate)


def remove_dc_offset(waveform: np.ndarray) -> tuple[np.ndarray, float]:
    """The v3-only first step of light audio cleaning; source WAVs stay untouched."""
    value = np.asarray(waveform, dtype=np.float32).reshape(-1)
    offset = float(value.mean()) if len(value) else 0.0
    return (value - offset).astype(np.float32), offset


def light_prepare_waveform(
    waveform: np.ndarray, source_rate: int, config: AudioPreparationConfig
) -> tuple[Any, float]:
    dc_removed, offset = remove_dc_offset(waveform)
    return prepare_waveform_segment(dc_removed, source_rate, config), offset


def _interpolate(value: np.ndarray, frames: int) -> np.ndarray:
    source = torch.from_numpy(np.asarray(value, dtype=np.float32)).unsqueeze(0)
    return F.interpolate(source, size=frames, mode="linear", align_corners=False).squeeze(0).numpy()


def _mfcc_from_mel(
    mel_db: np.ndarray, active: np.ndarray, valid: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw CMVN MFCC and a content-only canonical representation.

    The c0 coefficient is zeroed after utterance CMVN.  This deliberately
    removes absolute energy from the EEG target; v3 makes no loudness claim.
    """
    raw = dct(np.asarray(mel_db, dtype=np.float32), type=2, axis=0, norm="ortho")[:bins]
    support = np.asarray(active, dtype=bool) & np.asarray(valid, dtype=bool)
    if int(support.sum()) < 2:
        support = np.asarray(valid, dtype=bool)
    if int(support.sum()) < 2:
        zeros = np.zeros_like(raw, dtype=np.float32)
        return zeros, zeros, zeros, np.zeros(bins, dtype=np.float32), np.ones(bins, dtype=np.float32)
    mean = raw[:, support].mean(axis=1, keepdims=True)
    scale = np.maximum(raw[:, support].std(axis=1, keepdims=True), 1.0e-4)
    normalized = ((raw - mean) / scale).astype(np.float32)
    normalized[0] = 0.0
    indices = np.flatnonzero(support)
    canonical = _interpolate(normalized[:, indices[0] : indices[-1] + 1], raw.shape[1])
    return (
        raw.astype(np.float32), normalized, canonical.astype(np.float32),
        mean[:, 0].astype(np.float32), scale[:, 0].astype(np.float32),
    )


def _accepted_denoise_paths(config_path: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    value = cfg.get("paths", {}).get("denoise_manifest")
    if not value:
        return {}
    manifest_path = (config_path.parent / value).resolve()
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted: dict[str, Path] = {}
    for item in payload.get("records", []):
        if not bool(item.get("accepted", False)):
            continue
        path = Path(str(item.get("enhanced_wav", ""))).resolve()
        if not path.is_file():
            raise RuntimeError(f"accepted denoised audio is missing: {path}")
        accepted[str(item["sample_key"])] = path
    return accepted


def _canonical_mel(mel: np.ndarray, active: np.ndarray, valid: np.ndarray) -> np.ndarray:
    support = np.asarray(active, dtype=bool) & np.asarray(valid, dtype=bool)
    if int(support.sum()) < 2:
        support = np.asarray(valid, dtype=bool)
    if int(support.sum()) < 2:
        return np.full_like(mel, -80.0, dtype=np.float32)
    indices = np.flatnonzero(support)
    return _interpolate(np.asarray(mel)[:, indices[0] : indices[-1] + 1], mel.shape[1]).astype(np.float32)


@dataclass(frozen=True)
class PreparedRecords:
    arrays: dict[str, np.ndarray]
    roles: np.ndarray

    def __len__(self) -> int:
        return int(len(self.roles))


def prepare_records(config_path: Path, cfg: dict[str, Any]) -> tuple[PreparedRecords, list[dict[str, Any]]]:
    source_root = (config_path.parent / cfg["paths"]["source_cache_root"]).resolve()
    audio_root = (config_path.parent / cfg["data"]["audio_root"]).resolve()
    manifest = (config_path.parent / cfg["data"]["unified_manifest"]).resolve()
    arrays = merge_source(source_root)
    # Keep the immutable power-dB Mel from the v0728 source cache for V0.
    # ``mel`` below is replaced by the separate v3 canonical active target.
    arrays["vocoder_mel"] = np.asarray(arrays["mel"], dtype=np.float32).copy()
    arrays["vocoder_activity"] = np.asarray(arrays["activity"], dtype=bool).copy()
    for key in ("sample_keys", "audio_keys", "labels", "subjects", "source_split"):
        arrays[key] = arrays[key].astype(str)
    roles = assign_roles(
        arrays["source_split"], arrays["subjects"], arrays["labels"],
        subject_holdout=cfg["split"]["subject_holdout"], unseen_label=cfg["split"]["unseen_label"],
    )
    paths = _manifest_audio_paths(manifest)
    denoised_paths = _accepted_denoise_paths(config_path, cfg)
    frames = int(cfg["audio"]["canonical_frames"])
    sample_rate = int(cfg["audio"]["sample_rate"])
    prep_cfg = AudioPreparationConfig(
        sample_rate=sample_rate,
        max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
        target_rms=float(cfg["audio"]["target_rms"]),
    )
    feature_cfg = AcousticFeatureConfig(
        sample_rate=sample_rate,
        n_fft=int(cfg["audio"]["n_fft"]),
        mel_bins=int(cfg["audio"]["mel_bins"]),
        max_frames=frames,
        min_db=float(cfg["audio"]["mel_db_min"]),
        max_db=float(cfg["audio"]["mel_db_max"]),
    )
    raw_mfcc, normalized_mfcc, canonical_mfcc, mfcc_mean, mfcc_std = [], [], [], [], []
    raw_mel, canonical_mel = [], []
    frame_mask, activity_mask, valid_samples, active_seconds, fit_eligible = [], [], [], [], []
    audit: list[dict[str, Any]] = []
    manual_review = set(map(str, cfg["audio"].get("manual_review_sample_keys", ())))
    for index, key in enumerate(tqdm(arrays["sample_keys"].tolist(), desc="[v3 prepare] light audio audit", unit="trial", dynamic_ncols=True)):
        relative = paths.get(str(key))
        if relative is None:
            raise KeyError(f"KaraOne sample {key} absent from unified manifest")
        original_path = audio_root / relative
        selected_path = denoised_paths.get(str(key), original_path)
        waveform, native_rate = _read_waveform(selected_path)
        prepared, dc_offset = light_prepare_waveform(waveform, native_rate, prep_cfg)
        acoustic = extract_acoustic_features(
            prepared.waveform, valid_samples=prepared.valid_samples, config=feature_cfg
        )
        mfcc_unscaled, mfcc_normalized, mfcc_canonical, utterance_mean, utterance_std = _mfcc_from_mel(
            acoustic.log_mel_energy, acoustic.activity_mask, acoustic.frame_valid_mask,
            int(cfg["audio"]["mfcc_bins"]),
        )
        raw_mfcc.append(mfcc_unscaled)
        normalized_mfcc.append(mfcc_normalized)
        canonical_mfcc.append(mfcc_canonical)
        mfcc_mean.append(utterance_mean)
        mfcc_std.append(utterance_std)
        raw_mel.append(acoustic.log_mel_energy)
        canonical_mel.append(_canonical_mel(acoustic.log_mel_energy, acoustic.activity_mask, acoustic.frame_valid_mask))
        frame_mask.append(acoustic.frame_valid_mask)
        activity_mask.append(acoustic.activity_mask)
        valid_frames = np.asarray(acoustic.frame_valid_mask, dtype=bool)
        active_frames = np.asarray(acoustic.activity_mask, dtype=bool) & valid_frames
        inactive_frames = valid_frames & ~active_frames
        if active_frames.any() and inactive_frames.any():
            contrast_db = float(
                np.median(acoustic.log_rms_dbfs[active_frames])
                - np.median(acoustic.log_rms_dbfs[inactive_frames])
            )
        else:
            contrast_db = float("nan")
        low_contrast = bool(
            np.isfinite(contrast_db)
            and contrast_db < float(cfg["audio"]["low_contrast_db_threshold"])
        )
        pending_manual_review = str(key) in manual_review and str(key) not in denoised_paths
        eligible = bool(
            prepared.reconstruction_eligible
            and prepared.active_duration_seconds <= float(cfg["audio"]["max_active_seconds"])
            and not pending_manual_review
        )
        valid_samples.append(prepared.valid_samples)
        active_seconds.append(prepared.active_duration_seconds)
        fit_eligible.append(eligible)
        frame_rms = prepared.waveform[: max(1, prepared.valid_samples)]
        audit.append(
            {
                "sample_key": str(key), "audio_relpath": relative, "role": str(roles[index]),
                "feature_audio_path": str(selected_path),
                "used_accepted_denoising": str(key) in denoised_paths,
                "label": str(arrays["labels"][index]), "subject": str(arrays["subjects"][index]),
                "native_sample_rate": native_rate, "valid_samples": int(prepared.valid_samples),
                "dc_removed": True, "dc_offset": dc_offset,
                "active_duration_seconds": float(prepared.active_duration_seconds),
                "exceeds_2_56_seconds": bool(prepared.exceeds_max_active_seconds),
                "has_activity": bool(prepared.has_activity), "fit_eligible": eligible,
                "native_rms": float(prepared.native_rms), "normalization_gain": float(prepared.normalization_gain),
                "peak_after_normalization": float(np.max(np.abs(frame_rms), initial=0.0)),
                "active_inactive_contrast_db": contrast_db,
                "low_contrast": low_contrast,
                "low_contrast_threshold_db": float(cfg["audio"]["low_contrast_db_threshold"]),
                "manual_review_required": pending_manual_review,
                "action": (
                    "exclude_from_fit_pending_manual_review"
                    if pending_manual_review
                    else "exclude_from_fit_review_required"
                    if not eligible
                    else "light_clean_keep_in_low_contrast_audit_queue"
                    if low_contrast
                    else "light_clean_only"
                ),
            }
        )
    arrays.update(
        {
            "v3_preparation_schema": np.asarray(PREPARATION_SCHEMA),
            "mfcc_raw": np.stack(raw_mfcc).astype(np.float32),
            "mfcc_cmvn": np.stack(normalized_mfcc).astype(np.float32),
            "mfcc": np.stack(canonical_mfcc).astype(np.float32),
            "mfcc_mean": np.stack(mfcc_mean).astype(np.float32),
            "mfcc_std": np.stack(mfcc_std).astype(np.float32),
            "mel_raw": np.stack(raw_mel).astype(np.float32),
            "mel": np.stack(canonical_mel).astype(np.float32),
            "mfcc_mask": np.stack(frame_mask).astype(bool),
            "activity_v3": np.stack(activity_mask).astype(bool),
            "audio_valid_samples_v3": np.asarray(valid_samples, dtype=np.int32),
            "audio_active_seconds_v3": np.asarray(active_seconds, dtype=np.float32),
            "fit_eligible": np.asarray(fit_eligible, dtype=bool),
        }
    )
    return PreparedRecords(arrays=arrays, roles=roles), audit


def save_prepared(path: Path, records: PreparedRecords) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **records.arrays, roles=records.roles)


def load_prepared(path: Path) -> PreparedRecords:
    raw = np.load(path, allow_pickle=False)
    required = {
        "eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask", "mel",
        "mfcc", "mfcc_raw", "mfcc_cmvn", "mfcc_mean", "mfcc_std", "mfcc_mask",
        "activity_v3", "fit_eligible", "sample_keys",
        "audio_keys", "labels", "subjects", "roles", "v3_preparation_schema",
    }
    missing = required - set(raw.files)
    if missing:
        raise ValueError(f"v3 prepared cache lacks {sorted(missing)}")
    schema = str(np.asarray(raw["v3_preparation_schema"]).item())
    if schema != PREPARATION_SCHEMA:
        raise ValueError(
            f"v3 cache schema {schema!r} is stale; rerun prepare_open_vocab_v3.py --force"
        )
    arrays = {key: np.asarray(raw[key]) for key in raw.files if key != "roles"}
    return PreparedRecords(arrays=arrays, roles=np.asarray(raw["roles"]).astype(str))


class V3Dataset(Dataset[dict[str, Any]]):
    def __init__(self, records: PreparedRecords, roles: Iterable[str], *, eligible_only: bool = False):
        accepted = set(roles)
        selector = np.isin(records.roles, list(accepted))
        if eligible_only:
            selector &= records.arrays["fit_eligible"].astype(bool)
        self.records = records
        self.indices = np.flatnonzero(selector).tolist()
        if not self.indices:
            raise ValueError(f"empty v3 dataset roles={sorted(accepted)} eligible_only={eligible_only}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        value = self.records.arrays
        return {
            "eeg": value["eeg"][index], "channel_xyz": value["channel_xyz"][index],
            "channel_mask": value["channel_mask"][index], "time_mask": value["time_mask"][index],
            "hubert": value["hubert"][index], "hubert_mask": value["hubert_mask"][index],
            "mfcc": value["mfcc"][index], "mel": value["mel"][index],
            "mfcc_mask": value["mfcc_mask"][index], "activity": value["activity_v3"][index],
            "sample_key": str(value["sample_keys"][index]), "audio_key": str(value["audio_keys"][index]),
            "label": str(value["labels"][index]), "subject": str(value["subjects"][index]),
            "role": str(self.records.roles[index]),
            "speaker_reference": value["speaker_reference_embedding"][index] if "speaker_reference_embedding" in value else np.zeros(192, dtype=np.float32),
            "speaker_target": value["speaker_target_embedding"][index] if "speaker_target_embedding" in value else np.zeros(192, dtype=np.float32),
            "canonical_voice": value["canonical_voice"] if "canonical_voice" in value else np.zeros(192, dtype=np.float32),
            "target_mfcc_mean": value["mfcc_mean"][index],
            "target_mfcc_std": value["mfcc_std"][index],
            "speaker_reference_mfcc_mean": value["speaker_reference_mfcc_mean"][index] if "speaker_reference_mfcc_mean" in value else value["mfcc_mean"][index],
            "speaker_reference_mfcc_std": value["speaker_reference_mfcc_std"][index] if "speaker_reference_mfcc_std" in value else value["mfcc_std"][index],
            "canonical_mfcc_mean": value["canonical_mfcc_mean"] if "canonical_mfcc_mean" in value else value["mfcc_mean"].mean(0),
            "canonical_mfcc_std": value["canonical_mfcc_std"] if "canonical_mfcc_std" in value else value["mfcc_std"].mean(0),
        }


def collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensors = (
        "eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask",
        "mfcc", "mel", "mfcc_mask", "activity", "speaker_reference", "speaker_target",
        "canonical_voice", "target_mfcc_mean", "target_mfcc_std",
        "speaker_reference_mfcc_mean", "speaker_reference_mfcc_std",
        "canonical_mfcc_mean", "canonical_mfcc_std",
    )
    result: dict[str, Any] = {key: torch.as_tensor(np.stack([item[key] for item in items])) for key in tensors}
    for key in ("channel_mask", "time_mask", "hubert_mask", "mfcc_mask", "activity"):
        result[key] = result[key].bool()
    for key in ("sample_key", "audio_key", "label", "subject", "role"):
        result[key] = [str(item[key]) for item in items]
    return result


def _fixed_permutation(length: int, device: torch.device) -> torch.Tensor:
    """A deterministic non-identity permutation without an RNG side effect."""
    if length <= 1:
        return torch.arange(length, device=device)
    stride = next((candidate for candidate in range(2, length) if np.gcd(candidate, length) == 1), 1)
    return (torch.arange(length, device=device) * stride + 1) % length


def time_shuffled_eeg(eeg: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    """Shuffle only valid within-trial time samples; padding remains inert."""
    result = eeg.clone()
    for row in range(eeg.shape[0]):
        indices = torch.nonzero(time_mask[row], as_tuple=False).squeeze(1)
        if len(indices) > 1:
            result[row, :, indices] = eeg[row, :, indices[_fixed_permutation(len(indices), eeg.device)]]
    return result


def channel_shuffled_eeg(eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """Shuffle valid signal channels but intentionally keep coordinates fixed."""
    result = eeg.clone()
    for row in range(eeg.shape[0]):
        indices = torch.nonzero(channel_mask[row], as_tuple=False).squeeze(1)
        if len(indices) > 1:
            result[row, indices] = eeg[row, indices[_fixed_permutation(len(indices), eeg.device)]]
    return result


def role_counts(records: PreparedRecords) -> dict[str, int]:
    return {role: int((records.roles == role).sum()) for role in sorted(set(records.roles.tolist()))}


def canonical_mfcc_from_waveform(waveform: np.ndarray, sample_rate: int, cfg: dict[str, Any]) -> np.ndarray:
    """Apply exactly the v3 content-target transform to a generated waveform."""
    frames = int(cfg["audio"]["canonical_frames"])
    prep_cfg = AudioPreparationConfig(
        sample_rate=int(cfg["audio"]["sample_rate"]),
        max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
        target_rms=float(cfg["audio"]["target_rms"]),
    )
    feature_cfg = AcousticFeatureConfig(
        sample_rate=int(cfg["audio"]["sample_rate"]), n_fft=int(cfg["audio"]["n_fft"]),
        mel_bins=int(cfg["audio"]["mel_bins"]), max_frames=frames,
        min_db=float(cfg["audio"]["mel_db_min"]), max_db=float(cfg["audio"]["mel_db_max"]),
    )
    prepared, _ = light_prepare_waveform(waveform, sample_rate, prep_cfg)
    acoustic = extract_acoustic_features(prepared.waveform, valid_samples=prepared.valid_samples, config=feature_cfg)
    _, _, canonical, _, _ = _mfcc_from_mel(
        acoustic.log_mel_energy, acoustic.activity_mask, acoustic.frame_valid_mask,
        int(cfg["audio"]["mfcc_bins"]),
    )
    return canonical.astype(np.float32)
