from __future__ import annotations

import csv
import hashlib
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
PREPARATION_SCHEMA = "openvoice-v3-encodec-clip-mfcc-preparation-v2-native-mel-161"
CP_TEMPORAL_PREPARATION_SCHEMA = "openvoice-v3-cp-temporal-preparation-v1-161"
BRIDGE_PREPARATION_SCHEMA = "openvoice-v3-mfcc-encodec-bridge-preparation-v2-161"
RVQ_REPAIR_PREPARATION_SCHEMA = "openvoice-v3-mfcc-encodec-rvq-repair-preparation-v3-161"
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


def fit_source_keys(root: Path, *, subject_holdout: Sequence[str], unseen_label: str) -> set[str]:
    """Read train/validation metadata and return only final-role fit rows.

    This helper lets the strict audio audit avoid opening any validation,
    diagnostic, locked-test, held-subject, or unseen-label WAV.
    """
    paths = [root / f"records_{split}.npz" for split in ("train", "validation")]
    if not all(path.is_file() for path in paths):
        fallback = root / "prepared_encodec_bridge_v2.npz"
        if not fallback.is_file():
            raise FileNotFoundError(f"source records and prepared fallback are both missing under {root}")
        raw = np.load(fallback, allow_pickle=False); roles = np.asarray(raw["roles"]).astype(str)
        return set(np.asarray(raw["sample_keys"]).astype(str)[roles == "fit"].tolist())
    metadata = [np.load(path, allow_pickle=False) for path in paths]
    keys = np.concatenate([np.asarray(raw["sample_keys"]).astype(str) for raw in metadata])
    subjects = np.concatenate([np.asarray(raw["subjects"]).astype(str) for raw in metadata])
    labels = np.concatenate([np.asarray(raw["labels"]).astype(str) for raw in metadata])
    held = set(map(str, subject_holdout)); unseen = normalize_label(unseen_label)
    keep = np.asarray([(subject not in held) and normalize_label(label) != unseen for subject, label in zip(subjects, labels)])
    return set(keys[keep].tolist())


def merge_fit_source(root: Path, *, subject_holdout: Sequence[str], unseen_label: str) -> dict[str, np.ndarray]:
    """Load only final-role fit rows from train/validation source containers.

    The NPZ containers are monolithic, so NumPy may decompress source rows
    that are subsequently excluded. No excluded row is returned, cached,
    trained on, evaluated, rendered, or exposed through ``V3Dataset``.
    """
    paths = [root / f"records_{split}.npz" for split in ("train", "validation")]
    required = {
        "eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask",
        "mel", "activity", "duration", "sample_keys", "audio_keys", "labels", "subjects",
    }
    if not all(path.is_file() for path in paths):
        fallback = root / "prepared_encodec_bridge_v2.npz"
        if not fallback.is_file():
            raise FileNotFoundError(f"source records and prepared fallback are both missing under {root}")
        raw = np.load(fallback, allow_pickle=False);missing = required - set(raw.files)
        if missing:
            raise ValueError(f"prepared source fallback {fallback} lacks {sorted(missing)}")
        roles = np.asarray(raw["roles"]).astype(str);keep = roles == "fit"
        values = {key: np.asarray(raw[key])[keep] for key in required};values["source_split"] = np.asarray(raw["source_split"])[keep].astype(str)
        if len(values["sample_keys"]) != 1019:
            raise ValueError(f"prepared source fallback must yield 1,019 fit rows, found {len(values['sample_keys'])}")
        return values
    parts = [np.load(path, allow_pickle=False) for path in paths]
    for path, raw in zip(paths, parts):
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"source cache {path} lacks {sorted(missing)}")
    subjects = np.concatenate([np.asarray(raw["subjects"]).astype(str) for raw in parts]); labels = np.concatenate([np.asarray(raw["labels"]).astype(str) for raw in parts])
    held = set(map(str, subject_holdout)); unseen = normalize_label(unseen_label)
    keep = np.asarray([(subject not in held) and normalize_label(label) != unseen for subject, label in zip(subjects, labels)])
    values = {key: np.concatenate([np.asarray(raw[key]) for raw in parts])[keep] for key in required}
    source_split = np.concatenate([np.full(len(raw["sample_keys"]), split) for split, raw in zip(("train", "validation"), parts)])
    values["source_split"] = source_split[keep]
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


def _cp_temporal_targets(acoustic: Any, raw_mfcc: np.ndarray, duration: float,
                         maximum_duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    """Build bounded C/P targets on the unified acoustic grid.

    P-base is activity, relative log energy, and its first difference.  P-plus
    is deliberately audio-only: voicing probability and coarse log-F0.  The
    separate c0 target keeps absolute cepstral energy out of the C/CLIP space.
    """
    valid = np.asarray(acoustic.frame_valid_mask, dtype=bool)
    active = np.asarray(acoustic.activity_mask, dtype=bool) & valid
    support = active if int(active.sum()) >= 2 else valid
    rms = np.asarray(acoustic.log_rms_dbfs, dtype=np.float32)
    center = float(np.median(rms[support])) if support.any() else float(np.median(rms))
    relative = np.clip((rms - center) / 20.0, -3.0, 1.0).astype(np.float32)
    relative[~valid] = -3.0
    delta = np.diff(relative, prepend=relative[:1]).astype(np.float32)
    p_base = np.stack((active.astype(np.float32), relative, delta), axis=-1)

    voiced = np.asarray(acoustic.voicing, dtype=np.float32) * valid.astype(np.float32)
    log_f0 = np.asarray(acoustic.log_f0_hz, dtype=np.float32)
    lo, hi = np.log(50.0), np.log(500.0)
    coarse_f0 = np.clip((log_f0 - lo) / (hi - lo), 0.0, 1.0) * (voiced > 0).astype(np.float32)
    p_plus = np.stack((voiced, coarse_f0.astype(np.float32)), axis=-1)

    c0 = np.asarray(raw_mfcc[0], dtype=np.float32)
    if support.any():
        c0 = (c0 - float(c0[support].mean())) / max(float(c0[support].std()), 1.0e-4)
    else:
        c0 = np.zeros_like(c0)
    c0[~valid] = 0.0
    duration_fraction = np.float32(np.clip(float(duration) / max(float(maximum_duration), 1.0e-6), 0.0, 1.0))
    return p_base.astype(np.float32), p_plus.astype(np.float32), c0.astype(np.float32), duration_fraction


def _bridge_content_target(normalized_mfcc: np.ndarray, active: np.ndarray,
                           valid: np.ndarray, frames: int) -> tuple[np.ndarray, int, int]:
    """Build the bridge-v2 C target on the VAD-active, normalized-time grid.

    This is deliberately different from the CP-temporal cache: silence and
    duration are not carried by C.  ``P`` owns those factors, while C always
    occupies the complete 161-frame relative-time grid.
    """
    support = np.asarray(active, dtype=bool) & np.asarray(valid, dtype=bool)
    if int(support.sum()) < 2:
        support = np.asarray(valid, dtype=bool)
    if int(support.sum()) < 2:
        return np.zeros((normalized_mfcc.shape[0] - 1, frames), dtype=np.float32), 0, 0
    indices = np.flatnonzero(support)
    start, end = int(indices[0]), int(indices[-1])
    content = _interpolate(np.asarray(normalized_mfcc[1:, start:end + 1], dtype=np.float32), frames)
    return content.astype(np.float32), start, end


def _p_medoid_bank(p_base: np.ndarray, duration: np.ndarray, keys: np.ndarray,
                   count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, actual fit-train P medoids without label access."""
    values = np.asarray(p_base, dtype=np.float32)
    if len(values) < int(count):
        raise ValueError("fit-train P bank has fewer trials than requested medoids")
    flat = values.reshape(len(values), -1)
    # Standardize dimensions so activity, envelope, and delta contribute on a
    # comparable scale.  Add duration explicitly because it belongs to P.
    scale = np.maximum(flat.std(0, keepdims=True), 1.0e-4)
    flat = (flat - flat.mean(0, keepdims=True)) / scale
    feature = np.concatenate((flat, np.asarray(duration, dtype=np.float32)[:, None] * 5.0), axis=1)
    norm = (feature * feature).sum(1, keepdims=True)
    distance = np.maximum(norm + norm.T - 2.0 * (feature @ feature.T), 0.0)
    # First medoid is the actual point nearest the global P centre.  The other
    # slots use deterministic farthest-first initialisation, followed by a
    # short PAM update.  No labels or target trial are consulted.
    selected = [int(np.argmin(distance.sum(1)))]
    while len(selected) < int(count):
        nearest = distance[:, selected].min(1)
        candidates = np.flatnonzero(nearest == nearest.max())
        selected.append(int(sorted(candidates.tolist(), key=lambda i: str(keys[i]))[0]))
    for _ in range(4):
        assignment = distance[:, selected].argmin(1)
        replacement: list[int] = []
        for cluster, old in enumerate(selected):
            members = np.flatnonzero(assignment == cluster)
            if not len(members):
                replacement.append(old); continue
            costs = distance[np.ix_(members, members)].sum(1)
            minima = members[np.flatnonzero(costs == costs.min())]
            replacement.append(int(sorted(minima.tolist(), key=lambda i: str(keys[i]))[0]))
        if replacement == selected:
            break
        selected = replacement
    order = [selected[0]] + sorted(selected[1:], key=lambda i: str(keys[i]))
    return values[order].astype(np.float32), np.asarray(duration, dtype=np.float32)[order], np.asarray(keys, dtype=str)[order]


def _fit_internal_dev_mask(roles: np.ndarray, eligible: np.ndarray, subjects: np.ndarray,
                           labels: np.ndarray, sample_keys: np.ndarray, seed: int = 31) -> np.ndarray:
    """Deterministic 10% subject-label stratified dev split inside fit only."""
    result = np.zeros(len(roles), dtype=bool)
    fit = (np.asarray(roles).astype(str) == "fit") & np.asarray(eligible, dtype=bool)
    groups: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(fit):
        groups.setdefault((str(subjects[index]), normalize_label(str(labels[index]))), []).append(int(index))
    for group, indices in sorted(groups.items()):
        ordered = sorted(indices, key=lambda i: (hashlib.sha256(f"{seed}:{sample_keys[i]}".encode()).hexdigest(), str(sample_keys[i])))
        count = max(1, int(round(0.10 * len(ordered)))) if len(ordered) >= 5 else 0
        result[ordered[:count]] = True
    return result


@dataclass(frozen=True)
class PreparedRecords:
    arrays: dict[str, np.ndarray]
    roles: np.ndarray

    def __len__(self) -> int:
        return int(len(self.roles))


def prepare_records(config_path: Path, cfg: dict[str, Any], *, fit_only: bool = False) -> tuple[PreparedRecords, list[dict[str, Any]]]:
    cp_temporal = str(cfg.get("version", "")) == "openvoice-v3-cp-temporal-large-v1"
    bridge = str(cfg.get("version", "")) in {
        "openvoice-v3-mfcc-encodec-bridge-v2", "openvoice-v3-mfcc-encodec-rvq-repair-v3",
    }
    rvq_repair = str(cfg.get("version", "")) == "openvoice-v3-mfcc-encodec-rvq-repair-v3"
    temporal_schema = cp_temporal or bridge
    source_root = (config_path.parent / cfg["paths"]["source_cache_root"]).resolve()
    audio_root = (config_path.parent / cfg["data"]["audio_root"]).resolve()
    manifest = (config_path.parent / cfg["data"]["unified_manifest"]).resolve()
    arrays = merge_fit_source(
        source_root, subject_holdout=cfg["split"]["subject_holdout"], unseen_label=cfg["split"]["unseen_label"],
    ) if fit_only else merge_source(source_root)
    # Legacy v0728 Mel is retained only as lineage, never routed to SpeechT5.
    arrays["legacy_v0728_mel"] = np.asarray(arrays["mel"], dtype=np.float32).copy()
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
    raw_mel, canonical_mel, speech_t5_mel, speech_t5_mask = [], [], [], []
    frame_mask, activity_mask, valid_samples, active_seconds, fit_eligible = [], [], [], [], []
    content_mfcc_targets, p_base_targets, p_plus_targets, c0_targets, duration_fractions = [], [], [], [], []
    active_frame_start, active_frame_end = [], []
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
        # Long/manual-review trials must be excluded before the fixed native
        # SpeechT5-Mel cache shape is imposed.  They remain in the 1,913-row
        # cache only for split/lineage auditability, never as train targets.
        pending_manual_review = str(key) in manual_review and str(key) not in denoised_paths
        eligible = bool(
            prepared.reconstruction_eligible
            and prepared.active_duration_seconds <= float(cfg["audio"]["max_active_seconds"])
            and not pending_manual_review
        )
        from .native_mel import native_speecht5_mel
        native = native_speecht5_mel(
            torch.from_numpy(prepared.waveform[: max(1, prepared.valid_samples)]).unsqueeze(0), cfg,
        ).squeeze(0).cpu().numpy().astype(np.float32)
        maximum=int(cfg["audio"]["native_mel_frames"])
        source_native_frames = int(native.shape[-1])
        if native.shape[-1] > maximum:
            # SpeechT5 framing can contribute one or two boundary frames even
            # when the VAD-active duration satisfies 2.56 s. Normalize once,
            # directly onto the declared 161-frame acoustic grid. This is an
            # audited target transform, not the abandoned 96->256->161 model
            # interpolation chain.
            native = _interpolate(native, maximum)
        native_frames=native.shape[-1];native_padded=np.full((native.shape[0],maximum),float(native.min()),dtype=np.float32);native_padded[:,:native_frames]=native
        speech_t5_mel.append(native_padded)
        one_mask=np.zeros(maximum,dtype=bool);one_mask[:native_frames]=True;speech_t5_mask.append(one_mask)
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
        valid_samples.append(prepared.valid_samples)
        active_seconds.append(prepared.active_duration_seconds)
        if temporal_schema:
            if bridge:
                content_target, active_start, active_end = _bridge_content_target(
                    mfcc_normalized, acoustic.activity_mask, acoustic.frame_valid_mask, frames
                )
            else:
                content_target = mfcc_normalized[1:].copy()
                content_target[:, ~np.asarray(acoustic.frame_valid_mask, dtype=bool)] = 0.0
                active_start, active_end = 0, int(frames - 1)
            p_base, p_plus, c0, duration_fraction = _cp_temporal_targets(
                acoustic, mfcc_unscaled, prepared.active_duration_seconds,
                float(cfg["audio"]["max_active_seconds"]),
            )
            content_mfcc_targets.append(content_target.astype(np.float32))
            p_base_targets.append(p_base)
            p_plus_targets.append(p_plus)
            c0_targets.append(c0)
            duration_fractions.append(duration_fraction)
            active_frame_start.append(active_start)
            active_frame_end.append(active_end)
        fit_eligible.append(eligible)
        frame_rms = prepared.waveform[: max(1, prepared.valid_samples)]
        audit.append(
            {
                "sample_key": str(key), "audio_relpath": relative, "role": str(roles[index]),
                "feature_audio_path": str(selected_path),
                "used_accepted_denoising": str(key) in denoised_paths,
                "label": str(arrays["labels"][index]), "subject": str(arrays["subjects"][index]),
                "native_sample_rate": native_rate, "valid_samples": int(prepared.valid_samples),
                "speech_t5_native_frames_raw": source_native_frames,
                "speech_t5_native_resampled_to_161_contract": bool(source_native_frames > maximum),
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
    preparation_schema = (
        CP_TEMPORAL_PREPARATION_SCHEMA if cp_temporal else RVQ_REPAIR_PREPARATION_SCHEMA if rvq_repair else BRIDGE_PREPARATION_SCHEMA if bridge
        else PREPARATION_SCHEMA
    )
    arrays.update(
        {
            "v3_preparation_schema": np.asarray(preparation_schema),
            "mfcc_raw": np.stack(raw_mfcc).astype(np.float32),
            "mfcc_cmvn": np.stack(normalized_mfcc).astype(np.float32),
            "mfcc": np.stack(canonical_mfcc).astype(np.float32),
            "mfcc_mean": np.stack(mfcc_mean).astype(np.float32),
            "mfcc_std": np.stack(mfcc_std).astype(np.float32),
            "mel_raw": np.stack(raw_mel).astype(np.float32),
            "mel": np.stack(canonical_mel).astype(np.float32),
            "speech_t5_mel": np.stack(speech_t5_mel).astype(np.float32),
            "speech_t5_mel_mask": np.stack(speech_t5_mask).astype(bool),
            "mfcc_mask": np.stack(frame_mask).astype(bool),
            "activity_v3": np.stack(activity_mask).astype(bool),
            "audio_valid_samples_v3": np.asarray(valid_samples, dtype=np.int32),
            "audio_active_seconds_v3": np.asarray(active_seconds, dtype=np.float32),
            "fit_eligible": np.asarray(fit_eligible, dtype=bool),
        }
    )
    if temporal_schema:
        arrays.update({
            "content_mfcc": np.stack(content_mfcc_targets).astype(np.float32),
            "p_base": np.stack(p_base_targets).astype(np.float32),
            "p_plus": np.stack(p_plus_targets).astype(np.float32),
            "mfcc_c0": np.stack(c0_targets).astype(np.float32),
            "duration_fraction": np.asarray(duration_fractions, dtype=np.float32),
            "active_frame_start": np.asarray(active_frame_start, dtype=np.int16),
            "active_frame_end": np.asarray(active_frame_end, dtype=np.int16),
        })
        internal_dev = _fit_internal_dev_mask(
            roles, arrays["fit_eligible"], arrays["subjects"], arrays["labels"],
            arrays["sample_keys"], seed=int(cfg.get("training", {}).get("seed", 31)),
        )
        arrays["fit_internal_dev"] = internal_dev
        fit_train = (roles == "fit") & arrays["fit_eligible"] & ~internal_dev
        if bridge:
            bank, bank_duration, bank_keys = _p_medoid_bank(
                arrays["p_base"][fit_train], arrays["duration_fraction"][fit_train],
                arrays["sample_keys"][fit_train], int(cfg["audio"].get("canonical_p_bank_size", 4)),
            )
            arrays["canonical_p_bank"] = bank
            arrays["canonical_p_bank_duration_fraction"] = bank_duration.astype(np.float32)
            arrays["canonical_p_bank_keys"] = bank_keys.astype(str)
            arrays["canonical_p_base"] = bank[0]
            arrays["canonical_duration_fraction"] = np.asarray(bank_duration[0], dtype=np.float32)
            arrays["eeg_content_mfcc"] = arrays["content_mfcc"].copy()
            arrays["canonical_content_mask"] = np.ones(frames, dtype=bool)
        else:
            arrays["canonical_p_base"] = np.median(arrays["p_base"][fit_train], axis=0).astype(np.float32)
            arrays["canonical_duration_fraction"] = np.asarray(
                np.median(arrays["duration_fraction"][fit_train]), dtype=np.float32
            )
            canonical_length = int(np.clip(round(float(arrays["canonical_duration_fraction"]) * frames), 1, frames))
            eeg_targets = np.zeros((len(roles), int(cfg["audio"]["mfcc_bins"]) - 1, frames), dtype=np.float32)
            for index in range(len(roles)):
                eeg_targets[index, :, :canonical_length] = _interpolate(arrays["mfcc"][index, 1:], canonical_length)
            arrays["eeg_content_mfcc"] = eeg_targets
            canonical_mask = np.zeros(frames, dtype=bool); canonical_mask[:canonical_length] = True
            arrays["canonical_content_mask"] = canonical_mask
    return PreparedRecords(arrays=arrays, roles=roles), audit


def save_prepared(path: Path, records: PreparedRecords) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **records.arrays, roles=records.roles)


def load_prepared(path: Path, expected_schema: str | None = None) -> PreparedRecords:
    raw = np.load(path, allow_pickle=False)
    required = {
        "eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask", "mel", "speech_t5_mel", "speech_t5_mel_mask",
        "mfcc", "mfcc_raw", "mfcc_cmvn", "mfcc_mean", "mfcc_std", "mfcc_mask",
        "activity_v3", "fit_eligible", "sample_keys",
        "audio_keys", "labels", "subjects", "roles", "v3_preparation_schema",
    }
    missing = required - set(raw.files)
    if missing:
        raise ValueError(f"v3 prepared cache lacks {sorted(missing)}")
    schema = str(np.asarray(raw["v3_preparation_schema"]).item())
    accepted = {PREPARATION_SCHEMA, CP_TEMPORAL_PREPARATION_SCHEMA, BRIDGE_PREPARATION_SCHEMA, RVQ_REPAIR_PREPARATION_SCHEMA}
    if schema not in accepted or (expected_schema is not None and schema != expected_schema):
        raise ValueError(
            f"v3 cache schema {schema!r} is stale; rerun prepare_open_vocab_v3.py --force"
        )
    if schema in {CP_TEMPORAL_PREPARATION_SCHEMA, BRIDGE_PREPARATION_SCHEMA, RVQ_REPAIR_PREPARATION_SCHEMA}:
        cp_required = {
            "content_mfcc", "eeg_content_mfcc", "canonical_content_mask", "p_base", "p_plus", "mfcc_c0", "duration_fraction",
            "canonical_p_base", "canonical_duration_fraction", "fit_internal_dev",
        }
        cp_missing = cp_required - set(raw.files)
        if cp_missing:
            raise ValueError(f"CP-temporal cache lacks {sorted(cp_missing)}; rerun prepare --force")
    if schema in {BRIDGE_PREPARATION_SCHEMA, RVQ_REPAIR_PREPARATION_SCHEMA}:
        bridge_required = {
            "active_frame_start", "active_frame_end", "canonical_p_bank",
            "canonical_p_bank_duration_fraction", "canonical_p_bank_keys",
        }
        missing_bridge = bridge_required - set(raw.files)
        if missing_bridge:
            raise ValueError(f"EnCodec-bridge cache lacks {sorted(missing_bridge)}; rerun prepare --force")
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
        result = {
            # Preserve the immutable prepared-cache row identity through any
            # number of torch.utils.data.Subset wrappers. Downstream token
            # caches must key on this value, never on a subset-local position.
            "source_index": int(index),
            "eeg": value["eeg"][index], "channel_xyz": value["channel_xyz"][index],
            "channel_mask": value["channel_mask"][index], "time_mask": value["time_mask"][index],
            "hubert": value["hubert"][index], "hubert_mask": value["hubert_mask"][index],
            "mfcc": value["mfcc"][index], "mel": value["mel"][index],
            "speech_t5_mel": value["speech_t5_mel"][index], "speech_t5_mel_mask": value["speech_t5_mel_mask"][index],
            "mfcc_mask": value["mfcc_mask"][index], "activity": value["activity_v3"][index],
            "sample_key": str(value["sample_keys"][index]), "audio_key": str(value["audio_keys"][index]),
            "label": str(value["labels"][index]), "subject": str(value["subjects"][index]),
            "role": str(self.records.roles[index]),
            "speaker_reference": value["speaker_reference_embedding"][index] if "speaker_reference_embedding" in value else np.zeros(192, dtype=np.float32),
            "speaker_target": value["speaker_target_embedding"][index] if "speaker_target_embedding" in value else np.zeros(192, dtype=np.float32),
            "speaker_audit_reference": value["speaker_audit_reference_embedding"][index] if "speaker_audit_reference_embedding" in value else np.zeros(192, dtype=np.float32),
            "canonical_voice": value["canonical_voice"] if "canonical_voice" in value else np.zeros(192, dtype=np.float32),
            "target_mfcc_mean": value["mfcc_mean"][index],
            "target_mfcc_std": value["mfcc_std"][index],
            "speaker_reference_mfcc_mean": value["speaker_reference_mfcc_mean"][index] if "speaker_reference_mfcc_mean" in value else value["mfcc_mean"][index],
            "speaker_reference_mfcc_std": value["speaker_reference_mfcc_std"][index] if "speaker_reference_mfcc_std" in value else value["mfcc_std"][index],
            "canonical_mfcc_mean": value["canonical_mfcc_mean"] if "canonical_mfcc_mean" in value else value["mfcc_mean"].mean(0),
            "canonical_mfcc_std": value["canonical_mfcc_std"] if "canonical_mfcc_std" in value else value["mfcc_std"].mean(0),
        }
        if "p_base" in value:
            result.update({
                "content_mfcc": value["content_mfcc"][index] if "content_mfcc" in value else value["mfcc"][index, 1:],
                "eeg_content_mfcc": value["eeg_content_mfcc"][index],
                "canonical_content_mask": value["canonical_content_mask"],
                "p_base": value["p_base"][index],
                "p_plus": value["p_plus"][index],
                "mfcc_c0": value["mfcc_c0"][index],
                "duration_fraction": value["duration_fraction"][index],
                "canonical_p_base": value["canonical_p_base"],
                "canonical_duration_fraction": value["canonical_duration_fraction"],
                "fit_internal_dev": value["fit_internal_dev"][index],
                "active_frame_start": value["active_frame_start"][index] if "active_frame_start" in value else np.int16(0),
                "active_frame_end": value["active_frame_end"][index] if "active_frame_end" in value else np.int16(value["content_mfcc"].shape[-1] - 1),
                "canonical_p_bank": value["canonical_p_bank"] if "canonical_p_bank" in value else value["canonical_p_base"][None],
                "canonical_p_bank_duration_fraction": value["canonical_p_bank_duration_fraction"] if "canonical_p_bank_duration_fraction" in value else np.asarray([value["canonical_duration_fraction"]], dtype=np.float32),
            })
        return result


def collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensors = (
        # ``source_index`` is the immutable identity used by fit-only EnCodec
        # caches and by evaluation to recover the corresponding HuBERT target.
        # Keep it through collation rather than accidentally replacing it with
        # a DataLoader-local row number.
        "source_index", "eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask",
        "mfcc", "mel", "speech_t5_mel", "speech_t5_mel_mask", "mfcc_mask", "activity", "speaker_reference", "speaker_target", "speaker_audit_reference",
        "canonical_voice", "target_mfcc_mean", "target_mfcc_std",
        "speaker_reference_mfcc_mean", "speaker_reference_mfcc_std",
        "canonical_mfcc_mean", "canonical_mfcc_std",
    )
    result: dict[str, Any] = {key: torch.as_tensor(np.stack([item[key] for item in items])) for key in tensors}
    optional = (
        "content_mfcc", "eeg_content_mfcc", "canonical_content_mask", "p_base", "p_plus", "mfcc_c0", "duration_fraction",
        "canonical_p_base", "canonical_duration_fraction", "fit_internal_dev",
        "active_frame_start", "active_frame_end", "canonical_p_bank",
        "canonical_p_bank_duration_fraction",
    )
    for key in optional:
        if key in items[0]:
            result[key] = torch.as_tensor(np.stack([item[key] for item in items]))
    for key in ("channel_mask", "time_mask", "hubert_mask", "speech_t5_mel_mask", "mfcc_mask", "activity"):
        result[key] = result[key].bool()
    if "fit_internal_dev" in result:
        result["fit_internal_dev"] = result["fit_internal_dev"].bool()
    if "canonical_content_mask" in result:
        result["canonical_content_mask"] = result["canonical_content_mask"].bool()
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
