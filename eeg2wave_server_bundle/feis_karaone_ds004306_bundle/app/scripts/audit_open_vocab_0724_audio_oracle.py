#!/usr/bin/env python3
"""Gate the v0724 audio factorizer before any paired EEG training.

Only validation utterances are scored.  The audit reconstructs each eligible
KaraOne and FEIS validation utterance from its frozen audio teachers, then
compares that condition with a counterfactual that keeps content fixed while
borrowing realization and timbre from a different utterance with the same
label.  A train utterance may be used only as the non-scored counterfactual
source when a validation label is a singleton.  Rendered spectrogram images
are never involved in the metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
KARAONE_APP = APP.parents[1] / "karaone_overt_recon_bundle" / "app"
if str(KARAONE_APP) not in sys.path:
    sys.path.insert(0, str(KARAONE_APP))

from src.karaone_0715.codec import (  # noqa: E402
    DiscreteEncodec,
    DiscreteEncodecConfig,
)
from src.open_vocab_0722.audio_io import read_wav  # noqa: E402
from src.open_vocab_0724.audio_features import (  # noqa: E402
    ActiveSpeechConfig,
    AudioPreparationConfig,
    prepare_waveform_segment,
    resample_audio,
)
from src.open_vocab_0724.audio_gate import (  # noqa: E402
    AUDIO_FREEZE_SCHEMA,
    AUDIO_ORACLE_GATE_SCHEMA,
    require_frozen_audio_checkpoint,
)
from src.open_vocab_0724.data import (  # noqa: E402
    FactorizedAudioDataset,
    TeacherCacheV2,
    collate_factorized,
    load_context,
)
from src.open_vocab_0724.lineage import (  # noqa: E402
    build_lineage,
    file_sha256,
    validate_checkpoint,
)
from src.open_vocab_0724.metrics import (  # noqa: E402
    energy_structure_metrics,
    reconstruction_metrics,
    summarize,
)
from src.open_vocab_0724.model import FactorizedAudioModel  # noqa: E402
from src.open_vocab_0724.runtime import (  # noqa: E402
    audio_model_config,
    default_device,
    load_config,
    move_batch,
    resolve_config_path,
    seed_everything,
    write_json,
)


DATASETS = ("karaone", "feis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit validation-only v0724 audio reconstruction and freeze the "
            "checkpoint only when every absolute and counterfactual gate "
            "passes"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help=(
            "Diagnostic limit per dataset; limited audits can never pass or "
            "replace the frozen manifest"
        ),
    )
    parser.add_argument(
        "--verify-frozen",
        action="store_true",
        help=("Verify the existing gate/checkpoint/cache binding without " "decoding"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit status when the audit fails",
    )
    return parser.parse_args()


def preparation_config(cfg: dict[str, Any]) -> AudioPreparationConfig:
    audio = cfg["audio"]
    return AudioPreparationConfig(
        sample_rate=int(audio["sample_rate"]),
        max_active_seconds=float(audio["max_active_seconds"]),
        target_rms=float(audio["target_rms"]),
        active=ActiveSpeechConfig(
            sample_rate=int(audio["sample_rate"]),
            window_ms=float(audio["active_window_ms"]),
            hop_ms=float(audio["active_hop_ms"]),
            noise_margin_db=float(audio["active_noise_margin_db"]),
            peak_margin_db=float(audio["active_peak_margin_db"]),
            close_gap_ms=float(audio["active_close_gap_ms"]),
            context_ms=float(audio["active_context_ms"]),
        ),
    )


def reference_audio(
    metadata: dict[str, Any],
    cfg: dict[str, Any],
    codec_rate: int,
) -> np.ndarray:
    """Recreate and hash-check the exact cache-v2 waveform interval."""

    audio, source_rate = read_wav(Path(str(metadata["audio_path"])))
    prepared = prepare_waveform_segment(audio, source_rate, preparation_config(cfg))
    expected = str(
        metadata.get("segment_pcm_sha256") or metadata.get("pcm_sha256") or ""
    )
    if not expected:
        raise ValueError(
            f"Teacher cache has no segment hash for {metadata['audio_key']}"
        )
    if prepared.pcm_sha256 != expected:
        raise ValueError(
            f"Teacher/reference waveform mismatch for {metadata['audio_key']}"
        )
    value = prepared.waveform[: prepared.valid_samples]
    return resample_audio(value, prepared.sample_rate, codec_rate)


def same_label_controls(entries: list[dict[str, Any]]) -> list[int]:
    """Rotate in dataset/label groups; never use a label-changing control."""

    grouped: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        group = str(entry["dataset"]), str(entry["content_id"])
        grouped[group].append(index)
    controls = list(range(len(entries)))
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda index: str(entries[index]["audio_key"]))
        for position, index in enumerate(ordered):
            controls[index] = ordered[(position + 1) % len(ordered)]
    return controls


def code_accuracy(
    prediction: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray,
    codebook: int,
) -> float:
    mask = np.asarray(target_valid[codebook], dtype=bool)
    if not mask.any():
        return 0.0
    return float(
        np.mean(
            np.asarray(prediction[codebook])[mask] == np.asarray(target[codebook])[mask]
        )
    )


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def clustered_paired_bootstrap_lower(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    """Lower 95% bound after preserving every correct/control pair.

    Repeated utterances from one speaker are first averaged, then speakers are
    resampled.  This prevents a speaker with many trials from dominating the
    audio-only counterfactual gate.
    """

    finite = np.isfinite(values)
    values = np.asarray(values, dtype=np.float64)[finite]
    groups = np.asarray(groups, dtype=str)[finite]
    unique = np.unique(groups)
    if len(unique) == 0:
        return float("nan")
    cluster_values = np.asarray(
        [np.mean(values[groups == group]) for group in unique],
        dtype=np.float64,
    )
    if len(cluster_values) == 1:
        return float(cluster_values[0])
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(max(1, int(samples)), dtype=np.float64)
    for index in range(len(estimates)):
        estimates[index] = rng.choice(
            cluster_values, size=len(cluster_values), replace=True
        ).mean()
    return float(np.quantile(estimates, 0.025))


def concatenate_state(chunks: list[Any], name: str) -> torch.Tensor:
    values = [getattr(chunk, name).detach().cpu() for chunk in chunks]
    return torch.cat(values, dim=0)


def main() -> None:
    args = parse_args()
    if args.limit == 0 or args.limit < -1:
        raise ValueError("--limit must be -1 or a positive per-dataset count")

    config_path, cfg = load_config(args.config)
    context = load_context(config_path)
    checkpoint = resolve_config_path(config_path, cfg["paths"]["audio_checkpoint"])
    cache_path = resolve_config_path(config_path, cfg["paths"]["teacher_cache"])
    lineage = build_lineage(context)

    if args.verify_frozen:
        freeze = require_frozen_audio_checkpoint(config_path, cfg, lineage, checkpoint)
        print(
            json.dumps(
                {
                    "verified": True,
                    "audio_checkpoint": str(checkpoint),
                    "audio_checkpoint_sha256": file_sha256(checkpoint),
                    "teacher_cache": str(cache_path),
                    "teacher_cache_sha256": lineage["teacher_cache_sha256"],
                    "freeze": freeze,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    seed = int(cfg["training"]["seed"])
    seed_everything(seed)
    device = torch.device(args.device) if args.device else default_device()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, phase="audio", lineage=lineage, source=str(checkpoint))
    model = FactorizedAudioModel(audio_model_config(cfg)).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    teachers = TeacherCacheV2(cache_path)
    cache_audit = teachers.audit()
    evaluation_entries: list[dict[str, Any]] = []
    full_counts: dict[str, int] = {}
    for dataset_name in DATASETS:
        dataset = FactorizedAudioDataset(
            context,
            teachers,
            split="validation",
            datasets=(dataset_name,),
            allow_locked_test=False,
        )
        eligible = []
        for index in range(len(dataset)):
            item = dataset[index]
            if bool(item["audio_generation_eligible"].item()):
                eligible.append(item)
        full_counts[dataset_name] = len(eligible)
        if args.limit >= 0:
            eligible = eligible[: args.limit]
        evaluation_entries.extend(eligible)
    if not evaluation_entries:
        raise ValueError("No reconstruction-eligible validation audio was found")

    # FEIS has one held-out subject-label audio per validation label, so a
    # validation-only *reference* set cannot provide a different same-label
    # realization. Add at most one eligible training utterance per singleton
    # dataset/content group as a control source. It is never scored as a
    # reference and cannot affect validation normalization/statistics.
    entries = list(evaluation_entries)
    grouped_audio: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in entries:
        grouped_audio[(str(entry["dataset"]), str(entry["content_id"]))].add(
            str(entry["audio_key"])
        )
    for dataset_name in DATASETS:
        needed = {
            group
            for group, audio_keys in grouped_audio.items()
            if group[0] == dataset_name and len(audio_keys) < 2
        }
        if not needed:
            continue
        train_dataset = FactorizedAudioDataset(
            context,
            teachers,
            split="train",
            datasets=(dataset_name,),
            allow_locked_test=False,
        )
        for index in range(len(train_dataset)):
            item = train_dataset[index]
            group = str(item["dataset"]), str(item["content_id"])
            audio_key = str(item["audio_key"])
            if (
                group in needed
                and bool(item["audio_generation_eligible"].item())
                and audio_key not in grouped_audio[group]
            ):
                entries.append(item)
                grouped_audio[group].add(audio_key)
                needed.remove(group)
                if not needed:
                    break

    controls = same_label_controls(entries)
    batch_size = int(args.batch_size or cfg["training"]["audio_batch_size"])
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")

    state_chunks: list[Any] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(entries), batch_size),
            desc="[0724 audio-oracle] factorize",
            unit="batch",
        ):
            batch = move_batch(
                collate_factorized(entries[start : start + batch_size]), device
            )
            state_chunks.append(
                model.encode(
                    batch["content_tokens"],
                    batch["content_token_mask"],
                    batch["realization_features"],
                    batch["realization_frame_mask"],
                    batch["timbre_global"],
                )
            )

    content_tokens = concatenate_state(state_chunks, "content_tokens")
    realization_tokens = concatenate_state(state_chunks, "realization_tokens")
    timbre_global = concatenate_state(state_chunks, "timbre_global")
    content_mask = concatenate_state(state_chunks, "content_valid_mask").bool()
    realization_mask = concatenate_state(state_chunks, "realization_valid_mask").bool()
    correct_condition = concatenate_state(state_chunks, "fused_condition")
    correct_energy = concatenate_state(state_chunks, "log_mel_energy")
    correct_duration = concatenate_state(state_chunks, "duration_seconds")

    wrong_condition_chunks: list[torch.Tensor] = []
    wrong_energy_chunks: list[torch.Tensor] = []
    wrong_duration_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(entries), batch_size):
            stop = min(start + batch_size, len(entries))
            control_index = torch.tensor(controls[start:stop], dtype=torch.long)
            wrong = model.fuse(
                content_tokens[start:stop].to(device),
                realization_tokens[control_index].to(device),
                timbre_global[control_index].to(device),
                content_mask=content_mask[start:stop].to(device),
                realization_mask=realization_mask[control_index].to(device),
            )
            wrong_condition_chunks.append(wrong.fused_condition.cpu())
            wrong_energy_chunks.append(wrong.log_mel_energy.cpu())
            wrong_duration_chunks.append(wrong.duration_seconds.cpu())
    wrong_condition = torch.cat(wrong_condition_chunks)
    wrong_energy = torch.cat(wrong_energy_chunks)
    wrong_duration = torch.cat(wrong_duration_chunks)

    correct_codes: list[np.ndarray] = []
    wrong_codes: list[np.ndarray] = []
    correct_valid: list[np.ndarray] = []
    wrong_valid: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(entries), batch_size),
            desc="[0724 audio-oracle] MaskGIT",
            unit="batch",
        ):
            stop = min(start + batch_size, len(entries))
            codes, valid = model.decoder.generate(
                correct_condition[start:stop].to(device),
                correct_duration[start:stop].to(device),
            )
            shuffled_codes, shuffled_valid = model.decoder.generate(
                wrong_condition[start:stop].to(device),
                wrong_duration[start:stop].to(device),
            )
            correct_codes.extend(codes.cpu().numpy())
            wrong_codes.extend(shuffled_codes.cpu().numpy())
            correct_valid.extend(valid.cpu().numpy())
            wrong_valid.extend(shuffled_valid.cpu().numpy())

    codec = DiscreteEncodec(
        DiscreteEncodecConfig(
            model_path=str(
                resolve_config_path(config_path, cfg["paths"]["encodec_model"])
            ),
            sample_rate=int(cfg["codec"]["sample_rate"]),
            duration_sec=float(cfg["codec"]["max_duration_sec"]),
            bandwidth=float(cfg["codec"]["bandwidth"]),
        ),
        device,
    )

    records: list[dict[str, Any]] = []
    for index, entry in enumerate(
        tqdm(
            evaluation_entries,
            desc="[0724 audio-oracle] metrics",
            unit="audio",
        )
    ):
        control = entries[controls[index]]
        control_available = controls[index] != index
        if control_available and (
            str(control["dataset"]) != str(entry["dataset"])
            or str(control["content_id"]) != str(entry["content_id"])
            or str(control["audio_key"]) == str(entry["audio_key"])
        ):
            raise RuntimeError("same-label realization control is invalid")

        correct_steps = int(np.asarray(correct_valid[index], dtype=bool).sum())
        wrong_steps = int(np.asarray(wrong_valid[index], dtype=bool).sum())
        correct_waveform = np.asarray(
            codec.decode(correct_codes[index][:, :correct_steps], scale=None),
            dtype=np.float32,
        ).reshape(-1)
        wrong_waveform = np.asarray(
            codec.decode(wrong_codes[index][:, :wrong_steps], scale=None),
            dtype=np.float32,
        ).reshape(-1)
        metadata = teachers.metadata(str(entry["audio_key"]))
        reference = reference_audio(metadata, cfg, codec.codec_sample_rate)

        decoded_correct = reconstruction_metrics(
            reference,
            correct_waveform,
            codec.codec_sample_rate,
            max_lag_ms=float(cfg["evaluation"]["max_envelope_lag_ms"]),
        )
        decoded_wrong = reconstruction_metrics(
            reference,
            wrong_waveform,
            codec.codec_sample_rate,
            max_lag_ms=float(cfg["evaluation"]["max_envelope_lag_ms"]),
        )
        target_codes = entry["codes"].numpy()
        target_valid = entry["code_valid_mask"].numpy()
        decoded_correct["q0_accuracy"] = code_accuracy(
            correct_codes[index], target_codes, target_valid, 0
        )
        decoded_wrong["q0_accuracy"] = code_accuracy(
            wrong_codes[index], target_codes, target_valid, 0
        )

        reference_energy = entry["log_mel_energy"].numpy()
        direct_correct = energy_structure_metrics(
            reference_energy, correct_energy[index].numpy()
        )
        direct_wrong = energy_structure_metrics(
            reference_energy, wrong_energy[index].numpy()
        )
        soft_correct = float(direct_correct["soft_dtw_divergence"])
        soft_wrong = float(direct_wrong["soft_dtw_divergence"])
        record = {
            "audio_key": str(entry["audio_key"]),
            "dataset": str(entry["dataset"]),
            "label": str(entry["label"]),
            "content_id": str(entry["content_id"]),
            "subject_group_id": str(entry["eeg_subject_group_id"]),
            "pairing_scope": str(entry["pairing_scope"]),
            "same_label_control_available": control_available,
            "wrong_realization_audio_key": str(control["audio_key"]),
            "correct_condition": decoded_correct,
            "same_label_wrong_realization": decoded_wrong,
            "correct_energy": direct_correct,
            "same_label_wrong_energy": direct_wrong,
            "envelope_gain_over_same_label_wrong": float(
                decoded_correct["lag_envelope_correlation"]
                - decoded_wrong["lag_envelope_correlation"]
            ),
            "log_mel_gain_db_over_same_label_wrong": float(
                decoded_wrong["log_mel_mae_db"] - decoded_correct["log_mel_mae_db"]
            ),
            "morphology_gain_over_same_label_wrong": float(
                direct_correct["morphology_ssim"] - direct_wrong["morphology_ssim"]
            ),
            "soft_dtw_gain_over_same_label_wrong": soft_wrong - soft_correct,
            "soft_dtw_relative_gain_over_same_label_wrong": float(
                (soft_wrong - soft_correct) / max(abs(soft_wrong), 1.0e-8)
            ),
            "predicted_duration_seconds": float(correct_duration[index]),
            "wrong_realization_duration_seconds": float(wrong_duration[index]),
            "reference_duration_seconds": float(entry["duration_seconds"]),
        }
        records.append(record)

    absolute = cfg["evaluation"]["audio_oracle"]
    thresholds = {
        "minimum_median_lag_envelope_correlation": float(
            absolute["minimum_median_lag_envelope_correlation"]
        ),
        "minimum_median_modulation_correlation": float(
            absolute["minimum_median_modulation_correlation"]
        ),
        "maximum_median_log_mel_mae_db": float(
            absolute["maximum_median_log_mel_mae_db"]
        ),
        "minimum_median_q0_accuracy": float(absolute["minimum_median_q0_accuracy"]),
        "minimum_median_envelope_gain_over_shuffled": float(
            absolute["minimum_median_envelope_gain_over_shuffled"]
        ),
        "minimum_median_log_mel_gain_db_over_shuffled": float(
            absolute["minimum_median_log_mel_gain_db_over_shuffled"]
        ),
        "minimum_median_morphology_gain": float(
            cfg["evaluation"]["morphology_ssim_minimum_gain"]
        ),
        "minimum_median_soft_dtw_relative_gain": float(
            cfg["evaluation"]["soft_dtw_minimum_relative_gain"]
        ),
        "minimum_soft_dtw_paired_bootstrap_lower_95": float(
            cfg["evaluation"]["bootstrap_lower_bound"]
        ),
    }

    all_checks: dict[str, bool] = {
        "validation_not_limited": args.limit < 0,
        "checkpoint_not_diagnostic_smoke": not bool(
            payload.get("diagnostic_smoke", False)
        ),
        "checkpoint_is_primary_registered_run": payload.get("run")
        == {
            "seed": seed,
            "generalization": "g1",
            "holdout_label": None,
            "loso_subject": None,
        },
        "cache_hash_audit": bool(cache_audit["passed"]),
        "cache_schema_matches": (
            lineage.get("teacher_cache_schema") == TeacherCacheV2.SCHEMA_VERSION
        ),
    }
    dataset_reports: dict[str, Any] = {}
    bootstrap_samples = int(cfg["evaluation"]["bootstrap_samples"])
    for dataset_offset, dataset_name in enumerate(DATASETS):
        rows = [record for record in records if record["dataset"] == dataset_name]
        correct = [record["correct_condition"] for record in rows]
        wrong_records = [record["same_label_wrong_realization"] for record in rows]
        correct_maps = [record["correct_energy"] for record in rows]
        morphology = np.asarray(
            [record["morphology_gain_over_same_label_wrong"] for record in rows],
            dtype=np.float64,
        )
        soft_absolute = np.asarray(
            [record["soft_dtw_gain_over_same_label_wrong"] for record in rows],
            dtype=np.float64,
        )
        soft_relative = np.asarray(
            [record["soft_dtw_relative_gain_over_same_label_wrong"] for record in rows],
            dtype=np.float64,
        )
        groups = np.asarray([record["subject_group_id"] for record in rows], dtype=str)
        morphology_bootstrap = clustered_paired_bootstrap_lower(
            morphology,
            groups,
            samples=bootstrap_samples,
            seed=seed + dataset_offset * 17,
        )
        soft_dtw_bootstrap = clustered_paired_bootstrap_lower(
            soft_absolute,
            groups,
            samples=bootstrap_samples,
            seed=seed + dataset_offset * 17 + 1,
        )
        values = {
            "n_unique_validation_audio": len(rows),
            "n_full_unique_validation_audio": int(full_counts[dataset_name]),
            "same_label_control_coverage": (
                float(
                    np.mean([record["same_label_control_available"] for record in rows])
                )
                if rows
                else 0.0
            ),
            "median_lag_envelope_correlation": finite_median(
                value["lag_envelope_correlation"] for value in correct
            ),
            "median_modulation_correlation": finite_median(
                value["modulation_correlation"] for value in correct
            ),
            "median_log_mel_mae_db": finite_median(
                value["log_mel_mae_db"] for value in correct
            ),
            "median_energy_map_log_mel_mae_db": finite_median(
                value["native_log_mel_mae_db"] for value in correct_maps
            ),
            "median_q0_accuracy": finite_median(
                value["q0_accuracy"] for value in correct
            ),
            "median_envelope_gain_over_same_label_wrong": finite_median(
                record["envelope_gain_over_same_label_wrong"] for record in rows
            ),
            "median_log_mel_gain_db_over_same_label_wrong": finite_median(
                record["log_mel_gain_db_over_same_label_wrong"] for record in rows
            ),
            "median_morphology_gain_over_same_label_wrong": finite_median(morphology),
            "morphology_paired_bootstrap_lower_95": morphology_bootstrap,
            "median_soft_dtw_relative_gain_"
            "over_same_label_wrong": finite_median(soft_relative),
            "soft_dtw_paired_bootstrap_lower_95": soft_dtw_bootstrap,
        }
        checks = {
            "has_validation_audio": bool(rows),
            "full_same_label_control_coverage": bool(
                rows and values["same_label_control_coverage"] == 1.0
            ),
            "envelope_absolute": bool(
                np.isfinite(values["median_lag_envelope_correlation"])
                and values["median_lag_envelope_correlation"]
                >= thresholds["minimum_median_lag_envelope_correlation"]
            ),
            "modulation_absolute": bool(
                np.isfinite(values["median_modulation_correlation"])
                and values["median_modulation_correlation"]
                >= thresholds["minimum_median_modulation_correlation"]
            ),
            "log_mel_absolute": bool(
                np.isfinite(values["median_log_mel_mae_db"])
                and values["median_log_mel_mae_db"]
                <= thresholds["maximum_median_log_mel_mae_db"]
            ),
            "energy_map_log_mel_absolute": bool(
                np.isfinite(values["median_energy_map_log_mel_mae_db"])
                and values["median_energy_map_log_mel_mae_db"]
                <= thresholds["maximum_median_log_mel_mae_db"]
            ),
            "q0_absolute": bool(
                np.isfinite(values["median_q0_accuracy"])
                and values["median_q0_accuracy"]
                >= thresholds["minimum_median_q0_accuracy"]
            ),
            "envelope_condition_specific": bool(
                np.isfinite(values["median_envelope_gain_over_same_label_wrong"])
                and values["median_envelope_gain_over_same_label_wrong"]
                >= thresholds["minimum_median_envelope_gain_over_shuffled"]
            ),
            "log_mel_condition_specific": bool(
                np.isfinite(values["median_log_mel_gain_db_over_same_label_wrong"])
                and values["median_log_mel_gain_db_over_same_label_wrong"]
                >= thresholds["minimum_median_log_mel_gain_db_over_shuffled"]
            ),
            "morphology_condition_specific": bool(
                np.isfinite(values["median_morphology_gain_over_same_label_wrong"])
                and values["median_morphology_gain_over_same_label_wrong"]
                >= thresholds["minimum_median_morphology_gain"]
            ),
            "soft_dtw_relative_condition_specific": bool(
                np.isfinite(
                    values["median_soft_dtw_relative_gain_over_same_label_wrong"]
                )
                and values["median_soft_dtw_relative_gain_over_same_label_wrong"]
                >= thresholds["minimum_median_soft_dtw_relative_gain"]
            ),
            "soft_dtw_paired_bootstrap": bool(
                np.isfinite(values["soft_dtw_paired_bootstrap_lower_95"])
                and values["soft_dtw_paired_bootstrap_lower_95"]
                > thresholds["minimum_soft_dtw_paired_bootstrap_lower_95"]
            ),
        }
        all_checks.update(
            {f"{dataset_name}:{name}": value for name, value in checks.items()}
        )
        dataset_reports[dataset_name] = {
            "values": values,
            "checks": checks,
            "correct_summary": summarize(correct),
            "same_label_wrong_summary": summarize(wrong_records),
        }

    failed = sorted(name for name, passed in all_checks.items() if not passed)
    passed = not failed
    configured_gate_path = resolve_config_path(
        config_path, cfg["paths"]["audio_oracle_gate"]
    )
    gate_path = (
        configured_gate_path
        if args.limit < 0
        else configured_gate_path.with_name(
            f"{configured_gate_path.stem}.diagnostic_limit_{args.limit}.json"
        )
    )
    report = {
        "schema_version": AUDIO_ORACLE_GATE_SCHEMA,
        "passed": passed,
        "failed_checks": failed,
        "checks": all_checks,
        "thresholds": thresholds,
        "datasets": dataset_reports,
        "audio_checkpoint": str(checkpoint),
        "audio_checkpoint_sha256": file_sha256(checkpoint),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "teacher_cache": str(cache_path),
        "teacher_cache_sha256": lineage["teacher_cache_sha256"],
        "teacher_cache_schema": lineage["teacher_cache_schema"],
        "teacher_cache_audit": cache_audit,
        "lineage": lineage,
        "selection_split": "validation",
        "labels_used_for_generation": False,
        "labels_used_for_counterfactual_selection": True,
        "same_label_control_changes_content": False,
        "metrics_use_png_pixels": False,
        "frequency_axis_scaled": False,
        "test_accessed": False,
        "samples": records,
    }
    write_json(gate_path, report)

    freeze_path = resolve_config_path(
        config_path, cfg["paths"]["audio_freeze_manifest"]
    )
    if passed and args.limit < 0:
        freeze = {
            "schema_version": AUDIO_FREEZE_SCHEMA,
            "audio_checkpoint": str(checkpoint),
            "audio_checkpoint_sha256": file_sha256(checkpoint),
            "audio_oracle_gate": str(gate_path),
            "audio_oracle_gate_sha256": file_sha256(gate_path),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "teacher_cache": str(cache_path),
            "teacher_cache_sha256": lineage["teacher_cache_sha256"],
            "teacher_cache_schema": lineage["teacher_cache_schema"],
            "lineage": lineage,
            "policy": "freeze_audio_teacher_and_decoder_before_eeg_training",
            "test_accessed": False,
        }
        write_json(freeze_path, freeze)
    elif args.limit < 0 and freeze_path.exists():
        # A stale passing freeze must never survive a newly failed full audit.
        freeze_path.unlink()

    print(
        json.dumps(
            {
                "passed": passed,
                "gate": str(gate_path),
                "freeze": (str(freeze_path) if passed and args.limit < 0 else None),
                "failed_checks": failed,
                "datasets": {
                    name: report["datasets"][name]["values"] for name in DATASETS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
