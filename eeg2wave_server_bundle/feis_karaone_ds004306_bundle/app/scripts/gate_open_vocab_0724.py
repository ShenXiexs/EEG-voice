#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0724.audio_gate import (  # noqa: E402
    AUDIO_FREEZE_SCHEMA,
    AUDIO_ORACLE_GATE_SCHEMA,
    require_frozen_audio_checkpoint,
)
from src.open_vocab_0724.data import TeacherCacheV2, load_context  # noqa: E402
from src.open_vocab_0724.lineage import (  # noqa: E402
    VALIDATION_GATE_SCHEMA_VERSION,
    build_lineage,
    file_sha256,
    validate_lineage,
)
from src.open_vocab_0724.runtime import (  # noqa: E402
    load_config,
    resolve_config_path,
    resolve_evaluation_output,
    resolve_run_checkpoint,
    run_identifier,
    write_json,
)


SYNTHESIS_SCHEMA_VERSION = "openvoice-0724-synthesis-v1"
REQUIRED_COUNTERFACTUAL_MODES = (
    "correct_content_correct_realization",
    "correct_content_wrong_realization",
    "wrong_content_correct_realization",
    "wrong_content_wrong_realization",
    "content_only",
    "realization_only",
    "shuffled_eeg",
    "zero_eeg",
)


def resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
    )


def expected_sample_keys(
    context: Any,
    teachers: TeacherCacheV2,
    *,
    dataset: str,
    loso_subject: str | None = None,
) -> set[str]:
    keys: set[str] = set()
    for row in context.rows:
        if str(row["dataset"]) != dataset:
            continue
        metadata = teachers.metadata(str(row["audio_key"]))
        if not bool(metadata.get("reconstruction_eligible", False)):
            continue
        actual_split = context.split_for(row)
        if loso_subject is None:
            selected = actual_split == "validation"
        else:
            selected = actual_split == "train" and str(row["subject_group_id"]) == str(
                loso_subject
            )
        if selected:
            keys.add(str(row["sample_key"]))
    return keys


def required_loso_subjects(context: Any) -> list[str]:
    return sorted(
        {
            str(row["subject_group_id"])
            for row in context.rows
            if str(row["dataset"]) == "karaone" and context.split_for(row) == "train"
        }
    )


def default_loso_manifest_path(
    config_path: Path, cfg: dict[str, Any], subject: str
) -> Path:
    run_id = run_identifier(
        cfg,
        seed=int(cfg["training"]["seed"]),
        loso_subject=subject,
        generalization="g1",
        holdout_label=None,
    )
    if run_id is None:
        raise RuntimeError("A LOSO run must have an isolated run identifier")
    return (
        resolve_config_path(config_path, cfg["paths"]["output_root"])
        / "synthesis"
        / "karaone"
        / "validation"
        / "runs"
        / run_id
        / "synthesis_manifest.json"
    )


def records_have_required_modes(records: list[dict[str, Any]]) -> bool:
    required = set(REQUIRED_COUNTERFACTUAL_MODES)
    return bool(records) and all(
        required <= set((record.get("metrics") or {}).keys()) for record in records
    )


def records_have_valid_controls(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(
        bool((record.get("controls") or {}).get("same_label_control_available"))
        and bool((record.get("controls") or {}).get("wrong_label_control_available"))
        and bool((record.get("controls") or {}).get("shuffled_control_available"))
        for record in records
    )


def manifest_coverage(
    source: dict[str, Any], expected_keys: set[str]
) -> tuple[bool, list[str]]:
    observed = [
        str(record.get("sample_key", "")) for record in source.get("records", [])
    ]
    unique = set(observed)
    complete = bool(
        int(source.get("diagnostic_limit", -2)) == -1
        and int(source.get("full_dataset_record_count", -1)) == len(expected_keys)
        and len(observed) == len(unique) == len(expected_keys)
        and unique == expected_keys
    )
    return complete, sorted(expected_keys - unique)


def lineage_matches(saved: Any, expected: dict[str, Any]) -> bool:
    try:
        validate_lineage(saved, expected, source="v0724 synthesis manifest")
    except (TypeError, ValueError):
        return False
    return True


def development_artifact_audit(
    config_path: Path,
    cfg: dict[str, Any],
    lineage: dict[str, Any],
    subjects: list[str],
    audio_checkpoint: Path,
) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    settings = [(None, seed) for seed in seeds] + [
        (subject, seed) for subject in subjects for seed in seeds
    ]
    audio_hash = file_sha256(audio_checkpoint) if audio_checkpoint.is_file() else None
    for subject, seed in settings:
        run_name = f"{subject or 'official'}::seed_{seed}"
        checkpoint = resolve_run_checkpoint(
            config_path,
            cfg,
            "eeg_checkpoint",
            seed=seed,
            loso_subject=subject,
            generalization="g1",
            holdout_label=None,
        )
        evaluation = resolve_evaluation_output(
            config_path,
            cfg,
            split="validation",
            seed=seed,
            loso_subject=subject,
            generalization="g1",
            holdout_label=None,
        )
        payload: dict[str, Any] = {}
        if evaluation.is_file():
            try:
                payload = json.loads(evaluation.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                payload = {}
        expected_run = {
            "seed": seed,
            "generalization": "g1",
            "holdout_label": None,
            "loso_subject": subject,
        }
        loss = (payload.get("metrics") or {}).get("loss")
        checks = {
            "checkpoint_exists": checkpoint.is_file(),
            "evaluation_exists": evaluation.is_file(),
            "schema": payload.get("schema_version")
            == "openvoice-0724-latent-evaluation-v1",
            "validation_split": payload.get("split") == "validation",
            "run": payload.get("run") == expected_run,
            "not_diagnostic": not bool(payload.get("diagnostic_smoke", True)),
            "test_not_accessed": not bool(payload.get("test_accessed", True)),
            "lineage": lineage_matches(payload.get("lineage"), lineage),
            "finite_validation_loss": bool(
                isinstance(loss, (int, float)) and np.isfinite(float(loss))
            ),
            "audio_checkpoint_binding": bool(
                audio_hash is not None
                and payload.get("audio_checkpoint_sha256") == audio_hash
            ),
            "eeg_checkpoint_binding": bool(
                checkpoint.is_file()
                and payload.get("eeg_checkpoint_sha256") == file_sha256(checkpoint)
            ),
        }
        details[run_name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "checkpoint": str(checkpoint),
            "evaluation": str(evaluation),
        }
    return bool(details and all(item["passed"] for item in details.values())), details


def metric(record: dict[str, Any], mode: str, names: Iterable[str]) -> float:
    container = record.get("metrics", {}).get(mode)
    if container is None:
        container = record.get("modes", {}).get(mode, {}).get("metrics", {})
    for name in names:
        value = (container or {}).get(name)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return float("nan")


def subject_macro(values: np.ndarray, subjects: np.ndarray) -> float:
    return float(
        np.mean(
            [np.mean(values[subjects == subject]) for subject in np.unique(subjects)]
        )
    )


def subject_bootstrap_lower(
    values: np.ndarray,
    subjects: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    unique = np.unique(subjects)
    if len(unique) == 0:
        return float("nan")
    by_subject = np.asarray(
        [np.mean(values[subjects == subject]) for subject in unique], dtype=np.float64
    )
    if len(by_subject) == 1:
        return float(by_subject[0])
    rng = np.random.default_rng(seed)
    estimates = np.empty(max(1, int(samples)), dtype=np.float64)
    for index in range(len(estimates)):
        estimates[index] = rng.choice(
            by_subject, size=len(by_subject), replace=True
        ).mean()
    return float(np.quantile(estimates, 0.025))


def paired_values(
    records: list[dict[str, Any]],
    *,
    correct_mode: str,
    control_mode: str,
    metric_names: tuple[str, ...],
    lower_is_better: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correct, control, subjects = [], [], []
    for record in records:
        first = metric(record, correct_mode, metric_names)
        second = metric(record, control_mode, metric_names)
        if not np.isfinite(first) or not np.isfinite(second):
            continue
        correct.append(first)
        control.append(second)
        subjects.append(str(record.get("subject_group_id", "unknown")))
    first_array = np.asarray(correct, dtype=np.float64)
    second_array = np.asarray(control, dtype=np.float64)
    difference = (
        second_array - first_array if lower_is_better else first_array - second_array
    )
    return first_array, second_array, difference, np.asarray(subjects, dtype=str)


def selectivity_margin(
    records: list[dict[str, Any]],
    *,
    correct_mode: str,
    intended_control: str,
    cross_control: str,
    metric_names: tuple[str, ...],
    lower_is_better: bool,
) -> tuple[np.ndarray, np.ndarray]:
    margins, subjects = [], []
    for record in records:
        correct = metric(record, correct_mode, metric_names)
        intended = metric(record, intended_control, metric_names)
        cross = metric(record, cross_control, metric_names)
        if not all(np.isfinite(value) for value in (correct, intended, cross)):
            continue
        intended_gain = intended - correct if lower_is_better else correct - intended
        cross_change = abs(cross - correct)
        margins.append(intended_gain - cross_change)
        subjects.append(str(record.get("subject_group_id", "unknown")))
    return np.asarray(margins, dtype=np.float64), np.asarray(subjects, dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate v0724 factorized validation reconstruction"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synthesis-manifest", type=Path)
    parser.add_argument(
        "--loso-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Primary-seed KaraOne LOSO manifest; repeat for every train subject. "
            "When omitted, manifests are resolved from the standard run paths."
        ),
    )
    parser.add_argument(
        "--list-required-loso-subjects",
        action="store_true",
        help="Print the locked train-split KaraOne subjects and exit",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config_path, cfg = load_config(args.config)
    context = load_context(config_path)
    expected_subjects = required_loso_subjects(context)
    if args.list_required_loso_subjects:
        print("\n".join(expected_subjects))
        return
    if args.synthesis_manifest is None:
        parser.error("--synthesis-manifest is required unless listing LOSO subjects")
    teachers = TeacherCacheV2(
        resolve_config_path(config_path, cfg["paths"]["teacher_cache"])
    )
    source = json.loads(args.synthesis_manifest.resolve().read_text(encoding="utf-8"))
    if source.get("split") != "validation":
        raise ValueError(
            "Only a validation synthesis manifest may create the locked-test gate"
        )
    if source.get("dataset") != "karaone":
        raise ValueError(
            "The top-level v0724 reconstruction gate is KaraOne exact-pair only"
        )
    if source.get("loso_subject") is not None:
        raise ValueError("A development LOSO manifest cannot unlock the locked test")
    if int(source.get("seed", cfg["training"]["seed"])) != int(cfg["training"]["seed"]):
        raise ValueError(
            "Only the preregistered primary seed may create the locked-test gate"
        )
    records = list(source.get("records") or [])
    if not records:
        raise ValueError("Synthesis manifest has no records")
    current_lineage = build_lineage(context)
    expected_validation = expected_sample_keys(context, teachers, dataset="karaone")
    coverage_complete, missing_validation_keys = manifest_coverage(
        source, expected_validation
    )

    correct_mode = "correct_content_correct_realization"
    morphology = paired_values(
        records,
        correct_mode=correct_mode,
        control_mode="correct_content_wrong_realization",
        metric_names=(
            "morphology_ssim",
            "foreground_weighted_ssim",
            "time_normalized_ssim",
        ),
    )
    soft_dtw = paired_values(
        records,
        correct_mode=correct_mode,
        control_mode="correct_content_wrong_realization",
        metric_names=("soft_dtw_divergence", "mel_soft_dtw_divergence"),
        lower_is_better=True,
    )
    content = paired_values(
        records,
        correct_mode=correct_mode,
        control_mode="wrong_content_correct_realization",
        metric_names=("speech_bertscore", "hubert_frame_matching_f1"),
    )
    same_label_content = paired_values(
        records,
        correct_mode=correct_mode,
        control_mode="correct_content_wrong_realization",
        metric_names=("speech_bertscore", "hubert_frame_matching_f1"),
    )
    duration_margin_values, duration_margin_subjects = selectivity_margin(
        records,
        correct_mode=correct_mode,
        intended_control="correct_content_wrong_realization",
        cross_control="wrong_content_correct_realization",
        metric_names=("predicted_duration_error_seconds",),
        lower_is_better=True,
    )

    audio_checkpoint = resolve(config_path, cfg["paths"]["audio_checkpoint"])
    loso_paths = (
        [path.resolve() for path in args.loso_manifest]
        if args.loso_manifest
        else [
            default_loso_manifest_path(config_path, cfg, subject)
            for subject in expected_subjects
        ]
    )
    loso_records: list[dict[str, Any]] = []
    loso_details: dict[str, Any] = {}
    valid_loso_subjects: list[str] = []
    seen_loso_subjects: list[str] = []
    missing_loso_manifests: list[str] = []
    for path in loso_paths:
        if not path.is_file():
            missing_loso_manifests.append(str(path))
            continue
        loso_source = json.loads(path.read_text(encoding="utf-8"))
        subject = str(loso_source.get("loso_subject") or "")
        seen_loso_subjects.append(subject)
        expected_keys = expected_sample_keys(
            context,
            teachers,
            dataset="karaone",
            loso_subject=subject,
        )
        complete, missing_keys = manifest_coverage(loso_source, expected_keys)
        fold_records = list(loso_source.get("records") or [])
        expected_checkpoint = resolve_run_checkpoint(
            config_path,
            cfg,
            "eeg_checkpoint",
            seed=int(cfg["training"]["seed"]),
            loso_subject=subject,
            generalization="g1",
            holdout_label=None,
        )
        fold_checks = {
            "schema": loso_source.get("schema_version") == SYNTHESIS_SCHEMA_VERSION,
            "dataset": loso_source.get("dataset") == "karaone",
            "split": loso_source.get("split") == "validation",
            "generalization": loso_source.get("generalization") == "g1",
            "holdout_label": loso_source.get("holdout_label") is None,
            "primary_seed": int(loso_source.get("seed", -1))
            == int(cfg["training"]["seed"]),
            "known_subject": subject in expected_subjects,
            "full_fold_coverage": complete,
            "required_modes": records_have_required_modes(fold_records),
            "valid_controls": records_have_valid_controls(fold_records),
            "decoded_content_metric": bool(
                loso_source.get("decoded_content_metric_available")
            ),
            "test_not_accessed": not bool(loso_source.get("test_accessed", True)),
            "lineage": lineage_matches(loso_source.get("lineage"), current_lineage),
            "audio_checkpoint_binding": bool(
                audio_checkpoint.is_file()
                and loso_source.get("audio_checkpoint_sha256")
                == file_sha256(audio_checkpoint)
            ),
            "eeg_checkpoint_binding": bool(
                expected_checkpoint.is_file()
                and loso_source.get("eeg_checkpoint_sha256")
                == file_sha256(expected_checkpoint)
            ),
            "record_subject_binding": bool(
                fold_records
                and all(
                    str(record.get("subject_group_id")) == subject
                    for record in fold_records
                )
            ),
        }
        fold_passed = all(fold_checks.values())
        loso_details[subject or str(path)] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "checks": fold_checks,
            "passed": fold_passed,
            "record_count": len(fold_records),
            "missing_sample_keys": missing_keys,
        }
        if fold_passed:
            valid_loso_subjects.append(subject)
            loso_records.extend(fold_records)

    loso_morphology = paired_values(
        loso_records,
        correct_mode=correct_mode,
        control_mode="correct_content_wrong_realization",
        metric_names=(
            "morphology_ssim",
            "foreground_weighted_ssim",
            "time_normalized_ssim",
        ),
    )
    loso_soft_dtw = paired_values(
        loso_records,
        correct_mode=correct_mode,
        control_mode="correct_content_wrong_realization",
        metric_names=("soft_dtw_divergence", "mel_soft_dtw_divergence"),
        lower_is_better=True,
    )
    loso_content = paired_values(
        loso_records,
        correct_mode=correct_mode,
        control_mode="wrong_content_correct_realization",
        metric_names=("speech_bertscore", "hubert_frame_matching_f1"),
    )
    loso_duration_values, loso_duration_subjects = selectivity_margin(
        loso_records,
        correct_mode=correct_mode,
        intended_control="correct_content_wrong_realization",
        cross_control="wrong_content_correct_realization",
        metric_names=("predicted_duration_error_seconds",),
        lower_is_better=True,
    )
    development_complete, development_details = development_artifact_audit(
        config_path,
        cfg,
        current_lineage,
        expected_subjects,
        audio_checkpoint,
    )

    bootstrap_samples = int(cfg["evaluation"]["bootstrap_samples"])
    seed = int(cfg["training"]["seed"])
    morphology_gain = (
        float(np.median(morphology[2])) if len(morphology[2]) else float("nan")
    )
    morphology_ci = subject_bootstrap_lower(
        loso_morphology[2],
        loso_morphology[3],
        samples=bootstrap_samples,
        seed=seed,
    )
    morphology_win = (
        float(np.mean(morphology[2] > 0)) if len(morphology[2]) else float("nan")
    )
    soft_relative = (soft_dtw[1] - soft_dtw[0]) / np.maximum(np.abs(soft_dtw[1]), 1e-8)
    soft_gain = float(np.median(soft_relative)) if len(soft_relative) else float("nan")
    soft_ci = subject_bootstrap_lower(
        loso_soft_dtw[2],
        loso_soft_dtw[3],
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    content_gain = float(np.median(content[2])) if len(content[2]) else float("nan")
    content_ci = subject_bootstrap_lower(
        loso_content[2],
        loso_content[3],
        samples=bootstrap_samples,
        seed=seed + 2,
    )
    content_win = float(np.mean(content[2] > 0)) if len(content[2]) else float("nan")
    same_label_content_change = (
        float(np.median(np.abs(same_label_content[2])))
        if len(same_label_content[2])
        else float("nan")
    )
    content_selectivity_ratio = (
        same_label_content_change / max(abs(content_gain), 1e-8)
        if np.isfinite(same_label_content_change) and np.isfinite(content_gain)
        else float("nan")
    )
    duration_specificity = (
        float(np.median(duration_margin_values))
        if len(duration_margin_values)
        else float("nan")
    )
    duration_specificity_ci = subject_bootstrap_lower(
        loso_duration_values,
        loso_duration_subjects,
        samples=bootstrap_samples,
        seed=seed + 3,
    )
    loso_subject_count = len(set(valid_loso_subjects))

    retrieval = source.get("retrieval", {})
    top1 = float(retrieval.get("macro_top1", float("nan")))
    chance = float(retrieval.get("balanced_chance", float("nan")))
    chance_multiple = (
        top1 / chance
        if np.isfinite(top1) and np.isfinite(chance) and chance > 0
        else float("nan")
    )
    required = cfg["evaluation"]
    checks = {
        "synthesis_schema": source.get("schema_version") == SYNTHESIS_SCHEMA_VERSION,
        "primary_g1_run": bool(
            source.get("generalization") == "g1"
            and source.get("holdout_label") is None
            and source.get("loso_subject") is None
            and int(source.get("seed", -1)) == seed
        ),
        "validation_not_diagnostic": int(source.get("diagnostic_limit", -2)) == -1,
        "validation_full_coverage": coverage_complete,
        "validation_required_modes": records_have_required_modes(records),
        "validation_valid_controls": records_have_valid_controls(records),
        "validation_decoded_content_metric": bool(
            source.get("decoded_content_metric_available")
        ),
        "validation_test_not_accessed": not bool(source.get("test_accessed", True)),
        "validation_lineage": lineage_matches(source.get("lineage"), current_lineage),
        "morphology_metric_full_coverage": len(morphology[2]) == len(records),
        "soft_dtw_metric_full_coverage": len(soft_dtw[2]) == len(records),
        "content_metric_full_coverage": len(content[2]) == len(records),
        "same_label_content_metric_full_coverage": len(same_label_content[2])
        == len(records),
        "duration_metric_full_coverage": len(duration_margin_values) == len(records),
        "loso_manifest_files_complete": not missing_loso_manifests,
        "loso_subjects_complete": bool(
            len(seen_loso_subjects) == len(set(seen_loso_subjects))
            and set(valid_loso_subjects) == set(expected_subjects)
        ),
        "loso_subject_count": loso_subject_count
        >= int(required.get("minimum_subjects_for_bootstrap", 2)),
        "three_seed_and_loso_artifacts_complete": development_complete,
        "loso_morphology_metric_full_coverage": len(loso_morphology[2])
        == len(loso_records),
        "loso_soft_dtw_metric_full_coverage": len(loso_soft_dtw[2])
        == len(loso_records),
        "loso_content_metric_full_coverage": len(loso_content[2]) == len(loso_records),
        "loso_duration_metric_full_coverage": len(loso_duration_values)
        == len(loso_records),
        "morphology_gain": bool(
            np.isfinite(morphology_gain)
            and morphology_gain >= float(required["morphology_ssim_minimum_gain"])
        ),
        "morphology_subject_bootstrap": bool(
            loso_subject_count >= int(required.get("minimum_subjects_for_bootstrap", 2))
            and np.isfinite(morphology_ci)
            and morphology_ci > float(required["bootstrap_lower_bound"])
        ),
        "morphology_trial_win_rate": bool(
            np.isfinite(morphology_win)
            and morphology_win >= float(required["minimum_trial_win_rate"])
        ),
        "soft_dtw_relative_gain": bool(
            np.isfinite(soft_gain)
            and soft_gain >= float(required["soft_dtw_minimum_relative_gain"])
        ),
        "soft_dtw_subject_bootstrap": bool(
            loso_subject_count >= int(required.get("minimum_subjects_for_bootstrap", 2))
            and np.isfinite(soft_ci)
            and soft_ci > float(required["bootstrap_lower_bound"])
        ),
        "content_subject_bootstrap": bool(
            loso_subject_count >= int(required.get("minimum_subjects_for_bootstrap", 2))
            and np.isfinite(content_ci)
            and content_ci > float(required["bootstrap_lower_bound"])
        ),
        "content_trial_win_rate": bool(
            np.isfinite(content_win)
            and content_win >= float(required["minimum_trial_win_rate"])
        ),
        "content_retrieval": bool(
            np.isfinite(chance_multiple)
            and chance_multiple >= float(required["retrieval_minimum_chance_multiple"])
        ),
        "content_factor_selectivity": bool(
            np.isfinite(content_selectivity_ratio)
            and content_selectivity_ratio
            <= float(required["maximum_content_cross_factor_ratio"])
        ),
        "duration_factor_selectivity": bool(
            np.isfinite(duration_specificity)
            and duration_specificity
            > float(required["minimum_duration_specificity_margin"])
        ),
        "duration_factor_subject_bootstrap": bool(
            loso_subject_count >= int(required.get("minimum_subjects_for_bootstrap", 2))
            and np.isfinite(duration_specificity_ci)
            and duration_specificity_ci > float(required["bootstrap_lower_bound"])
        ),
    }

    audio_gate_path = resolve(config_path, cfg["paths"]["audio_oracle_gate"])
    audio_gate = (
        json.loads(audio_gate_path.read_text(encoding="utf-8"))
        if audio_gate_path.is_file()
        else {}
    )
    checks["audio_oracle"] = bool(
        audio_gate.get("schema_version") == AUDIO_ORACLE_GATE_SCHEMA
        and audio_gate.get("passed")
    )
    eeg_checkpoint = resolve(config_path, cfg["paths"]["eeg_checkpoint"])
    freeze_path = resolve(config_path, cfg["paths"]["audio_freeze_manifest"])
    freeze = (
        json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze_path.is_file()
        else {}
    )
    checks["audio_freeze_binding"] = bool(
        freeze.get("schema_version") == AUDIO_FREEZE_SCHEMA
        and audio_checkpoint.is_file()
        and audio_gate_path.is_file()
        and freeze.get("audio_checkpoint_sha256") == file_sha256(audio_checkpoint)
        and freeze.get("audio_oracle_gate_sha256") == file_sha256(audio_gate_path)
        and audio_gate.get("audio_checkpoint_sha256") == file_sha256(audio_checkpoint)
    )
    try:
        require_frozen_audio_checkpoint(
            config_path, cfg, current_lineage, audio_checkpoint
        )
    except (FileNotFoundError, PermissionError, TypeError, ValueError):
        checks["audio_freeze_lineage_binding"] = False
    else:
        checks["audio_freeze_lineage_binding"] = True
    checks["checkpoint_files"] = audio_checkpoint.is_file() and eeg_checkpoint.is_file()
    checks["synthesis_audio_checkpoint_binding"] = bool(
        audio_checkpoint.is_file()
        and source.get("audio_checkpoint_sha256") == file_sha256(audio_checkpoint)
    )
    checks["synthesis_eeg_checkpoint_binding"] = bool(
        eeg_checkpoint.is_file()
        and source.get("eeg_checkpoint_sha256") == file_sha256(eeg_checkpoint)
    )
    checks["synthesis_config_binding"] = bool(
        (source.get("lineage") or {}).get("config_sha256") == file_sha256(config_path)
    )
    failed = sorted(name for name, passed in checks.items() if not passed)

    report_path = resolve(config_path, cfg["paths"]["validation_report"])
    gate_path = resolve(config_path, cfg["paths"]["validation_gate"])
    report = {
        "schema_version": "openvoice-0724-validation-report-v1",
        "dataset": "karaone",
        "split": "validation",
        "synthesis_manifest": str(args.synthesis_manifest.resolve()),
        "synthesis_manifest_sha256": file_sha256(args.synthesis_manifest.resolve()),
        "record_count": len(records),
        "expected_record_count": len(expected_validation),
        "missing_validation_sample_keys": missing_validation_keys,
        "required_loso_subjects": expected_subjects,
        "valid_loso_subjects": sorted(set(valid_loso_subjects)),
        "missing_loso_manifests": missing_loso_manifests,
        "loso_manifests": loso_details,
        "development_artifacts": development_details,
        "metrics": {
            "morphology_median_gain": morphology_gain,
            "morphology_subject_bootstrap_lower_95": morphology_ci,
            "morphology_trial_win_rate": morphology_win,
            "soft_dtw_median_relative_gain": soft_gain,
            "soft_dtw_subject_bootstrap_lower_95": soft_ci,
            "content_median_gain": content_gain,
            "content_subject_bootstrap_lower_95": content_ci,
            "content_trial_win_rate": content_win,
            "retrieval_chance_multiple": chance_multiple,
            "same_label_content_median_absolute_change": same_label_content_change,
            "content_factor_selectivity_ratio": content_selectivity_ratio,
            "duration_factor_selectivity_median_margin": duration_specificity,
            "duration_factor_selectivity_subject_bootstrap_lower_95": duration_specificity_ci,
            "loso_subject_count": loso_subject_count,
        },
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
        "lineage": current_lineage,
        "audio_checkpoint_sha256": (
            file_sha256(audio_checkpoint) if audio_checkpoint.is_file() else "missing"
        ),
        "eeg_checkpoint_sha256": (
            file_sha256(eeg_checkpoint) if eeg_checkpoint.is_file() else "missing"
        ),
        "test_accessed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    gate = {
        "schema_version": VALIDATION_GATE_SCHEMA_VERSION,
        "passed": not failed,
        "failed_checks": failed,
        "validation_report": str(report_path),
        "validation_report_sha256": file_sha256(report_path),
        "lineage": current_lineage,
        "audio_checkpoint_sha256": (
            file_sha256(audio_checkpoint) if audio_checkpoint.is_file() else "missing"
        ),
        "eeg_checkpoint_sha256": (
            file_sha256(eeg_checkpoint) if eeg_checkpoint.is_file() else "missing"
        ),
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(gate_path, gate)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "gate": str(gate_path),
                "passed": not failed,
                "failed_checks": failed,
            },
            indent=2,
        )
    )
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
