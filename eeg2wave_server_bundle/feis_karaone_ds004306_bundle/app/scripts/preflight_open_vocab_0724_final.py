#!/usr/bin/env python3
"""Read-only preflight checks for the one-command v0724 formal run.

The ``before-test`` stage deliberately does not construct an EEG dataset or
open a teacher-cache shard.  It verifies only source/artifact metadata and the
validation-gate binding before the official runner atomically claims any locked
test component.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0724.lineage import authorize_locked_test_metadata  # noqa: E402
from src.open_vocab_0724.runtime import (  # noqa: E402
    load_config,
    resolve_config_path,
    resolve_run_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only prerequisites checks for the v0724 complete formal "
            "training/test/plot workflow"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("initial", "before-test"), default="initial"
    )
    parser.add_argument(
        "--print-output-root",
        action="store_true",
        help="Print the config-resolved artifact root and exit without other checks",
    )
    return parser.parse_args()


def writable_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def check_path(
    checks: dict[str, dict[str, Any]],
    name: str,
    path: Path,
    *,
    kind: str,
) -> None:
    expected = {"file": path.is_file, "directory": path.is_dir}[kind]
    checks[name] = {
        "path": str(path),
        "kind": kind,
        "passed": bool(expected()),
    }


def initial_checks(config_path: Path, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data = cfg["data"]
    paths = cfg["paths"]
    teachers = cfg["teachers"]
    eeg_root = resolve_config_path(config_path, data["eeg_output_root"])
    audio_root = resolve_config_path(config_path, data["audio_output_root"])
    output_root = resolve_config_path(config_path, paths["output_root"])
    checks: dict[str, dict[str, Any]] = {}
    for module in ("matplotlib", "numpy", "scipy", "torch", "tqdm", "yaml"):
        checks[f"python_module_{module}"] = {
            "kind": "importable_python_module",
            "passed": importlib.util.find_spec(module) is not None,
        }
    check_path(checks, "eeg_output_root", eeg_root, kind="directory")
    check_path(
        checks,
        "unified_trial_manifest",
        eeg_root / "manifests" / "unified_trials.csv",
        kind="file",
    )
    check_path(checks, "audio_output_root", audio_root, kind="directory")
    check_path(
        checks,
        "locked_subject_split",
        resolve_config_path(config_path, data["subject_split_file"]),
        kind="file",
    )
    check_path(
        checks,
        "label_holdout_folds",
        resolve_config_path(config_path, data["label_holdout_file"]),
        kind="file",
    )
    check_path(
        checks,
        "montage_registry",
        resolve_config_path(config_path, data["montage_registry"]),
        kind="file",
    )
    check_path(
        checks,
        "local_hubert_teacher",
        resolve_config_path(config_path, teachers["hubert_model"]),
        kind="directory",
    )
    check_path(
        checks,
        "local_encodec_decoder",
        resolve_config_path(config_path, paths["encodec_model"]),
        kind="directory",
    )
    writable_target = (
        output_root if output_root.exists() else writable_ancestor(output_root.parent)
    )
    checks["artifact_root_writable"] = {
        "path": str(writable_target),
        "kind": "writable_artifact_root_or_parent",
        "passed": bool(
            writable_target.is_dir() and os.access(writable_target, os.W_OK)
        ),
    }
    claim_path = (
        resolve_config_path(config_path, paths["validation_gate"]).parent
        / "locked_test_access"
        / "claim.json"
    )
    checks["locked_test_unclaimed"] = {
        "path": str(claim_path),
        "kind": "absent_file",
        "passed": not claim_path.exists(),
    }
    return checks


def before_test_checks(
    config_path: Path, cfg: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    paths = cfg["paths"]
    checks = initial_checks(config_path, cfg)
    teacher_cache = resolve_config_path(config_path, paths["teacher_cache"])
    audio_checkpoint = resolve_config_path(config_path, paths["audio_checkpoint"])
    eeg_checkpoint = resolve_run_checkpoint(config_path, cfg, "eeg_checkpoint")
    audio_gate = resolve_config_path(config_path, paths["audio_oracle_gate"])
    audio_freeze = resolve_config_path(config_path, paths["audio_freeze_manifest"])
    validation_gate = resolve_config_path(config_path, paths["validation_gate"])
    check_path(checks, "teacher_cache_index", teacher_cache / "index.json", kind="file")
    check_path(
        checks,
        "teacher_cache_train_statistics",
        teacher_cache / "train_statistics.npz",
        kind="file",
    )
    check_path(checks, "audio_checkpoint", audio_checkpoint, kind="file")
    check_path(checks, "eeg_checkpoint", eeg_checkpoint, kind="file")
    check_path(checks, "audio_oracle_gate", audio_gate, kind="file")
    check_path(checks, "audio_freeze_manifest", audio_freeze, kind="file")
    check_path(checks, "validation_gate", validation_gate, kind="file")

    required = (
        "audio_checkpoint",
        "eeg_checkpoint",
        "validation_gate",
        "locked_test_unclaimed",
    )
    if all(bool(checks[name]["passed"]) for name in required):
        try:
            # This reads checkpoint and gate hashes only.  It does not open any
            # test EEG payload, test teacher-cache shard, or test audio row.
            authorize_locked_test_metadata(
                validation_gate,
                config_path=config_path,
                audio_checkpoint=audio_checkpoint,
                eeg_checkpoint=eeg_checkpoint,
            )
        except (FileNotFoundError, PermissionError, TypeError, ValueError) as error:
            checks["locked_test_gate_binding"] = {
                "kind": "metadata_only_validation",
                "passed": False,
                "error": str(error),
            }
        else:
            checks["locked_test_gate_binding"] = {
                "kind": "metadata_only_validation",
                "passed": True,
            }
    else:
        checks["locked_test_gate_binding"] = {
            "kind": "metadata_only_validation",
            "passed": False,
            "error": "Skipped because a required precondition is missing",
        }
    return checks


def main() -> None:
    args = parse_args()
    config_path, cfg = load_config(args.config)
    output_root = resolve_config_path(config_path, cfg["paths"]["output_root"])
    if args.print_output_root:
        print(output_root)
        return

    checks = (
        initial_checks(config_path, cfg)
        if args.stage == "initial"
        else before_test_checks(config_path, cfg)
    )
    failed = sorted(name for name, value in checks.items() if not value["passed"])
    summary = {
        "schema_version": "openvoice-0724-formal-preflight-v1",
        "stage": args.stage,
        "config": str(config_path),
        "output_root": str(output_root),
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "test_data_opened": False,
        "locked_test_claim_created": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(
            "v0724 preflight failed before training/test access: " + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
