from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LINEAGE_SCHEMA_VERSION = "openvoice-0724-lineage-v1"
CHECKPOINT_SCHEMA_VERSION = "openvoice-0724-checkpoint-v1"
VALIDATION_GATE_SCHEMA_VERSION = "openvoice-0724-validation-gate-v1"
VALIDATION_REPORT_SCHEMA_VERSION = "openvoice-0724-validation-report-v1"
LOCKED_TEST_ACCESS_SCHEMA_VERSION = "openvoice-0724-locked-test-access-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_sha256(path: str | Path) -> str:
    value = Path(path)
    if value.is_file():
        return file_sha256(value)
    if not value.is_dir():
        raise FileNotFoundError(value)
    index = value / "index.json"
    if index.is_file():
        # The v2 index contains every shard digest. Hashing it is both faster
        # and stricter than repeatedly reading a multi-gigabyte cache.
        return file_sha256(index)
    digest = hashlib.sha256(b"openvoice-0724-directory-v1\0")
    for child in sorted(item for item in value.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(value)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def optional_sha256(path: str | Path | None) -> str:
    return path_sha256(path) if path is not None and Path(path).exists() else "absent"


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def eeg_payloads_sha256(root: Path, rows: Iterable[dict[str, str]]) -> str:
    digest = hashlib.sha256(b"openvoice-0724-eeg-payloads-v1\0")
    for relative in sorted({str(row["eeg_relpath"]) for row in rows}):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest-referenced EEG payload is missing: {path}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return (
        path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
    )


def _cache_schema(path: Path) -> str:
    index = path / "index.json" if path.is_dir() else None
    if index is not None and index.is_file():
        return str(
            json.loads(index.read_text(encoding="utf-8")).get(
                "schema_version", "unknown"
            )
        )
    if path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as raw:
            for key in ("schema_version", "cache_schema_version", "version"):
                if key in raw.files and np.asarray(raw[key]).size == 1:
                    return str(np.asarray(raw[key]).reshape(-1)[0])
    return "absent" if not path.exists() else "unknown"


def build_lineage(
    context: Any, *, require_teacher_cache: bool = True
) -> dict[str, Any]:
    cfg = context.config
    config_path = Path(context.config_path).resolve()
    teacher_cache = _resolve(config_path, cfg["paths"]["teacher_cache"])
    montage = _resolve(config_path, cfg["data"]["montage_registry"])
    if require_teacher_cache and not teacher_cache.exists():
        raise FileNotFoundError(f"v0724 teacher cache is missing: {teacher_cache}")
    model_cfg = cfg["model"]
    audio_model_cfg = {
        key: value for key, value in model_cfg.items() if not key.startswith("eeg_")
    }
    lineage = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "config_sha256": file_sha256(config_path),
        "subject_split_sha256": file_sha256(context.split_path),
        "subject_split_version": str(context.split.get("version", "unknown")),
        "manifest_sha256": file_sha256(context.manifest_path),
        "eeg_payloads_sha256": eeg_payloads_sha256(
            Path(context.eeg_root), context.rows
        ),
        "montage_registry_sha256": file_sha256(montage),
        "teacher_cache_sha256": optional_sha256(teacher_cache),
        "teacher_cache_schema": _cache_schema(teacher_cache),
        "pairing_policy_version": str(cfg["data"]["pairing_policy_version"]),
        "encodec_version": str(cfg["paths"]["encodec_model"]),
        "hubert_version": str(cfg["teachers"]["hubert_model"]),
        "hubert_layer": int(cfg["teachers"]["hubert_layer"]),
        "wavlm_version": str(cfg["teachers"].get("wavlm_model") or "disabled"),
        "audio_recipe_sha256": object_sha256(
            {
                "audio": cfg["audio"],
                "codec": cfg["codec"],
                "teachers": cfg["teachers"],
                "model": audio_model_cfg,
                "training": cfg["training"],
                "loss": cfg["loss"],
            }
        ),
        "eeg_recipe_sha256": object_sha256(
            {
                "data": cfg["data"],
                "model": cfg["model"],
                "loss": cfg["loss"],
                "training": cfg["training"],
                "evaluation": cfg["evaluation"],
                "experiment": cfg.get("experiment", {}),
            }
        ),
        "paths": {
            "config": str(config_path),
            "subject_split": str(context.split_path),
            "manifest": str(context.manifest_path),
            "montage_registry": str(montage),
            "teacher_cache": str(teacher_cache),
        },
    }
    return lineage


_AUDIO_KEYS = (
    "schema_version",
    "subject_split_sha256",
    "manifest_sha256",
    "teacher_cache_sha256",
    "teacher_cache_schema",
    "pairing_policy_version",
    "encodec_version",
    "hubert_version",
    "hubert_layer",
    "wavlm_version",
    "audio_recipe_sha256",
)
_FULL_KEYS = _AUDIO_KEYS + (
    "config_sha256",
    "eeg_payloads_sha256",
    "montage_registry_sha256",
    "eeg_recipe_sha256",
)


def comparable(lineage: dict[str, Any], *, scope: str = "full") -> dict[str, Any]:
    keys = _AUDIO_KEYS if scope == "audio" else _FULL_KEYS
    return {key: lineage.get(key) for key in keys}


def validate_lineage(
    saved: Any, expected: dict[str, Any], *, source: str, scope: str = "full"
) -> None:
    if not isinstance(saved, dict):
        raise ValueError(f"{source} has no v0724 lineage")
    saved_values = comparable(saved, scope=scope)
    expected_values = comparable(expected, scope=scope)
    mismatch = {
        key: {"saved": saved_values[key], "current": expected_values[key]}
        for key in expected_values
        if saved_values[key] != expected_values[key]
    }
    if mismatch:
        raise ValueError(
            f"{source} lineage mismatch: {json.dumps(mismatch, sort_keys=True)}"
        )


def checkpoint_payload(
    *,
    phase: str,
    lineage: dict[str, Any],
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    epoch: int,
    metrics: dict[str, Any],
    dependencies: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": str(phase),
        "lineage": lineage,
        "dependencies": dependencies or {},
        "epoch": int(epoch),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "metrics": metrics,
    }


def validate_checkpoint(
    payload: dict[str, Any],
    *,
    phase: str,
    lineage: dict[str, Any],
    source: str,
    dependencies: dict[str, str] | None = None,
) -> None:
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{source} is not a compatible v0724 checkpoint")
    if payload.get("phase") != phase:
        raise ValueError(
            f"{source} phase mismatch: {payload.get('phase')!r} != {phase!r}"
        )
    validate_lineage(
        payload.get("lineage"),
        lineage,
        source=source,
        scope="audio" if phase == "audio" else "full",
    )
    if (payload.get("dependencies") or {}) != (dependencies or {}):
        raise ValueError(f"{source} dependency mismatch")


def _validate_gate_report(gate: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(str(gate.get("validation_report") or ""))
    if not report_path.is_file():
        raise PermissionError(f"Validation report is missing: {report_path}")
    expected_hash = str(gate.get("validation_report_sha256") or "")
    observed_hash = file_sha256(report_path)
    if expected_hash != observed_hash:
        raise PermissionError("Validation report SHA256 no longer matches the gate")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != VALIDATION_REPORT_SCHEMA_VERSION
        or not bool(report.get("passed"))
        or report.get("split") != "validation"
        or bool(report.get("test_accessed", True))
    ):
        raise PermissionError("Validation report is not a passing pre-test report")
    synthesis_path = Path(str(report.get("synthesis_manifest") or ""))
    if not synthesis_path.is_file() or report.get(
        "synthesis_manifest_sha256"
    ) != file_sha256(synthesis_path):
        raise PermissionError(
            "Validation synthesis manifest no longer matches the report"
        )
    for details in (report.get("loso_manifests") or {}).values():
        loso_path = Path(str(details.get("path") or ""))
        if not loso_path.is_file() or details.get("sha256") != file_sha256(loso_path):
            raise PermissionError(
                "A LOSO synthesis manifest no longer matches the report"
            )
    gate_lineage = gate.get("lineage")
    if not isinstance(gate_lineage, dict):
        raise PermissionError("Validation gate has no lineage")
    try:
        validate_lineage(
            report.get("lineage"),
            gate_lineage,
            source="v0724 validation report",
        )
    except ValueError as error:
        raise PermissionError(str(error)) from error
    for key in ("audio_checkpoint_sha256", "eeg_checkpoint_sha256"):
        if report.get(key) != gate.get(key):
            raise PermissionError(f"Validation report {key} does not match the gate")
    return report


def authorize_locked_test(
    gate_path: str | Path,
    *,
    lineage: dict[str, Any],
    audio_checkpoint: str | Path,
    eeg_checkpoint: str | Path,
) -> dict[str, Any]:
    path = Path(gate_path)
    if not path.is_file():
        raise PermissionError(f"Validation gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != VALIDATION_GATE_SCHEMA_VERSION or not bool(
        gate.get("passed")
    ):
        raise PermissionError(
            f"v0724 validation gate has not passed: {gate.get('failed_checks', [])}"
        )
    _validate_gate_report(gate)
    validate_lineage(gate.get("lineage"), lineage, source="v0724 validation gate")
    bindings = {
        "audio_checkpoint_sha256": file_sha256(audio_checkpoint),
        "eeg_checkpoint_sha256": file_sha256(eeg_checkpoint),
    }
    mismatch = {
        key: {"saved": gate.get(key), "current": value}
        for key, value in bindings.items()
        if gate.get(key) != value
    }
    if mismatch:
        raise PermissionError(
            f"Validation gate checkpoint binding mismatch: {json.dumps(mismatch, sort_keys=True)}"
        )
    return gate


def authorize_locked_test_metadata(
    gate_path: str | Path,
    *,
    config_path: str | Path,
    audio_checkpoint: str | Path,
    eeg_checkpoint: str | Path,
) -> dict[str, Any]:
    """Authorize before any locked-test EEG payload or cache row is read."""

    path = Path(gate_path)
    if not path.is_file():
        raise PermissionError(f"Validation gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != VALIDATION_GATE_SCHEMA_VERSION or not bool(
        gate.get("passed")
    ):
        raise PermissionError(
            f"v0724 validation gate has not passed: {gate.get('failed_checks', [])}"
        )
    _validate_gate_report(gate)
    lineage = gate.get("lineage") or {}
    expected = {
        "config_sha256": file_sha256(config_path),
        "audio_checkpoint_sha256": file_sha256(audio_checkpoint),
        "eeg_checkpoint_sha256": file_sha256(eeg_checkpoint),
    }
    observed = {
        "config_sha256": lineage.get("config_sha256"),
        "audio_checkpoint_sha256": gate.get("audio_checkpoint_sha256"),
        "eeg_checkpoint_sha256": gate.get("eeg_checkpoint_sha256"),
    }
    mismatch = {
        key: {"saved": observed[key], "current": value}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatch:
        raise PermissionError(
            f"Validation gate metadata mismatch: {json.dumps(mismatch, sort_keys=True)}"
        )
    return gate


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def claim_locked_test_access(
    gate_path: str | Path,
    *,
    purpose: str,
    access_id: str | None = None,
) -> dict[str, Any]:
    """Atomically consume one component of the single final-test session.

    The claim is created before a test dataset or cache row is constructed. A
    shared ``access_id`` lets the official runner execute latent evaluation and
    the two dataset-specific synthesis components as one session, while every
    component remains single-use.
    """

    allowed = {"latent_evaluation", "reconstruction_karaone", "reconstruction_feis"}
    if purpose not in allowed:
        raise ValueError(f"Unknown locked-test purpose: {purpose!r}")
    chosen = str(access_id or secrets.token_hex(16))
    if not re.fullmatch(r"[A-Za-z0-9_.-]{8,128}", chosen):
        raise ValueError("Locked-test access ID must be 8-128 safe characters")
    gate = Path(gate_path).resolve()
    gate_hash = file_sha256(gate)
    ledger = gate.parent / "locked_test_access"
    ledger.mkdir(parents=True, exist_ok=True)
    claim_path = ledger / "claim.json"
    claim = {
        "schema_version": LOCKED_TEST_ACCESS_SCHEMA_VERSION,
        "access_id": chosen,
        "validation_gate": str(gate),
        "validation_gate_sha256": gate_hash,
        "policy": "single_final_test_session_component_claims",
    }
    try:
        _write_json_exclusive(claim_path, claim)
    except FileExistsError:
        observed = json.loads(claim_path.read_text(encoding="utf-8"))
        if (
            observed.get("schema_version") != LOCKED_TEST_ACCESS_SCHEMA_VERSION
            or observed.get("access_id") != chosen
            or observed.get("validation_gate_sha256") != gate_hash
        ):
            raise PermissionError(
                "The v0724 locked test has already been claimed by another "
                "final-test session"
            )
        claim = observed
    component_path = ledger / f"{purpose}.json"
    component = {**claim, "purpose": purpose}
    try:
        _write_json_exclusive(component_path, component)
    except FileExistsError as error:
        raise PermissionError(
            f"The v0724 locked-test component {purpose!r} was already accessed"
        ) from error
    return component


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "LINEAGE_SCHEMA_VERSION",
    "LOCKED_TEST_ACCESS_SCHEMA_VERSION",
    "VALIDATION_GATE_SCHEMA_VERSION",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "authorize_locked_test",
    "authorize_locked_test_metadata",
    "build_lineage",
    "claim_locked_test_access",
    "checkpoint_payload",
    "file_sha256",
    "object_sha256",
    "path_sha256",
    "validate_checkpoint",
    "validate_lineage",
]
