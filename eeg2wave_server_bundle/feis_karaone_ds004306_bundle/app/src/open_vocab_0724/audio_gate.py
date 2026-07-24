from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lineage import file_sha256, validate_lineage


AUDIO_ORACLE_GATE_SCHEMA = "openvoice-0724-audio-oracle-gate-v1"
AUDIO_FREEZE_SCHEMA = "openvoice-0724-audio-freeze-v1"


def _resolve(config_path: str | Path, value: str | Path) -> Path:
    base = Path(config_path).resolve().parent
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def require_frozen_audio_checkpoint(
    config_path: str | Path,
    cfg: dict[str, Any],
    lineage: dict[str, Any],
    audio_checkpoint: str | Path,
) -> dict[str, Any]:
    if not bool(cfg.get("gating", {}).get("require_audio_oracle_before_eeg", True)):
        return {}
    checkpoint = Path(audio_checkpoint).resolve()
    gate_path = _resolve(config_path, cfg["paths"]["audio_oracle_gate"])
    freeze_path = _resolve(config_path, cfg["paths"]["audio_freeze_manifest"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Audio checkpoint is missing: {checkpoint}")
    if not gate_path.is_file() or not freeze_path.is_file():
        raise PermissionError("Run the v0724 audio oracle audit before EEG training")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != AUDIO_ORACLE_GATE_SCHEMA or not bool(
        gate.get("passed")
    ):
        raise PermissionError(
            f"v0724 audio oracle gate failed: {gate.get('failed_checks', [])}"
        )
    if freeze.get("schema_version") != AUDIO_FREEZE_SCHEMA:
        raise ValueError(f"Unsupported v0724 audio freeze manifest: {freeze_path}")
    validate_lineage(
        gate.get("lineage"), lineage, source="v0724 audio oracle", scope="audio"
    )
    validate_lineage(
        freeze.get("lineage"), lineage, source="v0724 audio freeze", scope="audio"
    )
    checkpoint_hash = file_sha256(checkpoint)
    expected = {
        "audio_checkpoint_sha256": checkpoint_hash,
        "audio_oracle_gate_sha256": file_sha256(gate_path),
    }
    mismatch = {
        key: {"saved": freeze.get(key), "current": value}
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if gate.get("audio_checkpoint_sha256") != checkpoint_hash:
        mismatch["gate_audio_checkpoint_sha256"] = {
            "saved": gate.get("audio_checkpoint_sha256"),
            "current": checkpoint_hash,
        }
    if mismatch:
        raise PermissionError(
            f"v0724 frozen audio binding mismatch: {json.dumps(mismatch, sort_keys=True)}"
        )
    return freeze


__all__ = [
    "AUDIO_FREEZE_SCHEMA",
    "AUDIO_ORACLE_GATE_SCHEMA",
    "require_frozen_audio_checkpoint",
]
