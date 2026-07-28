from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import sha256_file, stable_hash, write_json


@dataclass(frozen=True)
class Lineage:
    config_sha256: str
    manifest_sha256: str
    split_sha256: str
    montage_sha256: str
    hubert_reference: str

    def as_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


def build_lineage(config_path: Path, cfg: dict[str, Any], *, manifest: Path, split: Path, montage: Path) -> Lineage:
    return Lineage(
        config_sha256=sha256_file(config_path),
        manifest_sha256=sha256_file(manifest),
        split_sha256=sha256_file(split),
        montage_sha256=sha256_file(montage),
        hubert_reference=str(cfg["teachers"]["hubert_model"]),
    )


def checkpoint_payload(*, state_dict: dict[str, Any], epoch: int, lineage: Lineage, extra: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "openvoice-0728-checkpoint-v1", "epoch": int(epoch), "state_dict": state_dict, "lineage": lineage.as_dict(), "extra": extra}


def validate_checkpoint(payload: dict[str, Any], lineage: Lineage) -> None:
    if payload.get("schema_version") != "openvoice-0728-checkpoint-v1":
        raise ValueError("unsupported v0728 checkpoint schema")
    if payload.get("lineage") != lineage.as_dict():
        raise ValueError("checkpoint lineage differs from current v0728 inputs")


def freeze_locked_test(path: Path, *, lineage: Lineage, fingerprints: dict[str, str]) -> dict[str, Any]:
    payload = {"schema_version": "openvoice-0728-locked-test-freeze-v1", "lineage": lineage.as_dict(), "fingerprints": dict(sorted(fingerprints.items()))}
    write_json(path, payload)
    return payload


def claim_locked_test_access(ledger_path: Path, *, freeze: dict[str, Any], access_id: str) -> dict[str, Any]:
    """Create/resume only the exact frozen formal-test transaction."""
    identity = stable_hash(json.dumps(freeze, sort_keys=True))
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("freeze_identity") != identity:
            raise PermissionError("locked test was already claimed by a different frozen run")
        if ledger.get("access_id") != access_id:
            raise PermissionError("locked test may only resume with its original access id")
        return ledger
    ledger = {"schema_version": "openvoice-0728-locked-test-ledger-v1", "freeze_identity": identity, "access_id": access_id, "completed_keys": [], "status": "running"}
    write_json(ledger_path, ledger)
    return ledger


def update_locked_test_ledger(ledger_path: Path, ledger: dict[str, Any], *, completed_key: str | None = None, complete: bool = False) -> None:
    if completed_key and completed_key not in ledger["completed_keys"]:
        ledger["completed_keys"].append(completed_key)
    if complete:
        ledger["status"] = "complete"
    write_json(ledger_path, ledger)
