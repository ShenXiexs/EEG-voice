"""Registered experiment gates shared by training and artifact preparation."""
from __future__ import annotations

import json
from pathlib import Path


def registered_m0_gate_status(project_root: Path, cfg: dict) -> dict[str, list[str]]:
    missing: list[str] = []
    failed: list[str] = []
    for seed in cfg["training"]["seeds"]:
        for mode, datasets in {
            "ds004940": ["ds004940"],
            "ds006104": ["ds006104"],
            "joint": ["ds004940", "ds006104"],
        }.items():
            root = project_root / "outputs" / "joint_pilot_v1" / "pilot" / "overfit" / mode / f"seed-{seed}"
            for dataset in datasets:
                path = root / f"evaluation_{dataset}_train.json"
                relative = str(path.relative_to(project_root))
                if not path.exists():
                    missing.append(relative)
                    continue
                payload = json.loads(path.read_text())
                if payload.get("run_kind") != "pilot" or not payload.get("gate", {}).get("passed", False):
                    failed.append(relative)
    return {"missing": missing, "failed": failed}


def require_registered_m0_gates(project_root: Path, cfg: dict) -> None:
    status = registered_m0_gate_status(project_root, cfg)
    if status["missing"] or status["failed"]:
        raise RuntimeError(
            "Stage 2 is gated by all registered M0 runs; "
            f"missing={status['missing']}, failed={status['failed']}"
        )
