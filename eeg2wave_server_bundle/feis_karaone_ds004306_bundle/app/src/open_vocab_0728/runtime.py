from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


VERSION = "openvoice-eeg-0728-duallatent-v1"


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("version") != VERSION:
        raise ValueError(f"unsupported v0728 config: {cfg.get('version')!r}")
    if list(cfg["training"]["seeds"]) != [15, 31, 47]:
        raise ValueError("v0728 preregisters seeds [15, 31, 47]")
    # An exploratory override must never contaminate the preregistered root.
    # Rewrite all v0728 artifact paths coherently, without mutating the YAML.
    env_name = str(cfg.get("gating", {}).get("allow_failed_gates_env", "ALLOW_FAILED_GATES"))
    if os.environ.get(env_name, "") == "1":
        original = str(cfg["paths"]["output_root"])
        exploratory = f"{original}/exploratory_failed_gate"
        for key, value in list(cfg["paths"].items()):
            if isinstance(value, str) and value.startswith(original):
                cfg["paths"][key] = exploratory + value[len(original):]
    ensure_output_firewall(config_path, cfg)
    return config_path, cfg


def resolve_config_path(config_path: str | Path, value: str | Path) -> Path:
    base = Path(config_path).resolve().parent
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def ensure_output_firewall(config_path: str | Path, cfg: dict[str, Any]) -> None:
    root = resolve_config_path(config_path, cfg["paths"]["output_root"])
    forbidden = {"open_vocab_0722", "open_vocab_0724", "open_vocab_0725"}
    if any(part in forbidden for part in root.parts) or not any("open_vocab_0728" in part for part in root.parts):
        raise ValueError(f"v0728 output root violates namespace firewall: {root}")
    for value in cfg.get("paths", {}).values():
        if not isinstance(value, str):
            continue
        candidate = resolve_config_path(config_path, value)
        if any(part in forbidden for part in candidate.parts):
            raise ValueError(f"v0728 may not write protected namespace: {candidate}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str |Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def failed_gate_root(config_path: str | Path, cfg: dict[str, Any]) -> Path:
    return resolve_config_path(config_path, cfg["paths"]["output_root"]) / "exploratory_failed_gate"


def allow_failed_gates(cfg: dict[str, Any]) -> bool:
    return os.environ.get(str(cfg["gating"]["allow_failed_gates_env"]), "") == "1"
