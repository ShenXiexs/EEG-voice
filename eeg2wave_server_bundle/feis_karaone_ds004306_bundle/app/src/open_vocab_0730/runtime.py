from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


VERSION = "openvoice-eeg-0730-explicit-cp-v1"


def resolve_config_path(config_path: str | Path, value: str | Path) -> Path:
    base = Path(config_path).resolve().parent
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def ensure_output_firewall(config_path: str | Path, cfg: dict[str, Any]) -> None:
    root = resolve_config_path(config_path, cfg["paths"]["output_root"])
    protected = {"open_vocab_0722", "open_vocab_0724", "open_vocab_0728"}
    if any(part in protected for part in root.parts) or "open_vocab_0730" not in str(root):
        raise ValueError(f"v0730 output root violates namespace firewall: {root}")
    for key, value in cfg.get("paths", {}).items():
        if not isinstance(value, str) or key == "source_cache_root":
            continue
        candidate = resolve_config_path(config_path, value)
        if any(part in protected for part in candidate.parts):
            raise ValueError(f"v0730 may not write protected namespace: {candidate}")


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("version") != VERSION:
        raise ValueError(f"unsupported v0730 config: {cfg.get('version')!r}")
    if "pot" not in str(cfg["split"]["unseen_label"]).lower():
        raise ValueError("v0730 requires the fixed unseen label 'pot'")
    if tuple(cfg["split"]["subject_holdout"]) != ("karaone:MM19", "karaone:MM20"):
        raise ValueError("v0730 preregisters MM19 and MM20 as the subject holdout")
    ensure_output_firewall(config_path, cfg)
    return config_path, cfg


def default_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value for key, value in batch.items()}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
