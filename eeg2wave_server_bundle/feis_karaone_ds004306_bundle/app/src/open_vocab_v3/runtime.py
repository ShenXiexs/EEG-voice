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

from . import BRIDGE_VERSION, CP_TEMPORAL_VERSION, LEGACY_VERSION, RVQ_REPAIR_VERSION, VERSION


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge a short repair config over the immutable v3 baseline config."""
    result = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_config_path(config_path: str | Path, value: str | Path) -> Path:
    base = Path(config_path).resolve().parent
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def content_schema(cfg: dict[str, Any]) -> str:
    return str(cfg.get("experiment", {}).get("schema", "openvoice-v3-encodec-clip-mfcc-v1"))


def checkpoint_schema(cfg: dict[str, Any], component: str) -> str:
    if content_schema(cfg) == RVQ_REPAIR_VERSION:
        values = {
            "rvq_bridge": "openvoice-v3-sequential-rvq-bridge-v3",
            "rvq_micro": "openvoice-v3-sequential-rvq-micro-v3",
            "audio_c": "openvoice-v3-audio-c-teacher-rvq-v3",
            "micro_m0a": "openvoice-v3-eeg-direct-mfcc-memorization-v3",
            "micro_m0b": "openvoice-v3-eeg-clip-memorization-v3",
            "micro_m1": "openvoice-v3-eeg-micro-generalization-v3",
            "label_evaluator": "openvoice-v3-fit-hubert-linear-probe-v3",
        }
        if component not in values:
            raise KeyError(f"unknown RVQ-repair checkpoint component: {component}")
        return values[component]
    if content_schema(cfg) == BRIDGE_VERSION:
        values = {
            "bridge": "openvoice-v3-encodec-latent-bridge-v2",
            "audio_c": "openvoice-v3-audio-c-teacher-v2",
            "micro_m0": "openvoice-v3-eeg-c-memorization-v2",
            "micro_m1": "openvoice-v3-eeg-c-micro-generalization-v2",
        }
        if component not in values:
            raise KeyError(f"unknown EnCodec-bridge checkpoint component: {component}")
        return values[component]
    if content_schema(cfg) == "openvoice-v3-cp-temporal-large-v1":
        values = {
            "oracle": "openvoice-v3-cp-temporal-oracle-v1",
            "prosody": "openvoice-v3-cp-temporal-prosody-v1",
            "content": "openvoice-v3-cp-temporal-content-v1",
            "cvae": "openvoice-v3-cp-temporal-residual-cvae-v1",
            "micro": "openvoice-v3-cp-temporal-eeg-micro-v1",
            "fit": "openvoice-v3-cp-temporal-eeg-fit-v1",
            "eeg_prosody": "openvoice-v3-cp-temporal-eeg-prosody-v1",
        }
        if component not in values:
            raise KeyError(f"unknown CP-temporal checkpoint component: {component}")
        return values[component]
    repair = content_schema(cfg) == "openvoice-v3-content-repair-v2"
    if repair:
        values = {
            "audio": "openvoice-v3-audio-content-v2-repair",
            "cvae": "openvoice-v3-native-mel-cvae-v2-repair",
            "micro": "openvoice-v3-eeg-encodec-clip-micro-v2-repair",
            "fit": "openvoice-v3-eeg-encodec-clip-fit-v2-repair",
        }
    else:
        values = {
            "audio": "openvoice-v3-audio-content-v1",
            "cvae": "openvoice-v3-native-mel-cvae-v1",
            "micro": "openvoice-v3-eeg-encodec-clip-micro-v1",
            "fit": "openvoice-v3-eeg-encodec-clip-fit-v1",
        }
    return values[component]


def artifact_root_name(cfg: dict[str, Any]) -> str:
    """Return the per-run artifact namespace, optionally selected by a runner.

    A timestamped namespace is deliberately supplied by the bridge runners so
    that ``--fresh`` can never erase a previous experiment.  Keep the
    override narrow: it is a directory *name*, never an arbitrary path.
    """
    configured = str(cfg.get("experiment", {}).get("artifact_root_name", "open_vocab_v3_mfcc_training_first"))
    override = os.environ.get("OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME")
    if not override:
        return configured
    valid = all(character.isalnum() or character in "._-" for character in override)
    if not valid or not override.startswith("open_vocab_v3_"):
        raise ValueError("OPEN_VOCAB_V3_ARTIFACT_ROOT_NAME must be a safe open_vocab_v3_* directory name")
    return override


def output_path(config_path: str | Path, cfg: dict[str, Any], key: str) -> Path:
    path = resolve_config_path(config_path, cfg["paths"][key])
    root_name = artifact_root_name(cfg)
    # A content-repair config inherits the legacy path map but each writable
    # artifact must be redirected to its own namespace before any cache/model
    # lookup.  Immutable source caches never carry the old artifact root name.
    parts = list(path.parts)
    if root_name != "open_vocab_v3_mfcc_training_first":
        # Configs intentionally inherit the earlier v3 path map.  Replace the
        # first *artifact* namespace, whether it comes from the original,
        # content-repair, or CP-temporal experiment.  Immutable v072x inputs
        # never begin with ``open_vocab_v3_`` and therefore cannot be rerouted.
        for index, part in enumerate(parts):
            if part.startswith("open_vocab_v3_"):
                parts[index] = root_name
                path = Path(*parts)
                break
    # The exploratory runner must never mix bypassed checkpoints/reports with
    # the fail-closed primary experiment.  Keep the configuration identical so
    # the feature contract is identical, but route every v3-output path to a
    # sibling artifact root when explicitly opted into by the shell runner.
    if os.environ.get("OPEN_VOCAB_V3_EXPLORATION") == "1":
        parts = list(path.parts)
        try:
            index = parts.index(root_name)
        except ValueError:
            return path
        parts[index] = f"{root_name}_explore"
        return Path(*parts).resolve()
    return path


def ensure_output_firewall(config_path: str | Path, cfg: dict[str, Any]) -> None:
    root = output_path(config_path, cfg, "output_root")
    configured = artifact_root_name(cfg)
    expected = f"{configured}_explore" if os.environ.get("OPEN_VOCAB_V3_EXPLORATION") == "1" else configured
    if root.name != expected:
        raise ValueError(f"v3 output root must end in {expected}, got {root}")
    protected = {"open_vocab_0722", "open_vocab_0724", "open_vocab_0728", "open_vocab_0730"}
    for key, value in cfg.get("paths", {}).items():
        if key == "source_cache_root" or not isinstance(value, str):
            continue
        candidate = resolve_config_path(config_path, value)
        if any(part in protected for part in candidate.parts):
            raise ValueError(f"v3 path {key} may not write protected namespace: {candidate}")


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    def read_recursive(candidate: Path, seen: set[Path]) -> dict[str, Any]:
        candidate = candidate.resolve()
        if candidate in seen:
            raise ValueError(f"cyclic v3 config inheritance: {candidate}")
        value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid YAML mapping: {candidate}")
        inherited = value.get("inherits")
        if not inherited:
            return value
        base = read_recursive(resolve_config_path(candidate, str(inherited)), seen | {candidate})
        return _deep_merge(base, value)
    cfg = read_recursive(config_path, set())
    if cfg.get("version") not in {VERSION, LEGACY_VERSION, CP_TEMPORAL_VERSION, BRIDGE_VERSION, RVQ_REPAIR_VERSION}:
        raise ValueError(f"unsupported v3 config: {cfg.get('version')!r}")
    if tuple(cfg["split"]["subject_holdout"]) != ("karaone:MM19", "karaone:MM20"):
        raise ValueError("v3 preregisters MM19/MM20 as the subject holdout")
    if str(cfg["split"]["unseen_label"]).strip().lower() != "pot":
        raise ValueError("v3 preregisters pot as the unseen label")
    cp_temporal = cfg.get("version") == CP_TEMPORAL_VERSION
    bridge = cfg.get("version") in {BRIDGE_VERSION, RVQ_REPAIR_VERSION}
    expected_frames = 161 if (cp_temporal or bridge) else 256
    if int(cfg["audio"]["canonical_frames"]) != expected_frames:
        raise ValueError(f"v3 content gate requires exactly {expected_frames} canonical MFCC frames")
    if int(cfg["audio"]["mfcc_bins"]) != 40:
        raise ValueError("v3 content gate requires exactly 40 MFCC coefficients")
    if int(cfg["model"]["audio_latent_dimension"]) <= 0:
        raise ValueError("v3 conditional variational decoder requires a positive latent dimension")
    if float(cfg["model"]["audio_residual_limit_log10"]) <= 0:
        raise ValueError("v3 native-Mel CVAE requires a positive log10 residual limit")
    if (int(cfg["audio"]["encodec_sample_rate"]), int(cfg["audio"]["encodec_codebooks"]),
            int(cfg["audio"]["encodec_codebook_size"]), int(cfg["audio"]["encodec_steps"])) != (24000, 8, 1024, 192):
        raise ValueError("v3 requires the declared 24kHz/6kbps/8x1024/192 EnCodec contract")
    expected_tokens = 96 if (cp_temporal or bridge) else 32
    if int(cfg["audio"]["content_tokens"]) != expected_tokens or int(cfg["audio"]["native_mel_frames"]) != 161:
        raise ValueError(f"v3 requires {expected_tokens} aligned content tokens and 161 native SpeechT5 Mel frames")
    if str(cfg["vocoder"].get("native_contract")) != "speecht5_native_log_mel_v1":
        raise ValueError("v3 rejects the legacy power-dB/10 SpeechT5 adapter contract")
    if float(cfg["training"].get("canonical_voice_dropout", 0.0)) != 0.0:
        raise ValueError("canonical voice dropout would pair the wrong voice with target Mel and is forbidden")
    denoiser = str(cfg["denoise"].get("backend", "")).lower()
    if denoiser == "deepfilternet3" and int(cfg["denoise"]["processing_sample_rate"]) != 48_000:
        raise ValueError("DeepFilterNet v3 must run at its native 48 kHz processing rate")
    if denoiser != "deepfilternet3" and int(cfg["denoise"]["processing_sample_rate"]) != int(cfg["audio"]["sample_rate"]):
        raise ValueError("deterministic v3 denoising must preserve the native model sample rate")
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
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def capture_lineage(
    config_path: Path, cfg: dict[str, Any], *, artifact_keys: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Capture the prepared-cache identity plus exact dependent artifacts."""
    manifest_path = output_path(config_path, cfg, "prepared_manifest")
    if not manifest_path.is_file():
        raise RuntimeError(f"prepared-cache manifest is missing: {manifest_path}")
    prepared = read_json(manifest_path)
    cache_path = output_path(config_path, cfg, "prepared_cache")
    if not cache_path.is_file():
        raise RuntimeError(f"prepared cache is missing: {cache_path}")
    if int(prepared.get("bytes", -1)) != cache_path.stat().st_size:
        raise RuntimeError("prepared cache size no longer matches its manifest; rerun prepare --force")
    if int(prepared.get("mtime_ns", -1)) != cache_path.stat().st_mtime_ns:
        raise RuntimeError("prepared cache timestamp no longer matches its manifest; rerun prepare --force")
    config_sha256 = sha256_file(config_path)
    if str(prepared.get("config_sha256", "")) != config_sha256:
        raise RuntimeError("v3 config no longer matches the prepared cache; rerun prepare --force")
    supporting = prepared.get("supporting_artifacts")
    if not isinstance(supporting, dict) or not supporting:
        raise RuntimeError("prepared manifest lacks bound audit artifacts; rerun prepare --force")
    for name, identity in supporting.items():
        if not isinstance(identity, dict):
            raise RuntimeError(f"invalid prepared supporting-artifact identity: {name}")
        path = Path(str(identity.get("path", "")))
        if not path.is_file():
            raise RuntimeError(f"prepared supporting artifact is missing: {path}")
        if int(identity.get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"prepared supporting artifact size changed: {path}")
        if str(identity.get("sha256", "")) != sha256_file(path):
            raise RuntimeError(f"prepared supporting artifact hash changed: {path}")
    artifacts: dict[str, Any] = {}
    for key in artifact_keys:
        path = output_path(config_path, cfg, key)
        if not path.is_file():
            raise RuntimeError(f"lineage artifact {key} is missing: {path}")
        artifacts[key] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "prepared_cache": str(cache_path),
        "prepared_cache_sha256": str(prepared["sha256"]),
        "prepared_manifest": str(manifest_path),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha256,
        "artifacts": artifacts,
    }


def require_passed_gate(
    config_path: Path,
    cfg: dict[str, Any],
    gate_key: str,
    *,
    lineage_artifact_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    gate_path = output_path(config_path, cfg, gate_key)
    if not gate_path.is_file():
        raise RuntimeError(f"v3 fail-closed: required gate is missing: {gate_path}")
    report = read_json(gate_path)
    if not report.get("passed", False):
        raise RuntimeError(f"v3 fail-closed: required gate did not pass: {gate_path}")
    expected = capture_lineage(config_path, cfg, artifact_keys=lineage_artifact_keys)
    observed = report.get("lineage")
    if observed != expected:
        raise RuntimeError(
            f"v3 fail-closed: stale or mismatched lineage for {gate_path}; rerun the preceding stages"
        )
    return report


def checkpoint_path(config_path: Path, cfg: dict[str, Any], name: str) -> Path:
    key = {
        "audio": "audio_checkpoint",
        "micro": "micro_checkpoint",
        "fit": "fit_checkpoint",
    }[name]
    return output_path(config_path, cfg, key)
