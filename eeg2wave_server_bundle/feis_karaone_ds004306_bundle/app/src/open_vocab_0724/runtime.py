from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .model import FactorizedAudioConfig, FactorizedEEGConfig


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if str(cfg.get("version")) != "openvoice-eeg-0724-factorized-v1":
        raise ValueError(f"Unsupported v0724 config version: {cfg.get('version')!r}")
    gating = cfg.get("gating") or {}
    fixed_routes = {
        "exact_reconstruction_datasets": ["karaone"],
        "weak_prototype_datasets": ["feis"],
        "audio_ineligible_datasets": ["ds004306"],
    }
    mismatched_routes = {
        key: {"configured": gating.get(key), "required": expected}
        for key, expected in fixed_routes.items()
        if list(gating.get(key) or []) != expected
    }
    if mismatched_routes:
        raise ValueError(
            "v0724 supervision routing is preregistered and cannot be changed: "
            f"{json.dumps(mismatched_routes, sort_keys=True)}"
        )
    training = cfg.get("training") or {}
    if int(training.get("seed", -1)) != 15 or list(training.get("seeds") or []) != [
        15,
        31,
        47,
    ]:
        raise ValueError("v0724 preregisters primary seed 15 and seeds [15, 31, 47]")
    return config_path, cfg


def resolve_config_path(config_path: str | Path, value: str | Path) -> Path:
    config = Path(config_path).resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config.parent / path).resolve()


def run_identifier(
    cfg: dict[str, Any],
    *,
    seed: int | None = None,
    loso_subject: str | None = None,
    generalization: str = "g1",
    holdout_label: str | None = None,
) -> str | None:
    """Return a stable artifact namespace for non-primary registered runs."""

    chosen_seed = int(cfg["training"]["seed"] if seed is None else seed)
    default_seed = int(cfg["training"]["seed"])
    if (
        loso_subject is None
        and chosen_seed == default_seed
        and generalization == "g1"
        and holdout_label is None
    ):
        return None
    components: list[str] = []
    if generalization != "g1" or holdout_label is not None:
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(holdout_label or "all")).strip("_")
        components.append(f"{generalization}_label_{label or 'label'}")
    if loso_subject is not None:
        subject = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(loso_subject)).strip("_")
        components.append(f"loso_{subject or 'subject'}")
    components.append(f"seed_{chosen_seed}")
    return "_".join(components)


def resolve_run_checkpoint(
    config_path: str | Path,
    cfg: dict[str, Any],
    path_key: str,
    *,
    seed: int | None = None,
    loso_subject: str | None = None,
    generalization: str = "g1",
    holdout_label: str | None = None,
) -> Path:
    """Resolve checkpoints without allowing seeds/LOSO folds to overwrite."""

    base = resolve_config_path(config_path, cfg["paths"][path_key])
    run_id = run_identifier(
        cfg,
        seed=seed,
        loso_subject=loso_subject,
        generalization=generalization,
        holdout_label=holdout_label,
    )
    if run_id is None:
        return base
    return base.parent.parent / "runs" / run_id / base.parent.name / base.name


def resolve_evaluation_output(
    config_path: str | Path,
    cfg: dict[str, Any],
    *,
    split: str,
    seed: int | None = None,
    loso_subject: str | None = None,
    generalization: str = "g1",
    holdout_label: str | None = None,
) -> Path:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    root = resolve_config_path(config_path, cfg["paths"]["output_root"]) / "evaluation"
    run_id = run_identifier(
        cfg,
        seed=seed,
        loso_subject=loso_subject,
        generalization=generalization,
        holdout_label=holdout_label,
    )
    if run_id is not None:
        root = root / "runs" / run_id
    return root / f"latent_{split}.json"


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def audio_model_config(cfg: dict[str, Any]) -> FactorizedAudioConfig:
    model = cfg["model"]
    codec = cfg["codec"]
    evaluation = cfg["evaluation"]
    audio = cfg["audio"]
    return FactorizedAudioConfig(
        codebooks=int(codec["codebooks"]),
        code_steps=int(codec["code_steps"]),
        code_rate_hz=float(codec["code_rate_hz"]),
        vocab_size=int(codec["vocab_size"]),
        d_model=int(model["d_model"]),
        condition_steps=int(model["condition_steps"]),
        mel_bins=int(model["mel_bins"]),
        energy_frames=int(model["energy_frames"]),
        content_input_dimension=int(model["content_input_dimension"]),
        timbre_input_dimension=int(model["timbre_input_dimension"]),
        realization_input_dimension=int(model["realization_input_dimension"]),
        audio_encoder_layers=int(model["audio_encoder_layers"]),
        fusion_layers=int(model.get("fusion_layers", 2)),
        decoder_layers=int(model["decoder_layers"]),
        heads=int(model["heads"]),
        dropout=float(model["dropout"]),
        branch_dropout_probability=float(model["branch_dropout_probability"]),
        mel_db_min=float(audio["mel_db_min"]),
        mel_db_max=float(audio["mel_db_max"]),
        min_duration_seconds=1.0 / float(codec["code_rate_hz"]),
        max_duration_sec=float(codec["max_duration_sec"]),
        generation_steps=int(evaluation["maskgit_steps"]),
        generation_temperature=float(evaluation["synthesis_temperature"]),
        use_content_condition=bool(
            model.get(
                "audio_use_content_condition",
                model.get("use_content_condition", True),
            )
        ),
        use_realization_condition=bool(
            model.get(
                "audio_use_realization_condition",
                model.get("use_realization_condition", True),
            )
        ),
        use_energy_feedback=bool(
            model.get(
                "audio_use_energy_feedback",
                model.get("use_energy_feedback", True),
            )
        ),
    )


def eeg_model_config(
    cfg: dict[str, Any],
    *,
    num_train_subjects: int,
    num_content_labels: int,
) -> FactorizedEEGConfig:
    model = cfg["model"]
    return FactorizedEEGConfig(
        eeg_samples=int(cfg["data"]["eeg_samples"]),
        patch_size=int(model["eeg_patch_size"]),
        patch_hop=int(model["eeg_patch_hop"]),
        d_model=int(model["d_model"]),
        condition_steps=int(model["condition_steps"]),
        mel_bins=int(model["mel_bins"]),
        mel_frames=int(model["energy_frames"]),
        heads=int(model["heads"]),
        latent_layers=int(model["eeg_latent_layers"]),
        fusion_layers=int(model.get("fusion_layers", 2)),
        dropout=float(model["eeg_dropout"]),
        specialists=int(model["specialists"]),
        specialist_bottleneck=int(model["specialist_bottleneck"]),
        soft_routing_epochs=int(model["soft_routing_epochs"]),
        top_k_specialists=int(model["top_k_specialists"]),
        expert_dropout=float(model["expert_dropout"]),
        num_datasets=3,
        num_train_subjects=max(1, int(num_train_subjects)),
        num_content_labels=max(1, int(num_content_labels)),
        adapter_moe_enabled=bool(model["adapter_moe_enabled"]),
        branch_dropout_probability=float(model["branch_dropout_probability"]),
        mel_min_db=float(cfg["audio"]["mel_db_min"]),
        mel_max_db=float(cfg["audio"]["mel_db_max"]),
        min_duration_seconds=1.0 / float(cfg["codec"]["code_rate_hz"]),
        max_duration_seconds=float(cfg["codec"]["max_duration_sec"]),
        use_content_condition=bool(
            model.get(
                "eeg_use_content_condition",
                model.get("use_content_condition", True),
            )
        ),
        use_realization_condition=bool(
            model.get(
                "eeg_use_realization_condition",
                model.get("use_realization_condition", True),
            )
        ),
        use_energy_feedback=bool(
            model.get(
                "eeg_use_energy_feedback",
                model.get("use_energy_feedback", True),
            )
        ),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if torch.is_tensor(value):
        return json_safe(
            value.detach().cpu().tolist() if value.ndim else value.detach().cpu().item()
        )
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
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
    destination.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "audio_model_config",
    "default_device",
    "eeg_model_config",
    "json_safe",
    "load_config",
    "move_batch",
    "resolve_config_path",
    "resolve_evaluation_output",
    "resolve_run_checkpoint",
    "run_identifier",
    "seed_everything",
    "write_json",
]
