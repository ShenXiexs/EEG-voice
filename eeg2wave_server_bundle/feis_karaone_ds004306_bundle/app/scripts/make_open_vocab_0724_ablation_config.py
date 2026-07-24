#!/usr/bin/env python3
"""Create isolated, same-parameter-count v0724 ablation configurations."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import yaml


ABLATIONS = (
    "full_v0724",
    "dual_token_no_structure",
    "dual_token_no_disentanglement",
    "content_only",
    "realization_only",
    "full_contentvec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--ablation", choices=ABLATIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contentvec-model",
        default=None,
        help="Local path/model id required only for full_contentvec",
    )
    return parser.parse_args()


def artifact_root(cfg: dict[str, Any], tag: str) -> str:
    base = Path(str(cfg["paths"]["output_root"]))
    return str(base.parent / f"open_vocab_0724_ablation_{tag}")


def make_ablation_config(
    source: dict[str, Any],
    ablation: str,
    *,
    contentvec_model: str | None = None,
) -> dict[str, Any]:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unsupported ablation: {ablation}")
    if ablation == "full_contentvec" and not contentvec_model:
        raise ValueError("full_contentvec requires --contentvec-model")
    cfg = copy.deepcopy(source)
    tag = re.sub(r"[^a-z0-9_.-]+", "_", ablation.lower())
    root = artifact_root(cfg, tag)
    cfg["experiment"] = {
        "ablation": ablation,
        "same_parameter_count_control": True,
        "baseline_0722_external": ablation == "full_v0724",
        "audio_prior": (
            "train_contentvec_ablation"
            if ablation == "full_contentvec"
            else "shared_primary_frozen"
        ),
    }
    cfg["model"]["eeg_use_content_condition"] = ablation != "realization_only"
    cfg["model"]["eeg_use_realization_condition"] = ablation != "content_only"
    cfg["model"]["eeg_use_energy_feedback"] = ablation != "dual_token_no_structure"
    paths = cfg["paths"]
    paths.update(
        {
            "output_root": root,
            "eeg_pretrain_checkpoint": f"{root}/eeg_pretrain/checkpoints/best.pt",
            "eeg_checkpoint": f"{root}/eeg/checkpoints/best.pt",
            "validation_report": f"{root}/evaluation/validation_report.json",
            "validation_gate": f"{root}/evaluation/validation_gate.json",
        }
    )
    if ablation == "full_contentvec":
        paths["teacher_cache"] = f"{root}/cache/teacher_v2"
        paths["audio_checkpoint"] = f"{root}/audio/checkpoints/best.pt"
        paths["audio_oracle_gate"] = f"{root}/audio/metrics/audio_oracle_gate.json"
        paths["audio_freeze_manifest"] = f"{root}/audio/frozen_checkpoint.json"
        cfg["teachers"]["hubert_model"] = str(contentvec_model)
        cfg["teachers"]["content_teacher_name"] = "contentvec"
    return cfg


def main() -> None:
    args = parse_args()
    source = yaml.safe_load(args.base_config.resolve().read_text(encoding="utf-8"))
    output = make_ablation_config(
        source,
        args.ablation,
        contentvec_model=args.contentvec_model,
    )
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(output, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
