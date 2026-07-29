#!/usr/bin/env python3
"""Build v0730-fixed C/P cache with physically isolated final-test roles."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0730.data_fixed import prepare_fixed_records, save_prepared, text_anchor
from src.open_vocab_0730.metrics import role_counts
from src.open_vocab_0730.runtime import load_config, resolve_config_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build split-safe v0730-fixed cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path, cfg = load_config(args.config)
    output = resolve_config_path(config_path, cfg["paths"]["prepared_cache"])
    audit_path = resolve_config_path(config_path, cfg["paths"]["split_audit"])
    if output.exists() and not args.force:
        print(f"[0730-fixed prepare] exists: {output}", flush=True)
        return

    records = prepare_fixed_records(
        resolve_config_path(config_path, cfg["paths"]["source_cache_root"]),
        subject_holdout=cfg["split"]["subject_holdout"],
        unseen_label=cfg["split"]["unseen_label"],
        pca_components=int(cfg["content"]["pca_components"]),
        clusters=int(cfg["content"]["codebook_size"]),
        seed=int(cfg["training"]["seed"]),
    )
    save_prepared(output, records)
    counts = role_counts(records.roles)
    audit = {
        "schema_version": "openvoice-0730-fixed-prepared-v2",
        "prepared_cache": str(output),
        "counts": counts,
        "pair_records": int(sum(counts.get(role, 0) for role in ("fit", "subject_holdout_seen", "label_holdout_seen_subject", "subject_and_label_holdout"))),
        "final_test_records": int(sum(counts.get(role, 0) for role in ("locked_test_seen_label", "locked_test_unseen_label", "diagnostic_subject_seen_label", "diagnostic_subject_unseen_label"))),
        "subjects_by_role": {
            role: sorted(set(records.arrays["subjects"][records.roles == role].tolist()))
            for role in counts
        },
        "content_teacher": {
            "source": "frozen HuBERT cache v3",
            "fit_role_only": True,
            "source_layer": int(cfg["content"]["source_hubert_layer"]),
            "pca_components": int(cfg["content"]["pca_components"]),
            "codebook_size": int(cfg["content"]["codebook_size"]),
        },
        "text_auxiliary": {
            "enabled": bool(cfg.get("text_reference")),
            "inference_input": False,
            "unseen_pot_anchor_present": bool(
                text_anchor(
                    ["pot"],
                    cfg.get("text_reference", {}),
                    dimension=int(cfg["content"]["pca_components"]),
                )[1][0]
            ),
        },
        "leakage_guards": {
            "locked_test_used_for_fit": False,
            "diagnostic_subjects_used_for_fit": False,
            "labels_as_forward_input": False,
        },
    }
    write_json(audit_path, audit)
    print(f"[0730-fixed prepare] counts={counts}", flush=True)


if __name__ == "__main__":
    main()
