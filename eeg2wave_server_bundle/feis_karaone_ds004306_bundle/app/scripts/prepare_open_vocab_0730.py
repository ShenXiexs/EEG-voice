#!/usr/bin/env python3
"""Create v0730's immutable explicit-C/P cache from the read-only v0728 v3 cache."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0730.data import prepare_records, save_prepared, text_anchor
from src.open_vocab_0730.metrics import role_counts
from src.open_vocab_0730.runtime import load_config, resolve_config_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a split-safe v0730 explicit C/P cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path, cfg = load_config(args.config)
    output = resolve_config_path(config_path, cfg["paths"]["prepared_cache"])
    audit_path = resolve_config_path(config_path, cfg["paths"]["split_audit"])
    if output.exists() and not args.force:
        print(f"[0730 prepare] exists: {output}")
        return
    records = prepare_records(
        resolve_config_path(config_path, cfg["paths"]["source_cache_root"]),
        subject_holdout=cfg["split"]["subject_holdout"],
        unseen_label=cfg["split"]["unseen_label"],
        pca_components=int(cfg["content"]["pca_components"]),
        clusters=int(cfg["content"]["codebook_size"]),
        seed=int(cfg["training"]["seed"]),
    )
    save_prepared(output, records)
    roles = role_counts(records.roles)
    audit = {
        "schema_version": "openvoice-0730-prepared-v1",
        "prepared_cache": str(output),
        "counts": roles,
        "subjects_by_role": {role: sorted(set(records.arrays["subjects"][records.roles == role].tolist())) for role in roles},
        "pot_counts_by_role": {role: int(sum(str(label) == "pot" for label in records.arrays["labels"][records.roles == role])) for role in roles},
        "content_teacher": {"source": "frozen HuBERT cache v3", "source_layer": int(cfg["content"]["source_hubert_layer"]), "pca_components": int(cfg["content"]["pca_components"]), "codebook_size": int(cfg["content"]["codebook_size"]), "labels_used_in_fit": False},
        "prosody": {"fields": ["duration_seconds", "global_log_rms", "activity_32", "lowpass_log_rms_envelope_32"], "excluded": ["F0", "voicing", "formants", "speaker_embedding", "full_mel"]},
        "text_auxiliary": {"enabled": bool(cfg.get("text_reference")), "kind": "frozen phoneme character-ngram anchor", "inference_input": False, "unseen_pot_anchor_present": bool(text_anchor(["pot"], cfg.get("text_reference", {}), dimension=int(cfg["content"]["pca_components"]))[1][0])},
        "labels_never_forward_input": True,
    }
    write_json(audit_path, audit)
    print(roles)


if __name__ == "__main__":
    main()
