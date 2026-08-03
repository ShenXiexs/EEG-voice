#!/usr/bin/env python3
"""Write and verify the compact lineage index for one CP-temporal run."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_v3.cp_temporal import SCHEMA
from src.open_vocab_v3.runtime import load_config, output_path, read_json, sha256_file, write_json


def identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("training_preview", "complete"), required=True)
    parser.add_argument("--explore", action="store_true")
    args = parser.parse_args()
    config_path, cfg = load_config(args.config)
    keys = (
        "prepared_cache", "prepared_manifest", "encodec_cache", "encodec_cache_manifest",
        "oracle_checkpoint", "prosody_checkpoint", "content_checkpoint", "cvae_checkpoint",
        "micro_checkpoint", "fit_checkpoint", "eeg_prosody_checkpoint", "t0_gate", "oracle_gate",
        "prosody_gate", "content_gate", "intervention_gate", "cvae_gate", "micro_gate", "fit_gate",
        "eeg_prosody_gate", "training_review", "validation_report", "locked_report",
        "locked_unseen_report", "micro_preview_manifest", "fit_preview_manifest",
    )
    artifacts = {}
    for key in keys:
        path = output_path(config_path, cfg, key)
        if path.is_file():
            artifacts[key] = identity(path)

    pair_manifests = []
    lineage_errors = []
    for key in ("micro_preview_manifest", "fit_preview_manifest"):
        path = output_path(config_path, cfg, key)
        if path.is_file():
            pair_manifests.append(path)
    final_manifest = output_path(config_path, cfg, "pair_root") / "export_manifest.json"
    if final_manifest.is_file():
        pair_manifests.append(final_manifest)
    pair_summary = {}
    for manifest_path in pair_manifests:
        manifest = read_json(manifest_path)
        verified = 0
        for row in manifest.get("pairs", []):
            metadata_path = Path(str(row.get("metadata", "")))
            if not metadata_path.is_file():
                lineage_errors.append(f"missing metadata: {metadata_path}")
                continue
            metadata = read_json(metadata_path)
            folder = metadata_path.parent
            valid = True
            for name, expected_hash in metadata.get("files", {}).items():
                file_path = folder / name
                if not file_path.is_file() or sha256_file(file_path) != expected_hash:
                    valid = False
                    lineage_errors.append(f"pair hash mismatch: {file_path}")
            source = Path(str(metadata.get("source_audio", "")))
            if not source.is_file() or sha256_file(source) != metadata.get("source_audio_sha256"):
                valid = False
                lineage_errors.append(f"source hash mismatch: {source}")
            verified += int(valid)
        pair_summary[str(manifest_path)] = {
            "sha256": sha256_file(manifest_path), "declared_pairs": len(manifest.get("pairs", [])),
            "verified_pairs": verified, "complete": bool(manifest.get("complete", False)),
            "csv": identity(manifest_path.parent / "manifest.csv") if (manifest_path.parent / "manifest.csv").is_file() else None,
        }
    payload = {
        "schema_version": SCHEMA,
        "phase": args.phase,
        "exploratory": bool(args.explore),
        "scientific_status": "exploratory_only" if args.explore else "strict_fail_closed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": identity(config_path),
        "artifacts": artifacts,
        "pair_manifests": pair_summary,
        "lineage_errors": lineage_errors,
        "lineage_valid": not lineage_errors,
    }
    destination = output_path(config_path, cfg, "run_manifest")
    write_json(destination, payload)
    if lineage_errors:
        raise RuntimeError(f"CP-temporal lineage validation failed with {len(lineage_errors)} errors; see {destination}")
    print(f"[v3 CP lineage] {args.phase}: {destination}", flush=True)


if __name__ == "__main__":
    main()
