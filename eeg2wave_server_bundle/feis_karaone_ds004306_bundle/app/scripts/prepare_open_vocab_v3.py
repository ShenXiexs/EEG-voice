#!/usr/bin/env python3
"""Prepare immutable-source v3 MFCC records and the audio QC audit."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_v3.data import PREPARATION_SCHEMA, load_prepared, prepare_records, role_counts, save_prepared
from src.open_vocab_v3.runtime import default_device, load_config, output_path, read_json, sha256_file, write_json
from src.open_vocab_v3.speaker import attach_speaker_embeddings


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare v3 content-first MFCC records")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--with-speaker", action="store_true", help="cache non-target ECAPA references for V1/V2 only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fit-only", action="store_true", help="exclude every held-out row before WAV feature extraction")
    return parser.parse_args()


def write_audit_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["sample_key"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_prepared_manifest(config_path: Path, cfg: dict, cache_path: Path, records) -> None:
    path = output_path(config_path, cfg, "prepared_manifest")
    supporting_artifacts = {}
    for key in (
        "raw_audio_audit", "raw_audio_audit_csv", "denoise_selection", "denoise_manifest",
        "audio_audit", "audio_audit_csv",
    ):
        artifact = output_path(config_path, cfg, key)
        if not artifact.is_file():
            raise RuntimeError(f"v3 preparation artifact is missing: {artifact}")
        supporting_artifacts[key] = {
            "path": str(artifact),
            "sha256": sha256_file(artifact),
            "bytes": artifact.stat().st_size,
        }
    speaker_manifest = output_path(config_path, cfg, "speaker_manifest")
    if "speaker_reference_embedding" in records.arrays:
        if not speaker_manifest.is_file():
            raise RuntimeError(f"v3 speaker manifest is missing: {speaker_manifest}")
        supporting_artifacts["speaker_manifest"] = {
            "path": str(speaker_manifest),
            "sha256": sha256_file(speaker_manifest),
            "bytes": speaker_manifest.stat().st_size,
        }
        # Once speaker conditioning is attached, this is the final prepared
        # cache and must be cryptographically bound to both fine-tuned audio
        # backbones and their fail-closed adaptation gate.
        dependencies = (
            () if str(cfg.get("experiment", {}).get("schema", "")) in {
                "openvoice-v3-cp-temporal-large-v1", "openvoice-v3-mfcc-encodec-bridge-v2", "openvoice-v3-mfcc-encodec-rvq-repair-v3"
            }
            else ("encodec_manifest", "vocoder_manifest", "speaker_adaptation_manifest", "audio_adaptation_gate")
        )
        for key in dependencies:
            artifact = output_path(config_path, cfg, key)
            if not artifact.is_file():
                raise RuntimeError(f"v3 adapted audio artifact is missing: {artifact}")
            supporting_artifacts[key] = {
                "path": str(artifact),
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
    content_only_mfcc = str(cfg.get("version", "")) in {
        "openvoice-v3-mfcc-encodec-bridge-v2",
        "openvoice-v3-mfcc-encodec-rvq-repair-v3",
    }
    write_json(
        path,
        {
            "schema_version": "openvoice-v3-prepared-manifest-v1",
            "preparation_schema": str(records.arrays["v3_preparation_schema"].item()),
            "prepared_cache": str(cache_path),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "sha256": sha256_file(cache_path),
            "bytes": cache_path.stat().st_size,
            "mtime_ns": cache_path.stat().st_mtime_ns,
            "records": len(records),
            "scope": "fit_only" if set(records.roles.tolist()) == {"fit"} else "all_protocol_roles",
            "role_counts": role_counts(records),
            "fit_eligible": int(((records.roles == "fit") & records.arrays["fit_eligible"]).sum()),
            "has_speaker_embeddings": "speaker_reference_embedding" in records.arrays,
            "feature_contract": {
                "sample_rate": int(cfg["audio"]["sample_rate"]),
                "n_fft": int(cfg["audio"]["n_fft"]),
                "mel_bins": int(cfg["audio"]["mel_bins"]),
                "mfcc_bins": int(cfg["audio"]["mfcc_bins"]),
                "content_coefficients": "c1..c39" if content_only_mfcc else "schema_specific",
                "c0_in_content": False if content_only_mfcc else None,
                "cmvn": "utterance_level_active_support" if content_only_mfcc else "schema_specific",
                "vad_crop": "active_start_to_active_end_then_relative_time_resample" if content_only_mfcc else "schema_specific",
                "target_frames": int(cfg["audio"]["canonical_frames"]),
                "interpolation": "torch_linear_align_corners_false",
            },
            "supporting_artifacts": supporting_artifacts,
        },
    )


def main() -> None:
    args = parse()
    config_path, cfg = load_config(args.config)
    cache_path = output_path(config_path, cfg, "prepared_cache")
    audit_json = output_path(config_path, cfg, "audio_audit")
    audit_csv = output_path(config_path, cfg, "audio_audit_csv")
    if cache_path.is_file() and not args.force:
        records = load_prepared(cache_path)
        cache_changed = False
        if not audit_json.is_file() or not audit_csv.is_file():
            raise RuntimeError("v3 audio audit is incomplete; rerun prepare_open_vocab_v3.py --force")
        if args.with_speaker and "speaker_reference_embedding" not in records.arrays:
            speaker = attach_speaker_embeddings(records, config_path=config_path, cfg=cfg, device=default_device(args.device))
            save_prepared(cache_path, records)
            write_json(output_path(config_path, cfg, "speaker_manifest"), speaker)
            cache_changed = True
        manifest_path = output_path(config_path, cfg, "prepared_manifest")
        manifest_valid = False
        if manifest_path.is_file():
            existing = read_json(manifest_path)
            if str(existing.get("config_sha256", "")) != sha256_file(config_path):
                raise RuntimeError("v3 config changed; rerun prepare_open_vocab_v3.py --force")
            cache_identity_valid = (
                int(existing.get("bytes", -1)) == cache_path.stat().st_size
                and int(existing.get("mtime_ns", -1)) == cache_path.stat().st_mtime_ns
            )
            if not cache_identity_valid and not cache_changed:
                raise RuntimeError("v3 prepared cache changed outside the preparation step; rerun --force")
            supporting = existing.get("supporting_artifacts")
            if isinstance(supporting, dict) and supporting:
                for name, identity in supporting.items():
                    artifact = Path(str(identity.get("path", "")))
                    if (
                        not artifact.is_file()
                        or int(identity.get("bytes", -1)) != artifact.stat().st_size
                        or str(identity.get("sha256", "")) != sha256_file(artifact)
                    ):
                        raise RuntimeError(
                            f"v3 supporting artifact {name} changed; rerun prepare_open_vocab_v3.py --force"
                        )
                manifest_valid = cache_identity_valid and not cache_changed
        if cache_changed or not manifest_valid:
            write_prepared_manifest(config_path, cfg, cache_path, records)
        print(f"[v3 prepare] exists: {cache_path}", flush=True)
        return
    records, audit = prepare_records(config_path, cfg, fit_only=args.fit_only)
    speaker = None
    if args.with_speaker:
        speaker = attach_speaker_embeddings(records, config_path=config_path, cfg=cfg, device=default_device(args.device))
        write_json(output_path(config_path, cfg, "speaker_manifest"), speaker)
    write_audit_csv(audit_csv, audit)
    counts = role_counts(records)
    flagged = [row for row in audit if not bool(row["fit_eligible"])]
    manual_review = [row for row in audit if bool(row["manual_review_required"])]
    low_contrast = [row for row in audit if bool(row["low_contrast"])]
    report = {
        "schema_version": "openvoice-v3-audio-audit-v3-selective-denoise",
        "prepared_cache": str(cache_path),
        "counts": counts,
        "fit_eligible": int(((records.roles == "fit") & records.arrays["fit_eligible"]).sum()),
        "fit_excluded": int(((records.roles == "fit") & ~records.arrays["fit_eligible"]).sum()),
        "no_automatic_deepfilternet": True,
        "accepted_denoised_count": sum(bool(row.get("used_accepted_denoising", False)) for row in audit),
        "flagged_count": len(flagged),
        "flagged_samples": flagged,
        "manual_review_count": len(manual_review),
        "manual_review_samples": manual_review,
        "manual_review_policy": "pending named anomalies are excluded from v3 fit until explicitly reviewed",
        "low_contrast_count": len(low_contrast),
        "low_contrast_samples": low_contrast,
        "low_contrast_policy": "audit queue only; no automatic DeepFilterNet and no automatic exclusion",
        "speaker": speaker,
    }
    write_json(audit_json, report)
    save_prepared(cache_path, records)
    write_prepared_manifest(config_path, cfg, cache_path, records)
    print(f"[v3 prepare] counts={counts} fit_eligible={report['fit_eligible']} flagged={len(flagged)}", flush=True)


if __name__ == "__main__":
    main()
