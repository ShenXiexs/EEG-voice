#!/usr/bin/env python3
"""File-integrity audit for all 1,341 v0730-fixed WAV pairs."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from scipy.io import wavfile

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0730.runtime import load_config, resolve_config_path, write_json


EXPECTED_ROLES = {
    "fit": 1019,
    "subject_holdout_seen": 200,
    "label_holdout_seen_subject": 102,
    "subject_and_label_holdout": 20,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all v0730-fixed WAV pairs")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path, cfg = load_config(args.config)
    root = resolve_config_path(config_path, cfg["paths"]["pair_root"])
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures: list[str] = []
    keys = [row["sample_key"] for row in rows]
    roles = Counter(row["evaluation_role"] for row in rows)
    if len(rows) != 1341:
        failures.append(f"expected 1341 rows, found {len(rows)}")
    if len(set(keys)) != len(keys):
        failures.append("duplicate sample_key")
    if dict(roles) != EXPECTED_ROLES:
        failures.append(f"role counts mismatch: {dict(roles)}")
    for row in rows:
        for field in ("reference_wav", "reconstruction_wav"):
            path = Path(row[field])
            if not path.is_file():
                failures.append(f"missing {field}: {path}")
                continue
            sample_rate, waveform = wavfile.read(path, mmap=True)
            if sample_rate != 16000 or waveform.size == 0:
                failures.append(f"invalid {field}: {path}")
    gate_values = [row.get("generated_gate_passed", "").lower() == "true" for row in rows]
    report = {
        "schema_version": "openvoice-0730-fixed-pair-audit-v2",
        "passed": not failures,
        "pair_count": len(rows),
        "role_counts": dict(roles),
        "generated_gate_passed": bool(rows) and all(gate_values),
        "failures": failures[:100],
        "note": "File integrity is separate from the generated-speech scientific gate.",
    }
    write_json(root / "pairs_audit.json", report)
    if failures:
        raise RuntimeError(f"v0730-fixed pair audit failed: {failures[:5]}")
    print(root / "pairs_audit.json", flush=True)


if __name__ == "__main__":
    main()
