#!/usr/bin/env python3
"""Fail-closed D0 audit for the DS004940 fixed EEG window artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=1178)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, keep_default_na=False, low_memory=False)
    frame = frame[(frame.dataset == "ds004940") & (frame.build_status == "included")]
    failures = []; rows = 0; shard_rows = 0
    for path in sorted(set(frame.shard_path.astype(str))):
        if not path:
            continue
        source = Path(path)
        if not source.is_absolute(): source = args.manifest.parents[4] / source
        if not source.exists():
            failures.append(f"missing shard {source}"); continue
        with h5py.File(source, "r") as shard:
            if str(shard.attrs.get("model_time_mask_policy", "")) != "fixed_full_epoch":
                failures.append(f"{source}: model_time_mask_policy is not fixed_full_epoch")
            eeg = shard["eeg"]; mask = shard["eeg_valid_mask"][:]
            if eeg.shape[-1] != args.expected_samples:
                failures.append(f"{source}: samples={eeg.shape[-1]} expected={args.expected_samples}")
            if mask.shape[-1] != args.expected_samples or not mask.all():
                failures.append(f"{source}: contains duration-derived invalid samples")
            shard_rows += len(mask)
    report = {"schema_version": "ds004940-fixed-window-d0-v1", "included_manifest_rows": int(len(frame)),
              "shard_rows": int(shard_rows), "expected_samples": args.expected_samples,
              "status": "pass" if not failures else "fail", "failures": failures,
              "interpretation": "D0 proves equal physical EEG windows/masks; it does not itself prove EEG decoding."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); return 0 if not failures else 2


if __name__ == "__main__": raise SystemExit(main())
