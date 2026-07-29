#!/usr/bin/env python3
"""Audit the complete v0724 KaraOne/FEIS exploratory pair export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    ("karaone", "train"): (1616, 1615),
    ("karaone", "validation"): (165, 165),
    ("karaone", "exploratory_test"): (132, 132),
    ("feis", "train"): (2832, 2832),
    ("feis", "validation"): (160, 160),
    ("feis", "exploratory_test"): (160, 160),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.synthesis_root.resolve()
    rows: list[dict[str, object]] = []
    for (dataset, output_split), (expected_input, expected_complete) in EXPECTED.items():
        base = root / dataset / output_split
        synthesis_path = base / "synthesis_manifest.json"
        comparison_path = base / "comparison_pairs" / "comparison_manifest.json"
        if not synthesis_path.is_file() or not comparison_path.is_file():
            raise FileNotFoundError(f"Incomplete export for {dataset}/{output_split}: {base}")
        synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        observed = (
            int(synthesis.get("input_dataset_record_count", -1)),
            int(synthesis.get("full_dataset_record_count", -1)),
            int(synthesis.get("completed_record_count", len(synthesis.get("records") or []))),
            int(comparison.get("plots_written", -1)),
        )
        expected = (expected_input, expected_complete, expected_complete, expected_complete)
        if observed != expected:
            raise RuntimeError(
                f"Count mismatch for {dataset}/{output_split}: observed={observed}, expected={expected}"
            )
        rows.append(
            {
                "dataset": dataset,
                "output_split": output_split,
                "input_trials": observed[0],
                "reconstruction_eligible": observed[1],
                "skipped": int(synthesis.get("skipped_record_count", 0)),
                "plots_written": observed[3],
                "evaluation_scope": synthesis.get("evaluation_scope"),
            }
        )
    report = {
        "schema_version": "openvoice-0724-full-pair-export-audit-v1",
        "synthesis_root": str(root),
        "partitions": rows,
        "total_reconstruction_pairs": sum(int(row["plots_written"]) for row in rows),
        "passed": True,
    }
    destination = root / "full_pair_export_audit.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
