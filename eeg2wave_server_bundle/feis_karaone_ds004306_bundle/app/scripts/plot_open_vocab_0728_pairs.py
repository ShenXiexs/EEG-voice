#!/usr/bin/env python3
"""Render per-trial numerical log-mel reference/reconstruction comparison pairs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openvoice_0728_pair_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


CONDITIONS = (
    ("audio_latent_oracle", "Audio-latent oracle"),
    ("label_median_baseline", "Train-label median"),
    ("correct", "Correct EEG"),
    ("realization_shuffle", "Same-label realization shuffle"),
    ("content_shuffle", "Wrong-content shuffle"),
    ("zero_eeg", "Zero EEG"),
)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render v0728 numerical mel reference/reconstruction pairs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True, help="records_<split>.npz containing paired overt reference mel")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 means every manifest record")
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def title(name: str, details: dict[str, float]) -> str:
    return f"{name}\nSTSS={details['stss']:.3f} | duration={details['duration_seconds']:.2f}s | evidence={details['evidence']:.3f}"


def safe_stem(sample_key: str) -> str:
    return sample_key.replace(":", "_").replace("/", "_")


def render(record: dict, reference: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(11.4, 13.0), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for row, (condition, name) in enumerate(CONDITIONS):
        ref_ax, generated_ax = axes[row]
        image = ref_ax.imshow(reference, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0, extent=(0, 4, 0, 8))
        generated = np.load(record["conditions"][condition]["mel_path"])
        generated_ax.imshow(generated, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0, extent=(0, 4, 0, 8))
        ref_ax.set_ylabel(f"{name}\nFrequency (kHz)")
        generated_ax.set_title(title(name, record["conditions"][condition]), loc="left", fontsize=8.1)
        ref_ax.set_title("Paired overt reference" if row == 0 else "", loc="left", fontsize=8.5)
        if row == len(CONDITIONS) - 1:
            ref_ax.set_xlabel("Time (s)")
            generated_ax.set_xlabel("Time (s)")
    fig.suptitle(f"v0728 energy-structure pair | label={record['label']} | sample={record['sample_key']}", fontsize=11, y=1.005)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.80, pad=0.015)
    colorbar.set_label("Log-mel energy (dB)")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse()
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = raw["records"] if args.limit <= 0 else raw["records"][: args.limit]
    cache = np.load(args.reference_cache, allow_pickle=False)
    if "sample_keys" not in cache.files or "mel" not in cache.files:
        raise ValueError("reference cache must contain sample_keys and mel")
    reference = {str(key): cache["mel"][index] for index, key in enumerate(cache["sample_keys"])}
    missing = [record["sample_key"] for record in records if record["sample_key"] not in reference]
    if missing:
        raise ValueError(f"reference cache is missing {len(missing)} synthesis keys; first={missing[0]}")
    output = args.output.resolve() if args.output else args.manifest.parent / "comparison_pairs"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for record in tqdm(records, desc="[0728 energy pairs]", unit="figure", mininterval=1.0):
        target = output / f"{safe_stem(record['sample_key'])}.png"
        if not (args.resume_existing and target.exists()):
            render(record, reference[record["sample_key"]], target)
        for condition, _ in CONDITIONS:
            values = record["conditions"][condition]
            rows.append({
                "sample_key": record["sample_key"], "label": record["label"], "subject": record["subject"],
                "condition": condition, "figure": str(target), "mel_path": values["mel_path"],
                "stss": values["stss"], "duration_seconds": values["duration_seconds"], "evidence": values["evidence"],
            })
    fieldnames = list(rows[0]) if rows else ["sample_key", "label", "subject", "condition", "figure", "mel_path", "stss", "duration_seconds", "evidence"]
    with (output / "pair_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    summary = {
        "schema_version": "openvoice-0728-energy-comparison-pairs-v1",
        "source_manifest": str(args.manifest.resolve()), "reference_cache": str(args.reference_cache.resolve()),
        "output": str(output), "plots_written": len(records), "conditions_per_plot": [name for name, _ in CONDITIONS],
        "plot_definition": "Each row pairs the identical paired-overt numerical reference log-mel (left) with a generated numerical log-mel condition (right); neither axis is resized.",
    }
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
