#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import h5py
import mne
import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path("/Users/samxie/Research/EEG-Voice/openneuro_downloads/ds006104-download")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "exploration_outputs" / "edf_full_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export full EEG matrices from hydrated EDF files into HDF5, with progress bars, "
            "while keeping outputs under exploration_outputs/edf_full_analysis."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the hydrated ds006104 dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Existing output root; each EDF keeps using its current per-file folder.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject filter, e.g. P01 S01 S02.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        help="Chunk size in seconds for reading/writing EEG data.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="gzip compression level for HDF5 dataset, 0-9.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a file if the HDF5 export already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue exporting remaining files after an error.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 export and copied sidecars.",
    )
    return parser.parse_args()


def discover_edf_paths(dataset_root: Path, subjects: list[str] | None) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    edf_paths = sorted(dataset_root.rglob("*_eeg.edf"))
    if subjects:
        subject_set = {subject.replace("sub-", "") for subject in subjects}
        edf_paths = [path for path in edf_paths if path.parts[-4].replace("sub-", "") in subject_set]
    return edf_paths


def compute_sidecar_paths(edf_path: Path) -> list[Path]:
    stem = edf_path.stem
    if not stem.endswith("_eeg"):
        raise ValueError(f"Unexpected EDF stem format: {stem}")

    prefix = stem.removesuffix("_eeg")
    session_prefix = "_".join(prefix.split("_")[:2])
    parent = edf_path.parent

    candidates = [
        parent / f"{prefix}_events.tsv",
        parent / f"{prefix}_events.json",
        parent / f"{prefix}_channels.tsv",
        parent / f"{prefix}_eeg.json",
        parent / f"{session_prefix}_coordsystem.json",
    ]
    return [path for path in candidates if path.exists()]


def copy_sidecars(sidecar_paths: list[Path], output_dir: Path, overwrite: bool) -> list[str]:
    copied: list[str] = []
    for source_path in sidecar_paths:
        target_path = output_dir / source_path.name
        if target_path.exists() and not overwrite:
            copied.append(str(target_path))
            continue
        shutil.copy2(source_path, target_path)
        copied.append(str(target_path))
    return copied


def export_one_edf(
    edf_path: Path,
    output_root: Path,
    chunk_seconds: float,
    compression_level: int,
    overwrite: bool,
) -> dict[str, object]:
    started = time.time()
    output_dir = output_root / edf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = output_dir / f"{edf_path.stem}_full_eeg.h5"
    manifest_path = output_dir / f"{edf_path.stem}_full_export_manifest.json"

    if h5_path.exists() and overwrite:
        h5_path.unlink()
    if manifest_path.exists() and overwrite:
        manifest_path.unlink()

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_total_channels = raw.info["nchan"]
    total_samples = int(raw.n_times)
    duration_seconds = float(total_samples / sfreq)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_channel_names = [raw.ch_names[idx] for idx in eeg_picks]
    non_eeg_channel_names = [raw.ch_names[idx] for idx in range(n_total_channels) if idx not in eeg_picks]

    chunk_samples = max(1, int(round(chunk_seconds * sfreq)))
    total_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    string_dtype = h5py.string_dtype(encoding="utf-8")
    chunk_shape = (min(len(eeg_picks), 8), min(chunk_samples, total_samples))

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.attrs["source_edf_path"] = str(edf_path)
        h5_file.attrs["export_started_unix"] = started
        h5_file.attrs["sampling_frequency_hz"] = sfreq
        h5_file.attrs["duration_seconds"] = duration_seconds
        h5_file.attrs["sample_count"] = total_samples
        h5_file.attrs["eeg_channel_count"] = len(eeg_picks)
        h5_file.attrs["data_unit"] = "uV"
        h5_file.attrs["mne_native_unit_before_scaling"] = "V"
        h5_file.attrs["chunk_seconds"] = chunk_seconds

        h5_file.create_dataset(
            "eeg_data_uV",
            shape=(len(eeg_picks), total_samples),
            dtype="float32",
            chunks=chunk_shape,
            compression="gzip",
            compression_opts=compression_level,
        )
        h5_file.create_dataset("eeg_channel_names", data=np.array(eeg_channel_names, dtype=object), dtype=string_dtype)
        h5_file.create_dataset("all_channel_names", data=np.array(raw.ch_names, dtype=object), dtype=string_dtype)
        h5_file.create_dataset(
            "non_eeg_channel_names",
            data=np.array(non_eeg_channel_names, dtype=object),
            dtype=string_dtype,
        )

        chunk_bar = tqdm(
            total=total_chunks,
            desc=edf_path.stem,
            unit="chunk",
            leave=False,
        )
        for chunk_index, start in enumerate(range(0, total_samples, chunk_samples), start=1):
            stop = min(start + chunk_samples, total_samples)
            data_v = raw.get_data(picks=eeg_picks, start=start, stop=stop)
            data_uv = (data_v * 1_000_000.0).astype(np.float32, copy=False)
            h5_file["eeg_data_uV"][:, start:stop] = data_uv
            chunk_bar.set_postfix(
                samples=f"{start}:{stop}",
                elapsed=f"{time.time() - started:.0f}s",
            )
            chunk_bar.update(1)
        chunk_bar.close()

    sidecar_paths = compute_sidecar_paths(edf_path)
    copied_sidecars = copy_sidecars(sidecar_paths, output_dir, overwrite=overwrite)

    manifest = {
        "source_edf_path": str(edf_path),
        "output_h5_path": str(h5_path),
        "sampling_frequency_hz": sfreq,
        "duration_seconds": duration_seconds,
        "sample_count": total_samples,
        "total_channel_count": n_total_channels,
        "eeg_channel_count": len(eeg_picks),
        "eeg_channel_names": eeg_channel_names,
        "non_eeg_channel_names": non_eeg_channel_names,
        "chunk_seconds": chunk_seconds,
        "total_chunks": total_chunks,
        "data_unit": "uV",
        "copied_sidecars": copied_sidecars,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    args = parse_args()
    started = time.time()
    edf_paths = discover_edf_paths(args.dataset_root, args.subjects)
    if not edf_paths:
        raise SystemExit("No EDF files found for the given dataset root / subject filter.")

    if not 0 <= args.compression_level <= 9:
        raise SystemExit("--compression-level must be between 0 and 9.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    file_bar = tqdm(edf_paths, desc="EDF exports", unit="file")
    for edf_path in file_bar:
        output_dir = output_root / edf_path.stem
        h5_path = output_dir / f"{edf_path.stem}_full_eeg.h5"

        if args.skip_existing and h5_path.exists() and not args.overwrite:
            completed.append(
                {
                    "source_edf_path": str(edf_path),
                    "output_h5_path": str(h5_path),
                    "status": "skipped_existing",
                }
            )
            continue

        file_bar.set_postfix(file=edf_path.stem)
        try:
            manifest = export_one_edf(
                edf_path=edf_path,
                output_root=output_root,
                chunk_seconds=args.chunk_seconds,
                compression_level=args.compression_level,
                overwrite=args.overwrite,
            )
            manifest["status"] = "exported"
            completed.append(manifest)
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_edf_path": str(edf_path), "error": str(exc)})
            if not args.continue_on_error:
                raise

    batch_summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(output_root),
        "edf_entries": len(edf_paths),
        "completed_count": len(completed),
        "failure_count": len(failures),
        "chunk_seconds": args.chunk_seconds,
        "compression_level": args.compression_level,
        "elapsed_seconds": round(time.time() - started, 2),
        "completed": completed,
        "failures": failures,
    }
    summary_path = output_root / "batch_full_export_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(batch_summary, fp, ensure_ascii=False, indent=2)

    print(f"Saved batch export summary: {summary_path}")


if __name__ == "__main__":
    main()
