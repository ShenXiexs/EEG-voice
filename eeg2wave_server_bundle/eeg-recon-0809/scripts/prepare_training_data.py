#!/usr/bin/env python3
"""Reproducible DS004940/DS006104 training-data preparation (v2).

This program deliberately separates *audit* (inventory and immutable locks),
*make-splits*, *build*, *validate*, and *fit-normalizer*.  It never edits a
raw BIDS input.  The only preprocessing profile implemented here is the
project's ``harmonized_v2`` profile; it is not represented as official data
preprocessing.

Install the optional runtime with ``pip install -r requirements-preprocess.txt``.
The pure provenance/split helpers have no optional-dependency import at module
load time, so they are also directly unit-testable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG = ROOT / "configs" / "training_data_v2.yaml"
SPLIT_ALGORITHM = "balanced-greedy-v1"
EVENT_TABLE_SHA256 = {
    "S01": "5c9323d3805b30bccc1698145f0adc651f8e4f1073234f3e614b128d6054736c",
    "S02": "6ce6643d7d59213dad2468ec08360a5427a70395248a070fb92f3bc0fb89a17c",
    "S03": "3c0c5e1bafadb250714e637a536583fbe9904b0bc0cbb74d4f8cd0b096cce3b3",
    "S04": "d954874a3e59a799b36abbafbcc20672cde8c7f654b610e250391c4b73c80ace",
    "S05": "0b84b0724a195c57a9e8d4757286e44efcd2ece74f10e36b6ea3d54ba6f8e5db",
    "S06": "1f449d0b18bd824e7171c5a6ea771849c44dee571ca7685b375c54a80a0d0630",
    "S07": "14f88dcd25bed34a90ee194a9942936ac022ada7658bc398219ca777708296f9",
    "S08": "1fba9eb0e587dd1cb514538b6fcd7273987833f7c97e21977acacf5be7f3bcf5",
    "S09": "9e13d7b744a421289529623a70c12bf076d984c5d4c88df9edd6fe5c362aa24f",
    "S10": "d3d57bf88cfbb9f0249740accbd9aeec3ba724f410bf04736bd7f1ba27b6990a",
    "S11": "4142ea24e2c766def6ebc821e1d9587aae383af5593736cfaec5c34db16c6bcc",
    "S12": "e5b2e0a00f418fb91582846bb44de2026eca7487332ee56483bcb807d97e32c0",
    "S13": "86ede0fad758fe9ec41ed172b92bc19b4cbb2e899cfa388b3e6eb90851364451",
    "S14": "5b13d654f1414069b0405017244b6f7ec59751c50b8db7d5bf85ffcbef1c4079",
    "S15": "97b2aabfe88add53f49d5c933cec9e1d556d1f57fc26c5d7a7c375d4e5382390",
    "S16": "39ca3f87c21e577b0074b6cf8cf7e91359f8d3802ba253183f839c121cf038ac",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def channel_order_hash(channels: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(channels) + "\n").encode("utf-8"))


def round_half_up(value: float) -> int:
    """Round non-negative values; exact .5 is rounded upward, never bankers."""
    if value < 0:
        return -round_half_up(-value)
    return math.floor(value + 0.5)


def trial_id(dataset: str, subject: str, task: str, run: str, event_row: int, onset: float) -> str:
    text = f"{dataset}|{subject}|{task}|{run}|{event_row}|{onset:.9f}"
    return f"{dataset}-{sha256_bytes(text.encode())[:20]}"


def source_interval_to_target_mask(
    *, source_zero: int, output_zero: int, target_length: int,
    source_sfreq: int, target_sfreq: int, intervals: list[tuple[int, int]],
) -> list[bool]:
    """Map direct source interpolation intervals to output samples only.

    It intentionally does not claim to represent filtering-ring effects.
    """
    out = []
    for index in range(target_length):
        source = source_zero + round_half_up((index - output_zero) * source_sfreq / target_sfreq)
        out.append(any(start <= source < end for start, end in intervals))
    return out


def clean_perception_mask(target_length: int, start: int, end: int, mixed: bool) -> list[bool]:
    if not mixed:
        return [True] * target_length
    return [start <= i < end for i in range(target_length)]


def balanced_group_assignment(weights: dict[str, int], folds: int, seed: str, namespace: str) -> dict[str, dict[str, Any]]:
    """Deterministic LPT placement; weights then stable hashes define ordering."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    fold_weight = [0] * folds
    fold_groups = [0] * folds
    groups = sorted(weights, key=lambda g: (-weights[g], sha256_bytes(f"{namespace}|{seed}|{g}".encode()), g))
    result: dict[str, dict[str, Any]] = {}
    for position, group in enumerate(groups):
        fold = min(range(folds), key=lambda f: (fold_weight[f], fold_groups[f], sha256_bytes(f"{namespace}|{seed}|{group}|{f}".encode()), f))
        result[group] = {"fold": fold, "trial_weight": int(weights[group]), "sort_position": position,
                         "tie_sha256": sha256_bytes(f"{namespace}|{seed}|{group}".encode())}
        fold_weight[fold] += weights[group]
        fold_groups[fold] += 1
    return result


def split_role(protocol: str, fold: int, subject_fold: int, audio_fold: int | None,
               supervision_type: str, folds: int = 5) -> tuple[str, str]:
    """Return frozen role and explicit exclusion explanation when appropriate."""
    val = (fold + 1) % folds
    if protocol == "subject_ood":
        return ("test", "") if subject_fold == fold else (("validation", "") if subject_fold == val else ("train", ""))
    if supervision_type != "paired_audio" or audio_fold is None:
        return "excluded", "not_audio_supervised"
    if protocol == "audio_ood":
        return ("test", "") if audio_fold == fold else (("validation", "") if audio_fold == val else ("train", ""))
    if protocol != "joint_ood":
        raise ValueError(f"unknown protocol {protocol}")
    if subject_fold == fold and audio_fold == fold:
        return "test", ""
    if subject_fold == val and audio_fold == val:
        return "validation", ""
    held_subject = subject_fold in (fold, val)
    held_audio = audio_fold in (fold, val)
    if held_subject or held_audio:
        return "excluded", "joint_cross_quadrant"
    return "train", ""


def audio_semantics_ds006104(wav_sha: str | None, clean_hashes: set[str], presentation_evidence: str | None = None) -> tuple[str, str]:
    if wav_sha and wav_sha in clean_hashes:
        return "clean_stimulus", "sha256_matches_cleaned_inventory"
    if presentation_evidence:
        return "presented_waveform", presentation_evidence
    return "unknown", "no_pinned_presentation_manifest"


def stimulus_content_id(dataset: str, value: str) -> str:
    """Content-level identity, deliberately coarser than the waveform file id.

    DS006104 emotional variants such as ``Bo_happy1`` and ``Bo_angry2`` are
    grouped as ``Bo``.  The original string stays in the manifest, so this
    grouping is auditable rather than destructive.
    """
    stem = Path(value).stem
    if dataset == "ds006104":
        stem = re.sub(r"_(?:happy|angry|neutral|sad|fear|fearful)\d*$", "", stem, flags=re.I)
    return f"{dataset}:content:{stem}"


def parse_bdf_header(path: Path) -> tuple[int, float]:
    """Read duration information without loading raw data, for split-run auditing."""
    with path.open("rb") as handle:
        header = handle.read(256)
    records = int(header[236:244].decode("ascii", "ignore").strip() or "0")
    duration = float(header[244:252].decode("ascii", "ignore").strip() or "1")
    return records, duration


def git_provenance() -> tuple[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    current_sources = "".join(sha256_file(p) for p in (HERE / "prepare_training_data.py", HERE / "training_data_loader.py") if p.exists())
    return run(["git", "rev-parse", "HEAD"]), run(["git", "diff", "--no-ext-diff", "--binary", "HEAD"]) + current_sources


def runtime():
    try:
        import yaml  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install optional runtime: pip install -r requirements-preprocess.txt") from exc
    return yaml, pd


def progress(iterable, *, desc: str, total: int | None = None):
    """Use tqdm when installed, without making provenance helpers depend on it."""
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, unit="item")
    except ImportError:
        return iterable


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    yaml, _ = runtime()
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = sha256_bytes(raw)
    return config, config["_config_sha256"]


def output_root(config: dict[str, Any]) -> Path:
    return ROOT / config["output_root"]


def as_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_tsv(path: Path, pd):
    return pd.read_csv(path, sep="\t")


def normalise_ds004_channel(name: str) -> str:
    return name.split("_")[0].strip()


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def find_ds004_audio(root: Path, stim_file: str) -> Path | None:
    base = Path(str(stim_file)).name
    return first_existing([root / "stimuli" / base, root / "stimuli" / "audio" / base, *root.glob(f"**/{base}")])


def find_ds006_audio(root: Path, stimulus: str) -> Path | None:
    stem = Path(str(stimulus)).stem
    audio_root = root / "audio_internal" / "stimuli"
    return first_existing([audio_root / f"{stem}.wav", *audio_root.glob(f"**/{stem}.wav")])


def source_lock_entry(path: Path, kind: str) -> dict[str, Any]:
    return {"path": as_relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "kind": kind}


def download_pinned_url(url: str, destination: Path, *, retries: int = 3) -> None:
    """Download a pinned source robustly on macOS Conda/OpenSSL setups.

    Some Conda Python builds fail the TLS handshake against raw.githubusercontent
    while the system curl trust store succeeds.  We try urllib first, then use
    curl with retries and write atomically in both cases.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        temporary: Path | None = None
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as tmp:
                shutil.copyfileobj(response, tmp)
                temporary = Path(tmp.name)
            os.replace(temporary, destination)
            return
        except Exception as exc:
            last_error = exc
            if temporary and temporary.exists(): temporary.unlink()
    temporary = destination.with_name(destination.name + ".download")
    if temporary.exists(): temporary.unlink()
    try:
        subprocess.run(["curl", "--fail", "--location", "--retry", str(retries), "--retry-all-errors",
                        "--connect-timeout", "20", "--max-time", "180", "--output", str(temporary), url],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.replace(temporary, destination)
        return
    except Exception as exc:
        if temporary.exists(): temporary.unlink()
        raise RuntimeError(f"unable to download pinned source after urllib/curl attempts: {url}; last error: {exc}") from last_error


def inventory_sha256(root: Path, suffixes: tuple[str, ...] = (".wav",)) -> str:
    """Hash a deterministic path+content inventory, not merely concatenated WAV bytes."""
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes):
        entries.append(f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\n")
    return sha256_bytes("".join(entries).encode("utf-8"))


def pinned_hash_status(path: Path, expected: str) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    actual = sha256_file(path)
    return actual == expected, actual


def resume_compatible(attrs: dict[str, Any], *, config_sha: str, source_lock_sha: str,
                      channel_hash: str | None = None, split_hash: str | None = None) -> bool:
    """Single gate for deterministic reuse of derived artifacts."""
    if attrs.get("preprocess_config_sha256", "") != config_sha or attrs.get("source_lock_sha256", "") != source_lock_sha:
        return False
    if channel_hash is not None and attrs.get("channel_order_hash", "") != channel_hash:
        return False
    return split_hash is None or attrs.get("split_index_sha256", "") == split_hash


def _event_column(frame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in frame.columns}
    return next((lower[c.lower()] for c in candidates if c.lower() in lower), None)


def _ds004_trial_rows(config: dict[str, Any], lock: dict[str, Any], qc: dict[str, Any]) -> list[dict[str, Any]]:
    _, pd = runtime()
    spec = config["sources"]["ds004940"]
    data_root = ROOT / spec["data_root"]
    rows: list[dict[str, Any]] = []
    event_files = sorted(data_root.glob("sub-*/ses-*/eeg/*_events.tsv"))
    subject_seen = set()
    for events_path in progress(event_files, desc="audit ds004940 event files"):
        parts = events_path.name.split("_")
        task = next((x.removeprefix("task-") for x in parts if x.startswith("task-")), "unknown")
        run = next((x.removeprefix("run-") for x in parts if x.startswith("run-")), "01")
        subject = next((p.name for p in events_path.parents if p.name.startswith("sub-")), "unknown")
        session = next((p.name for p in events_path.parents if p.name.startswith("ses-")), "")
        subject_seen.add(subject)
        frame = read_tsv(events_path, pd)
        onset_col = _event_column(frame, ["stim_onset_s_", "stim_onset_s", "onset"])
        stim_col = _event_column(frame, ["stim_file", "stimulus_file"])
        type_col = _event_column(frame, ["type"])
        if not onset_col:
            qc["exclusions"]["missing_onset"] += len(frame)
            continue
        eeg_base = events_path.name.replace("_events.tsv", "")
        bdf = first_existing([events_path.parent / f"{eeg_base}_eeg.bdf", events_path.parent / f"{eeg_base}_eeg.edf"])
        if bdf is None:
            qc["exclusions"]["missing_eeg"] += len(frame)
        for row_index, event in frame.iterrows():
            # Sound rows are experimental stimuli.  Prompts, instructions and
            # response rows are retained only as neither source trials nor
            # silently counted trials.
            if type_col and str(event[type_col]) != "Sound":
                qc["exclusions"]["non_stimulus_event"] += 1
                continue
            try:
                onset = float(event[onset_col])
            except (TypeError, ValueError):
                qc["exclusions"]["invalid_stimulus_onset"] += 1
                continue
            audio = find_ds004_audio(data_root, str(event[stim_col])) if stim_col and str(event[stim_col]) not in ("nan", "") else None
            reason = "" if bdf and audio else ("missing_eeg" if not bdf else "missing_audio")
            if reason:
                qc["exclusions"][reason] += 1
            identifier = trial_id("ds004940", subject, task, run, int(row_index), onset)
            entry = {
                "trial_id": identifier, "dataset": "ds004940", "dataset_version": spec["openneuro_version"],
                "subject": subject, "session": session, "task": task, "run": run,
                "source_eeg_path": as_relative(bdf) if bdf else "", "source_eeg_sha256": "",
                "source_event_path": as_relative(events_path), "source_event_sha256": sha256_file(events_path),
                "source_event_row": int(row_index), "event_onset_seconds": onset,
                "event_to_sample_error": abs(onset * 512 - round_half_up(onset * 512)), "stim_file": str(event[stim_col]) if stim_col else "",
                "stimulus_content_id": stimulus_content_id("ds004940", str(event[stim_col])) if stim_col else "",
                "audio_path": as_relative(audio) if audio else "", "audio_sha256": sha256_file(audio) if audio else "",
                "supervision_type": "paired_audio", "audio_semantics": "presented_waveform",
                "audio_semantics_evidence": "bids_stim_file_direct_reference" if audio else "missing_stim_file_target",
                "neural_task": "perception", "response_onset_relative_s": None,
                "response_onset_output_index": None, "response_onset_provenance": "",
                "production_contaminated": False, "clean_perception_start_index": 0,
                "clean_perception_end_index": int(spec.get("epoch_target_length", config["harmonized"]["epoch"]["ds004940"]["total_samples_target"])),
                "source_sfreq_hz": 512, "target_sfreq_hz": 256,
                "eeg_zero_index": 64, "audio_start_relative_to_eeg_samples": 64,
                "source_zero_sample": round_half_up(onset * 512),
                "source_start_sample": round_half_up(onset * 512) - 128,
                "source_end_sample": round_half_up(onset * 512) - 128 + 2356,
                "source_run_offsets": json.dumps([{"path": as_relative(bdf), "start_sample": 0}]) if bdf else "[]", "boundary_overlap": False,
                "qc_pass": not bool(reason), "build_status": "included" if not reason else "excluded",
                "exclusion_reason": reason, "channel_order_hash": channel_order_hash(spec["channel_order"]),
            }
            rows.append(entry)
    qc["actual_subjects"]["ds004940"] = len(subject_seen)
    return rows


def _fetch_aux_event(subject: str, config: dict[str, Any]) -> Path:
    spec = config["sources"]["ds006104"]["auxiliary_repository"]
    cache = output_root(config) / "auxiliary" / spec["commit"]
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{subject}_Tab.csv"
    expected = EVENT_TABLE_SHA256[subject]
    if not target.exists():
        url = f"https://raw.githubusercontent.com/mcjpedro/speech_decoding/{spec['commit']}/events_information/{subject}_Tab.csv"
        download_pinned_url(url, target)
    actual = sha256_file(target)
    if actual != expected:
        raise RuntimeError(f"official event table hash mismatch {subject}: {actual} != {expected}")
    return target


def _fetch_analysis_bids(config: dict[str, Any], allow_download: bool) -> Path | None:
    spec = config["sources"]["ds006104"]["auxiliary_repository"]
    cache = output_root(config) / "auxiliary" / spec["commit"]
    path = cache / "analysis_bids.m"
    if allow_download and not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/mcjpedro/speech_decoding/{spec['commit']}/matlab_code/analysis_bids.m"
        download_pinned_url(url, path)
    if not path.exists():
        return None
    actual = sha256_file(path)
    if actual != spec["analysis_bids_sha256"]:
        raise RuntimeError(f"analysis_bids.m hash mismatch: {actual} != {spec['analysis_bids_sha256']}")
    return path


def _ds006_clean_hashes(root: Path) -> set[str]:
    return {sha256_file(p) for p in root.glob("audio_internal/stimuli/**/cleaned/**/*.wav")}


def _ds006_trial_rows(config: dict[str, Any], lock: dict[str, Any], qc: dict[str, Any], fetch_aux: bool) -> list[dict[str, Any]]:
    _, pd = runtime()
    spec = config["sources"]["ds006104"]
    data_root = ROOT / spec["data_root"]
    clean_hashes = _ds006_clean_hashes(data_root)
    rows: list[dict[str, Any]] = []
    subjects = sorted(p.name.replace("sub-", "S") for p in data_root.glob("sub-*"))
    # BIDS subject S01 is named sub-01. Keep the official identifier in auxiliary provenance.
    bids_subjects = sorted(data_root.glob("sub-*"))
    for bids_subject in progress(bids_subjects, desc="audit ds006104 subjects"):
        suffix = bids_subject.name.removeprefix("sub-")
        subject = f"S{int(suffix):02d}" if suffix.isdigit() else suffix
        try:
            aux = _fetch_aux_event(subject, config) if fetch_aux else output_root(config) / "auxiliary" / spec["auxiliary_repository"]["commit"] / f"{subject}_Tab.csv"
        except RuntimeError as exc:
            qc["warnings"].append(f"ds006104 {subject}: {exc}")
            qc["exclusions"]["official_aux_download_failed"] += 1
            aux = output_root(config) / "auxiliary" / spec["auxiliary_repository"]["commit"] / f"{subject}_Tab.csv"
        if aux.exists():
            lock["official_aux"][f"events_information/{subject}_Tab.csv"] = source_lock_entry(aux, "official_event_table")
        aux_frame = pd.read_csv(aux) if aux.exists() else None
        event_files = sorted(bids_subject.glob("ses-*/eeg/*_events.tsv"))
        for events_path in event_files:
            task = next((x.removeprefix("task-") for x in events_path.name.split("_") if x.startswith("task-")), "unknown")
            run = next((x.removeprefix("run-") for x in events_path.name.split("_") if x.startswith("run-")), "01")
            session = next((p.name for p in events_path.parents if p.name.startswith("ses-")), "")
            frame = read_tsv(events_path, pd)
            onset_col = _event_column(frame, ["onset"])
            trial_col = _event_column(frame, ["trial_type", "value"])
            eeg_path = first_existing([events_path.parent / events_path.name.replace("_events.tsv", "_eeg.edf")])
            # Stimulus events are joined by their sequential TrialN when available.  In raw BIDS,
            # TMS and stimulus can be separate rows; only stimulus rows become model trials.
            for row_index, event in frame.iterrows():
                value = str(event[trial_col]) if trial_col else ""
                if task.lower() not in value.lower() and "stim" not in value.lower() and trial_col:
                    continue
                onset = float(event[onset_col]) if onset_col else float("nan")
                if not math.isfinite(onset):
                    qc["exclusions"]["missing_onset"] += 1
                    continue
                aux_row = None
                if aux_frame is not None:
                    # Events tables use task names (phonemes, singlephoneme, Words) and TrialN.
                    candidates = aux_frame
                    task_col = _event_column(aux_frame, ["Task", "task"])
                    if task_col:
                        candidates = candidates[candidates[task_col].astype(str).str.lower() == task.lower()]
                    # TrialN is the only intended join key.  Do not choose an
                    # apparently close event time, which could silently pair a
                    # different trial after a dropped BIDS row.
                    event_trial_col = _event_column(frame, ["trial"])
                    aux_trial_col = _event_column(candidates, ["TrialN", "trial"])
                    if event_trial_col and aux_trial_col:
                        candidates = candidates[candidates[aux_trial_col].astype(str) == str(event[event_trial_col])]
                    else:
                        candidates = candidates.iloc[0:0]
                    if len(candidates):
                        aux_row = candidates.iloc[0]
                stimulus = str(aux_row.get("Stimulus", "")) if aux_row is not None else ""
                is_single = task.lower() == "singlephoneme"
                audio = None if is_single else find_ds006_audio(data_root, stimulus)
                audio_sha = sha256_file(audio) if audio else ""
                semantics, evidence = audio_semantics_ds006104(audio_sha or None, clean_hashes)
                reason = ""
                if eeg_path is None:
                    reason = "missing_eeg"
                elif aux_row is None:
                    reason = "missing_official_aux_row"
                elif not is_single and audio is None:
                    reason = "missing_audio"
                if reason:
                    qc["exclusions"][reason] += 1
                source_zero = round_half_up(onset * 2000)
                identifier = trial_id("ds006104", subject, task, run, int(row_index), onset)
                tms = bool(int(aux_row.get("TMS", 0))) if aux_row is not None and str(aux_row.get("TMS", "")).strip() not in ("", "nan") else False
                entry = {
                    "trial_id": identifier, "dataset": "ds006104", "dataset_version": spec["openneuro_version"],
                    "subject": subject, "session": session, "task": task, "run": run,
                    "source_eeg_path": as_relative(eeg_path) if eeg_path else "", "source_eeg_sha256": "",
                    "source_event_path": as_relative(events_path), "source_event_sha256": sha256_file(events_path), "source_event_row": int(row_index),
                    "official_aux_path": as_relative(aux) if aux.exists() else "", "official_aux_sha256": sha256_file(aux) if aux.exists() else "", "official_aux_row": int(aux_row.name) if aux_row is not None else None,
                    "event_onset_seconds": onset, "event_to_sample_error": abs(onset * 2000 - source_zero),
                    "stimulus": stimulus, "audio_path": as_relative(audio) if audio else "", "audio_sha256": audio_sha,
                    "stimulus_content_id": stimulus_content_id("ds006104", stimulus) if stimulus else "",
                    "supervision_type": "label_only" if is_single else "paired_audio", "audio_semantics": "unknown" if is_single else semantics,
                    "audio_semantics_evidence": "singlephoneme_no_paired_wav" if is_single else evidence,
                    "neural_task": "mixed" if is_single else "perception", "response_onset_relative_s": 0.3 if is_single else None,
                    "response_onset_output_index": 140 if is_single else None,
                    "response_onset_provenance": "paper_protocol_nominal" if is_single else "",
                    "production_contaminated": is_single, "clean_perception_start_index": 64 if is_single else 0,
                    "clean_perception_end_index": 140 if is_single else 384,
                    "source_sfreq_hz": 2000, "target_sfreq_hz": 256, "eeg_zero_index": 64,
                    "audio_start_relative_to_eeg_samples": 64, "source_zero_sample": source_zero,
                    "source_start_sample": source_zero - 500, "source_end_sample": source_zero + 2500,
                    "source_run_offsets": json.dumps([{"path": as_relative(eeg_path), "start_sample": 0}]) if eeg_path else "[]", "boundary_overlap": False, "tms_applied": tms,
                    "tms_pulse_1_source_sample": None,
                    "tms_pulse_2_source_sample": None, "tms_intervals_source_half_open": "[]",
                    "qc_pass": not bool(reason) and abs(onset * 2000 - source_zero) <= .5,
                    "build_status": "included" if not reason else "excluded", "exclusion_reason": reason,
                    "channel_order_hash": channel_order_hash(spec["channel_order"]),
                }
                p1_index = aux_row.get("P1_TSidx", float("nan")) if aux_row is not None else float("nan")
                if tms and p1_index is not None and str(p1_index).lower() not in ("", "nan"):
                    # Verified pinned Matlab: pulses occur at P1_TSidx-0.100fs
                    # and P1_TSidx-0.050fs; P1_TSidx itself is not a pulse.
                    p1 = int(float(p1_index)) - 200
                    p2 = int(float(p1_index)) - 100
                    entry["tms_pulse_1_source_sample"] = p1
                    entry["tms_pulse_2_source_sample"] = p2
                    entry["tms_intervals_source_half_open"] = json.dumps([[p1 - 10, p1 + 51], [p2 - 10, p2 + 51]])
                rows.append(entry)
    qc["actual_subjects"]["ds006104"] = len(subjects)
    actual_inventory = sha256_bytes("".join(
        f"{subject}\t{EVENT_TABLE_SHA256[subject]}\n" for subject in sorted(EVENT_TABLE_SHA256)
    ).encode())
    lock["official_aux"]["events_inventory_computed_sha256"] = actual_inventory
    lock["official_aux"]["events_inventory_pinned_sha256"] = spec["auxiliary_repository"]["events_inventory_sha256"]
    # The release-level inventory SHA is pinned above.  The source release does
    # not specify its concatenation encoding, so audit verifies every pinned
    # table SHA256 individually (and records a transparent local aggregate).
    return rows


def write_frame(frame, path: Path, pd) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception as exc:
        # CSV is always written; parquet requires an optional engine.
        (path.with_suffix(".parquet.unavailable.txt")).write_text(str(exc) + "\n")


def audit(config: dict[str, Any], strict: bool, fetch_aux: bool) -> int:
    _, pd = runtime()
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    audit_timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    qc: dict[str, Any] = {"schema_version": config["schema_version"], "created_at": time.time(), "actual_subjects": {},
                           "exclusions": Counter(), "warnings": [], "preprocessing_statement": "harmonized_v2 is project preprocessing, not official preprocessing"}
    free_bytes = shutil.disk_usage(ROOT).free
    qc["disk_free_bytes"] = free_bytes
    qc["estimated_derived_bytes_range"] = [12 * 1024**3, 18 * 1024**3]
    if free_bytes < 18 * 1024**3:
        qc["warnings"].append("available disk is below the documented 18 GiB derived-data estimate")
    # Do not put wall-clock state in this immutable lock: its content hash must
    # remain identical when the same raw data are audited again.
    lock: dict[str, Any] = {"schema_version": config["schema_version"], "config_sha256": config["_config_sha256"], "files": [], "official_aux": {}}
    for dataset, spec in progress(config["sources"].items(), desc="audit pinned sources", total=len(config["sources"])):
        data_root = ROOT / spec["data_root"]
        description = data_root / "dataset_description.json"
        if description.exists():
            actual = sha256_file(description)
            lock["files"].append(source_lock_entry(description, "dataset_description"))
            if actual != spec["dataset_description_sha256"]:
                qc["warnings"].append(f"{dataset}: dataset_description hash mismatch")
        else:
            qc["warnings"].append(f"{dataset}: missing dataset_description")
        if dataset == "ds006104":
            participants = data_root / "participants.tsv"
            ok, value = pinned_hash_status(participants, spec["participants_sha256"])
            if participants.exists(): lock["files"].append(source_lock_entry(participants, "participants"))
            if not ok: qc["warnings"].append(f"{dataset}: participants.tsv hash {value}, expected pinned SHA256")
        stimulus_root = data_root / ("stimuli" if dataset == "ds004940" else "audio_internal/stimuli")
        if stimulus_root.exists():
            actual_inventory = inventory_sha256(stimulus_root)
            lock["stimulus_inventory"] = lock.get("stimulus_inventory", {}) | {dataset: {"root": as_relative(stimulus_root), "sha256": actual_inventory}}
            if actual_inventory != spec["stimulus_inventory_sha256"]:
                qc["warnings"].append(f"{dataset}: stimulus inventory hash mismatch")
        else:
            qc["warnings"].append(f"{dataset}: missing stimulus inventory")
    try:
        auxiliary_code = _fetch_analysis_bids(config, fetch_aux)
    except RuntimeError as exc:
        auxiliary_code = None
        qc["warnings"].append(f"ds006104 official Matlab source: {exc}")
    if auxiliary_code is None:
        qc["warnings"].append("ds006104: pinned analysis_bids.m is unavailable; use --fetch-aux")
    else:
        lock["official_aux"]["analysis_bids.m"] = source_lock_entry(auxiliary_code, "official_matlab_code")
    try:
        rows = _ds004_trial_rows(config, lock, qc) + _ds006_trial_rows(config, lock, qc, fetch_aux)
    except Exception as exc:
        # QC must be written even when a dependency/network/source issue is
        # encountered.  Keep the DS004 inventory if DS006 scanning is blocked.
        qc["warnings"].append(f"DS006104 scan failed: {type(exc).__name__}: {exc}")
        rows = _ds004_trial_rows(config, lock, qc)
    # Add every actually referenced source only once; raw hashes are intentionally computed here,
    # before build, so resume can reject a mutated raw dataset.
    sources = sorted({r[k] for r in rows for k in ("source_eeg_path", "source_event_path", "official_aux_path", "audio_path") if r.get(k)})
    for relative in progress(sources, desc="SHA256 source lock"):
        path = ROOT / relative
        if path.exists():
            entry = source_lock_entry(path, "source")
            lock["files"].append(entry)
            for row in rows:
                if row.get("source_eeg_path") == relative: row["source_eeg_sha256"] = entry["sha256"]
    lock["official_aux"]["repository_commit"] = config["sources"]["ds006104"]["auxiliary_repository"]["commit"]
    for dataset, spec in config["sources"].items():
        observed = sum(r["dataset"] == dataset for r in rows)
        if observed != spec["expected_trials"]:
            qc["warnings"].append(f"{dataset}: observed {observed} trials; pinned expectation {spec['expected_trials']}")
        observed_subjects = qc["actual_subjects"].get(dataset, 0)
        if observed_subjects != spec["expected_subjects"]:
            qc["warnings"].append(f"{dataset}: observed {observed_subjects} subjects; pinned expectation {spec['expected_subjects']}")
    lock["files"].sort(key=lambda x: x["path"])
    lock["source_lock_sha256"] = sha256_bytes(stable_json({k: v for k, v in lock.items() if k != "source_lock_sha256"}))
    for row in rows:
        row["source_lock_sha256"] = lock["source_lock_sha256"]
        row["preprocess_config_sha256"] = config["_config_sha256"]
        row["code_commit"], diff = git_provenance()
        row["code_diff_hash"] = sha256_bytes(diff.encode())
        row["audit_timestamp_utc"] = audit_timestamp_utc
    frame = pd.DataFrame(rows)
    write_frame(frame, root / "manifests" / "manifest_all", pd)
    (root / "source_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    qc["exclusions"] = dict(qc["exclusions"])
    qc["status"] = "warning" if qc["warnings"] else "pass"
    (root / "qc").mkdir(exist_ok=True)
    (root / "qc" / "audit.json").write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([{"reason": k, "count": v} for k, v in qc["exclusions"].items()]).to_csv(root / "qc" / "exclusions.csv", index=False)
    print(f"audit wrote {len(rows)} inventoried trials, source lock {lock['source_lock_sha256']}")
    return 2 if strict and qc["warnings"] else 0


def read_manifest(config: dict[str, Any]):
    _, pd = runtime()
    path = output_root(config) / "manifests" / "manifest_all.csv"
    if not path.exists():
        raise RuntimeError("audit first: manifest_all.csv is missing")
    return pd.read_csv(path, keep_default_na=False), pd


def make_splits(config: dict[str, Any]) -> int:
    frame, pd = read_manifest(config)
    root = output_root(config)
    lock = json.loads((root / "source_lock.json").read_text())
    commit, diff = git_provenance()
    eligible = frame[(frame["build_status"] == "included") & (frame["qc_pass"].astype(str).str.lower() == "true")].copy()
    subject_weights = eligible.groupby("subject").size().astype(int).to_dict()
    audio_eligible = eligible[eligible["supervision_type"] == "paired_audio"].copy()
    audio_eligible["audio_group"] = audio_eligible.apply(lambda r: r.get("stimulus_content_id") or f"{r.dataset}:file:{r.audio_sha256}", axis=1)
    audio_weights = audio_eligible.groupby("audio_group").size().astype(int).to_dict()
    subject_assign = balanced_group_assignment(subject_weights, config["folds"], config["split_seed"], "subject")
    audio_assign = balanced_group_assignment(audio_weights, config["folds"], config["split_seed"], "audio_content")
    records = []
    audio_groups_by_id = dict(zip(audio_eligible["trial_id"], audio_eligible["audio_group"]))
    for _, row in eligible.iterrows():
        s = subject_assign[row.subject]
        group = audio_groups_by_id.get(row.trial_id)
        a = audio_assign.get(group) if group else None
        for protocol in ("subject_ood", "audio_ood", "joint_ood"):
            for fold in range(config["folds"]):
                role, reason = split_role(protocol, fold, s["fold"], a["fold"] if a else None, row.supervision_type, config["folds"])
                records.append({"trial_id": row.trial_id, "protocol": protocol, "fold": fold, "role": role,
                                "exclusion_reason": reason, "subject_group": row.subject, "subject_fold": s["fold"],
                                "subject_group_trial_weight": s["trial_weight"], "subject_sort_position": s["sort_position"],
                                "audio_group": group or "", "audio_fold": a["fold"] if a else "",
                                "audio_group_trial_weight": a["trial_weight"] if a else "", "audio_sort_position": a["sort_position"] if a else "",
                                "assignment_algorithm": SPLIT_ALGORITHM, "assignment_seed": config["split_seed"],
                                "source_lock_sha256": lock["source_lock_sha256"], "preprocess_config_sha256": config["_config_sha256"],
                                "code_commit": commit, "code_diff_hash": sha256_bytes(diff.encode())})
    result = pd.DataFrame(records)
    split_root = root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for protocol in ("subject_ood", "audio_ood", "joint_ood"):
        for fold in range(config["folds"]):
            subset = result[(result.protocol == protocol) & (result.fold == fold)].copy()
            content = subset.to_csv(index=False).encode()
            file_hash = sha256_bytes(content)
            subset["split_csv_sha256"] = file_hash
            target = split_root / f"{protocol}_fold-{fold}.csv"
            subset.to_csv(target, index=False)
    split_hashes = {p.name: sha256_file(p) for p in sorted(split_root.glob("*_fold-*.csv"))}
    split_index = {"algorithm": SPLIT_ALGORITHM, "seed": config["split_seed"], "subject": subject_assign, "audio_content": audio_assign, "split_csv_sha256": split_hashes,
                   "source_lock_sha256": lock["source_lock_sha256"], "preprocess_config_sha256": config["_config_sha256"], "code_commit": commit, "code_diff_hash": sha256_bytes(diff.encode())}
    split_index["split_index_sha256"] = sha256_bytes(stable_json(split_index))
    (split_root / "assignment.json").write_text(json.dumps(split_index, indent=2, sort_keys=True) + "\n")
    print(f"wrote frozen split CSVs for {len(eligible)} included QC-pass trials")
    return 0


def build_audio_bank(config: dict[str, Any], resume: bool) -> int:
    """Materialise a deduplicated, explicitly parameterised audio bank.

    This is intentionally a separate phase: it updates only derived manifest
    fields (audio_id and audio metadata), never BIDS stimuli.  HDF5 groups are
    ragged by audio_id, avoiding hidden padding or waveform normalization.
    """
    h5py, _, np = require_build_runtime()
    try:
        import torchaudio  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("audio bank requires torch and torchaudio from requirements-preprocess.txt") from exc
    frame, pd = read_manifest(config)
    root = output_root(config)
    lock = json.loads((root / "source_lock.json").read_text())
    target = root / "audio_bank" / "audio_bank.h5"
    if target.exists() and resume:
        with h5py.File(target, "r") as bank:
            if not resume_compatible(dict(bank.attrs), config_sha=config["_config_sha256"], source_lock_sha=lock["source_lock_sha256"]):
                raise RuntimeError("resume refuses incompatible audio bank")
        print(f"reused {target}")
        return 0
    paired = frame[(frame.supervision_type == "paired_audio") & (frame.audio_path != "")].copy()
    # The same bytes can be cited under distinct experimental semantics.  Keep
    # separate logical bank records in that rare case rather than collapsing a
    # clean stimulus and a presented waveform into one semantic category.
    unique = paired.drop_duplicates(["audio_sha256", "audio_semantics"])
    partial = target.with_suffix(".h5.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists(): partial.unlink()
    melcfg = config["audio"]["mel"]
    with h5py.File(partial, "w") as bank:
        bank.attrs["schema_version"] = config["schema_version"]
        bank.attrs["preprocess_config_sha256"] = config["_config_sha256"]
        bank.attrs["source_lock_sha256"] = lock["source_lock_sha256"]
        commit, diff = git_provenance()
        bank.attrs["code_commit"] = commit
        bank.attrs["code_diff_hash"] = sha256_bytes(diff.encode())
        bank.attrs["audio_config"] = json.dumps(config["audio"], sort_keys=True)
        index = []
        for _, row in progress(unique.iterrows(), desc="audio bank", total=len(unique)):
            source = ROOT / row.audio_path
            wave, source_rate = torchaudio.load(str(source))
            original_channels, original_samples = int(wave.shape[0]), int(wave.shape[1])
            # Explicit arithmetic mean; no peak/RMS/DC/clipping transformations.
            wave = wave.mean(dim=0, keepdim=True)
            if source_rate != config["audio"]["target_sample_rate"]:
                wave = torchaudio.functional.resample(wave, source_rate, config["audio"]["target_sample_rate"], resampling_method="sinc_interp_kaiser")
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=config["audio"]["target_sample_rate"], n_fft=melcfg["n_fft"], win_length=melcfg["win_length"],
                hop_length=melcfg["hop_length"], window_fn=torch.hann_window, power=melcfg["power"], normalized=melcfg["normalized"],
                center=melcfg["center"], pad=melcfg["pad"], pad_mode=melcfg["pad_mode"], onesided=melcfg["onesided"],
                n_mels=melcfg["n_mels"], f_min=melcfg["f_min"], f_max=melcfg["f_max"], norm=melcfg["norm"], mel_scale=melcfg["mel_scale"],
            )(wave)
            log_mel = torch.log(mel.clamp_min(float(melcfg["log_epsilon"]))).squeeze(0).cpu().numpy().astype("float32")
            waveform = wave.squeeze(0).cpu().numpy().astype("float32")
            audio_id = f"audio-{row.audio_sha256[:16]}-{row.audio_semantics}"
            group = bank.create_group(audio_id)
            group.create_dataset("waveform", data=waveform, compression="gzip", shuffle=True)
            group.create_dataset("log_mel", data=log_mel, compression="gzip", shuffle=True)
            for key, value in {"source_path":row.audio_path, "source_sha256":row.audio_sha256, "source_sample_rate_hz":int(source_rate),
                               "source_channels":original_channels, "source_samples":original_samples, "target_sample_rate_hz":config["audio"]["target_sample_rate"],
                               "audio_semantics":row.audio_semantics, "audio_semantics_evidence":row.audio_semantics_evidence,
                               "stimulus_content_id":row.stimulus_content_id}.items():
                group.attrs[key] = value
            index.append({"audio_id":audio_id, "audio_sha256":row.audio_sha256, "audio_semantics":row.audio_semantics, "source_path":row.audio_path,
                          "audio_semantics_evidence":row.audio_semantics_evidence,
                          "stimulus_content_id":row.stimulus_content_id, "source_sample_rate_hz":source_rate,
                          "source_channels":original_channels, "source_samples":original_samples, "target_samples":len(waveform), "target_duration_s":len(waveform) / config["audio"]["target_sample_rate"], "mel_frames":log_mel.shape[1]})
    os.replace(partial, target)
    index_frame = pd.DataFrame(index)
    write_frame(index_frame, root / "audio_bank" / "audio_inventory", pd)
    audio_id_map = {(r.audio_sha256, r.audio_semantics): r.audio_id for _, r in index_frame.iterrows()}
    frame["audio_id"] = frame.apply(lambda r: audio_id_map.get((r.audio_sha256, r.audio_semantics), ""), axis=1)
    audio_duration_map = {(r.audio_sha256, r.audio_semantics): r.target_duration_s for _, r in index_frame.iterrows()}
    frame["audio_target_duration_s"] = frame.apply(lambda r: audio_duration_map.get((r.audio_sha256, r.audio_semantics), ""), axis=1)
    write_frame(frame, root / "manifests" / "manifest_all", pd)
    print(f"audio bank wrote {len(index_frame)} deduplicated files: {target}")
    return 0


def require_build_runtime():
    try:
        import h5py  # type: ignore
        import mne  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install optional runtime: pip install -r requirements-preprocess.txt") from exc
    return h5py, mne, np


def _h5_write_strings(group, name: str, values: list[str], h5py) -> None:
    group.create_dataset(name, data=values, dtype=h5py.string_dtype("utf-8"))


def _descriptive_stats(array, np):
    finite = np.isfinite(array)
    safe = np.where(finite, array, np.nan)
    med = np.nanmedian(safe, axis=2)
    return {"count": finite.sum(axis=2), "mean": np.nanmean(safe, axis=2), "std": np.nanstd(safe, axis=2),
            "median": med, "mad": np.nanmedian(np.abs(safe - med[:, :, None]), axis=2),
            "min": np.nanmin(safe, axis=2), "max": np.nanmax(safe, axis=2),
            "rms": np.sqrt(np.nanmean(safe ** 2, axis=2)), "nonfinite_count": (~finite).sum(axis=2)}


def _atomic_shard(path: Path, arrays: dict[str, Any], attrs: dict[str, Any], strings: dict[str, list[str]]) -> str:
    h5py, _, np = require_build_runtime()
    partial = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    with h5py.File(partial, "w") as output:
        for key, value in arrays.items():
            output.create_dataset(key, data=value, compression="gzip", shuffle=True)
        provenance = output.create_group("provenance")
        for key, values in strings.items():
            _h5_write_strings(provenance, key, values, h5py)
        stats = output.create_group("statistics")
        for key, value in _descriptive_stats(arrays["eeg"], np).items():
            stats.create_dataset(key, data=value.astype("float64"))
        stats.attrs["qc_descriptive_only"] = True
        for key, value in attrs.items():
            output.attrs[key] = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
        output.flush()
    os.replace(partial, path)
    return sha256_file(path)


def _raw_to_canonical(raw, channels: list[str], dataset: str, config: dict[str, Any], source_intervals: list[tuple[int, int]] | None = None):
    """Apply source TMS interpolation first, then only project harmonized choices."""
    _, mne, np = require_build_runtime()
    canonical = channels
    aliases = {normalise_ds004_channel(c): c for c in raw.ch_names} if dataset == "ds004940" else {c: c for c in raw.ch_names}
    missing = [c for c in canonical if c not in aliases]
    picked = [aliases[c] for c in canonical if c in aliases]
    raw.pick(picked)
    # TMS is source-rate, before filtering/resampling.  Endpoint interpolation is
    # a stated harmonized substitute for unavailable official fillgaps.
    if source_intervals:
        data = raw.get_data()
        for start, end in source_intervals:
            start, end = max(1, start), min(data.shape[1] - 1, end)
            if start < end:
                data[:, start:end] = np.linspace(data[:, start - 1], data[:, end], end - start, endpoint=False).T
        raw._data = data
    name_by_canonical = {normalise_ds004_channel(name) if dataset == "ds004940" else name: name for name in raw.ch_names}
    bad = [name_by_canonical[c] for c in canonical if c in name_by_canonical and c in raw.info.get("bads", [])]
    interpolated = []
    zero = list(missing)
    try:
        montage = mne.channels.make_standard_montage(config["harmonized"]["interpolation"][f"{dataset}_montage"])
        raw.set_montage(montage, on_missing="ignore")
        if bad and len(bad) / len(canonical) <= config["harmonized"]["interpolation"]["max_bad_fraction"]:
            raw.info["bads"] = bad
            raw.interpolate_bads(reset_bads=False)
            interpolated = [normalise_ds004_channel(c) if dataset == "ds004940" else c for c in bad]
        elif bad:
            zero += [normalise_ds004_channel(c) if dataset == "ds004940" else c for c in bad]
    except Exception:
        zero += [normalise_ds004_channel(c) if dataset == "ds004940" else c for c in bad]
    raw.set_eeg_reference("average", projection=False)
    raw.filter(*config["harmonized"]["bandpass_hz"], verbose="ERROR")
    raw.resample(config["harmonized"]["target_sfreq_hz"], npad="auto", verbose="ERROR")
    data = raw.get_data()
    aligned = np.zeros((len(canonical), data.shape[1]), dtype=np.float64)
    available = {normalise_ds004_channel(c) if dataset == "ds004940" else c: i for i, c in enumerate(raw.ch_names)}
    for i, c in enumerate(canonical):
        if c in available and c not in zero:
            aligned[i] = data[available[c]]
    badmask = np.array([c in set(interpolated) | set(zero) for c in canonical], dtype=bool)
    return aligned, badmask, np.array([c in interpolated for c in canonical], bool), np.array([c in zero for c in canonical], bool)


def _rows_for_shards(frame, requested_dataset: str, requested_subjects: set[str] | None):
    selected = frame[(frame.build_status == "included") & (frame.qc_pass.astype(str).str.lower() == "true")]
    if requested_dataset != "all": selected = selected[selected.dataset == requested_dataset]
    if requested_subjects: selected = selected[selected.subject.isin(requested_subjects)]
    return selected


def build(config: dict[str, Any], dataset: str, subjects: str, resume: bool, allow_audit_warnings: bool) -> int:
    h5py, mne, np = require_build_runtime()
    frame, pd = read_manifest(config)
    root = output_root(config)
    qc_audit = json.loads((root / "qc" / "audit.json").read_text())
    if qc_audit["status"] != "pass" and not allow_audit_warnings:
        raise RuntimeError("audit has warnings; inspect QC and use --allow-audit-warnings to record an explicit override")
    lock = json.loads((root / "source_lock.json").read_text())
    split_index_path = root / "splits" / "assignment.json"
    if not split_index_path.exists():
        raise RuntimeError("make-splits first; HDF5 shards must pin a frozen split index")
    split_index = json.loads(split_index_path.read_text())
    requested = None if subjects == "all" else set(subjects.split(","))
    selected = _rows_for_shards(frame, dataset, requested)
    built_rows = []
    build_timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    shard_groups = list(selected.groupby(["dataset", "subject", "task"], sort=True))
    for (ds, subject, task), group in progress(shard_groups, desc="EEG shards", total=len(shard_groups)):
        spec = config["sources"][ds]
        canonical = spec["channel_order"]
        target_len = config["harmonized"]["epoch"][ds]["total_samples_target"]
        target = root / "shards" / ds / subject / f"task-{task}.h5"
        if target.exists() and resume:
            with h5py.File(target, "r") as previous:
                if not resume_compatible(dict(previous.attrs), config_sha=config["_config_sha256"], source_lock_sha=lock["source_lock_sha256"], channel_hash=channel_order_hash(canonical), split_hash=split_index["split_index_sha256"]):
                    raise RuntimeError(f"resume refuses incompatible shard {target}")
            continue
        eegs=[]; valids=[]; cleans=[]; audio_losses=[]; tmsm=[]; bads=[]; interps=[]; zeros=[]; ids=[]; retained=[]
        for _, row in progress(group.iterrows(), desc=f"{ds}/{subject}/{task}", total=len(group)):
            raw_path = ROOT / row.source_eeg_path
            try:
                raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose="ERROR") if raw_path.suffix.lower() == ".bdf" else mne.io.read_raw_edf(raw_path, preload=True, verbose="ERROR")
                if abs(float(raw.info["sfreq"]) - float(row.source_sfreq_hz)) > 1e-6:
                    raise ValueError(f"source sampling rate {raw.info['sfreq']} != locked {row.source_sfreq_hz}")
                interval_text = str(row.get("tms_intervals_source_half_open", ""))
                intervals = json.loads(interval_text) if ds == "ds006104" and interval_text.startswith("[") else []
                # intervals are in full-run samples; raw files are one run.  Audit records offset
                # provenance and out-of-bound intervals simply remain absent from this output mask.
                tms_enabled = str(row.get("tms_applied", "")).strip().lower() in ("1", "true", "yes")
                data, bad, interp, zero = _raw_to_canonical(raw, canonical, ds, config, intervals if tms_enabled else [])
                start_target = round_half_up(int(row.source_start_sample) * 256 / int(row.source_sfreq_hz))
                end_target = start_target + target_len
                if start_target < 0 or end_target > data.shape[1]:
                    raise ValueError("epoch crosses raw-run boundary")
                ep = data[:, start_target:end_target]
                if ep.shape != (len(canonical), target_len): raise ValueError("integer epoch shape mismatch")
                eegs.append(ep.astype("float32")); valids.append(np.ones(target_len, dtype=bool))
                is_mixed = row.neural_task == "mixed"
                cleans.append(np.array(clean_perception_mask(target_len, int(row.clean_perception_start_index), int(row.clean_perception_end_index), is_mixed), bool))
                if ds == "ds004940" and str(row.get("audio_target_duration_s", "")) not in ("", "nan"):
                    loss_end = min(target_len, 64 + math.ceil((float(row.audio_target_duration_s) + .5) * 256))
                    audio_losses.append(np.array([i < loss_end for i in range(target_len)], bool))
                else:
                    # Label-only trials have no waveform loss; other paired trials
                    # retain their fixed epoch unless a dataset-specific duration is known.
                    audio_losses.append(np.array([row.supervision_type == "paired_audio"] * target_len, bool))
                ivals = [(int(x[0]), int(x[1])) for x in intervals]
                tmsm.append(np.array(source_interval_to_target_mask(source_zero=int(row.source_zero_sample), output_zero=int(row.eeg_zero_index), target_length=target_len, source_sfreq=int(row.source_sfreq_hz), target_sfreq=256, intervals=ivals), bool))
                bads.append(bad); interps.append(interp); zeros.append(zero); ids.append(row.trial_id); retained.append(row.to_dict())
            except Exception as exc:
                row = row.copy(); row["build_status"] = "excluded"; row["exclusion_reason"] = f"build:{type(exc).__name__}:{exc}"; row["build_timestamp_utc"] = build_timestamp_utc; built_rows.append(row.to_dict())
        if not eegs:
            continue
        arrays={"eeg": np.stack(eegs), "eeg_valid_mask": np.stack(valids), "clean_perception_mask": np.stack(cleans), "audio_loss_mask": np.stack(audio_losses), "tms_output_mask": np.stack(tmsm), "bad_channel_mask":np.stack(bads), "interpolated_channel_mask":np.stack(interps), "zero_filled_channel_mask":np.stack(zeros), "channel_valid_mask":~np.stack(zeros)}
        commit, diff = git_provenance()
        attrs={"schema_version": config["schema_version"], "preprocessing_profile": "harmonized_v2", "eeg_unit": "V", "eeg_dtype": "float32", "channel_order": canonical, "channel_order_hash": channel_order_hash(canonical), "preprocess_config_sha256":config["_config_sha256"], "source_lock_sha256":lock["source_lock_sha256"], "split_index_sha256":split_index["split_index_sha256"], "split_hash_required": True, "code_commit":commit, "code_diff_hash":sha256_bytes(diff.encode()), "audit_override_allow_warnings": bool(allow_audit_warnings), "tms_interpolation_algorithm": config["harmonized"]["tms"]["interpolation_algorithm"], "official_tms_code_sha256": config["harmonized"]["tms"]["source_code_sha256"]}
        checksum=_atomic_shard(target, arrays, attrs, {"trial_id":ids})
        for index, row in enumerate(retained):
            row.update({"shard_path":as_relative(target), "shard_row":index, "shard_sha256":checksum, "build_status":"included", "source_lock_sha256":lock["source_lock_sha256"], "preprocess_config_sha256":config["_config_sha256"], "split_index_sha256":split_index["split_index_sha256"], "audit_override_allow_warnings":bool(allow_audit_warnings), "build_timestamp_utc":build_timestamp_utc, "bad_channel_count":int(bads[index].sum()), "interpolated_channel_count":int(interps[index].sum()), "zero_filled_channel_count":int(zeros[index].sum())})
            built_rows.append(row)
    built = pd.DataFrame(built_rows)
    write_frame(built, root / "manifests" / "manifest_built", pd)
    print(f"build wrote {sum(built.get('build_status', []) == 'included') if len(built) else 0} trials")
    return 0


def fit_normalizer(config: dict[str, Any], split_csv: Path, fold: int, allow_mixed_production: bool) -> int:
    h5py, _, np = require_build_runtime()
    frame, pd = read_manifest(config)
    split = pd.read_csv(split_csv)
    train_ids = set(split[(split.fold == fold) & (split.role == "train")].trial_id)
    if not train_ids: raise RuntimeError("selected split/fold contains no training trials")
    built_path = output_root(config) / "manifests" / "manifest_built.csv"
    built = pd.read_csv(built_path, keep_default_na=False) if built_path.exists() else frame
    chosen = built[(built.trial_id.isin(train_ids)) & (built.build_status == "included")]
    if not allow_mixed_production: chosen = chosen[chosen.neural_task != "mixed"]
    count=None; mean=None; m2=None
    for shard, entries in chosen.groupby("shard_path"):
        with h5py.File(ROOT / shard, "r") as h5:
            if h5.attrs.get("eeg_unit") != "V": raise RuntimeError("normalizer refuses non-Volt EEG")
            data=h5["eeg"]
            for _, entry in entries.iterrows():
                x=data[int(entry.shard_row)].astype("float64")
                n=x.shape[1]; m=x.mean(axis=1); v=((x-m[:,None])**2).sum(axis=1)
                if count is None: count=np.full(x.shape[0], n, dtype=np.int64); mean=m; m2=v
                else:
                    total=count+n; delta=m-mean; m2 += v + delta**2*count*n/total; mean += delta*n/total; count=total
    result={"schema_version":config["schema_version"], "normalization_fit_role":"train_only", "protocol_split_csv":str(split_csv), "split_csv_sha256":sha256_file(split_csv), "fold":fold, "exclude_mixed_production":not allow_mixed_production, "count":count.tolist(), "mean":mean.tolist(), "std":np.sqrt(m2 / np.maximum(count-1,1)).tolist()}
    target=output_root(config)/"normalizers"/f"{split_csv.stem}_fold-{fold}.json"; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(target)
    return 0


def validate(config: dict[str, Any], strict: bool) -> int:
    h5py, _, np = require_build_runtime()
    _, pd = runtime()
    root=output_root(config); errors=[]
    split_index_path = root / "splits" / "assignment.json"
    expected_split_hash = json.loads(split_index_path.read_text())["split_index_sha256"] if split_index_path.exists() else ""
    built_path=root/"manifests"/"manifest_built.csv"
    if not built_path.exists(): raise RuntimeError("build first")
    frame=pd.read_csv(built_path, keep_default_na=False)
    for shard, rows in frame[frame.build_status == "included"].groupby("shard_path"):
        path=ROOT/shard
        if not path.exists(): errors.append(f"missing shard {shard}"); continue
        with h5py.File(path,"r") as h5:
            ds=rows.iloc[0].dataset; c=len(config["sources"][ds]["channel_order"]); t=config["harmonized"]["epoch"][ds]["total_samples_target"]
            if h5.attrs.get("eeg_unit") != "V" or h5["eeg"].dtype != np.dtype("float32"): errors.append(f"unit/dtype {shard}")
            if h5["eeg"].shape[1:] != (c,t): errors.append(f"shape {shard}: {h5['eeg'].shape}")
            for required in ("eeg_valid_mask", "clean_perception_mask", "audio_loss_mask", "tms_output_mask", "bad_channel_mask", "interpolated_channel_mask", "zero_filled_channel_mask", "channel_valid_mask"):
                if required not in h5: errors.append(f"missing {required} {shard}")
            if h5.attrs.get("channel_order_hash") != channel_order_hash(config["sources"][ds]["channel_order"]): errors.append(f"channel hash {shard}")
            if not expected_split_hash or h5.attrs.get("split_index_sha256", "") != expected_split_hash: errors.append(f"split hash {shard}")
            if np.any(~np.isfinite(h5["eeg"][:])): errors.append(f"nonfinite {shard}")
            for _, row in rows[rows.neural_task == "mixed"].iterrows():
                mask=h5["clean_perception_mask"][int(row.shard_row)]
                if mask.shape[0] != 384 or mask.sum()!=76 or not (mask[64:140].all() and not mask[:64].any() and not mask[140:].any()): errors.append(f"singlephoneme mask {row.trial_id}")
    for path in sorted((root/"splits").glob("*.csv")):
        split=pd.read_csv(path)
        for role_a, role_b, column in (("train","test","subject_group"),("train","test","audio_group"),("train","validation","subject_group"),("train","validation","audio_group")):
            # Empty audio group represents label-only rows and is not a group.
            a=set(split[(split.role==role_a)&(split[column]!="")][column]); b=set(split[(split.role==role_b)&(split[column]!="")][column])
            if a & b: errors.append(f"split leakage {path.name} {column} {role_a}/{role_b}: {sorted(a & b)[:3]}")
    report={"status":"pass" if not errors else "fail", "errors":errors, "created_at":time.time()}; (root/"qc"/"validate.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 2 if errors and strict else 0


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("audit"); a.add_argument("--strict",action="store_true"); a.add_argument("--fetch-aux",action="store_true",help="download pinned official DS006104 event tables")
    ab=sub.add_parser("build-audio-bank"); ab.add_argument("--resume", action="store_true")
    sub.add_parser("make-splits")
    b=sub.add_parser("build"); b.add_argument("--dataset",choices=["all","ds004940","ds006104"],default="all"); b.add_argument("--subjects",default="all"); b.add_argument("--resume",action="store_true"); b.add_argument("--allow-audit-warnings",action="store_true")
    n=sub.add_parser("fit-normalizer"); n.add_argument("--split-csv",type=Path,required=True); n.add_argument("--fold",type=int,required=True); n.add_argument("--allow-mixed-production",action="store_true")
    v=sub.add_parser("validate"); v.add_argument("--strict",action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args=parser().parse_args(argv); config,_=load_config(args.config)
    if args.command=="audit": return audit(config,args.strict,args.fetch_aux)
    if args.command=="build-audio-bank": return build_audio_bank(config,args.resume)
    if args.command=="make-splits": return make_splits(config)
    if args.command=="build": return build(config,args.dataset,args.subjects,args.resume,args.allow_audit_warnings)
    if args.command=="fit-normalizer": return fit_normalizer(config,args.split_csv,args.fold,args.allow_mixed_production)
    return validate(config,args.strict)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except RuntimeError as exc: print(f"error: {exc}",file=sys.stderr); raise SystemExit(2)
