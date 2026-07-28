from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .runtime import resolve_config_path, stable_hash


def normalize_label(value: str) -> str:
    return str(value).strip().strip("/").lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


@dataclass(frozen=True)
class Montage:
    names: tuple[str, ...]
    xyz: np.ndarray


@dataclass(frozen=True)
class Context:
    config_path: Path
    cfg: dict[str, Any]
    eeg_root: Path
    manifest_path: Path
    split_path: Path
    montage_path: Path
    rows: tuple[dict[str, str], ...]
    montages: dict[str, Montage]
    recording_to_montage: dict[str, str]

    @property
    def development_subjects(self) -> tuple[str, ...]:
        return tuple(self.cfg["gating"]["development_subjects"])


def load_context(config_path: Path, cfg: dict[str, Any]) -> Context:
    eeg_root = resolve_config_path(config_path, cfg["data"]["eeg_output_root"])
    manifest_path = eeg_root / "manifests" / "unified_trials.csv"
    split_path = resolve_config_path(config_path, cfg["data"]["subject_split_file"])
    montage_path = resolve_config_path(config_path, cfg["data"]["montage_registry"])
    rows = read_csv(manifest_path)
    registry = json.loads(montage_path.read_text(encoding="utf-8"))
    montages = {
        key: Montage(tuple(value["channel_names"]), np.asarray(value["channel_xyz"], dtype=np.float32))
        for key, value in registry["montages"].items()
    }
    selected = []
    required = {"sample_key", "dataset", "subject_group_id", "trial_index", "label", "eeg_relpath", "eeg_row", "eeg_valid_samples", "audio_key", "audio_relpath", "pairing_confidence"}
    for row in rows:
        if row["dataset"] != "karaone" or row["subject_group_id"] not in cfg["gating"]["development_subjects"] + cfg["gating"]["diagnostic_subjects"]:
            continue
        if required - set(row):
            raise ValueError("unified manifest lacks v0728-required fields")
        if row["pairing_confidence"] != "karaone_same_trial_overt":
            raise ValueError(f"unexpected KaraOne pairing: {row['pairing_confidence']}")
        selected.append(row)
    if not selected:
        raise ValueError("no KaraOne rows selected for v0728")
    return Context(config_path, cfg, eeg_root, manifest_path, split_path, montage_path, tuple(selected), montages, dict(registry["recordings"]))


def internal_split(rows: Iterable[dict[str, str]], *, seed: int, include_diagnostic: bool = False, development_subjects: Sequence[str] = ()) -> dict[str, str]:
    """Deterministic subject-label split, with no chronological allocation bias."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not include_diagnostic and row["subject_group_id"] not in development_subjects:
            continue
        grouped[(row["subject_group_id"], normalize_label(row["label"]))].append(row)
    quotas = {15: (10, 2, 3), 12: (8, 2, 2), 11: (7, 2, 2)}
    result: dict[str, str] = {}
    for (subject, label), items in grouped.items():
        if len(items) not in quotas:
            raise ValueError(f"unsupported trial count for {subject}/{label}: {len(items)}")
        train_n, valid_n, _ = quotas[len(items)]
        ordered = sorted(items, key=lambda row: stable_hash("v0728", seed, subject, label, row["sample_key"]))
        for row in ordered[:train_n]: result[row["sample_key"]] = "train"
        for row in ordered[train_n:train_n + valid_n]: result[row["sample_key"]] = "validation"
        for row in ordered[train_n + valid_n:]: result[row["sample_key"]] = "locked_test"
    return result


class CacheV3:
    """Split-isolated read-only cache. Locked data requires explicit authorization."""
    def __init__(self, root: Path, split: str, *, allow_locked: bool = False):
        if split == "locked_test" and not allow_locked:
            raise PermissionError("v0728 locked-test cache requires formal authorization")
        if split not in {"train", "validation", "locked_test", "diagnostic"}:
            raise ValueError(f"unknown cache split: {split}")
        path = root / f"records_{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        self.raw = np.load(path, allow_pickle=False)
        self.keys = np.asarray(self.raw["sample_keys"]).astype(str)
        self.index = {key: i for i, key in enumerate(self.keys.tolist())}
        if len(self.index) != len(self.keys):
            raise ValueError("duplicate v0728 cache sample key")
        for name in ("eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "mel", "activity", "duration", "labels", "subjects", "audio_keys"):
            if name not in self.raw.files:
                raise ValueError(f"cache lacks {name}")

    def __len__(self) -> int: return len(self.keys)

    def item(self, index: int) -> dict[str, Any]:
        return {"sample_key": str(self.keys[index]), "eeg": self.raw["eeg"][index], "channel_xyz": self.raw["channel_xyz"][index], "channel_mask": self.raw["channel_mask"][index], "time_mask": self.raw["time_mask"][index], "hubert": self.raw["hubert"][index], "hubert_mask": self.raw["hubert_mask"][index], "mel": self.raw["mel"][index], "activity": self.raw["activity"][index], "duration": self.raw["duration"][index], "label": str(self.raw["labels"][index]), "subject": str(self.raw["subjects"][index]), "audio_key": str(self.raw["audio_keys"][index])}


class DualLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, cache: CacheV3, *, labels: Sequence[str] | None = None, exclude_subject: str | None = None, only_subject: str | None = None):
        normalized = None if labels is None else {normalize_label(label) for label in labels}
        self.cache = cache
        self.indices = [i for i in range(len(cache)) if (normalized is None or normalize_label(str(cache.raw["labels"][i])) in normalized) and (exclude_subject is None or str(cache.raw["subjects"][i]) != exclude_subject) and (only_subject is None or str(cache.raw["subjects"][i]) == only_subject)]
        if not self.indices: raise ValueError("dataset selection is empty")
        classes = sorted({normalize_label(str(cache.raw["labels"][i])) for i in self.indices})
        self.label_to_index = {label: position for position, label in enumerate(classes)}

    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, item: int) -> dict[str, Any]:
        value = self.cache.item(self.indices[item])
        value["label_index"] = self.label_to_index[normalize_label(value["label"])]
        return value


def collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_names = ("eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask", "mel", "activity", "duration", "label_index")
    output: dict[str, Any] = {name: torch.as_tensor(np.stack([item[name] for item in items])) for name in tensor_names}
    for name in ("sample_key", "label", "subject", "audio_key"):
        output[name] = [str(item[name]) for item in items]
    output["channel_mask"] = output["channel_mask"].bool(); output["time_mask"] = output["time_mask"].bool(); output["hubert_mask"] = output["hubert_mask"].bool(); output["activity"] = output["activity"].bool()
    return output


def balanced_indices(dataset: DualLatentDataset, seed: int) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for position, source in enumerate(dataset.indices): groups[dataset.label_to_index[normalize_label(str(dataset.cache.raw["labels"][source]))]].append(position)
    rng = np.random.default_rng(seed)
    maximum = max(map(len, groups.values()))
    selected: list[int] = []
    for values in groups.values(): selected.extend(rng.choice(values, size=maximum, replace=True).tolist())
    rng.shuffle(selected)
    return selected
