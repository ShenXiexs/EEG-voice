"""Dataset adapters for ds006104 and ds005345.

These classes define the batch schema used by v0. They intentionally avoid
starting any training loop. EEG loading requires MNE-Python at access time.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - lets static inspection work without torch
    torch = None
    Dataset = object

from .audio_features import build_ds005345_voice_stats
from .config import load_simple_yaml


def _require_mne():
    try:
        import mne
    except Exception as exc:  # pragma: no cover
        raise ImportError("MNE-Python is required for EEG loading: python -m pip install mne") from exc
    return mne


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _unit_label(row: dict[str, str]) -> str:
    phones = [row.get(f"phoneme{i}", "").replace("\x00", "").strip() for i in (1, 2, 3)]
    phones = [p for p in phones if p and p != "n/a"]
    if len(phones) == 1:
        return "single"
    if len(phones) == 2:
        return "pair"
    if len(phones) == 3:
        return "triplet"
    return "unknown"


def _sensor_pos_from_raw(raw: Any):
    names = raw.ch_names
    pos = []
    for name in names:
        loc = raw.info["chs"][raw.ch_names.index(name)].get("loc")
        if loc is None:
            pos.append([0.0, 0.0, 0.0])
        else:
            pos.append([float(loc[0]), float(loc[1]), float(loc[2])])
    return pos


class DS006104EpochDataset(Dataset):
    """Stimulus-locked ds006104 epochs with phoneme/articulatory labels."""

    def __init__(
        self,
        root: str | Path,
        subjects: list[str] | None = None,
        sample_rate: int = 250,
        epoch_sec: float = 1.0,
        preload: bool = False,
    ):
        self.root = Path(root)
        self.subjects = subjects
        self.sample_rate = sample_rate
        self.epoch_sec = epoch_sec
        self.preload = preload
        self.items = self._index_items()

    def _index_items(self) -> list[dict[str, Any]]:
        items = []
        for events_path in sorted(self.root.glob("sub-*/ses-*/eeg/*events.tsv")):
            subject = events_path.parts[-4]
            if self.subjects and subject not in self.subjects:
                continue
            eeg_path = events_path.with_name(events_path.name.replace("_events.tsv", "_eeg.edf"))
            if not eeg_path.exists():
                continue
            for row in _read_tsv(events_path):
                if row.get("trial_type") != "stimulus":
                    continue
                items.append({"events_path": events_path, "eeg_path": eeg_path, "subject": subject, "row": row})
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch is required for dataset tensors")
        mne = _require_mne()
        item = self.items[index]
        row = item["row"]
        onset = float(row["onset"])
        raw = mne.io.read_raw_edf(item["eeg_path"], preload=self.preload, verbose=False)
        raw.pick_types(eeg=True)
        raw.load_data()
        raw.resample(self.sample_rate, verbose=False)
        segment = raw.copy().crop(tmin=onset, tmax=onset + self.epoch_sec, include_tmax=False)
        data = segment.get_data()
        sensor_pos = _sensor_pos_from_raw(raw)
        return {
            "eeg": torch.tensor(data, dtype=torch.float32),
            "sensor_pos": torch.tensor(sensor_pos, dtype=torch.float32),
            "channel_mask": torch.ones(len(raw.ch_names), dtype=torch.bool),
            "channel_names": raw.ch_names,
            "dataset_id": "ds006104",
            "labels": {
                "unit": _unit_label(row),
                "phoneme": "".join(
                    p for p in [row.get("phoneme1", ""), row.get("phoneme2", ""), row.get("phoneme3", "")]
                    if p and p not in {"n/a", "\x00"}
                ),
                "category": row.get("category", "n/a"),
                "manner": row.get("manner", "n/a"),
                "place": row.get("place", "n/a"),
                "voicing": row.get("voicing", "n/a"),
                "tms_target": row.get("tms_target", "n/a"),
            },
            "metadata": {"subject": item["subject"], "onset": onset, "eeg_path": str(item["eeg_path"])},
        }


class DS005345StreamDataset(Dataset):
    """Run-level ds005345 stream windows with configured audio targets."""

    def __init__(
        self,
        root: str | Path,
        run_config: str | Path,
        subjects: list[str] | None = None,
        sample_rate: int = 250,
        window_sec: float = 10.0,
        preload: bool = False,
    ):
        self.root = Path(root)
        self.run_config = load_simple_yaml(Path(run_config))
        self.subjects = subjects
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.preload = preload
        self.voice_stats = build_ds005345_voice_stats(self.root)
        self.items = self._index_items()

    def _index_items(self) -> list[dict[str, Any]]:
        items = []
        run_map = self.run_config.get("runs", {})
        for fif_path in sorted((self.root / "derivatives").glob("sub-*/eeg/*_eeg_preprocessed.fif")):
            subject = fif_path.parts[-3]
            if self.subjects and subject not in self.subjects:
                continue
            run_key = next((part for part in fif_path.name.split("_") if part.startswith("run-")), None)
            if not run_key or run_key not in run_map:
                continue
            items.append({"subject": subject, "eeg_path": fif_path, "run": run_key, "run_info": run_map[run_key]})
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch is required for dataset tensors")
        mne = _require_mne()
        item = self.items[index]
        raw = mne.io.read_raw_fif(item["eeg_path"], preload=self.preload, verbose=False)
        raw.pick_types(eeg=True)
        raw.load_data()
        raw.resample(self.sample_rate, verbose=False)
        segment = raw.copy().crop(tmin=0.0, tmax=self.window_sec, include_tmax=False)
        data = segment.get_data()
        stream = item["run_info"]["positive_stream"]
        audio_embedding = self.voice_stats[stream].vector()
        sensor_pos = _sensor_pos_from_raw(raw)
        return {
            "eeg": torch.tensor(data, dtype=torch.float32),
            "sensor_pos": torch.tensor(sensor_pos, dtype=torch.float32),
            "channel_mask": torch.ones(len(raw.ch_names), dtype=torch.bool),
            "channel_names": raw.ch_names,
            "dataset_id": "ds005345",
            "audio_embedding": torch.tensor(audio_embedding, dtype=torch.float32),
            "labels": {"condition": item["run_info"]["condition"], "positive_stream": stream},
            "metadata": {"subject": item["subject"], "run": item["run"], "eeg_path": str(item["eeg_path"])},
        }
