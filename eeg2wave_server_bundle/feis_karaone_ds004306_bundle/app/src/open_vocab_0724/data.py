from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.open_vocab_0722.data import (
    COMMON_CHANNELS,
    DATASETS,
    DATASET_IDS,
    MontageRegistry,
    OpenVoiceContext,
    load_context,
    normalize_label,
    resolve_config_path,
)


TEACHER_CACHE_SCHEMA_VERSION = "openvoice-0724-teacher-v2"
PAIRING_POLICY_VERSION = "openvoice-factorized-pairing-v2"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSupervision:
    content: bool
    realization: bool
    timbre: bool
    energy: bool
    code: bool
    feis_prototype: bool
    exact_pair: bool
    audio_generation_eligible: bool

    def as_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            "content_supervision": torch.tensor(self.content, dtype=torch.bool),
            "realization_supervision": torch.tensor(self.realization, dtype=torch.bool),
            "timbre_supervision": torch.tensor(self.timbre, dtype=torch.bool),
            "energy_supervision": torch.tensor(self.energy, dtype=torch.bool),
            "code_supervision": torch.tensor(self.code, dtype=torch.bool),
            "feis_prototype_supervision": torch.tensor(
                self.feis_prototype, dtype=torch.bool
            ),
            "exact_pair_supervision": torch.tensor(self.exact_pair, dtype=torch.bool),
            "audio_generation_eligible": torch.tensor(
                self.audio_generation_eligible, dtype=torch.bool
            ),
        }


def supervision_for_dataset(
    dataset: str, *, reconstruction_eligible: bool = True
) -> DatasetSupervision:
    """Return the preregistered EEG/audio supervision route for one dataset."""

    name = str(dataset).lower()
    eligible = bool(reconstruction_eligible)
    if name == "karaone":
        return DatasetSupervision(
            content=True,
            realization=eligible,
            timbre=eligible,
            energy=eligible,
            code=eligible,
            feis_prototype=False,
            exact_pair=eligible,
            audio_generation_eligible=eligible,
        )
    if name == "feis":
        return DatasetSupervision(
            content=True,
            realization=False,
            timbre=eligible,
            energy=False,
            code=False,
            feis_prototype=eligible,
            exact_pair=False,
            audio_generation_eligible=eligible,
        )
    if name == "ds004306":
        return DatasetSupervision(
            content=False,
            realization=False,
            timbre=False,
            energy=False,
            code=False,
            feis_prototype=False,
            exact_pair=False,
            audio_generation_eligible=False,
        )
    raise ValueError(f"Unknown v0724 dataset: {dataset!r}")


@dataclass(frozen=True)
class AudioRecordSpec:
    audio_key: str
    audio_path: Path
    audio_relpath: str
    dataset: str
    label: str
    content_id: str
    audio_utterance_id: str
    audio_speaker_id: str
    eeg_subject_group_id: str
    eeg_subject_group_ids: tuple[str, ...]
    pairing_scope: str
    pairing_confidence: str
    split_names: tuple[str, ...]
    fit_split: bool
    source_valid_samples: int | None
    row_count: int
    audio_generation_eligible: bool

    def index_metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value["audio_path"] = str(self.audio_path)
        value["eeg_subject_group_ids"] = list(self.eeg_subject_group_ids)
        value["split_names"] = list(self.split_names)
        return value


def _pairing_scope(dataset: str, confidence: str) -> str:
    if dataset == "karaone" and confidence == "karaone_same_trial_overt":
        return "exact_trial"
    if dataset == "feis" and confidence == "feis_subject_label":
        return "unique_subject_label_prototype"
    if dataset == "ds004306":
        return "none"
    raise ValueError(
        f"Unexpected pairing metadata: dataset={dataset!r}, confidence={confidence!r}"
    )


def build_project_records(context: OpenVoiceContext) -> list[AudioRecordSpec]:
    """Deduplicate manifest rows into auditable, split-aware audio records."""

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in context.rows:
        grouped.setdefault(str(row["audio_key"]), []).append(row)
    records: list[AudioRecordSpec] = []
    for key in sorted(grouped):
        rows = grouped[key]
        first = rows[0]
        invariant = ("dataset", "label", "audio_relpath", "pairing_confidence")
        inconsistent = [
            name
            for name in invariant
            if len({str(row.get(name, "")) for row in rows}) != 1
        ]
        if inconsistent:
            raise ValueError(
                f"audio_key {key!r} has inconsistent metadata: {inconsistent}"
            )
        dataset = str(first["dataset"])
        label = normalize_label(first["label"])
        confidence = str(first["pairing_confidence"])
        subjects = tuple(sorted({str(row["subject_group_id"]) for row in rows}))
        split_names = tuple(sorted({context.split_for(row) for row in rows}))
        if dataset != "ds004306" and len(split_names) != 1:
            raise ValueError(
                f"Supervised audio key {key!r} crosses subject splits: {split_names}"
            )
        if dataset != "ds004306" and len(subjects) != 1:
            raise ValueError(
                f"Supervised audio key {key!r} crosses speakers: {subjects}"
            )
        valid_values = {str(row.get("audio_valid_samples", "")).strip() for row in rows}
        valid_values.discard("")
        if len(valid_values) > 1:
            raise ValueError(f"audio_key {key!r} has inconsistent audio_valid_samples")
        source_valid = int(next(iter(valid_values))) if valid_values else None
        subject = subjects[0] if len(subjects) == 1 else "multiple"
        speaker = subject if dataset in {"karaone", "feis"} else "unavailable"
        records.append(
            AudioRecordSpec(
                audio_key=key,
                audio_path=(context.audio_root / str(first["audio_relpath"])).resolve(),
                audio_relpath=str(first["audio_relpath"]),
                dataset=dataset,
                label=label,
                content_id=label,
                audio_utterance_id=key,
                audio_speaker_id=speaker,
                eeg_subject_group_id=subject,
                eeg_subject_group_ids=subjects,
                pairing_scope=_pairing_scope(dataset, confidence),
                pairing_confidence=confidence,
                split_names=split_names,
                fit_split=bool(split_names == ("train",) and dataset != "ds004306"),
                source_valid_samples=source_valid,
                row_count=len(rows),
                audio_generation_eligible=dataset != "ds004306",
            )
        )
    return records


class TeacherCacheV2:
    """Lazy, read-only loader for the sharded factorized teacher cache."""

    SCHEMA_VERSION = TEACHER_CACHE_SCHEMA_VERSION
    ARRAY_FIELDS = (
        "content_tokens",
        "content_token_mask",
        "realization_features",
        "realization_frame_mask",
        "log_mel_energy",
        "f0_log_hz",
        "voicing",
        "log_rms_dbfs",
        "activity_mask",
        "timbre_global",
        "has_timbre",
        "encodec_codes",
        "encodec_scale",
        "encodec_scale_valid",
        "code_valid_mask",
        "has_codec",
    )

    def __init__(
        self, path: str | Path, *, max_open_shards: int = 2, verify_hashes: bool = False
    ):
        self.path = Path(path).resolve()
        index_path = self.path / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"v0724 teacher index is missing: {index_path}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        if self.index.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported v0724 teacher cache: {self.index.get('schema_version')!r}"
            )
        self.record_index = {
            str(key): (str(location[0]), int(location[1]))
            for key, location in self.index.get("record_index", {}).items()
        }
        self.records = {
            str(key): dict(value)
            for key, value in self.index.get("records", {}).items()
        }
        if not self.record_index or set(self.record_index) != set(self.records):
            raise ValueError(
                "v0724 teacher record_index and records are empty or inconsistent"
            )
        self.keys = tuple(sorted(self.record_index))
        self.key_to_index = {key: index for index, key in enumerate(self.keys)}
        self.content_steps = int(self.index["content_steps"])
        self.content_dimension = int(self.index["content_dimension"])
        self.realization_frames = int(self.index["realization_frames"])
        self.realization_dimension = int(self.index.get("realization_dimension", 84))
        self.mel_bins = int(self.index["mel_bins"])
        self.timbre_dimension = int(self.index["timbre_dimension"])
        self.codebooks = int(self.index["codebooks"])
        self.code_steps = int(self.index["code_steps"])
        self.sample_rate = int(self.index["sample_rate"])
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        train_speakers = sorted(
            {
                str(record["audio_speaker_id"])
                for record in self.records.values()
                if bool(record.get("fit_split"))
                and str(record.get("audio_speaker_id")) != "unavailable"
            }
        )
        self.speaker_to_index = {
            speaker: index for index, speaker in enumerate(train_speakers)
        }
        self.realization_mean = np.zeros(self.realization_dimension, dtype=np.float32)
        self.realization_std = np.ones(self.realization_dimension, dtype=np.float32)
        self.feis_timbre_prototypes: dict[str, np.ndarray] = {}
        statistics_file = self.index.get("statistics_file")
        if statistics_file:
            statistics_path = self.path / str(statistics_file)
            if not statistics_path.is_file():
                raise FileNotFoundError(
                    f"v0724 train-only statistics are missing: {statistics_path}"
                )
            with np.load(statistics_path, allow_pickle=False) as statistics:
                if not bool(np.asarray(statistics["fit_split_only"]).reshape(-1)[0]):
                    raise ValueError(
                        "v0724 normalization statistics were not marked train-only"
                    )
                self.realization_mean = np.asarray(
                    statistics["realization_mean"], dtype=np.float32
                )
                self.realization_std = np.asarray(
                    statistics["realization_std"], dtype=np.float32
                )
                prototype_ids = np.asarray(statistics["feis_prototype_ids"]).astype(str)
                prototype_values = np.asarray(
                    statistics["feis_timbre_prototypes"], dtype=np.float32
                )
            if self.realization_mean.shape != (
                self.realization_dimension,
            ) or self.realization_std.shape != (self.realization_dimension,):
                raise ValueError(
                    "v0724 realization statistics have incompatible dimensions"
                )
            if prototype_values.shape != (len(prototype_ids), self.timbre_dimension):
                raise ValueError("v0724 FEIS prototype arrays are inconsistent")
            self.feis_timbre_prototypes = {
                key: prototype_values[index]
                for index, key in enumerate(prototype_ids.tolist())
            }
        if verify_hashes:
            report = self.audit()
            if not report["passed"]:
                raise ValueError(
                    f"v0724 teacher cache hash audit failed: {report['failed_files']}"
                )

    def __len__(self) -> int:
        return len(self.record_index)

    def __contains__(self, audio_key: object) -> bool:
        return str(audio_key) in self.record_index

    def metadata(self, audio_key: str) -> dict[str, Any]:
        try:
            return dict(self.records[str(audio_key)])
        except KeyError as error:
            raise KeyError(f"Unknown v0724 teacher audio key: {audio_key!r}") from error

    def _load_shard(self, name: str) -> dict[str, np.ndarray]:
        if name not in self._shards:
            path = self.path / name
            if not path.is_file():
                raise FileNotFoundError(f"v0724 teacher shard is missing: {path}")
            with np.load(path, allow_pickle=False) as raw:
                missing = {"keys", *self.ARRAY_FIELDS} - set(raw.files)
                if missing:
                    raise ValueError(
                        f"v0724 teacher shard {name} lacks fields: {sorted(missing)}"
                    )
                shard = {key: np.asarray(raw[key]) for key in raw.files}
            size = len(shard["keys"])
            if any(
                np.asarray(shard[field]).ndim == 0 or len(shard[field]) != size
                for field in self.ARRAY_FIELDS
            ):
                raise ValueError(
                    f"v0724 teacher shard {name} has inconsistent first dimensions"
                )
            self._shards[name] = shard
            self._shards.move_to_end(name)
            while len(self._shards) > self.max_open_shards:
                self._shards.popitem(last=False)
        return self._shards[name]

    def get(self, audio_key: str, default: Any = None) -> dict[str, Any] | Any:
        location = self.record_index.get(str(audio_key))
        if location is None:
            return default
        shard_name, row = location
        shard = self._load_shard(shard_name)
        if (
            row < 0
            or row >= len(shard["keys"])
            or str(shard["keys"][row]) != str(audio_key)
        ):
            raise ValueError(
                f"v0724 teacher index points to the wrong row for {audio_key!r}"
            )
        output: dict[str, Any] = {
            field: np.asarray(shard[field][row]) for field in self.ARRAY_FIELDS
        }
        output.update(self.metadata(str(audio_key)))
        output["audio_key"] = str(audio_key)
        return output

    lookup = get

    def audit(self) -> dict[str, Any]:
        expected = {
            str(key): str(value)
            for key, value in self.index.get("file_sha256", {}).items()
        }
        failed: list[str] = []
        missing: list[str] = []
        for name, digest in expected.items():
            path = self.path / name
            if not path.is_file():
                missing.append(name)
            elif _file_sha256(path) != digest:
                failed.append(name)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "passed": bool(expected and not failed and not missing),
            "files_checked": len(expected),
            "failed_files": failed,
            "missing_files": missing,
        }


def _row_selected(
    context: OpenVoiceContext,
    row: Mapping[str, str],
    split: str,
    generalization: str,
    holdout_label: str | None,
    loso_subject: str | None = None,
) -> bool:
    actual = context.split_for(dict(row))
    label = normalize_label(str(row["label"]))
    holdout = normalize_label(holdout_label) if holdout_label else None
    if loso_subject is not None:
        if generalization != "g1":
            raise ValueError("subject-LOSO is only defined for g1")
        if split == "test":
            raise PermissionError("subject-LOSO may not read the locked test split")
        if actual != "train":
            return False
        is_holdout = str(row["subject_group_id"]) == str(loso_subject)
        return is_holdout if split == "validation" else not is_holdout
    if generalization == "g1":
        return actual == split
    if holdout is None:
        raise ValueError(f"{generalization} requires holdout_label")
    if split == "train":
        return actual == "train" and label != holdout
    if generalization == "g2":
        return actual == "train" and label == holdout
    if generalization == "g3":
        return actual == split and label == holdout
    raise ValueError(f"Unknown generalization setting: {generalization!r}")


def _teacher_tensors(record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    codes = np.asarray(record["encodec_codes"], dtype=np.int64)
    code_mask = np.asarray(record["code_valid_mask"], dtype=bool)
    if code_mask.ndim == 1:
        code_mask = np.broadcast_to(code_mask, codes.shape).copy()
    return {
        "content_tokens": torch.from_numpy(
            np.ascontiguousarray(record["content_tokens"])
        ).float(),
        "content_token_mask": torch.from_numpy(
            np.ascontiguousarray(record["content_token_mask"], dtype=bool)
        ),
        "realization_features": torch.from_numpy(
            np.ascontiguousarray(record["realization_features"])
        ).float(),
        "realization_frame_mask": torch.from_numpy(
            np.ascontiguousarray(record["realization_frame_mask"], dtype=bool)
        ),
        "log_mel_energy": torch.from_numpy(
            np.ascontiguousarray(record["log_mel_energy"])
        ).float(),
        "f0_log_hz": torch.from_numpy(
            np.ascontiguousarray(record["f0_log_hz"])
        ).float(),
        "voicing": torch.from_numpy(np.ascontiguousarray(record["voicing"])).float(),
        "log_rms_dbfs": torch.from_numpy(
            np.ascontiguousarray(record["log_rms_dbfs"])
        ).float(),
        "activity_mask": torch.from_numpy(
            np.ascontiguousarray(record["activity_mask"], dtype=bool)
        ),
        "timbre_global": torch.from_numpy(
            np.ascontiguousarray(record["timbre_global"])
        ).float(),
        "has_timbre": torch.tensor(
            bool(np.asarray(record["has_timbre"]).reshape(-1)[0]), dtype=torch.bool
        ),
        "codes": torch.from_numpy(np.ascontiguousarray(codes)).long(),
        "code_valid_mask": torch.from_numpy(np.ascontiguousarray(code_mask)),
        "has_codec": torch.tensor(
            bool(np.asarray(record["has_codec"]).reshape(-1)[0]), dtype=torch.bool
        ),
    }


def _empty_teacher(cache: TeacherCacheV2) -> dict[str, Any]:
    return {
        "content_tokens": np.zeros(
            (cache.content_steps, cache.content_dimension), dtype=np.float32
        ),
        "content_token_mask": np.zeros(cache.content_steps, dtype=bool),
        "realization_features": np.zeros(
            (cache.realization_frames, cache.realization_dimension), dtype=np.float32
        ),
        "realization_frame_mask": np.zeros(cache.realization_frames, dtype=bool),
        "log_mel_energy": np.full(
            (cache.mel_bins, cache.realization_frames), -80.0, dtype=np.float32
        ),
        "f0_log_hz": np.zeros(cache.realization_frames, dtype=np.float32),
        "voicing": np.zeros(cache.realization_frames, dtype=np.float32),
        "log_rms_dbfs": np.full(cache.realization_frames, -80.0, dtype=np.float32),
        "activity_mask": np.zeros(cache.realization_frames, dtype=bool),
        "timbre_global": np.zeros(cache.timbre_dimension, dtype=np.float32),
        "has_timbre": np.asarray(False),
        "encodec_codes": np.zeros((cache.codebooks, cache.code_steps), dtype=np.int64),
        "code_valid_mask": np.zeros((cache.codebooks, cache.code_steps), dtype=bool),
        "has_codec": np.asarray(False),
        "reconstruction_eligible": False,
        "audio_generation_eligible": False,
        "active_duration_seconds": 0.0,
        "content_id": "unavailable",
        "audio_utterance_id": "unavailable",
        "audio_speaker_id": "unavailable",
        "pairing_scope": "none",
    }


class FactorizedAudioDataset(Dataset[dict[str, Any]]):
    """Unique-audio dataset for the v0724 audio-only factorization phase."""

    def __init__(
        self,
        context: OpenVoiceContext,
        teachers: TeacherCacheV2,
        *,
        split: str,
        datasets: Sequence[str] = ("karaone", "feis"),
        allow_locked_test: bool = False,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if split == "test" and not allow_locked_test:
            raise PermissionError(
                "Locked-test audio requires an already-validated gate"
            )
        selected = set(map(str, datasets))
        if not selected <= set(DATASETS):
            raise ValueError(f"Unknown datasets: {sorted(selected - set(DATASETS))}")
        self.context = context
        self.teachers = teachers
        self.split = split
        self.records = []
        for key in teachers.keys:
            metadata = teachers.metadata(key)
            if str(metadata.get("dataset")) not in selected:
                continue
            split_names = tuple(metadata.get("split_names", ()))
            if split in split_names and len(split_names) == 1:
                self.records.append(key)
        if not self.records:
            raise ValueError(
                f"No v0724 audio records for split={split}, datasets={sorted(selected)}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        key = self.records[index]
        record = self.teachers.get(key)
        assert record is not None
        eligible = (
            bool(record.get("reconstruction_eligible", False))
            and str(record["dataset"]) != "ds004306"
        )
        output: dict[str, Any] = _teacher_tensors(record)
        # In the audio-only phase an eligible real waveform supervises all
        # reconstruction branches. FEIS weakening applies only to EEG pairing.
        for name in ("content", "realization", "timbre", "energy", "code"):
            output[f"{name}_supervision"] = torch.tensor(
                name == "content" or eligible, dtype=torch.bool
            )
        output.update(
            {
                "audio_generation_eligible": torch.tensor(eligible, dtype=torch.bool),
                "exact_pair_supervision": torch.tensor(eligible, dtype=torch.bool),
                "feis_prototype_supervision": torch.tensor(False, dtype=torch.bool),
                "audio_balance_weight": torch.tensor(1.0, dtype=torch.float32),
                "feis_audio_weight": torch.tensor(1.0, dtype=torch.float32),
                "audio_idx": torch.tensor(
                    self.teachers.key_to_index[key], dtype=torch.long
                ),
                "label_idx": torch.tensor(
                    self.context.label_to_index[normalize_label(str(record["label"]))],
                    dtype=torch.long,
                ),
                "speaker_idx": torch.tensor(
                    self.teachers.speaker_to_index.get(
                        str(record.get("audio_speaker_id", "unavailable")), -1
                    ),
                    dtype=torch.long,
                ),
                "duration_seconds": torch.tensor(
                    float(record.get("active_duration_seconds", 0.0)),
                    dtype=torch.float32,
                ),
                "segment_valid_samples": torch.tensor(
                    int(record.get("segment_valid_samples", 0)), dtype=torch.long
                ),
                "native_sample_count": torch.tensor(
                    int(record.get("native_sample_count", 0)), dtype=torch.long
                ),
                "native_rms": torch.tensor(
                    float(record.get("native_rms", 0.0)), dtype=torch.float32
                ),
                "normalization_gain": torch.tensor(
                    float(record.get("normalization_gain", 1.0)), dtype=torch.float32
                ),
                "active_start_sample": torch.tensor(
                    int(record.get("active_start_sample", 0)), dtype=torch.long
                ),
                "active_end_sample": torch.tensor(
                    int(record.get("active_end_sample", 0)), dtype=torch.long
                ),
                "segment_source_start_sample": torch.tensor(
                    int(record.get("segment_source_start_sample", 0)), dtype=torch.long
                ),
                "segment_source_end_sample": torch.tensor(
                    int(record.get("segment_source_end_sample", 0)), dtype=torch.long
                ),
            }
        )
        for name in (
            "audio_key",
            "dataset",
            "label",
            "content_id",
            "audio_utterance_id",
            "audio_speaker_id",
            "eeg_subject_group_id",
            "pairing_scope",
            "pairing_confidence",
            "audio_path",
            "audio_relpath",
            "segment_pcm_sha256",
            "source_audio_sha256",
        ):
            output[name] = record.get(name, "")
        return output


class FactorizedEEGDataset(Dataset[dict[str, Any]]):
    """Variable-channel EEG dataset with explicit v0724 supervision routing."""

    def __init__(
        self,
        context: OpenVoiceContext,
        teachers: TeacherCacheV2,
        *,
        split: str,
        generalization: str = "g1",
        holdout_label: str | None = None,
        loso_subject: str | None = None,
        datasets: Sequence[str] = DATASETS,
        eeg_samples: int | None = None,
        max_open_payloads: int | None = None,
        allow_locked_test: bool = False,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if split == "test" and not allow_locked_test:
            raise PermissionError("Locked-test EEG requires an already-validated gate")
        if generalization not in {"g1", "g2", "g3"}:
            raise ValueError("generalization must be g1, g2, or g3")
        if loso_subject is not None and (split == "test" or generalization != "g1"):
            raise ValueError("subject-LOSO is development-only and requires g1")
        selected = set(map(str, datasets))
        if not selected <= set(DATASETS):
            raise ValueError(f"Unknown datasets: {sorted(selected - set(DATASETS))}")
        self.context = context
        self.teachers = teachers
        self.split = split
        self.generalization = generalization
        self.holdout_label = holdout_label
        self.loso_subject = loso_subject
        self.eeg_samples = int(eeg_samples or context.config["data"]["eeg_samples"])
        self.max_open_payloads = max(
            1,
            int(
                max_open_payloads or context.config["data"].get("max_open_payloads", 4)
            ),
        )
        self.rows = tuple(
            row
            for row in context.rows
            if row["dataset"] in selected
            and _row_selected(
                context,
                row,
                split,
                generalization,
                holdout_label,
                loso_subject,
            )
        )
        if not self.rows:
            raise ValueError(
                f"No v0724 EEG rows for split={split}, generalization={generalization}"
            )
        supervised_missing = sorted(
            {
                row["audio_key"]
                for row in self.rows
                if row["dataset"] != "ds004306" and row["audio_key"] not in teachers
            }
        )
        if supervised_missing:
            raise ValueError(
                f"v0724 teacher cache misses {len(supervised_missing)} supervised audio keys"
            )
        self._payloads: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._reuse_count = Counter(
            row["audio_key"] for row in self.rows if row["dataset"] == "feis"
        )
        train_speakers = teachers.speaker_to_index
        self.speaker_to_index = train_speakers

    def __len__(self) -> int:
        return len(self.rows)

    def _payload(self, relative: str) -> dict[str, np.ndarray]:
        if relative not in self._payloads:
            path = self.context.eeg_root / relative
            with np.load(path, allow_pickle=False) as raw:
                self._payloads[relative] = {
                    name: np.asarray(raw[name]) for name in raw.files
                }
            self._payloads.move_to_end(relative)
            while len(self._payloads) > self.max_open_payloads:
                self._payloads.popitem(last=False)
        return self._payloads[relative]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        relative = str(row["eeg_relpath"])
        raw = self._payload(relative)
        eeg_row = int(row["eeg_row"])
        eeg = np.asarray(raw["eeg"][eeg_row], dtype=np.float32)
        montage = self.context.montage_registry.for_recording(relative)
        raw_names = tuple(
            str(value) for value in np.asarray(raw["channel_names"]).reshape(-1)
        )
        by_name = {name.upper(): position for position, name in enumerate(raw_names)}
        missing_channels = [
            name for name in montage.channel_names if name.upper() not in by_name
        ]
        if missing_channels:
            raise ValueError(f"EEG/montage mismatch in {relative}: {missing_channels}")
        eeg = eeg[
            [by_name[name.upper()] for name in montage.channel_names],
            : self.eeg_samples,
        ]
        valid = min(max(int(row["eeg_valid_samples"]), 1), eeg.shape[-1])
        if eeg.shape[-1] < self.eeg_samples:
            eeg = np.pad(eeg, ((0, 0), (0, self.eeg_samples - eeg.shape[-1])))
        time_mask = np.arange(self.eeg_samples) < valid
        eeg[:, ~time_mask] = 0.0

        cached = self.teachers.get(str(row["audio_key"]))
        has_teacher = bool(
            cached is not None
            and np.asarray(cached["content_token_mask"], dtype=bool).any()
        )
        record: dict[str, Any] = (
            dict(cached) if cached is not None else _empty_teacher(self.teachers)
        )
        eligible = bool(record.get("reconstruction_eligible", False))
        route = supervision_for_dataset(
            str(row["dataset"]), reconstruction_eligible=eligible
        )
        label_key = normalize_label(str(row["label"]))
        audio_speaker_id = str(record.get("audio_speaker_id", "unavailable"))
        feis_weight = 1.0
        if row["dataset"] == "feis":
            feis_weight = 1.0 / max(1, int(self._reuse_count[row["audio_key"]]))
        teacher_tensors = _teacher_tensors(record)
        if row["dataset"] == "feis" and self.split == "train":
            prototype_id = f"{row['subject_group_id']}::{label_key}"
            prototype = self.teachers.feis_timbre_prototypes.get(prototype_id)
            if prototype is None:
                raise ValueError(
                    f"Missing train-only FEIS timbre prototype: {prototype_id}"
                )
            teacher_tensors["timbre_global"] = torch.from_numpy(
                np.ascontiguousarray(prototype)
            ).float()
            teacher_tensors["has_timbre"] = torch.tensor(True, dtype=torch.bool)
        common_names = {name.upper() for name in COMMON_CHANNELS}
        common = np.asarray(
            [name.upper() in common_names for name in montage.channel_names], dtype=bool
        )

        output: dict[str, Any] = {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "channel_xyz": torch.from_numpy(np.ascontiguousarray(montage.channel_xyz)),
            "channel_mask": torch.ones(len(montage.channel_names), dtype=torch.bool),
            "common_channel_mask": torch.from_numpy(common),
            "time_mask": torch.from_numpy(time_mask),
            **teacher_tensors,
            **route.as_tensor_dict(),
            "has_audio_teacher": torch.tensor(has_teacher, dtype=torch.bool),
            "audio_balance_weight": torch.tensor(feis_weight, dtype=torch.float32),
            "feis_audio_weight": torch.tensor(feis_weight, dtype=torch.float32),
            "audio_idx": torch.tensor(
                self.teachers.key_to_index.get(str(row["audio_key"]), -1),
                dtype=torch.long,
            ),
            "label_idx": torch.tensor(
                self.context.label_to_index[label_key], dtype=torch.long
            ),
            "dataset_idx": torch.tensor(
                DATASET_IDS[str(row["dataset"])], dtype=torch.long
            ),
            "subject_idx": torch.tensor(
                self.context.subject_to_index.get(str(row["subject_group_id"]), -1),
                dtype=torch.long,
            ),
            "speaker_idx": torch.tensor(
                self.speaker_to_index.get(audio_speaker_id, -1), dtype=torch.long
            ),
            "duration_seconds": torch.tensor(
                float(record.get("active_duration_seconds", 0.0)), dtype=torch.float32
            ),
            "segment_valid_samples": torch.tensor(
                int(record.get("segment_valid_samples", 0)), dtype=torch.long
            ),
            "native_sample_count": torch.tensor(
                int(record.get("native_sample_count", 0)), dtype=torch.long
            ),
            "native_rms": torch.tensor(
                float(record.get("native_rms", 0.0)), dtype=torch.float32
            ),
            "normalization_gain": torch.tensor(
                float(record.get("normalization_gain", 1.0)), dtype=torch.float32
            ),
            "active_start_sample": torch.tensor(
                int(record.get("active_start_sample", 0)), dtype=torch.long
            ),
            "active_end_sample": torch.tensor(
                int(record.get("active_end_sample", 0)), dtype=torch.long
            ),
            "segment_source_start_sample": torch.tensor(
                int(record.get("segment_source_start_sample", 0)), dtype=torch.long
            ),
            "segment_source_end_sample": torch.tensor(
                int(record.get("segment_source_end_sample", 0)), dtype=torch.long
            ),
            "sample_key": str(row["sample_key"]),
            "audio_key": str(row["audio_key"]),
            "dataset": str(row["dataset"]),
            "label": str(row["label"]),
            "label_key": label_key,
            "content_id": str(record.get("content_id", label_key)),
            "audio_utterance_id": str(
                record.get("audio_utterance_id", row["audio_key"])
            ),
            "audio_speaker_id": audio_speaker_id,
            "eeg_subject_group_id": str(row["subject_group_id"]),
            "subject_group_id": str(row["subject_group_id"]),
            "pairing_scope": str(record.get("pairing_scope", "none")),
            "pairing_confidence": str(row["pairing_confidence"]),
            "timbre_prototype_id": (
                f"{row['subject_group_id']}::{label_key}"
                if row["dataset"] == "feis"
                else "none"
            ),
            "channel_names": montage.channel_names,
            "eeg_relpath": relative,
            "eeg_row": eeg_row,
            "audio_path": str(record.get("audio_path", "")),
            "audio_relpath": str(record.get("audio_relpath", "")),
            "segment_pcm_sha256": str(record.get("segment_pcm_sha256", "")),
            "source_audio_sha256": str(record.get("source_audio_sha256", "")),
        }
        return output


def collate_factorized(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty v0724 batch")
    output: dict[str, Any] = {}
    consumed: set[str] = set()
    if "eeg" in samples[0]:
        batch = len(samples)
        max_channels = max(int(sample["eeg"].shape[0]) for sample in samples)
        max_time = max(int(sample["eeg"].shape[1]) for sample in samples)
        eeg = torch.zeros(batch, max_channels, max_time, dtype=torch.float32)
        xyz = torch.zeros(batch, max_channels, 3, dtype=torch.float32)
        channel_mask = torch.zeros(batch, max_channels, dtype=torch.bool)
        common_mask = torch.zeros_like(channel_mask)
        time_mask = torch.zeros(batch, max_time, dtype=torch.bool)
        for position, sample in enumerate(samples):
            channels, times = sample["eeg"].shape
            eeg[position, :channels, :times] = sample["eeg"]
            xyz[position, :channels] = sample["channel_xyz"]
            channel_mask[position, :channels] = sample["channel_mask"]
            common_mask[position, :channels] = sample["common_channel_mask"]
            time_mask[position, :times] = sample["time_mask"]
        output.update(
            eeg=eeg,
            channel_xyz=xyz,
            channel_mask=channel_mask,
            common_channel_mask=common_mask,
            time_mask=time_mask,
        )
        consumed.update(output)
    for key in samples[0]:
        if key in consumed:
            continue
        values = [sample[key] for sample in samples]
        output[key] = (
            torch.stack(values) if isinstance(values[0], torch.Tensor) else values
        )
    return output


# Short compatibility alias used in the plan and training scripts.
collate_openvoice_0724 = collate_factorized


__all__ = [
    "AudioRecordSpec",
    "COMMON_CHANNELS",
    "DATASETS",
    "DATASET_IDS",
    "DatasetSupervision",
    "FactorizedAudioDataset",
    "FactorizedEEGDataset",
    "MontageRegistry",
    "OpenVoiceContext",
    "PAIRING_POLICY_VERSION",
    "TEACHER_CACHE_SCHEMA_VERSION",
    "TeacherCacheV2",
    "build_project_records",
    "collate_factorized",
    "collate_openvoice_0724",
    "load_context",
    "normalize_label",
    "resolve_config_path",
    "supervision_for_dataset",
]
