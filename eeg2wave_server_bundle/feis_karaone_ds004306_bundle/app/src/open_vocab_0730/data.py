from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


def normalize_label(value: str) -> str:
    return str(value).strip().strip("/").lower()


SOURCE_SPLITS = ("train", "validation")
P_CHANNELS = 66  # duration, loudness, 32-bin activity, 32-bin low-frequency envelope


def text_anchor(labels: Sequence[str], references: dict[str, str], *, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Frozen phoneme/text anchors for a train-only auxiliary CLIP target.

    They are deterministic signed character-ngram vectors, not trainable label
    embeddings. This lets the lexical form provide weak structure without
    becoming an inference input or a closed-vocabulary decoder.
    """
    anchors = np.zeros((len(labels), dimension), dtype=np.float32)
    available = np.zeros(len(labels), dtype=bool)
    for row, label in enumerate(labels):
        text = references.get(normalize_label(label))
        if not text:
            continue
        available[row] = True
        sequence = f"^{text.lower()}$"
        grams = [sequence[index:index + width] for width in (1, 2, 3) for index in range(max(0, len(sequence) - width + 1))]
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            column = int.from_bytes(digest[:4], "little") % dimension
            anchors[row, column] += 1.0 if digest[4] % 2 else -1.0
        norm = np.linalg.norm(anchors[row])
        if norm > 0:
            anchors[row] /= norm
    return anchors, available


@dataclass(frozen=True)
class PreparedRecords:
    arrays: dict[str, np.ndarray]
    roles: np.ndarray
    codebook: dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.arrays["sample_keys"].shape[0])


def _load_source(root: Path, split: str) -> dict[str, np.ndarray]:
    path = root / f"records_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=False)
    required = {"eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask", "mel", "activity", "duration", "sample_keys", "audio_keys", "labels", "subjects"}
    if required - set(value.files):
        raise ValueError(f"source cache {path} lacks {sorted(required - set(value.files))}")
    return {name: np.asarray(value[name]) for name in required}


def merge_source_cache(root: Path) -> dict[str, np.ndarray]:
    parts = [_load_source(root, split) for split in SOURCE_SPLITS]
    arrays = {name: np.concatenate([part[name] for part in parts], axis=0) for name in parts[0]}
    keys = arrays["sample_keys"].astype(str)
    if len(keys) != 1341 or len(set(keys.tolist())) != len(keys):
        raise ValueError(f"expected exactly 1,341 unique non-locked v0728 records, found {len(keys)}")
    return arrays


def assign_roles(subjects: Sequence[str], labels: Sequence[str], *, subject_holdout: Sequence[str], unseen_label: str) -> np.ndarray:
    held = {str(value) for value in subject_holdout}
    unseen = normalize_label(unseen_label)
    roles: list[str] = []
    for subject, label in zip(subjects, labels):
        label_is_unseen = normalize_label(str(label)) == unseen
        subject_is_held = str(subject) in held
        if subject_is_held and label_is_unseen:
            roles.append("subject_and_label_holdout")
        elif subject_is_held:
            roles.append("subject_holdout_seen")
        elif label_is_unseen:
            roles.append("label_holdout_seen_subject")
        else:
            roles.append("fit")
    return np.asarray(roles)


def _resize(values: np.ndarray, steps: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).view(1, 1, -1)
    return F.interpolate(tensor, size=steps, mode="linear", align_corners=False).view(-1).numpy()


def prosody_from_mel(mel: np.ndarray, activity: np.ndarray, duration: float) -> np.ndarray:
    """Explicit weak-P target; it never contains F0, formants, speaker ID, or label."""
    mel = np.asarray(mel, dtype=np.float32)
    active = np.asarray(activity, dtype=np.float32)
    amplitude = np.mean(np.power(10.0, mel / 20.0), axis=0)
    log_envelope = 20.0 * np.log10(np.maximum(amplitude, 1e-7))
    lowpass = np.convolve(log_envelope, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    activity_32 = _resize(active, 32)
    envelope_32 = _resize(lowpass, 32)
    loudness = float(np.mean(log_envelope[active > 0.5])) if np.any(active > 0.5) else float(np.mean(log_envelope))
    envelope_32 = envelope_32 - loudness
    return np.concatenate(([float(duration), loudness], activity_32, envelope_32)).astype(np.float32)


def fit_content_codebook(hubert: np.ndarray, roles: np.ndarray, *, components: int, clusters: int, seed: int) -> dict[str, np.ndarray]:
    fit = np.asarray(hubert[roles == "fit"], dtype=np.float32).reshape(-1, hubert.shape[-1])
    if not len(fit):
        raise ValueError("cannot fit codebook without fit records")
    pca = PCA(n_components=components, whiten=True, random_state=seed)
    reduced = pca.fit_transform(fit)
    kmeans = MiniBatchKMeans(n_clusters=clusters, batch_size=2048, n_init=10, random_state=seed)
    kmeans.fit(reduced)
    return {"pca_mean": pca.mean_.astype(np.float32), "pca_components": pca.components_.astype(np.float32), "pca_scale": np.maximum(pca.explained_variance_ ** 0.5, 1e-6).astype(np.float32), "centers": kmeans.cluster_centers_.astype(np.float32)}


def content_tokens(hubert: np.ndarray, codebook: dict[str, np.ndarray], *, steps: int = 16) -> np.ndarray:
    source = np.asarray(hubert, dtype=np.float32)
    reduced = (source - codebook["pca_mean"]) @ codebook["pca_components"].T / codebook["pca_scale"]
    distances = np.square(reduced[:, None, :] - codebook["centers"][None, :, :]).sum(-1)
    units = distances.argmin(-1).astype(np.int64)
    positions = np.linspace(0, len(units) - 1, steps).round().astype(int)
    return units[positions]


def prepare_records(source_root: Path, *, subject_holdout: Sequence[str], unseen_label: str, pca_components: int, clusters: int, seed: int) -> PreparedRecords:
    arrays = merge_source_cache(source_root)
    arrays["sample_keys"] = arrays["sample_keys"].astype(str)
    arrays["audio_keys"] = arrays["audio_keys"].astype(str)
    arrays["labels"] = arrays["labels"].astype(str)
    arrays["subjects"] = arrays["subjects"].astype(str)
    roles = assign_roles(arrays["subjects"], arrays["labels"], subject_holdout=subject_holdout, unseen_label=unseen_label)
    expected = {"fit": 1019, "subject_holdout_seen": 200, "label_holdout_seen_subject": 102, "subject_and_label_holdout": 20}
    actual = {role: int((roles == role).sum()) for role in expected}
    if actual != expected:
        raise ValueError(f"v0730 split integrity failure: expected {expected}, got {actual}")
    codebook = fit_content_codebook(arrays["hubert"], roles, components=pca_components, clusters=clusters, seed=seed)
    arrays["content_tokens"] = np.stack([content_tokens(value, codebook) for value in arrays["hubert"]])
    arrays["prosody"] = np.stack([prosody_from_mel(mel, activity, duration) for mel, activity, duration in zip(arrays["mel"], arrays["activity"], arrays["duration"])])
    return PreparedRecords(arrays=arrays, roles=roles, codebook=codebook)


def save_prepared(path: Path, values: PreparedRecords) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values.arrays, roles=values.roles, **{f"codebook_{key}": item for key, item in values.codebook.items()})


def load_prepared(path: Path) -> PreparedRecords:
    raw = np.load(path, allow_pickle=False)
    required = {"eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask", "mel", "activity", "duration", "sample_keys", "audio_keys", "labels", "subjects", "content_tokens", "prosody", "roles"}
    if required - set(raw.files):
        raise ValueError(f"prepared cache lacks {sorted(required - set(raw.files))}")
    arrays = {name: np.asarray(raw[name]) for name in required - {"roles"}}
    codebook = {name.removeprefix("codebook_"): np.asarray(raw[name]) for name in raw.files if name.startswith("codebook_")}
    return PreparedRecords(arrays=arrays, roles=np.asarray(raw["roles"]).astype(str), codebook=codebook)


class CPDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: PreparedRecords, roles: Iterable[str]):
        accepted = set(roles)
        self.records = records
        self.indices = np.flatnonzero(np.isin(records.roles, list(accepted))).tolist()
        if not self.indices:
            raise ValueError(f"empty v0730 dataset for roles {sorted(accepted)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        arrays = self.records.arrays
        return {"eeg": arrays["eeg"][index], "channel_xyz": arrays["channel_xyz"][index], "channel_mask": arrays["channel_mask"][index], "time_mask": arrays["time_mask"][index], "hubert": arrays["hubert"][index], "hubert_mask": arrays["hubert_mask"][index], "content_tokens": arrays["content_tokens"][index], "prosody": arrays["prosody"][index], "mel": arrays["mel"][index], "activity": arrays["activity"][index], "duration": arrays["duration"][index], "sample_key": str(arrays["sample_keys"][index]), "audio_key": str(arrays["audio_keys"][index]), "label": str(arrays["labels"][index]), "subject": str(arrays["subjects"][index]), "role": str(self.records.roles[index])}


def collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_names = ("eeg", "channel_xyz", "channel_mask", "time_mask", "hubert", "hubert_mask", "content_tokens", "prosody", "mel", "activity", "duration")
    result: dict[str, Any] = {name: torch.as_tensor(np.stack([item[name] for item in items])) for name in tensor_names}
    result["channel_mask"] = result["channel_mask"].bool()
    result["time_mask"] = result["time_mask"].bool()
    result["hubert_mask"] = result["hubert_mask"].bool()
    result["content_tokens"] = result["content_tokens"].long()
    for name in ("sample_key", "audio_key", "label", "subject", "role"):
        result[name] = [str(item[name]) for item in items]
    return result
