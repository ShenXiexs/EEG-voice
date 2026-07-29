from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .data import (
    CPDataset,
    PreparedRecords,
    _load_source,
    assign_roles,
    collate,
    content_tokens,
    fit_content_codebook,
    load_prepared,
    normalize_label,
    prosody_from_mel,
    save_prepared,
    text_anchor,
)


FIXED_SOURCE_SPLITS = ("train", "validation", "locked_test", "diagnostic")
PAIR_ROLES = (
    "fit",
    "subject_holdout_seen",
    "label_holdout_seen_subject",
    "subject_and_label_holdout",
)
FINAL_TEST_ROLES = (
    "locked_test_seen_label",
    "locked_test_unseen_label",
    "diagnostic_subject_seen_label",
    "diagnostic_subject_unseen_label",
)


def merge_fixed_source_cache(root: Path) -> dict[str, np.ndarray]:
    parts = [_load_source(root, split) for split in FIXED_SOURCE_SPLITS]
    arrays = {name: np.concatenate([part[name] for part in parts], axis=0) for name in parts[0]}
    arrays["source_split"] = np.concatenate(
        [np.full(len(part["sample_keys"]), split) for split, part in zip(FIXED_SOURCE_SPLITS, parts)]
    )
    keys = arrays["sample_keys"].astype(str)
    if len(keys) != 1913 or len(set(keys.tolist())) != len(keys):
        raise ValueError(f"expected 1,913 unique v0730-fixed records, found {len(keys)}")
    return arrays


def assign_fixed_roles(
    source_split: Sequence[str],
    subjects: Sequence[str],
    labels: Sequence[str],
    *,
    subject_holdout: Sequence[str],
    unseen_label: str,
) -> np.ndarray:
    base = assign_roles(
        subjects, labels, subject_holdout=subject_holdout, unseen_label=unseen_label
    )
    unseen = normalize_label(unseen_label)
    roles: list[str] = []
    for split, label, base_role in zip(source_split, labels, base):
        label_is_unseen = normalize_label(label) == unseen
        if split == "locked_test":
            roles.append("locked_test_unseen_label" if label_is_unseen else "locked_test_seen_label")
        elif split == "diagnostic":
            roles.append(
                "diagnostic_subject_unseen_label" if label_is_unseen else "diagnostic_subject_seen_label"
            )
        else:
            roles.append(str(base_role))
    return np.asarray(roles)


def prepare_fixed_records(
    source_root: Path,
    *,
    subject_holdout: Sequence[str],
    unseen_label: str,
    pca_components: int,
    clusters: int,
    seed: int,
) -> PreparedRecords:
    arrays = merge_fixed_source_cache(source_root)
    for key in ("sample_keys", "audio_keys", "labels", "subjects", "source_split"):
        arrays[key] = arrays[key].astype(str)
    roles = assign_fixed_roles(
        arrays["source_split"],
        arrays["subjects"],
        arrays["labels"],
        subject_holdout=subject_holdout,
        unseen_label=unseen_label,
    )
    required = {
        "fit": 1019,
        "subject_holdout_seen": 200,
        "label_holdout_seen_subject": 102,
        "subject_and_label_holdout": 20,
        "locked_test_seen_label": 250,
        "locked_test_unseen_label": 25,
        "diagnostic_subject_seen_label": 270,
        "diagnostic_subject_unseen_label": 27,
    }
    actual = {role: int((roles == role).sum()) for role in required}
    if actual != required:
        raise ValueError(f"v0730-fixed split integrity failure: expected {required}, got {actual}")

    codebook = fit_content_codebook(
        arrays["hubert"], roles, components=pca_components, clusters=clusters, seed=seed
    )
    arrays["content_tokens"] = np.stack(
        [content_tokens(value, codebook) for value in arrays["hubert"]]
    )
    arrays["prosody"] = np.stack(
        [
            prosody_from_mel(mel, activity, duration)
            for mel, activity, duration in zip(
                arrays["mel"], arrays["activity"], arrays["duration"]
            )
        ]
    )
    return PreparedRecords(arrays=arrays, roles=roles, codebook=codebook)


__all__ = [
    "CPDataset",
    "FINAL_TEST_ROLES",
    "PAIR_ROLES",
    "PreparedRecords",
    "collate",
    "load_prepared",
    "prepare_fixed_records",
    "save_prepared",
    "text_anchor",
]
