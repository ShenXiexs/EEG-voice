from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def envelope_from_mel(mel: np.ndarray) -> np.ndarray:
    return np.mean(np.power(10.0, np.asarray(mel, dtype=np.float32) / 20.0), axis=-2)


def envelope_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def activity_f1(logits_or_probability: np.ndarray, target: np.ndarray) -> float:
    value = np.asarray(logits_or_probability)
    prediction = value >= 0.5
    reference = np.asarray(target) >= 0.5
    tp = np.logical_and(prediction, reference).sum()
    return float(2 * tp / max(2 * tp + np.logical_and(prediction, ~reference).sum() + np.logical_and(~prediction, reference).sum(), 1))


def bootstrap_subject_gain(subjects: Iterable[str], gain: Iterable[float], *, samples: int, seed: int) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for subject, value in zip(subjects, gain):
        grouped[str(subject)].append(float(value))
    names = sorted(grouped)
    if not names:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "subjects": 0}
    per_subject = np.asarray([np.mean(grouped[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(per_subject, size=len(per_subject), replace=True).mean() for _ in range(samples)])
    return {"mean": float(per_subject.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "subjects": int(len(names))}


def role_counts(roles: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for role in roles:
        result[str(role)] = result.get(str(role), 0) + 1
    return dict(sorted(result.items()))
