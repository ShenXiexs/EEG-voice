"""Numerical speech-structure metrics for v0724.

All morphology comparisons operate on floating-point log-mel matrices.  Plot
files are presentation artifacts only and are never read by this module.
Frequency bins are kept fixed; only the time axis may be normalized.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable

import numpy as np
from scipy.signal import stft


LOG_MEL_BINS = 80
LOG_MEL_FLOOR_DB = -80.0
LOG_MEL_CEILING_DB = 0.0
DEFAULT_WINDOW_MS = 25.0
DEFAULT_HOP_MS = 10.0
DEFAULT_MORPHOLOGY_FRAMES = 128


def _as_audio(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains NaN or infinite values")
    return audio


def _unit_rms(value: np.ndarray) -> np.ndarray:
    audio = _as_audio(value)
    if len(audio) == 0:
        return audio
    rms = math.sqrt(float(np.mean(audio * audio)) + 1e-12)
    return audio / rms if rms > 1e-8 else np.zeros_like(audio)


def _frame_signal(audio: np.ndarray, window: int, hop: int) -> np.ndarray:
    value = _as_audio(audio)
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    frame_count = max(1, int(math.ceil(max(len(value), 1) / hop)))
    required = (frame_count - 1) * hop + window
    padded = np.pad(value, (0, max(0, required - len(value))))
    starts = np.arange(frame_count)[:, None] * hop
    offsets = np.arange(window)[None, :]
    return padded[starts + offsets]


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    size = min(np.asarray(first).size, np.asarray(second).size)
    if size < 2:
        return 0.0
    raw_x = np.asarray(first, dtype=np.float64).reshape(-1)[:size]
    raw_y = np.asarray(second, dtype=np.float64).reshape(-1)[:size]
    if np.allclose(raw_x, raw_y, atol=1e-10, rtol=1e-8):
        return 1.0
    x = raw_x - float(np.mean(raw_x))
    y = raw_y - float(np.mean(raw_y))
    denominator = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denominator) if denominator > 1e-12 else 0.0


def rms_envelope(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    window = max(1, round(sample_rate * window_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    frames = _frame_signal(audio, window, hop)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


def _fill_short_false_runs(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if maximum_gap <= 0 or not result.any():
        return result
    true_indices = np.flatnonzero(result)
    for left, right in zip(true_indices[:-1], true_indices[1:]):
        if 1 < right - left <= maximum_gap + 1:
            result[left : right + 1] = True
    return result


def active_frame_mask(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    noise_margin_db: float = 6.0,
    peak_margin_db: float = 40.0,
    merge_gap_ms: float = 50.0,
) -> np.ndarray:
    """Detect active speech using the preregistered RMS threshold rule."""

    envelope = rms_envelope(audio, sample_rate, window_ms, hop_ms)
    if envelope.size == 0 or float(np.max(envelope)) <= 1e-8:
        return np.zeros_like(envelope, dtype=bool)
    level_db = 20.0 * np.log10(np.maximum(envelope, 1e-12))
    peak = float(np.max(level_db))
    noise_floor = float(np.percentile(level_db, 10))
    threshold = max(noise_floor + float(noise_margin_db), peak - float(peak_margin_db))
    active = level_db >= threshold
    gap_frames = max(0, round(float(merge_gap_ms) / float(hop_ms)))
    return _fill_short_false_runs(active, gap_frames)


def detect_active_region(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    margin_ms: float = 100.0,
    merge_gap_ms: float = 50.0,
) -> tuple[int, int]:
    """Return ``[start,end)`` active-region sample indices, including margins."""

    value = _as_audio(audio)
    mask = active_frame_mask(
        value,
        sample_rate,
        window_ms=window_ms,
        hop_ms=hop_ms,
        merge_gap_ms=merge_gap_ms,
    )
    if not mask.any():
        return 0, 0
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    window = max(1, round(sample_rate * window_ms / 1000.0))
    margin = max(0, round(sample_rate * margin_ms / 1000.0))
    active_indices = np.flatnonzero(mask)
    start = max(0, int(active_indices[0] * hop) - margin)
    end = min(len(value), int(active_indices[-1] * hop + window) + margin)
    return start, end


def crop_active_audio(
    audio: np.ndarray, sample_rate: int, **kwargs: float
) -> tuple[np.ndarray, tuple[int, int]]:
    value = _as_audio(audio)
    region = detect_active_region(value, sample_rate, **kwargs)
    return value[region[0] : region[1]], region


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    bins: int = LOG_MEL_BINS,
    *,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    if sample_rate <= 0 or n_fft <= 0 or bins <= 0:
        raise ValueError("sample_rate, n_fft, and bins must be positive")
    maximum = (
        sample_rate / 2.0 if f_max is None else min(float(f_max), sample_rate / 2.0)
    )
    if not 0 <= f_min < maximum:
        raise ValueError("mel frequency limits are invalid")
    frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    points = _mel_to_hz(
        np.linspace(_hz_to_mel(float(f_min)), _hz_to_mel(maximum), bins + 2)
    )
    bank = np.zeros((bins, len(frequencies)), dtype=np.float64)
    for index in range(bins):
        left, center, right = points[index : index + 3]
        lower = (frequencies - left) / max(center - left, 1e-12)
        upper = (right - frequencies) / max(right - center, 1e-12)
        bank[index] = np.maximum(0.0, np.minimum(lower, upper))
    # Match the cache-v2 acoustic teacher's Slaney-style area normalization.
    widths = np.maximum(points[2:] - points[:-2], 1e-12)
    return bank * (2.0 / widths[:, None])


def log_mel(
    audio: np.ndarray,
    sample_rate: int,
    bins: int = LOG_MEL_BINS,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    top_db: float = 80.0,
    normalize_rms: bool = True,
) -> np.ndarray:
    """Return a numerical ``[mel,time]`` power map clipped to ``[-top_db,0]``."""

    if top_db <= 0:
        raise ValueError("top_db must be positive")
    value = _unit_rms(audio) if normalize_rms else _as_audio(audio)
    window = max(1, round(sample_rate * window_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    n_fft = 1 << max(7, int(math.ceil(math.log2(window))))
    if len(value) == 0 or float(np.max(np.abs(value), initial=0.0)) <= 1e-10:
        frame_count = max(1, int(math.ceil(max(len(value), 1) / hop)))
        return np.full((bins, frame_count), -float(top_db), dtype=np.float32)
    frames = _frame_signal(value, window, hop) * np.hanning(window)[None, :]
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = np.square(np.abs(spectrum)).T / max(
        float(np.square(np.hanning(window)).sum()), 1e-12
    )
    mel_power = mel_filterbank(sample_rate, n_fft, bins) @ power
    db = 10.0 * np.log10(np.maximum(mel_power, 1e-12))
    db -= float(np.max(db))
    return np.clip(db, -float(top_db), 0.0).astype(np.float32)


def pad_or_trim_log_mel(
    energy: np.ndarray,
    frames: int = 400,
    *,
    floor_db: float = LOG_MEL_FLOOR_DB,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(energy, dtype=np.float32)
    if value.ndim != 2 or frames <= 0:
        raise ValueError("energy must be [mel,time] and frames must be positive")
    output = np.full((value.shape[0], frames), float(floor_db), dtype=np.float32)
    valid = min(value.shape[1], frames)
    output[:, :valid] = value[:, :valid]
    mask = np.zeros(frames, dtype=bool)
    mask[:valid] = True
    return output, mask


def crop_active_energy(
    energy: np.ndarray,
    activity: np.ndarray | None = None,
    *,
    foreground_floor_db: float = -60.0,
) -> np.ndarray:
    value = np.asarray(energy, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("energy must be [frequency,time]")
    if value.shape[1] == 0:
        return value.copy()
    if activity is None:
        active = np.max(value, axis=0) > float(foreground_floor_db)
    else:
        active = np.asarray(activity, dtype=bool).reshape(-1)
        if active.shape != (value.shape[1],):
            raise ValueError("activity must match the energy time axis")
    if not active.any():
        return value[:, :0].copy()
    indices = np.flatnonzero(active)
    return value[:, indices[0] : indices[-1] + 1].copy()


# Concise plan-language alias.
crop_active_region = crop_active_energy


def time_normalize_energy(
    energy: np.ndarray,
    frames: int = DEFAULT_MORPHOLOGY_FRAMES,
    *,
    floor_db: float = LOG_MEL_FLOOR_DB,
) -> np.ndarray:
    """Interpolate only the time axis; the frequency axis is never resized."""

    value = np.asarray(energy, dtype=np.float64)
    if value.ndim != 2 or frames <= 0:
        raise ValueError("energy must be [frequency,time] and frames must be positive")
    if value.shape[1] == 0:
        return np.full((value.shape[0], frames), float(floor_db), dtype=np.float64)
    if value.shape[1] == 1:
        return np.repeat(value, frames, axis=1)
    source = np.linspace(0.0, 1.0, value.shape[1])
    target = np.linspace(0.0, 1.0, frames)
    return np.vstack([np.interp(target, source, row) for row in value])


time_axis_normalize = time_normalize_energy


def _db_to_unit(value: np.ndarray, floor_db: float) -> np.ndarray:
    return np.clip(
        (np.asarray(value, dtype=np.float64) - float(floor_db)) / -float(floor_db),
        0.0,
        1.0,
    )


def _temporal_pool(value: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or value.shape[1] < factor:
        return value
    usable = value.shape[1] - value.shape[1] % factor
    if usable == 0:
        return value
    return value[:, :usable].reshape(value.shape[0], -1, factor).mean(axis=2)


def _weighted_ssim(first: np.ndarray, second: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight))
    if denominator <= 1e-12:
        return 0.0
    mean_first = float(np.sum(weight * first) / denominator)
    mean_second = float(np.sum(weight * second) / denominator)
    centered_first = first - mean_first
    centered_second = second - mean_second
    variance_first = float(np.sum(weight * np.square(centered_first)) / denominator)
    variance_second = float(np.sum(weight * np.square(centered_second)) / denominator)
    covariance = float(np.sum(weight * centered_first * centered_second) / denominator)
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_first * mean_second + c1) * (2 * covariance + c2)) / (
        (mean_first**2 + mean_second**2 + c1) * (variance_first + variance_second + c2)
    )
    return float(np.clip(score, -1.0, 1.0))


def foreground_weighted_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    normalize_frames: int = DEFAULT_MORPHOLOGY_FRAMES,
    floor_db: float = LOG_MEL_FLOOR_DB,
    temporal_scales: tuple[int, ...] = (1, 2, 4),
) -> float:
    """Dependency-free multi-scale SSIM-like score weighted by reference energy."""

    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("energy matrices must share their frequency dimension")
    first = _db_to_unit(
        time_normalize_energy(first, normalize_frames, floor_db=floor_db), floor_db
    )
    second = _db_to_unit(
        time_normalize_energy(second, normalize_frames, floor_db=floor_db), floor_db
    )
    if float(first.sum()) <= 1e-10:
        return 0.0
    scores = []
    for factor in temporal_scales:
        pooled_first = _temporal_pool(first, factor)
        pooled_second = _temporal_pool(second, factor)
        # Reference-derived weights prevent a silent/generic prediction from
        # looking good merely because most time-frequency pixels are background.
        weight = np.maximum(pooled_first, 1e-4 * (pooled_first > 0))
        scores.append(_weighted_ssim(pooled_first, pooled_second, weight))
    return float(np.mean(scores))


def soft_iou(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    normalize_frames: int = DEFAULT_MORPHOLOGY_FRAMES,
    floor_db: float = LOG_MEL_FLOOR_DB,
) -> float:
    first = _db_to_unit(
        time_normalize_energy(reference, normalize_frames, floor_db=floor_db), floor_db
    )
    second = _db_to_unit(
        time_normalize_energy(candidate, normalize_frames, floor_db=floor_db), floor_db
    )
    union = float(np.maximum(first, second).sum())
    return float(np.minimum(first, second).sum() / union) if union > 1e-12 else 0.0


def _soft_dtw_cost(cost: np.ndarray, gamma: float, band_ratio: float | None) -> float:
    rows, columns = cost.shape
    table = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    table[0, 0] = 0.0
    resolution = max(1.0 / max(rows, 1), 1.0 / max(columns, 1))
    for row in range(1, rows + 1):
        row_position = (row - 1) / max(rows - 1, 1)
        for column in range(1, columns + 1):
            if band_ratio is not None:
                column_position = (column - 1) / max(columns - 1, 1)
                if abs(row_position - column_position) > float(band_ratio) + resolution:
                    continue
            previous = np.asarray(
                (
                    table[row - 1, column],
                    table[row, column - 1],
                    table[row - 1, column - 1],
                )
            )
            minimum = float(np.min(previous))
            if not np.isfinite(minimum):
                continue
            if gamma == 0:
                soft_minimum = minimum
            else:
                soft_minimum = minimum - float(gamma) * math.log(
                    float(np.exp(-(previous - minimum) / float(gamma)).sum())
                )
            table[row, column] = cost[row - 1, column - 1] + soft_minimum
    if not np.isfinite(table[-1, -1]):
        raise ValueError(
            "Sakoe-Chiba band is too narrow for the supplied sequence lengths"
        )
    return float(table[-1, -1])


def soft_dtw_divergence(
    first: np.ndarray,
    second: np.ndarray,
    *,
    gamma: float = 0.05,
    band_ratio: float | None = 0.25,
) -> float:
    """Self-cost-corrected soft-DTW divergence for scalar or vector frames."""

    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if band_ratio is not None and not 0 <= band_ratio <= 1:
        raise ValueError("band_ratio must be between zero and one")
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or len(x) == 0
        or len(y) == 0
    ):
        raise ValueError("soft-DTW inputs must be non-empty [T,D] arrays with common D")
    cost_xy = np.mean((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    cost_xx = np.mean((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    cost_yy = np.mean((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    value = _soft_dtw_cost(cost_xy, gamma, band_ratio)
    value -= 0.5 * (
        _soft_dtw_cost(cost_xx, gamma, band_ratio)
        + _soft_dtw_cost(cost_yy, gamma, band_ratio)
    )
    return max(0.0, float(value) / float(len(x) + len(y)))


def soft_dtw_distance(
    first: np.ndarray, second: np.ndarray, gamma: float = 0.05
) -> float:
    """Compatibility name; unlike v0722 this is a nonnegative divergence."""

    return soft_dtw_divergence(first, second, gamma=gamma)


def energy_structure_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    hop_seconds: float = DEFAULT_HOP_MS / 1000.0,
    normalize_frames: int = DEFAULT_MORPHOLOGY_FRAMES,
    foreground_floor_db: float = -60.0,
) -> dict[str, float]:
    """Compare active log-mel morphology while retaining native timing errors."""

    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError(
            "energy matrices must be [frequency,time] with common frequency bins"
        )
    first_activity = np.max(first, axis=0) > float(foreground_floor_db)
    second_activity = np.max(second, axis=0) > float(foreground_floor_db)
    first_active = crop_active_energy(first, first_activity)
    second_active = crop_active_energy(second, second_activity)
    first_duration = first_active.shape[1] * float(hop_seconds)
    second_duration = second_active.shape[1] * float(hop_seconds)
    stretch = second_duration / first_duration if first_duration > 0 else float("nan")

    frames = max(first_active.shape[1], second_active.shape[1], 1)
    padded_first = np.full((first.shape[0], frames), LOG_MEL_FLOOR_DB, dtype=np.float64)
    padded_second = np.full(
        (second.shape[0], frames), LOG_MEL_FLOOR_DB, dtype=np.float64
    )
    padded_first[:, : first_active.shape[1]] = first_active
    padded_second[:, : second_active.shape[1]] = second_active
    normalized_first = time_normalize_energy(first_active, normalize_frames)
    normalized_second = time_normalize_energy(second_active, normalize_frames)
    unit_first = _db_to_unit(normalized_first, LOG_MEL_FLOOR_DB).T
    unit_second = _db_to_unit(normalized_second, LOG_MEL_FLOOR_DB).T

    activity_length = max(len(first_activity), len(second_activity), 1)
    activity_first = np.zeros(activity_length, dtype=bool)
    activity_second = np.zeros(activity_length, dtype=bool)
    activity_first[: len(first_activity)] = first_activity
    activity_second[: len(second_activity)] = second_activity
    activity_intersection = int(np.logical_and(activity_first, activity_second).sum())
    activity_dice = (
        2.0
        * activity_intersection
        / max(int(activity_first.sum() + activity_second.sum()), 1)
    )
    morphology_ssim = foreground_weighted_ssim(
        first_active, second_active, normalize_frames=normalize_frames
    )
    mel_soft_dtw = soft_dtw_divergence(
        unit_first, unit_second, gamma=0.05, band_ratio=0.25
    )
    return {
        "native_log_mel_mae_db": float(np.mean(np.abs(padded_first - padded_second))),
        "time_normalized_log_mel_mae_db": float(
            np.mean(np.abs(normalized_first - normalized_second))
        ),
        "morphology_ssim": morphology_ssim,
        "foreground_weighted_ssim": morphology_ssim,
        "time_normalized_ssim": morphology_ssim,
        "soft_iou": soft_iou(
            first_active, second_active, normalize_frames=normalize_frames
        ),
        "energy_soft_dtw_divergence": mel_soft_dtw,
        "mel_soft_dtw_divergence": mel_soft_dtw,
        "soft_dtw_divergence": mel_soft_dtw,
        "activity_dice": float(activity_dice),
        "reference_active_duration_seconds": float(first_duration),
        "candidate_active_duration_seconds": float(second_duration),
        "active_duration_error_seconds": float(abs(first_duration - second_duration)),
        "stretch_factor": float(stretch),
    }


def best_lag_envelope_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    max_lag_ms: float = 250.0,
) -> tuple[float, float]:
    first = rms_envelope(reference, sample_rate)
    second = rms_envelope(candidate, sample_rate)
    max_steps = round(max_lag_ms / DEFAULT_HOP_MS)
    scores: list[tuple[float, int]] = []
    for lag in range(-max_steps, max_steps + 1):
        if lag < 0:
            x, y = first[-lag:], second[:lag]
        elif lag > 0:
            x, y = first[:-lag], second[lag:]
        else:
            x, y = first, second
        scores.append((_correlation(x, y), lag))
    score, lag = max(scores, key=lambda item: item[0])
    return float(score), float(lag * DEFAULT_HOP_MS)


def modulation_correlation(
    reference: np.ndarray, candidate: np.ndarray, sample_rate: int
) -> float:
    first = np.log1p(np.abs(np.fft.rfft(rms_envelope(reference, sample_rate))))
    second = np.log1p(np.abs(np.fft.rfft(rms_envelope(candidate, sample_rate))))
    return _correlation(first, second)


def log_mel_mae(
    reference: np.ndarray, candidate: np.ndarray, sample_rate: int
) -> float:
    return energy_structure_metrics(
        log_mel(reference, sample_rate), log_mel(candidate, sample_rate)
    )["native_log_mel_mae_db"]


def multi_resolution_stft_distance(
    reference: np.ndarray, candidate: np.ndarray, sample_rate: int
) -> float:
    first, second = _unit_rms(reference), _unit_rms(candidate)
    size = max(len(first), len(second), 1)
    first = np.pad(first, (0, size - len(first)))
    second = np.pad(second, (0, size - len(second)))
    values = []
    for window_ms in (20.0, 50.0, 100.0):
        window = max(16, round(sample_rate * window_ms / 1000.0))
        _, _, x = stft(
            first, fs=sample_rate, nperseg=window, noverlap=window // 2, boundary=None
        )
        _, _, y = stft(
            second, fs=sample_rate, nperseg=window, noverlap=window // 2, boundary=None
        )
        magnitude_x, magnitude_y = np.abs(x), np.abs(y)
        values.append(
            float(np.linalg.norm(magnitude_x - magnitude_y))
            / max(float(np.linalg.norm(magnitude_x)), 1e-8)
        )
    return float(np.mean(values))


def estimate_log_f0(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = 40.0,
    hop_ms: float = DEFAULT_HOP_MS,
    f_min: float = 60.0,
    f_max: float = 500.0,
    voicing_threshold: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Small dependency-free autocorrelation F0 estimator for evaluation only."""

    window = max(4, round(sample_rate * window_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    frames = _frame_signal(_unit_rms(audio), window, hop)
    minimum_lag = max(1, int(sample_rate / f_max))
    maximum_lag = min(window - 1, int(sample_rate / f_min))
    log_f0 = np.zeros(len(frames), dtype=np.float64)
    voiced = np.zeros(len(frames), dtype=bool)
    taper = np.hanning(window)
    for index, frame in enumerate(frames):
        centered = (frame - float(np.mean(frame))) * taper
        energy = float(np.dot(centered, centered))
        if energy <= 1e-8:
            continue
        autocorrelation = np.correlate(centered, centered, mode="full")[window - 1 :]
        segment = autocorrelation[minimum_lag : maximum_lag + 1]
        lag_offset = int(np.argmax(segment))
        lag = minimum_lag + lag_offset
        strength = float(segment[lag_offset] / max(autocorrelation[0], 1e-12))
        if strength >= voicing_threshold:
            voiced[index] = True
            log_f0[index] = math.log(sample_rate / lag)
    return log_f0, voiced


def _pad_waveforms(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    size = max(len(first), len(second), 1)
    return np.pad(first, (0, size - len(first))), np.pad(
        second, (0, size - len(second))
    )


def reconstruction_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    max_lag_ms: float = 250.0,
) -> dict[str, float]:
    """Waveform, timing, content-adjacent, and scale-robust morphology metrics."""

    raw_reference = _as_audio(reference)
    raw_candidate = _as_audio(candidate)
    padded_reference, padded_candidate = _pad_waveforms(raw_reference, raw_candidate)
    envelope = rms_envelope(raw_reference, sample_rate)
    candidate_envelope = rms_envelope(raw_candidate, sample_rate)
    lag_corr, lag = best_lag_envelope_correlation(
        raw_reference, raw_candidate, sample_rate, max_lag_ms
    )

    reference_zero = padded_reference - float(np.mean(padded_reference))
    candidate_zero = padded_candidate - float(np.mean(padded_candidate))
    projection = (
        float(
            np.dot(candidate_zero, reference_zero)
            / max(np.dot(reference_zero, reference_zero), 1e-12)
        )
        * reference_zero
    )
    residual = candidate_zero - projection
    si_sdr = 10.0 * math.log10(
        max(float(np.dot(projection, projection)), 1e-12)
        / max(float(np.dot(residual, residual)), 1e-12)
    )

    reference_energy = log_mel(raw_reference, sample_rate)
    candidate_energy = log_mel(raw_candidate, sample_rate)
    structure = energy_structure_metrics(reference_energy, candidate_energy)
    reference_f0, reference_voiced = estimate_log_f0(raw_reference, sample_rate)
    candidate_f0, candidate_voiced = estimate_log_f0(raw_candidate, sample_rate)
    f0_frames = DEFAULT_MORPHOLOGY_FRAMES
    reference_f0_normalized = np.interp(
        np.linspace(0, 1, f0_frames),
        np.linspace(0, 1, max(len(reference_f0), 1)),
        reference_f0 if len(reference_f0) else np.zeros(1),
    )
    candidate_f0_normalized = np.interp(
        np.linspace(0, 1, f0_frames),
        np.linspace(0, 1, max(len(candidate_f0), 1)),
        candidate_f0 if len(candidate_f0) else np.zeros(1),
    )
    reference_voiced_normalized = (
        np.interp(
            np.linspace(0, 1, f0_frames),
            np.linspace(0, 1, max(len(reference_voiced), 1)),
            reference_voiced.astype(float) if len(reference_voiced) else np.zeros(1),
        )
        >= 0.5
    )
    candidate_voiced_normalized = (
        np.interp(
            np.linspace(0, 1, f0_frames),
            np.linspace(0, 1, max(len(candidate_voiced), 1)),
            candidate_voiced.astype(float) if len(candidate_voiced) else np.zeros(1),
        )
        >= 0.5
    )
    joint_voiced = reference_voiced_normalized & candidate_voiced_normalized
    f0_log_mae = (
        float(
            np.mean(
                np.abs(
                    reference_f0_normalized[joint_voiced]
                    - candidate_f0_normalized[joint_voiced]
                )
            )
        )
        if joint_voiced.any()
        else float("nan")
    )
    voicing_intersection = int(joint_voiced.sum())
    voicing_f1 = (
        2.0
        * voicing_intersection
        / max(
            int(reference_voiced_normalized.sum() + candidate_voiced_normalized.sum()),
            1,
        )
    )

    envelope_first = envelope / max(float(np.max(envelope, initial=0.0)), 1e-8)
    envelope_second = candidate_envelope / max(
        float(np.max(candidate_envelope, initial=0.0)), 1e-8
    )
    output = {
        "waveform_correlation": _correlation(padded_reference, padded_candidate),
        "si_sdr_db": float(si_sdr),
        "envelope_correlation": _correlation(envelope, candidate_envelope),
        "lag_envelope_correlation": lag_corr,
        "envelope_best_lag_ms": lag,
        "soft_dtw_envelope_divergence": soft_dtw_divergence(
            envelope_first, envelope_second
        ),
        "soft_dtw_envelope_distance": soft_dtw_divergence(
            envelope_first, envelope_second
        ),
        "modulation_correlation": modulation_correlation(
            raw_reference, raw_candidate, sample_rate
        ),
        "log_mel_mae_db": structure["native_log_mel_mae_db"],
        "multi_resolution_stft_distance": multi_resolution_stft_distance(
            raw_reference, raw_candidate, sample_rate
        ),
        "raw_reference_duration_seconds": len(raw_reference) / float(sample_rate),
        "raw_candidate_duration_seconds": len(raw_candidate) / float(sample_rate),
        "raw_duration_error_seconds": abs(len(raw_reference) - len(raw_candidate))
        / float(sample_rate),
        "raw_rms_error": abs(
            float(np.sqrt(np.mean(np.square(raw_reference)) + 1e-12))
            - float(np.sqrt(np.mean(np.square(raw_candidate)) + 1e-12))
        ),
        "f0_log_mae": f0_log_mae,
        "voicing_f1": float(voicing_f1),
    }
    output.update(structure)
    output["energy_morphology_ssim"] = structure["morphology_ssim"]
    output["energy_soft_iou"] = structure["soft_iou"]
    output["energy_stretch_factor"] = structure["stretch_factor"]
    return output


def summarize(
    records: Iterable[dict[str, float | None]],
) -> dict[str, dict[str, float]]:
    rows = list(records)
    keys = sorted({key for row in rows for key in row})
    output: dict[str, dict[str, float]] = {}
    for key in keys:
        # Undefined per-trial metrics (for example F0 error when neither signal
        # has a voiced frame) are serialized as JSON null.  They are missing
        # observations, not zeros, and must be excluded from the aggregate.
        values = np.asarray(
            [
                float(value)
                for row in rows
                if key in row
                for value in (row[key],)
                if isinstance(value, Real) and np.isfinite(float(value))
            ],
            dtype=np.float64,
        )
        output[key] = {
            "mean": float(np.mean(values)) if len(values) else float("nan"),
            "median": float(np.median(values)) if len(values) else float("nan"),
            "p05": float(np.percentile(values, 5)) if len(values) else float("nan"),
            "min": float(np.min(values)) if len(values) else float("nan"),
        }
    return output


__all__ = [
    "DEFAULT_MORPHOLOGY_FRAMES",
    "LOG_MEL_BINS",
    "LOG_MEL_CEILING_DB",
    "LOG_MEL_FLOOR_DB",
    "active_frame_mask",
    "best_lag_envelope_correlation",
    "crop_active_audio",
    "crop_active_energy",
    "crop_active_region",
    "detect_active_region",
    "energy_structure_metrics",
    "estimate_log_f0",
    "foreground_weighted_ssim",
    "log_mel",
    "log_mel_mae",
    "mel_filterbank",
    "multi_resolution_stft_distance",
    "pad_or_trim_log_mel",
    "reconstruction_metrics",
    "rms_envelope",
    "soft_dtw_distance",
    "soft_dtw_divergence",
    "soft_iou",
    "summarize",
    "time_axis_normalize",
    "time_normalize_energy",
]
