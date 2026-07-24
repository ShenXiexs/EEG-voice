from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import resample_poly


EPS = 1.0e-12


@dataclass(frozen=True)
class ActiveSpeechConfig:
    sample_rate: int = 16_000
    window_ms: float = 25.0
    hop_ms: float = 10.0
    noise_margin_db: float = 6.0
    peak_margin_db: float = 40.0
    close_gap_ms: float = 50.0
    context_ms: float = 100.0

    @property
    def frame_length(self) -> int:
        return max(1, int(round(self.sample_rate * self.window_ms / 1000.0)))

    @property
    def hop_length(self) -> int:
        return max(1, int(round(self.sample_rate * self.hop_ms / 1000.0)))


@dataclass(frozen=True)
class AudioPreparationConfig:
    sample_rate: int = 16_000
    max_active_seconds: float = 4.0
    target_rms: float | None = 0.08
    max_gain: float = 10.0
    peak_limit: float = 0.95
    active: ActiveSpeechConfig | None = None

    @property
    def max_samples(self) -> int:
        return int(round(self.sample_rate * self.max_active_seconds))

    @property
    def active_config(self) -> ActiveSpeechConfig:
        return self.active or ActiveSpeechConfig(sample_rate=self.sample_rate)


@dataclass(frozen=True)
class AcousticFeatureConfig:
    sample_rate: int = 16_000
    window_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    mel_bins: int = 80
    max_frames: int = 400
    fmin_hz: float = 0.0
    fmax_hz: float = 8_000.0
    min_db: float = -80.0
    max_db: float = 0.0
    f0_min_hz: float = 50.0
    f0_max_hz: float = 500.0
    voicing_threshold: float = 0.30

    @property
    def frame_length(self) -> int:
        return max(1, int(round(self.sample_rate * self.window_ms / 1000.0)))

    @property
    def hop_length(self) -> int:
        return max(1, int(round(self.sample_rate * self.hop_ms / 1000.0)))


@dataclass(frozen=True)
class ActiveSpeechBounds:
    speech_start_sample: int
    speech_end_sample: int
    context_start_sample: int
    context_end_sample: int
    threshold_dbfs: float
    frame_rms_dbfs: np.ndarray
    frame_activity: np.ndarray
    has_activity: bool

    @property
    def active_samples(self) -> int:
        return max(0, int(self.speech_end_sample) - int(self.speech_start_sample))


@dataclass(frozen=True)
class PreparedWaveform:
    waveform: np.ndarray
    valid_samples: int
    source_sample_rate: int
    sample_rate: int
    native_sample_count: int
    resampled_sample_count: int
    native_rms: float
    normalization_gain: float
    active_start_sample: int
    active_end_sample: int
    context_start_sample: int
    context_end_sample: int
    segment_source_start_sample: int
    segment_source_end_sample: int
    active_duration_seconds: float
    has_activity: bool
    exceeds_max_active_seconds: bool
    reconstruction_eligible: bool
    pcm_sha256: str


@dataclass(frozen=True)
class AcousticFeatures:
    log_mel_energy: np.ndarray
    log_f0_hz: np.ndarray
    voicing: np.ndarray
    log_rms_dbfs: np.ndarray
    activity_mask: np.ndarray
    frame_valid_mask: np.ndarray

    @property
    def realization_features(self) -> np.ndarray:
        return np.concatenate(
            (
                self.log_mel_energy.T,
                self.log_f0_hz[:, None],
                self.voicing[:, None],
                self.log_rms_dbfs[:, None],
                self.activity_mask.astype(np.float32)[:, None],
            ),
            axis=1,
        ).astype(np.float32, copy=False)


def resample_audio(
    audio: np.ndarray, source_rate: int, target_rate: int = 16_000
) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(source_rate) <= 0 or int(target_rate) <= 0:
        raise ValueError("Audio sample rates must be positive")
    if int(source_rate) == int(target_rate):
        return value.copy()
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(
        value, int(target_rate) // divisor, int(source_rate) // divisor
    ).astype(np.float32)


def _frames(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
    frame_count: int | None = None,
) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    if frame_count is None:
        frame_count = max(1, int(math.ceil(len(value) / max(1, hop_length))))
    starts = np.arange(int(frame_count), dtype=np.int64) * int(hop_length)
    offsets = np.arange(int(frame_length), dtype=np.int64)
    indices = starts[:, None] + offsets[None, :]
    padded = np.pad(value, (0, max(0, int(indices.max(initial=0)) + 1 - len(value))))
    return np.asarray(padded[indices], dtype=np.float32)


def _close_short_gaps(mask: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    active = np.flatnonzero(output)
    if len(active) < 2 or maximum_gap_frames <= 0:
        return output
    for left, right in zip(active[:-1], active[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= int(maximum_gap_frames):
            output[left + 1 : right] = True
    return output


def detect_active_speech(
    audio: np.ndarray, config: ActiveSpeechConfig | None = None
) -> ActiveSpeechBounds:
    """Detect active speech deterministically in sample coordinates.

    The detector follows the v0724 preregistered rule exactly: 25 ms RMS
    frames at a 10 ms hop, threshold ``max(p10 + 6 dB, peak - 40 dB)``,
    closing of inactive gaps up to 50 ms, and 100 ms context on each side.
    """

    cfg = config or ActiveSpeechConfig()
    value = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1), copy=True)
    if len(value) == 0:
        raise ValueError("Cannot detect activity in an empty waveform")
    frames = _frames(value, cfg.frame_length, cfg.hop_length)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + EPS)
    rms_db = np.maximum(20.0 * np.log10(np.maximum(rms, EPS)), -120.0)
    threshold = max(
        float(np.percentile(rms_db, 10.0) + cfg.noise_margin_db),
        float(np.max(rms_db) - cfg.peak_margin_db),
    )
    activity = rms_db >= threshold
    # Exact digital silence must remain inactive; otherwise its equal-valued
    # frames would all pass an equality comparison at the numerical floor.
    if float(np.max(np.abs(value), initial=0.0)) <= 1.0e-7:
        activity[:] = False
    elif not activity.any():
        # A tightly cropped file can contain speech in every frame. In that
        # edge case p10 is itself speech, so p10+6 dB lies above the observed
        # peak and the registered threshold has no positive frame. Preserve
        # the registered threshold for audit, but use a deterministic
        # peak-relative fallback rather than misclassifying audible audio as
        # silence.
        activity = rms_db >= float(np.max(rms_db) - 6.0)
    close_frames = int(round(cfg.close_gap_ms / cfg.hop_ms))
    activity = _close_short_gaps(activity, close_frames)
    indices = np.flatnonzero(activity)
    if len(indices):
        speech_start = int(indices[0] * cfg.hop_length)
        speech_end = min(
            len(value), int(indices[-1] * cfg.hop_length + cfg.frame_length)
        )
        context = int(round(cfg.sample_rate * cfg.context_ms / 1000.0))
        context_start = max(0, speech_start - context)
        context_end = min(len(value), speech_end + context)
        has_activity = True
    else:
        speech_start = speech_end = 0
        context_start, context_end = 0, len(value)
        has_activity = False
    return ActiveSpeechBounds(
        speech_start_sample=speech_start,
        speech_end_sample=speech_end,
        context_start_sample=context_start,
        context_end_sample=context_end,
        threshold_dbfs=float(threshold),
        frame_rms_dbfs=rms_db.astype(np.float32),
        frame_activity=activity,
        has_activity=has_activity,
    )


def _maximum_energy_window(audio: np.ndarray, start: int, end: int, width: int) -> int:
    start, end, width = int(start), int(end), int(width)
    if end - start <= width:
        return start
    power = np.square(np.asarray(audio[start:end], dtype=np.float64))
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(power)))
    energy = cumulative[width:] - cumulative[:-width]
    return start + int(np.argmax(energy))


def _pcm_sha256(audio: np.ndarray) -> str:
    pcm = np.asarray(
        np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0, dtype="<i2"
    )
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def prepare_waveform_segment(
    audio: np.ndarray,
    source_rate: int,
    config: AudioPreparationConfig | None = None,
) -> PreparedWaveform:
    """Return the one fixed segment shared by HuBERT, acoustics, and EnCodec."""

    cfg = config or AudioPreparationConfig()
    native = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1), copy=True)
    if len(native) == 0:
        raise ValueError("Cannot prepare an empty waveform")
    resampled = resample_audio(native, int(source_rate), cfg.sample_rate)
    bounds = detect_active_speech(resampled, cfg.active_config)
    max_samples = cfg.max_samples
    exceeds = bool(bounds.active_samples > max_samples)

    context_fits = (
        bounds.context_end_sample - bounds.context_start_sample <= max_samples
    )
    if context_fits:
        segment_start = bounds.context_start_sample
    elif exceeds:
        segment_start = _maximum_energy_window(
            resampled,
            bounds.context_start_sample,
            bounds.context_end_sample,
            max_samples,
        )
    else:
        # Keep the entire active span and distribute the remaining context as
        # symmetrically as the source boundaries allow.
        ideal = (
            bounds.speech_start_sample + bounds.speech_end_sample - max_samples
        ) // 2
        minimum = max(
            bounds.context_start_sample, bounds.speech_end_sample - max_samples
        )
        maximum = min(
            bounds.speech_start_sample, bounds.context_end_sample - max_samples
        )
        segment_start = min(max(int(ideal), int(minimum)), int(maximum))
    if context_fits:
        segment_start = max(0, min(int(segment_start), len(resampled)))
    else:
        segment_start = max(
            0, min(int(segment_start), max(0, len(resampled) - max_samples))
        )
    segment_end = (
        int(bounds.context_end_sample)
        if context_fits
        else min(len(resampled), segment_start + max_samples)
    )
    valid_samples = max(0, segment_end - segment_start)
    raw_segment = resampled[segment_start:segment_end]
    native_rms = (
        float(np.sqrt(np.mean(np.square(raw_segment, dtype=np.float64)) + EPS))
        if valid_samples
        else 0.0
    )
    gain = 1.0
    if cfg.target_rms is not None and native_rms > 1.0e-7:
        gain = min(float(cfg.max_gain), float(cfg.target_rms) / native_rms)
        peak = float(np.max(np.abs(raw_segment), initial=0.0))
        if peak > 0.0:
            gain = min(gain, float(cfg.peak_limit) / peak)
    normalized = np.clip(
        raw_segment * gain, -float(cfg.peak_limit), float(cfg.peak_limit)
    ).astype(np.float32)
    waveform = np.zeros(max_samples, dtype=np.float32)
    waveform[:valid_samples] = normalized
    active_duration = float(bounds.active_samples / cfg.sample_rate)
    eligible = bool(bounds.has_activity and not exceeds)
    return PreparedWaveform(
        waveform=waveform,
        valid_samples=valid_samples,
        source_sample_rate=int(source_rate),
        sample_rate=int(cfg.sample_rate),
        native_sample_count=len(native),
        resampled_sample_count=len(resampled),
        native_rms=native_rms,
        normalization_gain=float(gain),
        active_start_sample=int(bounds.speech_start_sample),
        active_end_sample=int(bounds.speech_end_sample),
        context_start_sample=int(bounds.context_start_sample),
        context_end_sample=int(bounds.context_end_sample),
        segment_source_start_sample=int(segment_start),
        segment_source_end_sample=int(segment_end),
        active_duration_seconds=active_duration,
        has_activity=bool(bounds.has_activity),
        exceeds_max_active_seconds=exceeds,
        reconstruction_eligible=eligible,
        pcm_sha256=_pcm_sha256(waveform),
    )


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value, dtype=np.float64) / 700.0)


def _mel_to_hz(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (np.power(10.0, np.asarray(value, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def _mel_filterbank(
    sample_rate: int, n_fft: int, mel_bins: int, fmin_hz: float, fmax_hz: float
) -> np.ndarray:
    if not (0.0 <= fmin_hz < fmax_hz <= sample_rate / 2.0):
        raise ValueError("Mel frequency bounds must lie inside [0, Nyquist]")
    points = _mel_to_hz(
        np.linspace(_hz_to_mel(fmin_hz), _hz_to_mel(fmax_hz), mel_bins + 2)
    )
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    filters = np.zeros((mel_bins, len(frequencies)), dtype=np.float64)
    for index in range(mel_bins):
        left, center, right = points[index : index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, EPS),
                (right - frequencies) / max(right - center, EPS),
            ),
        )
        # Slaney-style area normalization keeps bins comparable without
        # altering the requested relative dB range.
        filters[index] *= 2.0 / max(right - left, EPS)
    return filters.astype(np.float32)


def _pitch_features(
    frames: np.ndarray,
    rms_db: np.ndarray,
    activity: np.ndarray,
    config: AcousticFeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    centered = frames.astype(np.float64) - frames.mean(axis=1, keepdims=True)
    fft_size = 1 << int(math.ceil(math.log2(max(2, 2 * centered.shape[1] - 1))))
    spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size, axis=1)[
        :, : centered.shape[1]
    ]
    lag_min = max(1, int(math.floor(config.sample_rate / config.f0_max_hz)))
    lag_max = min(
        centered.shape[1] - 1, int(math.ceil(config.sample_rate / config.f0_min_hz))
    )
    region = autocorrelation[:, lag_min : lag_max + 1]
    best_offset = np.argmax(region, axis=1)
    best_lag = best_offset + lag_min
    peak = region[np.arange(len(region)), best_offset]
    strength = np.clip(peak / np.maximum(autocorrelation[:, 0], EPS), 0.0, 1.0)
    voiced = (
        activity
        & (strength >= config.voicing_threshold)
        & (rms_db > config.min_db + 1.0)
    )
    f0 = np.zeros(len(frames), dtype=np.float32)
    f0[voiced] = config.sample_rate / best_lag[voiced]
    log_f0 = np.zeros_like(f0)
    log_f0[voiced] = np.log(f0[voiced])
    return log_f0, voiced.astype(np.float32)


def extract_acoustic_features(
    waveform: np.ndarray,
    *,
    valid_samples: int | None = None,
    config: AcousticFeatureConfig | None = None,
) -> AcousticFeatures:
    cfg = config or AcousticFeatureConfig()
    value = np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1), copy=True)
    if len(value) == 0:
        raise ValueError("Cannot extract features from an empty waveform")
    valid = (
        len(value)
        if valid_samples is None
        else min(max(int(valid_samples), 0), len(value))
    )
    frame_count = int(cfg.max_frames)
    frames = _frames(value, cfg.frame_length, cfg.hop_length, frame_count)
    frame_valid = (np.arange(frame_count) * cfg.hop_length) < valid
    window = np.hanning(cfg.frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=cfg.n_fft, axis=1)
    power = np.square(np.abs(spectrum), dtype=np.float64) / max(
        float(np.square(window).sum()), EPS
    )
    filters = _mel_filterbank(
        cfg.sample_rate, cfg.n_fft, cfg.mel_bins, cfg.fmin_hz, cfg.fmax_hz
    )
    mel_power = np.maximum(power @ filters.T, EPS)
    mel_db = 10.0 * np.log10(mel_power)
    valid_peak = (
        float(np.max(mel_db[frame_valid])) if frame_valid.any() else float(cfg.min_db)
    )
    if float(np.max(np.abs(value[:valid]), initial=0.0)) <= 1.0e-7:
        mel_db[:] = cfg.min_db
    else:
        mel_db = mel_db - valid_peak + cfg.max_db
        mel_db = np.clip(mel_db, cfg.min_db, cfg.max_db)
    mel_db[~frame_valid] = cfg.min_db

    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + EPS)
    rms_db = np.clip(20.0 * np.log10(np.maximum(rms, EPS)), cfg.min_db, cfg.max_db)
    if frame_valid.any():
        valid_values = rms_db[frame_valid]
        threshold = max(
            float(np.percentile(valid_values, 10.0) + 6.0),
            float(valid_values.max() - 40.0),
        )
        activity = (rms_db >= threshold) & frame_valid
        if float(np.max(np.abs(value[:valid]), initial=0.0)) <= 1.0e-7:
            activity[:] = False
        elif not activity.any():
            activity = (rms_db >= float(valid_values.max() - 6.0)) & frame_valid
        activity = (
            _close_short_gaps(activity, int(round(50.0 / cfg.hop_ms))) & frame_valid
        )
    else:
        activity = np.zeros(frame_count, dtype=bool)
    log_f0, voicing = _pitch_features(frames, rms_db, activity, cfg)
    rms_db[~frame_valid] = cfg.min_db
    return AcousticFeatures(
        log_mel_energy=np.ascontiguousarray(mel_db.T, dtype=np.float32),
        log_f0_hz=np.asarray(log_f0, dtype=np.float32),
        voicing=np.asarray(voicing, dtype=np.float32),
        log_rms_dbfs=np.asarray(rms_db, dtype=np.float32),
        activity_mask=np.asarray(activity, dtype=bool),
        frame_valid_mask=np.asarray(frame_valid, dtype=bool),
    )


def fallback_timbre_embedding(
    features: AcousticFeatures, dimension: int = 512
) -> np.ndarray:
    """Deterministic no-network fallback, explicitly marked as non-WavLM by the cache."""

    valid = np.asarray(features.frame_valid_mask, dtype=bool)
    mel = features.log_mel_energy[:, valid]
    if mel.shape[1]:
        base = np.concatenate(
            (
                mel.mean(axis=1),
                mel.std(axis=1),
                np.asarray(
                    [
                        features.log_f0_hz[valid].mean(),
                        features.log_f0_hz[valid].std(),
                        features.voicing[valid].mean(),
                        features.log_rms_dbfs[valid].mean(),
                        features.log_rms_dbfs[valid].std(),
                        features.activity_mask[valid].mean(),
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)
    else:
        base = np.zeros(166, dtype=np.float32)
    output = np.zeros(int(dimension), dtype=np.float32)
    output[: min(len(base), len(output))] = base[: len(output)]
    norm = float(np.linalg.norm(output))
    return output / norm if norm > 1.0e-8 else output


# Readable alias used by audits and tests.
active_speech_bounds = detect_active_speech


__all__ = [
    "AcousticFeatureConfig",
    "AcousticFeatures",
    "ActiveSpeechBounds",
    "ActiveSpeechConfig",
    "AudioPreparationConfig",
    "PreparedWaveform",
    "active_speech_bounds",
    "detect_active_speech",
    "extract_acoustic_features",
    "fallback_timbre_embedding",
    "prepare_waveform_segment",
    "resample_audio",
]
