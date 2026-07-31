from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import correlate, resample_poly
from scipy.signal import istft, stft

from src.open_vocab_0724.audio_features import ActiveSpeechConfig, detect_active_speech


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "apply", "approved"}


def waveform_sha256(waveform: np.ndarray) -> str:
    pcm = np.asarray(np.clip(waveform, -1.0, 1.0) * 32767.0, dtype="<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def resample_waveform(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    value = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if int(source_rate) == int(target_rate):
        return value.copy()
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(value, int(target_rate) // divisor, int(source_rate) // divisor).astype(np.float32)


def rms_envelope(waveform: np.ndarray, sample_rate: int, *, hop_ms: float = 10.0) -> np.ndarray:
    value = np.asarray(waveform, dtype=np.float32).reshape(-1)
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    frame = max(hop, int(round(sample_rate * 25.0 / 1000.0)))
    count = max(1, int(np.ceil(len(value) / hop)))
    starts = np.arange(count) * hop
    output = np.empty(count, dtype=np.float32)
    for index, start in enumerate(starts.tolist()):
        block = value[start : start + frame]
        output[index] = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)) + 1.0e-10))
    return output


def envelope_lag_ms(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    left = rms_envelope(reference, sample_rate)
    right = rms_envelope(candidate, sample_rate)
    left = (left - left.mean()) / max(float(left.std()), 1.0e-8)
    right = (right - right.mean()) / max(float(right.std()), 1.0e-8)
    score = correlate(right, left, mode="full", method="fft")
    lag_frames = int(np.argmax(score) - (len(left) - 1))
    return float(lag_frames * 10.0)


def vad_boundary_seconds(waveform: np.ndarray, sample_rate: int) -> tuple[float, float]:
    cfg = ActiveSpeechConfig(sample_rate=int(sample_rate))
    bounds = detect_active_speech(np.asarray(waveform, dtype=np.float32), cfg)
    return bounds.speech_start_sample / sample_rate, bounds.speech_end_sample / sample_rate


class DeepFilterNetEnhancer:
    """Isolated file-level DeepFilterNet3 wrapper with official delay padding.

    A fresh state is initialized per trial.  This is slower but prevents the
    adaptive normalization/state of one participant's recording from leaking
    into the next trial.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model_identity: dict[str, Any] = {}
        self.processing_rate = int(cfg["denoise"]["processing_sample_rate"])
        self.backend = str(cfg["denoise"].get("backend", "DeterministicSpectralGateV1"))
        if self.backend.lower() == "deepfilternet3" and self.processing_rate != 48_000:
            raise ValueError("official DeepFilterNet models require a 48 kHz processing rate")

    def _deterministic_spectral_gate(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Conservative, non-pretrained denoiser for explicitly selected WAVs."""
        value = np.asarray(waveform, dtype=np.float32).reshape(-1)
        nperseg = min(512, max(64, 2 ** int(np.floor(np.log2(max(len(value) // 8, 64))))))
        noverlap = int(0.75 * nperseg)
        _, _, spectrum = stft(value, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, boundary="zeros")
        magnitude = np.abs(spectrum)
        frame_energy = np.mean(magnitude, axis=0)
        cutoff = float(np.quantile(frame_energy, 0.20))
        noise_frames = magnitude[:, frame_energy <= cutoff]
        noise = np.median(noise_frames, axis=1, keepdims=True) if noise_frames.size else np.median(magnitude, axis=1, keepdims=True)
        residual = np.maximum(magnitude - 1.25 * noise, 0.0)
        mask = np.clip(residual / np.maximum(magnitude, 1.0e-8), 0.15, 1.0)
        _, enhanced = istft(spectrum * mask, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, input_onesided=True)
        enhanced = np.asarray(enhanced, dtype=np.float32)
        if len(enhanced) < len(value):
            enhanced = np.pad(enhanced, (0, len(value) - len(enhanced)))
        self.model_identity = {
            "requested_model": self.backend,
            "pretrained": False,
            "algorithm": "STFT soft spectral subtraction, bottom-20% noise estimate",
        }
        return enhanced[: len(value)]

    def enhance(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.backend.lower() != "deepfilternet3":
            value = np.asarray(waveform, dtype=np.float32).reshape(-1)
            return self._deterministic_spectral_gate(value - float(value.mean()), int(sample_rate))
        try:
            from df import enhance, init_df
        except ImportError as error:
            raise RuntimeError(
                "selected denoising requires deepfilternet; run ./bootstrap_open_vocab_v3.sh"
            ) from error
        value = np.asarray(waveform, dtype=np.float32).reshape(-1)
        value = value - float(value.mean())
        at_48k = resample_waveform(value, int(sample_rate), self.processing_rate)
        model_name = self.backend
        model_base_dir = None if model_name.lower() == "deepfilternet3" else model_name
        initialized = init_df(
            model_base_dir=model_base_dir,
            post_filter=bool(self.cfg["denoise"].get("post_filter", False)),
            log_level="ERROR",
            log_file=None,
        )
        model, state = initialized[0], initialized[1]
        if not self.model_identity:
            digest = hashlib.sha256()
            for name, tensor_value in sorted(model.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(tensor_value.detach().cpu().contiguous().numpy().tobytes())
            self.model_identity = {
                "requested_model": model_name,
                "suffix": str(initialized[2]) if len(initialized) > 2 else None,
                "epoch": int(initialized[3]) if len(initialized) > 3 else None,
                "state_dict_sha256": digest.hexdigest(),
            }
        tensor = torch.from_numpy(at_48k).unsqueeze(0)
        output = enhance(
            model,
            state,
            tensor,
            pad=bool(self.cfg["denoise"].get("compensate_delay", True)),
            atten_lim_db=float(self.cfg["denoise"].get("attenuation_limit_db", 18.0)),
        )
        result = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        result = resample_waveform(result, self.processing_rate, int(sample_rate))
        if len(result) < len(value):
            result = np.pad(result, (0, len(value) - len(result)))
        return np.asarray(result[: len(value)], dtype=np.float32)
