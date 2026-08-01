from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def selected_audio_indices(records, scope: str) -> list[int]:
    """Resolve the exact audio-domain adaptation corpus.

    ``fit`` is the only scope permitted for the scientific pipeline. ``all``
    exists for the explicitly transductive, audio-demonstration-only runner.
    """
    eligible = np.asarray(records.arrays["fit_eligible"], dtype=bool)
    if scope == "fit":
        selector = eligible & (np.asarray(records.roles).astype(str) == "fit")
    elif scope == "all":
        selector = eligible
    else:
        raise ValueError(f"unsupported audio adaptation scope: {scope!r}")
    indices = np.flatnonzero(selector).tolist()
    if not indices:
        raise ValueError(f"audio adaptation scope {scope!r} selected no eligible WAVs")
    return indices


def tensor_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_change(before: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, float | int]:
    changed = 0
    total = 0
    squared_delta = 0.0
    squared_base = 0.0
    current = dict(model.named_parameters())
    for name, initial in before.items():
        if name not in current:
            raise KeyError(f"adapted model lost parameter {name}")
        value = current[name].detach().cpu()
        delta = value - initial
        norm = float(torch.linalg.vector_norm(delta))
        changed += int(norm > 0.0)
        total += 1
        squared_delta += float(torch.sum(delta.double().square()))
        squared_base += float(torch.sum(initial.double().square()))
    return {
        "parameter_tensors": total,
        "changed_parameter_tensors": changed,
        "changed_parameter_fraction": float(changed / max(total, 1)),
        "relative_parameter_l2": float(np.sqrt(squared_delta) / max(np.sqrt(squared_base), 1.0e-12)),
    }


def module_state(modules: Iterable[tuple[str, torch.nn.Module]]) -> dict[str, torch.Tensor]:
    flat: dict[str, torch.Tensor] = {}
    for module_name, module in modules:
        for parameter_name, parameter in module.named_parameters():
            if parameter.requires_grad:
                flat[f"{module_name}.{parameter_name}"] = parameter.detach().cpu().clone()
    return flat


def module_parameter_change(
    before: dict[str, torch.Tensor], modules: Iterable[tuple[str, torch.nn.Module]]
) -> dict[str, float | int]:
    current: dict[str, torch.nn.Parameter] = {}
    for module_name, module in modules:
        for parameter_name, parameter in module.named_parameters():
            current[f"{module_name}.{parameter_name}"] = parameter
    changed = 0
    squared_delta = 0.0
    squared_base = 0.0
    for name, initial in before.items():
        value = current[name].detach().cpu()
        delta = value - initial
        changed += int(float(torch.linalg.vector_norm(delta)) > 0.0)
        squared_delta += float(torch.sum(delta.double().square()))
        squared_base += float(torch.sum(initial.double().square()))
    return {
        "parameter_tensors": len(before),
        "changed_parameter_tensors": changed,
        "changed_parameter_fraction": float(changed / max(len(before), 1)),
        "relative_parameter_l2": float(np.sqrt(squared_delta) / max(np.sqrt(squared_base), 1.0e-12)),
    }


def _wave_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalize waveform tensors to the 2-D shape accepted by torch.stft.

    Generator APIs are inconsistent here: HiFi-GAN returns ``[B, T]`` while
    Hugging Face EnCodec returns ``[B, C, T]``.  Treat channels as independent
    batch entries for mono/multichannel-safe reconstruction losses.
    """
    if value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim == 2:
        return value
    if value.ndim == 3:
        return value.reshape(-1, value.shape[-1])
    raise ValueError(f"{name} waveform must be [T], [B,T], or [B,C,T], got {tuple(value.shape)}")


def multi_resolution_stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    fft_sizes: Iterable[int],
    hop_sizes: Iterable[int],
) -> torch.Tensor:
    """Differentiable spectral loss suitable for generator-only adaptation."""
    prediction = _wave_batch(prediction, name="prediction")
    target = _wave_batch(target, name="target")
    if prediction.shape != target.shape:
        raise ValueError(
            f"STFT prediction/target shapes must match after channel flattening: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    values = []
    for fft_size, hop_size in zip(fft_sizes, hop_sizes):
        fft_size = int(fft_size)
        hop_size = int(hop_size)
        window = torch.hann_window(fft_size, device=prediction.device, dtype=prediction.dtype)
        pred = torch.stft(
            prediction, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
            window=window, return_complex=True,
        ).abs().clamp_min(1.0e-5)
        truth = torch.stft(
            target, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
            window=window, return_complex=True,
        ).abs().clamp_min(1.0e-5)
        spectral_convergence = torch.linalg.vector_norm(pred - truth) / torch.linalg.vector_norm(truth).clamp_min(1.0e-5)
        log_magnitude = F.l1_loss(torch.log(pred), torch.log(truth))
        values.append(spectral_convergence + log_magnitude)
    if not values:
        raise ValueError("at least one STFT resolution is required")
    return torch.stack(values).mean()


def _segment_mean_pool1d(value: torch.Tensor, frames: int) -> torch.Tensor:
    """Mean-pool ``[B,T]`` into arbitrary frame counts on CPU/CUDA/MPS.

    PyTorch's MPS adaptive-average-pool currently requires the input length to
    be divisible by the output length.  Cumulative sums plus integer segment
    boundaries implement the intended non-overlapping envelope averages for
    any waveform length while preserving autograd on the original device.
    """
    if value.ndim != 2:
        raise ValueError(f"segment mean pool expects [B,T], got {tuple(value.shape)}")
    frames = int(frames)
    length = int(value.shape[-1])
    if frames <= 0 or length < frames:
        raise ValueError(f"envelope frames must satisfy 0 < frames <= samples, got {frames} and {length}")
    edges = torch.div(
        torch.arange(frames + 1, device=value.device, dtype=torch.long) * length,
        frames,
        rounding_mode="floor",
    )
    integral = F.pad(torch.cumsum(value, dim=-1), (1, 0))
    totals = integral.index_select(-1, edges[1:]) - integral.index_select(-1, edges[:-1])
    widths = (edges[1:] - edges[:-1]).to(dtype=value.dtype).clamp_min(1)
    return totals / widths.unsqueeze(0)


def envelope_loss(prediction: torch.Tensor, target: torch.Tensor, frames: int = 160) -> torch.Tensor:
    prediction = _wave_batch(prediction, name="prediction")
    target = _wave_batch(target, name="target")
    if prediction.shape != target.shape:
        raise ValueError(
            f"envelope prediction/target shapes must match after channel flattening: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    pred = _segment_mean_pool1d(prediction.abs(), int(frames))
    truth = _segment_mean_pool1d(target.abs(), int(frames))
    return F.l1_loss(pred, truth)


def si_sdr_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SDR loss for [B,T] or [B,C,T] generator output."""
    estimate = _wave_batch(prediction, name="prediction")
    reference = _wave_batch(target, name="target")[..., :estimate.shape[-1]]
    estimate = estimate - estimate.mean(-1, keepdim=True)
    reference = reference - reference.mean(-1, keepdim=True)
    projection = (estimate * reference).sum(-1, keepdim=True) * reference / reference.square().sum(-1, keepdim=True).clamp_min(1.0e-8)
    residual = estimate - projection
    return -10.0 * torch.log10((projection.square().sum(-1) / residual.square().sum(-1).clamp_min(1.0e-8)).clamp_min(1.0e-8)).mean()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
