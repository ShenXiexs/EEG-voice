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


def multi_resolution_stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    fft_sizes: Iterable[int],
    hop_sizes: Iterable[int],
) -> torch.Tensor:
    """Differentiable spectral loss suitable for generator-only adaptation."""
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


def envelope_loss(prediction: torch.Tensor, target: torch.Tensor, frames: int = 160) -> torch.Tensor:
    pred = F.adaptive_avg_pool1d(prediction.abs().unsqueeze(1), int(frames)).squeeze(1)
    truth = F.adaptive_avg_pool1d(target.abs().unsqueeze(1), int(frames)).squeeze(1)
    return F.l1_loss(pred, truth)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

