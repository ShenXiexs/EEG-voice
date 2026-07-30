from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly


class HubertMetric:
    """Frozen HuBERT embedding extractor used only for V0/V1/V2 evaluation."""

    def __init__(self, root: Path, *, layer: int, device: torch.device):
        try:
            from transformers import HubertModel
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("transformers HubertModel is required") from error
        if not root.is_dir():
            raise FileNotFoundError(f"HuBERT model is not cached at {root}")
        self.model = HubertModel.from_pretrained(str(root), local_files_only=True, output_hidden_states=True).to(device).eval()
        self.layer = int(layer)
        self.device = device

    @torch.no_grad()
    def encode(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        value = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != 16_000:
            divisor = np.gcd(sample_rate, 16_000)
            value = resample_poly(value, 16_000 // divisor, sample_rate // divisor).astype(np.float32)
        if len(value) < 400:
            value = np.pad(value, (0, 400 - len(value)))
        hidden = self.model(torch.from_numpy(value).to(self.device).unsqueeze(0), output_hidden_states=True).hidden_states[self.layer]
        return hidden.squeeze(0).detach().cpu().numpy().astype(np.float32)


def dtw_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Mean cosine along a deterministic full DTW path, not a frame F1."""
    a = np.asarray(left, dtype=np.float32); b = np.asarray(right, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.0e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1.0e-8)
    similarity = np.clip(a @ b.T, -1.0, 1.0)
    cost = 1.0 - similarity
    rows, cols = cost.shape
    dynamic = np.full((rows + 1, cols + 1), np.inf, dtype=np.float64)
    dynamic[0, 0] = 0.0
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            dynamic[row, col] = float(cost[row - 1, col - 1]) + min(dynamic[row - 1, col], dynamic[row, col - 1], dynamic[row - 1, col - 1])
    row, col, values = rows, cols, []
    while row and col:
        values.append(similarity[row - 1, col - 1])
        candidates = (dynamic[row - 1, col - 1], dynamic[row - 1, col], dynamic[row, col - 1])
        move = int(np.argmin(candidates))
        if move == 0:
            row, col = row - 1, col - 1
        elif move == 1:
            row -= 1
        else:
            col -= 1
    return float(np.mean(values)) if values else 0.0

