from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


class SpeechT5PowerDbHiFiGan:
    """Frozen SpeechT5 HiFi-GAN with the v2 power-dB conversion (/10)."""

    def __init__(self, root: Path, *, device: torch.device):
        if not root.is_dir():
            raise FileNotFoundError(f"SpeechT5 HiFi-GAN is not cached at {root}")
        try:
            from transformers import SpeechT5HifiGan
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("transformers SpeechT5HifiGan support is required") from error
        self.root = root
        self.device = device
        self.model = SpeechT5HifiGan.from_pretrained(str(root), local_files_only=True).to(device).eval()

    @torch.no_grad()
    def synthesize(self, mel_power_db: torch.Tensor) -> torch.Tensor:
        if mel_power_db.ndim != 3 or mel_power_db.shape[1] != 80:
            raise ValueError("SpeechT5 backend expects power-dB mel [B,80,T]")
        waveform = self.model((mel_power_db / 10.0).transpose(1, 2).to(self.device))
        return waveform


def pcm16(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    source = waveform.detach().cpu().numpy() if torch.is_tensor(waveform) else np.asarray(waveform)
    return np.clip(source, -1.0, 1.0).astype(np.float32)


def model_manifest(root: Path) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest})
    return {"backend": "microsoft/speecht5_hifigan", "local_path": str(root), "files": files}

