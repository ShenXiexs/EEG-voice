from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


class SpeechT5HiFiGan:
    """Pinned SpeechT5 HiFi-GAN backend.

    The renderer works in dB log-mel space. SpeechT5 expects log-mel values with
    time first, so the adapter converts dB to log10-amplitude (dB / 20). It is a
    fixed interface, never a trainable EEG component.
    """

    def __init__(self, model_path: Path, *, device: torch.device):
        if not model_path.is_dir():
            raise FileNotFoundError(f"SpeechT5 HiFi-GAN is not cached at {model_path}; run download_speecht5_hifigan.py first")
        try:
            from transformers import SpeechT5HifiGan
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("transformers with SpeechT5HifiGan support is required") from error
        self.model_path = model_path
        self.device = device
        self.model = SpeechT5HifiGan.from_pretrained(str(model_path), local_files_only=True).to(device).eval()

    @torch.no_grad()
    def synthesize(self, mel_db: torch.Tensor) -> torch.Tensor:
        if mel_db.ndim != 3 or mel_db.shape[1] != 80:
            raise ValueError("SpeechT5 backend expects mel_db[B,80,T]")
        spectrogram = (mel_db / 20.0).transpose(1, 2).to(self.device)
        waveform = self.model(spectrogram)
        return waveform.squeeze(0) if waveform.ndim == 2 and waveform.shape[0] == 1 else waveform


def model_manifest(path: Path) -> dict[str, object]:
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        files.append({"relative_path": str(item.relative_to(path)), "bytes": item.stat().st_size, "sha256": digest})
    return {"backend": "microsoft/speecht5_hifigan", "local_path": str(path), "files": files}


def pcm16(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    source = waveform.detach().cpu().numpy() if torch.is_tensor(waveform) else np.asarray(waveform)
    return np.clip(source, -1.0, 1.0).astype(np.float32)
