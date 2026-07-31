from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ADAPTER_FILE = "karaone_mel_adapter.pt"


class KaraOneMelAdapter(nn.Module):
    """Learned bridge from the v3 10-ms power-dB Mel to SpeechT5 Mel.

    SpeechT5 HiFi-GAN advances 256 waveform samples per input frame, whereas
    the KaraOne feature contract advances 160 samples.  The deterministic
    interpolation preserves duration and the residual convolutions learn the
    corpus-specific filterbank/value-range correction during fine-tuning.
    """

    def __init__(self, *, bins: int = 80, input_hop_samples: int = 160, output_hop_samples: int = 256):
        super().__init__()
        self.bins = int(bins)
        self.input_hop_samples = int(input_hop_samples)
        self.output_hop_samples = int(output_hop_samples)
        self.residual = nn.Sequential(
            nn.Conv1d(bins, bins, 5, padding=2), nn.GELU(),
            nn.Conv1d(bins, bins, 1),
        )
        # Start from the previously used power-dB / 10 interface.  Fine-tuning
        # must move these parameters; the adaptation gate verifies that it did.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, mel_power_db: torch.Tensor) -> torch.Tensor:
        if mel_power_db.ndim != 3 or mel_power_db.shape[1] != self.bins:
            raise ValueError(f"KaraOne Mel adapter expects [B,{self.bins},T]")
        frames = max(
            1,
            int(round(mel_power_db.shape[-1] * self.input_hop_samples / self.output_hop_samples)),
        )
        base = F.interpolate(mel_power_db / 10.0, size=frames, mode="linear", align_corners=False)
        return base + 0.25 * torch.tanh(self.residual(base))


class SpeechT5PowerDbHiFiGan:
    """KaraOne-adapted SpeechT5 HiFi-GAN used by every v3 synthesis path."""

    def __init__(self, root: Path, *, device: torch.device):
        if not root.is_dir():
            raise FileNotFoundError(f"SpeechT5 HiFi-GAN is not cached at {root}")
        try:
            from transformers import SpeechT5HifiGan
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("transformers SpeechT5HifiGan support is required") from error
        self.root = root
        self.device = device
        adapter_path = root / ADAPTER_FILE
        if not adapter_path.is_file():
            raise FileNotFoundError(
                f"KaraOne vocoder adaptation is missing: {adapter_path}. "
                "Run scripts/finetune_open_vocab_v3_audio_models.py first."
            )
        self.model = SpeechT5HifiGan.from_pretrained(str(root), local_files_only=True).to(device).eval()
        payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
        self.adapter = KaraOneMelAdapter(
            bins=int(payload["bins"]),
            input_hop_samples=int(payload["input_hop_samples"]),
            output_hop_samples=int(payload["output_hop_samples"]),
        ).to(device)
        self.adapter.load_state_dict(payload["state_dict"], strict=True)
        self.adapter.eval()

    @torch.no_grad()
    def synthesize(self, mel_power_db: torch.Tensor) -> torch.Tensor:
        if mel_power_db.ndim != 3 or mel_power_db.shape[1] != 80:
            raise ValueError("SpeechT5 backend expects power-dB mel [B,80,T]")
        speech_t5_mel = self.adapter(mel_power_db.to(self.device))
        waveform = self.model(speech_t5_mel.transpose(1, 2))
        return waveform


def pcm16(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    source = waveform.detach().cpu().numpy() if torch.is_tensor(waveform) else np.asarray(waveform)
    return np.clip(source, -1.0, 1.0).astype(np.float32)


def model_manifest(root: Path, *, adapted: bool | None = None) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest})
    return {
        "backend": "microsoft/speecht5_hifigan",
        "local_path": str(root),
        "karaone_adapted": bool((root / ADAPTER_FILE).is_file()) if adapted is None else bool(adapted),
        "files": files,
    }
