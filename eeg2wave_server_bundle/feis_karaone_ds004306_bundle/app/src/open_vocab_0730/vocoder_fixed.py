from __future__ import annotations

from pathlib import Path

import torch

from .vocoder import SpeechT5HiFiGan, model_manifest, pcm16


class SpeechT5HiFiGanFixed(SpeechT5HiFiGan):
    """SpeechT5 adapter for power-dB Mel features.

    The cached v3 Mel is 10*log10(power), while SpeechT5 consumes log10 Mel
    power.  Therefore the correct conversion is dB/10 (v1 used dB/20, which is
    the amplitude-dB conversion).
    """

    @torch.no_grad()
    def synthesize(self, mel_db: torch.Tensor) -> torch.Tensor:
        if mel_db.ndim != 3 or mel_db.shape[1] != 80:
            raise ValueError("SpeechT5 backend expects mel_db[B,80,T]")
        spectrogram = (mel_db / 10.0).transpose(1, 2).to(self.device)
        waveform = self.model(spectrogram)
        return waveform.squeeze(0) if waveform.ndim == 2 and waveform.shape[0] == 1 else waveform


__all__ = ["SpeechT5HiFiGanFixed", "model_manifest", "pcm16"]
