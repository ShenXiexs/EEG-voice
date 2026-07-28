"""Independent v0728 dual-latent EEG-to-speech implementation.

The package intentionally does not import the protected v0724 namespace.
"""

from .model import DualLatentAudioModel, DualLatentEEGToSpeech

__all__ = ["DualLatentAudioModel", "DualLatentEEGToSpeech"]
