"""BrainOmni-style EEG voice model v0.

Heavy torch modules are loaded lazily so lightweight helpers such as
`audio_features` can still be used before PyTorch is installed.
"""

__all__ = [
    "AudioContrastiveHead",
    "BrainStyleEEGTokenizerConfig",
    "BrainStyleEEGTokenizerV0",
    "ProbeHead",
    "VoiceAttributeHead",
]


def __getattr__(name: str):
    if name in {"BrainStyleEEGTokenizerV0", "BrainStyleEEGTokenizerConfig"}:
        from .tokenizer import BrainStyleEEGTokenizerConfig, BrainStyleEEGTokenizerV0

        return {
            "BrainStyleEEGTokenizerConfig": BrainStyleEEGTokenizerConfig,
            "BrainStyleEEGTokenizerV0": BrainStyleEEGTokenizerV0,
        }[name]
    if name in {"AudioContrastiveHead", "ProbeHead", "VoiceAttributeHead"}:
        from .heads import AudioContrastiveHead, ProbeHead, VoiceAttributeHead

        return {
            "AudioContrastiveHead": AudioContrastiveHead,
            "ProbeHead": ProbeHead,
            "VoiceAttributeHead": VoiceAttributeHead,
        }[name]
    raise AttributeError(name)
