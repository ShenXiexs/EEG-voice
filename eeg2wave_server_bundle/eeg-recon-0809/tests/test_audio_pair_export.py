import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.io import wavfile


ROOT = Path(__file__).parents[1]
MODULE = ROOT / "app" / "export_audio_pair_comparisons.py"
sys.path.insert(0, str(ROOT / "app"))
spec = importlib.util.spec_from_file_location("audio_pair_export", MODULE)
exporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)


class TestAudioPairExport(unittest.TestCase):
    def test_griffin_lim_diagnostic_contract_and_wav_write(self):
        # A finite zero log-mel must still yield a finite, fixed-duration
        # diagnostic waveform; this does not invoke data loading or a model.
        waveform = exporter.inverse_log_mel(torch.zeros(80, 161), iterations=1, seed=31)
        self.assertEqual(waveform.shape, (160 * 160,))
        self.assertTrue(np.isfinite(waveform).all())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "diagnostic.wav"
            exporter.write_pcm16(target, waveform, peak_normalize=True)
            rate, data = wavfile.read(target)
            self.assertEqual(rate, 16_000)
            self.assertEqual(data.dtype, np.int16)
            self.assertEqual(len(data), len(waveform))

    def test_export_layout_matches_the_reference_pair_semantics(self):
        self.assertEqual(exporter.WAV_NAMES["target"], "01_target_logmel_griffinlim_oracle.wav")
        self.assertEqual(exporter.WAV_NAMES["joint"], "03_joint_eeg_mfcc_griffinlim.wav")
        self.assertIn("zero", exporter.WAV_NAMES)
        self.assertIn("time_shuffle", exporter.WAV_NAMES)
        self.assertIn("channel_shuffle", exporter.WAV_NAMES)

    def test_single_only_layout_does_not_claim_a_joint_model(self):
        self.assertEqual(exporter.SINGLE_ONLY_WAV_NAMES["single"], "02_ds004940_eeg_mfcc_griffinlim.wav")
        self.assertNotIn("joint", " ".join(exporter.SINGLE_ONLY_WAV_NAMES.values()))
        self.assertEqual(exporter.SINGLE_ONLY_DISPLAY["single"], "DS004940 EEG")


if __name__ == "__main__":
    unittest.main()
