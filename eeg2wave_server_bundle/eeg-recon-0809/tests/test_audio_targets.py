import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[1] / "scripts" / "cache_speech_targets.py"
sys.path.insert(0, str(MODULE.parent))
spec = importlib.util.spec_from_file_location("speech_targets", MODULE)
speech_targets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speech_targets)


class TestAudioTargets(unittest.TestCase):
    def test_waveform_rms_preserves_gain_and_matches_mel_frames(self):
        wave = np.zeros(1600, dtype=np.float32)
        wave[400:1200] = 0.25
        rms, activity = speech_targets.frame_rms_activity(wave)
        mel = speech_targets.log_mel(wave)
        self.assertEqual(len(rms), mel.shape[1])
        self.assertEqual(len(activity), mel.shape[1])
        self.assertGreater(float(rms.max()), 0.2)
        self.assertLess(float(rms[0]), 1e-4)
        self.assertTrue(activity.any())
        self.assertFalse(activity[0])

    def test_content_features_are_fixed_length_and_cmvn(self):
        time = np.arange(16000, dtype=np.float32) / 16000.0
        wave = np.sin(2 * np.pi * 220.0 * time).astype(np.float32)
        mfcc, mask = speech_targets.content_features(wave, frames=161)
        self.assertEqual(mfcc.shape, (39, 161))
        self.assertEqual(mask.shape, (161,))
        self.assertTrue(np.isfinite(mfcc).all())
        # Linear resampling after CMVN introduces a small endpoint-weighting
        # offset while preserving the intended utterance-scale normalization.
        self.assertLess(float(np.abs(mfcc.mean(axis=1)).max()), 5e-3)


if __name__ == "__main__":
    unittest.main()
