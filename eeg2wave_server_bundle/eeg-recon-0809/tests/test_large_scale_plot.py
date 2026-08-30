import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE = ROOT / "app" / "plot_ds004940_large_scale.py"
sys.path.insert(0, str(ROOT / "app"))
spec = importlib.util.spec_from_file_location("plot_ds004940_large_scale", MODULE)
plot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plot
spec.loader.exec_module(plot)


class TestLargeScalePlot(unittest.TestCase):
    def test_subject_control_bootstrap_uses_paired_subject_errors(self):
        records = [{
            "scale": "large_3380", "role": "test",
            "subject_mfcc_l1": {"sub-a": 0.3, "sub-b": 0.4},
            "subject_control_mfcc_l1": {
                "zero": {"sub-a": 0.5, "sub-b": 0.6},
                "time_shuffle": {"sub-a": 0.45, "sub-b": 0.55},
                "channel_shuffle": {"sub-a": 0.4, "sub-b": 0.5},
            },
        }]
        summary = plot.subject_control_summary(records, repetitions=100)
        test_zero = next(row for row in summary if row["role"] == "test" and row["control"] == "zero")
        self.assertEqual(test_zero["correct_eeg_margin_bootstrap"]["n"], 2)
        self.assertAlmostEqual(test_zero["correct_eeg_margin_bootstrap"]["mean"], 0.2)
        validation_zero = next(row for row in summary if row["role"] == "validation" and row["control"] == "zero")
        self.assertEqual(validation_zero["correct_eeg_margin_bootstrap"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
