import collections
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE = ROOT / "scripts" / "prepare_training_data.py"
spec = importlib.util.spec_from_file_location("prepare_integration", MODULE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TestRealManifestAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "data/ds004940").exists() or not (ROOT / "data/ds006104").exists():
            raise unittest.SkipTest("raw datasets are not installed")
        cls.config, _ = prepare.load_config(ROOT / "configs/training_data_v3.yaml")
        cls.qc = {"actual_subjects": {}, "exclusions": collections.Counter(), "warnings": []}
        cls.lock = {"official_aux": {}}
        cls.ds004 = prepare._ds004_trial_rows(cls.config, cls.lock, cls.qc)
        cls.ds006 = prepare._ds006_trial_rows(cls.config, cls.lock, cls.qc, False)

    def test_release_counts_and_boundary_exclusion(self):
        self.assertEqual(len(self.ds004), 17491)
        self.assertEqual(sum(row["qc_pass"] for row in self.ds004), 17489)
        self.assertEqual(sum(row["boundary_overlap"] for row in self.ds004), 2)
        self.assertEqual(self.qc["actual_subjects"]["ds004940"], 22)

    def test_ds006_preceding_tms_join_and_s15_is_explicit(self):
        self.assertEqual(len(self.ds006), 10888)
        self.assertEqual(sum(row["qc_pass"] for row in self.ds006), 10888)
        self.assertEqual(self.qc["exclusions"].get("missing_official_aux_row", 0), 0)
        matched = [row for row in self.ds006 if row["qc_pass"]]
        self.assertTrue(all(row["official_timing_error_samples"] == 0 for row in matched))

    def test_pairing_levels_do_not_upgrade_ds006_audio(self):
        levels = collections.Counter(row["pairing_level"] for row in self.ds006 if row["qc_pass"])
        self.assertEqual(levels["candidate_filename_timing"], 7533)
        self.assertEqual(levels["label_only"], 3355)
        self.assertFalse(any(row["pairing_level"] == "verified_exact" for row in self.ds006))

    def test_both_datasets_are_perception(self):
        self.assertTrue(all(row["neural_task"] == "perception" for row in self.ds004 + self.ds006))
        self.assertFalse(any(row["production_contaminated"] for row in self.ds004 + self.ds006))


if __name__ == "__main__":
    unittest.main()
