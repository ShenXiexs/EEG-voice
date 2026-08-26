import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODULE = ROOT / "scripts" / "prepare_m0_artifacts.py"
spec = importlib.util.spec_from_file_location("prepare_m0_artifacts", MODULE)
m0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m0)

from prepare_training_data import load_config


class TestRunScripts(unittest.TestCase):
    def test_all_shell_entry_points_are_executable_and_parse(self):
        scripts = sorted((ROOT / "app").glob("run_joint_*.sh"))
        self.assertGreaterEqual(len(scripts), 8)
        for script in scripts:
            self.assertTrue(os.access(script, os.X_OK), script)
            subprocess.run(["bash", "-n", str(script)], check=True)
        subprocess.run(["bash", "-n", str(ROOT / "app/lib/joint_pilot_common.sh")], check=True)

    def test_registered_m0_selection_uses_real_frozen_split(self):
        manifest = ROOT / "artifacts/training_data/v3/manifests/manifest_all.csv"
        split = ROOT / "artifacts/training_data/v3/splits/joint_ood_fold-0.csv"
        if not manifest.exists() or not split.exists():
            self.skipTest("installed dataset artifacts are unavailable")
        config, _ = load_config(ROOT / "configs/training_data_v3.yaml")
        pilot = yaml.safe_load((ROOT / "configs/joint_pilot_v1.yaml").read_text())
        grids = m0.select_registered_grids(config, pilot)
        self.assertEqual(len(grids["ds004940"]), 50)
        self.assertEqual(len(grids["ds006104"]), 50)
        self.assertEqual(len(grids["ds006104_label_only"]), 30)
        for frame in grids.values():
            self.assertTrue(frame.groupby(["subject", "linguistic_content_id"]).size().eq(1).all())

    def test_complete_runner_has_no_automatic_human_approval(self):
        complete = (ROOT / "app/run_joint_pilot_all.sh").read_text()
        stage0 = (ROOT / "app/run_joint_stage0.sh").read_text()
        self.assertIn("exit 3", complete)
        self.assertIn("human_listen_transcript_status=pass only after real human verification", stage0)
        self.assertNotIn("human_listen_transcript_status=pass\"", complete)

    def test_explore_runner_is_explicitly_isolated_from_registered_outputs(self):
        explore = (ROOT / "app/run_joint_explore.sh").read_text()
        self.assertIn("--explore", explore)
        self.assertIn("$RUN_ROOT/explore", explore)
        self.assertIn("--explore --materialize", explore)


if __name__ == "__main__":
    unittest.main()
