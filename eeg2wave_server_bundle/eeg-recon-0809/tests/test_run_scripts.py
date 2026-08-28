import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODULE = ROOT / "scripts" / "prepare_m0_artifacts.py"
spec = importlib.util.spec_from_file_location("prepare_m0_artifacts", MODULE)
m0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m0)

from prepare_training_data import load_config

TRAIN_MODULE = ROOT / "app" / "train_joint.py"
sys.path.insert(0, str(ROOT / "app" / "src"))
train_spec = importlib.util.spec_from_file_location("train_joint", TRAIN_MODULE)
train = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train)


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
        self.assertIn("--checkpoint-every", explore)
        self.assertIn("EXPLORE_STAGE2_MAX_STEPS", explore)

    def test_eight_hour_runner_is_isolated_resumable_and_renders_figures(self):
        explore = (ROOT / "app/run_joint_explore_8h.sh").read_text()
        self.assertIn("joint_explore_8h_v1.yaml", explore)
        self.assertIn("explore_8h_v1", explore)
        self.assertIn("EXPLORE_8H_MAX_STEPS", explore)
        self.assertIn("--checkpoint-every", explore)
        self.assertIn("--output-root", explore)
        self.assertIn("plot_joint_comparison.py", explore)
        self.assertNotIn("--stage overfit", explore)

    def test_atomic_training_state_and_contract_mismatch_detection(self):
        contract = {"mode": "joint", "seed": 31, "artifact_hashes": {"manifest": "abc"}}
        self.assertEqual(train.contract_mismatches(contract, dict(contract)), [])
        changed = dict(contract); changed["seed"] = 47
        self.assertEqual(train.contract_mismatches(contract, changed), ["seed"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "training_state.pt"
            train.atomic_torch_save({"completed_steps": 25, "resume_contract": contract}, target)
            self.assertTrue(target.exists())
            self.assertFalse(target.with_suffix(".pt.tmp").exists())
            self.assertEqual(torch.load(target, map_location="cpu", weights_only=False)["completed_steps"], 25)

    def test_legacy_explore_contract_allows_only_control_plane_upgrade(self):
        legacy = {"mode": "joint", "run_kind": "explore", "artifact_hashes": {"manifest": "abc"},
                  "runtime_code_sha256": "old-control-and-model-code"}
        migrated = {"mode": "joint", "run_kind": "explore", "artifact_hashes": {"manifest": "abc"},
                    "model_code_sha256": "model-code"}
        self.assertEqual(train.contract_mismatches(
            legacy, migrated, allow_legacy_explore_control_upgrade=True,
        ), [])
        changed_artifact = dict(migrated); changed_artifact["artifact_hashes"] = {"manifest": "changed"}
        self.assertEqual(train.contract_mismatches(
            legacy, changed_artifact, allow_legacy_explore_control_upgrade=True,
        ), ["artifact_hashes"])
        self.assertEqual(train.contract_mismatches(legacy, migrated), ["legacy_runtime_code_sha256"])

    def test_resume_keeps_original_finish_line_when_new_budget_is_smaller(self):
        self.assertEqual(train.resume_maximum_steps(2700, {"completed_steps": 4050, "maximum_steps": 5000}), (5000, True))
        self.assertEqual(train.resume_maximum_steps(5000, {"completed_steps": 2500, "maximum_steps": 2700}), (5000, False))
        with self.assertRaisesRegex(RuntimeError, "invalid completed_steps"):
            train.resume_maximum_steps(5000, {"completed_steps": 8, "maximum_steps": 7})


if __name__ == "__main__":
    unittest.main()
