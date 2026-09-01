import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "app/src"))
MODULE = ROOT / "scripts" / "prepare_stage2_split.py"
spec = importlib.util.spec_from_file_location("prepare_stage2_split", MODULE)
stage2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage2)

from eeg2speech.data import _complete_grid
from eeg2speech.gates import registered_m0_gate_status
from prepare_training_data import semantic_preprocessing_contract


class TestStage2Split(unittest.TestCase):
    def test_custom_explore_names_are_isolated(self):
        pilot={"stage2":{"protocol":"stage2_joint_ood_explore_8h_v1",
                         "explore_artifact_set":"explore_stage2_8h_v1",
                         "explore_target_name":"speech_targets_explore_stage2_8h_v1",
                         "explore_normalizer_name":"explore_stage2_8h_v1_joint_ood_fold-0"}}
        names=stage2.stage2_names(pilot,True)
        self.assertEqual(names["protocol"],"stage2_joint_ood_explore_8h_v1")
        self.assertEqual(names["artifact_set"],"explore_stage2_8h_v1")
        self.assertNotEqual(names["artifact_set"],"explore_stage2")

    def test_stage2_can_select_a_single_dataset_without_changing_default(self):
        self.assertEqual(stage2.stage2_datasets({}), ("ds004940", "ds006104"))
        self.assertEqual(stage2.stage2_datasets({"stage2": {"datasets": ["ds004940"]}}), ("ds004940",))
        with self.assertRaises(ValueError):
            stage2.stage2_datasets({"stage2": {"datasets": ["ds004940", "ds004940"]}})

    def test_large_scale_config_uses_all_ds004940_contents_in_a_double_ood_grid(self):
        config_path = ROOT / "configs/ds004940_large_scale_v1.yaml"
        if not config_path.exists():
            self.fail(f"missing large-scale config: {config_path}")
        import yaml
        pilot = yaml.safe_load(config_path.read_text())["pilot"]
        self.assertEqual(pilot["generalization_subjects_by_role"], {"train": 10, "validation": 2, "test": 2})
        self.assertEqual(pilot["generalization_contents_by_role"], {"train": 338, "validation": 32, "test": 32})
        self.assertEqual(pilot["max_train_trials_per_dataset"], 3380)
        self.assertEqual(pilot["max_validation_trials_per_dataset"], 64)
        self.assertEqual(pilot["max_test_trials_per_dataset"], 64)
        self.assertEqual(sum(pilot["generalization_contents_by_role"].values()), 402)

    def test_real_ds004940_active_manifest_supports_the_large_complete_grid(self):
        manifest = ROOT / "artifacts/training_data/v3/manifests/manifest_all.csv"
        if not manifest.exists(): self.skipTest("DS004940 manifest is unavailable")
        frame = pd.read_csv(manifest, keep_default_na=False, low_memory=False)
        from prepare_training_data import load_config
        data_config, _ = load_config(ROOT / "configs/training_data_v3.yaml")
        active = frame[(frame.dataset == "ds004940") & (frame.build_status == "included") &
                       (frame.qc_pass.astype(str).str.lower() == "true") &
                       (frame.task == "N400Active") & frame.supervision_type.isin(["paired_audio", "weak_audio"])]
        active = active[stage2.declared_channel_qc_mask(active, data_config)]
        subjects, contents = stage2.choose_subjects(active, 14, 402, "test-large-ds004940")
        self.assertEqual(len(subjects), 14)
        self.assertEqual(len(contents), 402)
        subject_roles = stage2.assign(subjects, {"train": 10, "validation": 2, "test": 2}, "subject")
        content_roles = stage2.assign(contents, {"train": 338, "validation": 32, "test": 32}, "content")
        self.assertEqual(pd.Series(subject_roles).value_counts().to_dict(), {"train": 10, "validation": 2, "test": 2})
        self.assertEqual(pd.Series(content_roles).value_counts().to_dict(), {"train": 338, "validation": 32, "test": 32})

    def test_declared_bad_channel_qc_is_applied_before_subject_selection(self):
        frame = pd.DataFrame([
            {"dataset": "ds004940", "bad_channels": '["A1", "A2"]'},
            {"dataset": "ds004940", "bad_channels": '["A1", "A2", "A3"]'},
            {"dataset": "ds006104", "bad_channels": "not-json"},
        ])
        config = {
            "sources": {
                "ds004940": {"channel_order": ["A1", "A2", "A3", "A4", "A5",
                                                 "A6", "A7", "A8", "A9", "A10"]},
                "ds006104": {"channel_order": ["C1", "C2", "C3", "C4"]},
            },
            "harmonized": {"interpolation": {"max_bad_fraction": 0.20}},
        }
        self.assertEqual(stage2.declared_channel_qc_mask(frame, config).tolist(), [True, False, False])

    def test_exact_assignment_counts(self):
        values=[f"g{i}" for i in range(40)]
        assigned=stage2.assign(values,{"train":28,"validation":6,"test":6},"test")
        counts=pd.Series(list(assigned.values())).value_counts().to_dict()
        self.assertEqual(counts,{"train":28,"validation":6,"test":6})
        self.assertEqual(assigned,stage2.assign(values,{"train":28,"validation":6,"test":6},"test"))

    def test_semantic_preprocessing_contract_excludes_repository_commit(self):
        contract = semantic_preprocessing_contract(
            config_sha="config", source_lock_sha="sources", split_hash="split", code_diff_hash="transform",
        )
        self.assertEqual(contract, {
            "preprocess_config_sha256": "config", "source_lock_sha256": "sources",
            "split_index_sha256": "split", "code_diff_hash": "transform",
        })
        self.assertNotIn("code_commit", contract)

    def test_real_stage2_split_has_exact_double_ood_grids(self):
        artifact=ROOT/"artifacts/training_data/v3"
        split_path=artifact/"splits/stage2_joint_ood_fold-0.csv"
        manifest_path=artifact/"manifests/manifest_all.csv"
        if not split_path.exists() or not manifest_path.exists(): self.skipTest("Stage2 artifacts are not installed")
        frame=pd.read_csv(manifest_path,keep_default_na=False,low_memory=False)
        split=pd.read_csv(split_path,keep_default_na=False)
        merged=frame.merge(split[["trial_id","role"]],on="trial_id")
        subject_counts={"train":4,"validation":1,"test":1}; content_counts={"train":28,"validation":6,"test":6}
        for dataset in ("ds004940","ds006104"):
            for role in subject_counts:
                selected=merged[(merged.dataset==dataset)&(merged.role==role)&
                                merged.supervision_type.isin(["paired_audio","weak_audio"])]
                grid=_complete_grid(selected,subject_counts[role],content_counts[role],f"test|{dataset}|{role}")
                self.assertEqual(len(grid),subject_counts[role]*content_counts[role])
                self.assertEqual(len(selected), subject_counts[role] * content_counts[role])
                self.assertTrue(selected.groupby(["subject", "linguistic_content_id"]).size().eq(1).all())
        label_counts={"train":7,"validation":2,"test":2}
        for role in subject_counts:
            selected=merged[(merged.dataset=="ds006104")&(merged.role==role)&
                            (merged.supervision_type=="label_only")]
            self.assertEqual(len(selected),subject_counts[role]*label_counts[role])
            self.assertTrue(selected.groupby(["subject","linguistic_content_id"]).size().eq(1).all())

    def test_stage2_gate_requires_every_mode_dataset_and_seed(self):
        config={"training":{"seeds":[31]}}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            status=registered_m0_gate_status(root,config)
            self.assertEqual(len(status["missing"]),4)
            self.assertEqual(status["failed"],[])
            for mode,datasets in {"ds004940":["ds004940"],"ds006104":["ds006104"],
                                  "joint":["ds004940","ds006104"]}.items():
                folder=root/"outputs/joint_pilot_v1/pilot/overfit"/mode/"seed-31"
                folder.mkdir(parents=True,exist_ok=True)
                for dataset in datasets:
                    (folder/f"evaluation_{dataset}_train.json").write_text(
                        json.dumps({"run_kind":"pilot","gate":{"passed":True}})
                    )
            self.assertEqual(registered_m0_gate_status(root,config),{"missing":[],"failed":[]})


if __name__=="__main__": unittest.main()
