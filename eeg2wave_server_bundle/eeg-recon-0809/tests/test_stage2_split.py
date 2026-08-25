import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "app/src"))
MODULE = ROOT / "scripts" / "prepare_stage2_split.py"
spec = importlib.util.spec_from_file_location("prepare_stage2_split", MODULE)
stage2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage2)

from eeg2speech.data import _complete_grid


class TestStage2Split(unittest.TestCase):
    def test_exact_assignment_counts(self):
        values=[f"g{i}" for i in range(40)]
        assigned=stage2.assign(values,{"train":28,"validation":6,"test":6},"test")
        counts=pd.Series(list(assigned.values())).value_counts().to_dict()
        self.assertEqual(counts,{"train":28,"validation":6,"test":6})
        self.assertEqual(assigned,stage2.assign(values,{"train":28,"validation":6,"test":6},"test"))

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


if __name__=="__main__": unittest.main()
