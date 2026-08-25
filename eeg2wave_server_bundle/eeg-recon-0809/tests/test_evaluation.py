import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/"app/src"))
MODULE=ROOT/"app/evaluate_joint.py"
spec=importlib.util.spec_from_file_location("evaluate_joint",MODULE)
evaluate=importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluate)


class TestEvaluationContracts(unittest.TestCase):
    def test_registered_dataset_mean_gate_does_not_require_degenerate_same_content(self):
        target=torch.stack([torch.zeros(2,3),torch.zeros(2,3),torch.ones(2,3),torch.ones(2,3)])
        prediction=target.clone()
        metrics=evaluate.template_metrics(prediction,target,["a","a","b","b"])
        self.assertFalse(metrics["same_content_template_gate_applicable"])
        name,passed=evaluate.registered_collapse_check(metrics,{"collapse_baseline":"dataset_mean","template_improvement_min":0.5})
        self.assertEqual(name,"registered_dataset_mean_collapse_baseline"); self.assertTrue(passed)

    def test_subject_probe_reports_chance(self):
        embeddings=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        result=evaluate.leave_one_out_subject_probe(embeddings,["s1","s1","s2","s2"])
        self.assertEqual(result["accuracy"],1.0); self.assertEqual(result["chance"],0.5)


if __name__=="__main__": unittest.main()
