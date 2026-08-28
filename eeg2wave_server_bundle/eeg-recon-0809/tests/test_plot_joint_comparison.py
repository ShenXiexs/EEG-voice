import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "app"))
MODULE = ROOT / "app" / "plot_joint_comparison.py"
spec = importlib.util.spec_from_file_location("plot_joint_comparison", MODULE)
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)


def evaluation(dataset: str, role: str, offset: float) -> dict:
    subjects = ["s1", "s2", "s3"]
    values = {subject: 0.8 + index * 0.01 + offset for index, subject in enumerate(subjects)}
    return {
        "dataset": dataset, "role": role, "pairs": 18, "mfcc_l1": sum(values.values()) / len(values),
        "delta_l1": 0.2 + offset,
        "retrieval": {"r1": 0.3 - offset, "mrr": 0.5, "chance_r1": 1 / 6, "unique_contents": 6},
        "controls": {"correct": 0.8 + offset, "zero": 0.85 + offset,
                     "time_shuffle": 0.84 + offset, "channel_shuffle": 0.83 + offset},
        "subject_mfcc_l1": values,
        "templates": {"dataset_mean_template_improvement": 0.05 - offset},
    }


class TestJointComparisonFigures(unittest.TestCase):
    def test_summary_and_all_figure_formats_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "explore_8h_v1"
            for seed in (31, 47):
                for dataset in plotter.DATASETS:
                    for mode, offset in ((dataset, 0.05), ("joint", 0.0)):
                        folder = root / "generalization" / mode / f"seed-{seed}"
                        folder.mkdir(parents=True, exist_ok=True)
                        for role in plotter.ROLES:
                            (folder / f"evaluation_{dataset}_{role}.json").write_text(
                                json.dumps(evaluation(dataset, role, offset))
                            )
                        retrieval_100 = ({key: 0.2 for key in plotter.DATASETS}
                                         if mode == "joint" else {dataset: 0.2})
                        retrieval_200 = ({key: 0.3 for key in plotter.DATASETS}
                                         if mode == "joint" else {dataset: 0.3})
                        metrics = {"history": [
                            {"step": 100, "full_content_retrieval_r1": retrieval_100},
                            {"step": 200, "full_content_retrieval_r1": retrieval_200},
                        ]}
                        (folder / "metrics.json").write_text(json.dumps(metrics))
            records, _ = plotter.collect_results(root, (31, 47))
            summary = plotter.summarize(records, 200)
            self.assertTrue(all(group["subject_bootstrap_gain"]["estimable"] for group in summary))
            output = root / "generalization" / "figures"
            plotter.configure()
            paths = []
            paths += plotter.plot_mfcc(records, output, ("png", "pdf"), 100)
            paths += plotter.plot_controls(records, output, ("png", "pdf"), 100)
            paths += plotter.plot_retrieval(records, output, ("png", "pdf"), 100)
            sources = []
            paths += plotter.training_curves(root, (31, 47), output, ("png", "pdf"), 100, sources)
            self.assertEqual(len(paths), 8)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertEqual(len(sources), 8)


if __name__ == "__main__":
    unittest.main()
