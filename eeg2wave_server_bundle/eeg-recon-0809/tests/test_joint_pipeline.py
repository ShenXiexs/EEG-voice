import sys
import tempfile
import unittest
from pathlib import Path

import torch
import pandas as pd


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "app/src"))

from eeg2speech.data import AlternatingBatchIterator, JointManifestDataset, homogeneous_collate, pilot_indices
from eeg2speech.losses import counterfactual_eeg, joint_content_loss, masked_mfcc_loss
from eeg2speech.model import AudioMFCCRenderer, JointEEGContentModel


class TestJointPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = JointEEGContentModel(dimension=24, heads=4, layers=1, local_layers=1,
                                          dropout=0.0, phoneme_classes=8)

    def _inputs(self, channels, samples, dataset_id, batch=2):
        eeg = torch.randn(batch, channels, samples)
        xyz = torch.randn(batch, channels, 3)
        xyz = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return eeg, xyz, torch.ones(batch, channels, dtype=torch.bool), torch.ones(batch, samples, dtype=torch.bool), torch.full((batch,), dataset_id)

    def test_both_channel_spaces_use_one_model(self):
        first = self.model(*self._inputs(128, 256, 0))
        second = self.model(*self._inputs(61, 192, 1))
        self.assertEqual(first.mfcc.shape, (2, 39, 161))
        self.assertEqual(second.local.shape, (2, 96, 24))
        (first.mfcc.mean() + second.mfcc.mean()).backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()))

    def test_supervision_weight_masks_label_only_content(self):
        state = self.model(*self._inputs(61, 192, 1, batch=3))
        batch = {
            "content_mfcc": torch.randn(3,39,161), "content_mask": torch.ones(3,161,dtype=torch.bool),
            "pairing_weight": torch.tensor([1.0,0.35,0.0]), "hubert_local": torch.zeros(3,96,768),
            "hubert_mask": torch.zeros(3,96,dtype=torch.bool), "phoneme_index": torch.tensor([-1,-1,2]),
            "linguistic_content_id": ["a","b","c"],
        }
        weights = {"mfcc":1.0,"delta":0.2,"local_alignment":0.5,"global_clip":0.5,"phoneme_auxiliary":0.1}
        loss, metrics = joint_content_loss(state,batch,self.model,weights)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["mfcc"],0)
        self.assertGreater(metrics["phoneme_auxiliary"],0)

    def test_counterfactual_controls_preserve_shape(self):
        eeg = torch.randn(2,61,128)
        for control in ("zero","time_shuffle","channel_shuffle"):
            self.assertEqual(counterfactual_eeg(eeg,control).shape,eeg.shape)

    def test_controls_only_shuffle_valid_support(self):
        eeg = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
        time_mask = torch.tensor([[1,1,1,1,0,0,0,0],[1,1,1,1,1,1,0,0]], dtype=torch.bool)
        channel_mask = torch.tensor([[1,1,1,0],[1,1,0,0]], dtype=torch.bool)
        time = counterfactual_eeg(eeg, "time_shuffle", time_mask=time_mask)
        channel = counterfactual_eeg(eeg, "channel_shuffle", channel_mask=channel_mask)
        self.assertTrue(torch.equal(time, counterfactual_eeg(eeg, "time_shuffle", time_mask=time_mask)))
        self.assertTrue(torch.equal(channel, counterfactual_eeg(eeg, "channel_shuffle", channel_mask=channel_mask)))
        self.assertTrue(torch.equal(time[0,:,4:], eeg[0,:,4:]))
        self.assertTrue(torch.equal(time[1,:,6:], eeg[1,:,6:]))
        self.assertTrue(torch.equal(channel[0,3], eeg[0,3]))
        self.assertTrue(torch.equal(channel[1,2:], eeg[1,2:]))

    def test_weak_supervision_really_scales_homogeneous_batch(self):
        prediction = torch.zeros(2, 39, 161)
        target = torch.ones_like(prediction)
        mask = torch.ones(2, 161, dtype=torch.bool)
        exact = masked_mfcc_loss(prediction, target, mask, torch.ones(2))[0]
        weak = masked_mfcc_loss(prediction, target, mask, torch.full((2,), 0.35))[0]
        self.assertAlmostEqual(float(weak / exact), 0.35, places=5)

    def test_padding_values_are_inert(self):
        eeg, xyz, channel_mask, time_mask, dataset_id = self._inputs(8, 128, 0)
        time_mask[:, 80:] = False
        first = self.model(eeg, xyz, channel_mask, time_mask, dataset_id).mfcc
        changed = eeg.clone(); changed[:, :, 80:] = 1e6
        second = self.model(changed, xyz, channel_mask, time_mask, dataset_id).mfcc
        self.assertTrue(torch.allclose(first, second, atol=1e-5, rtol=1e-5))

    def test_m0_selection_enforces_complete_subject_content_grid(self):
        rows = [{"dataset":"ds004940", "subject":f"s{subject}", "linguistic_content_id":f"c{content}",
                 "trial_id":f"trial-{subject}-{content}", "tms_applied":False}
                for subject in range(5) for content in range(10)]
        dataset = type("Dataset", (), {"frame": pd.DataFrame(rows)})()
        config = {"pilot":{"overfit_pairs_per_dataset":50, "overfit_subjects_per_dataset":5,
                           "overfit_contents_per_dataset":10, "primary_ds006104_tms":False}}
        self.assertEqual(len(pilot_indices(dataset, config, "overfit")), 50)
        dataset.frame = dataset.frame.iloc[:-1]
        with self.assertRaises(RuntimeError):
            pilot_indices(dataset, config, "overfit")

    def test_m1_selection_enforces_registered_role_grid(self):
        pilot = {"generalization_subjects_per_dataset":6, "generalization_contents_per_dataset":40,
                 "generalization_subjects_by_role":{"train":4,"validation":1,"test":1},
                 "generalization_contents_by_role":{"train":28,"validation":6,"test":6},
                 "max_train_trials_per_dataset":512,"max_validation_trials_per_dataset":128,
                 "max_test_trials_per_dataset":128,"primary_ds006104_tms":False}
        counts = {"train":(4,28,112), "validation":(1,6,6), "test":(1,6,6)}
        for role,(subjects,contents,pairs) in counts.items():
            rows = [{"dataset":"ds004940", "subject":f"{role}-s{s}",
                     "linguistic_content_id":f"{role}-c{c}", "trial_id":f"{role}-{s}-{c}",
                     "tms_applied":False} for s in range(subjects) for c in range(contents)]
            dataset = type("Dataset", (), {"frame": pd.DataFrame(rows)})()
            selected = dataset.frame.iloc[pilot_indices(dataset, {"pilot":pilot}, "generalization", role)]
            self.assertEqual((len(selected), selected.subject.nunique(), selected.linguistic_content_id.nunique()),
                             (pairs, subjects, contents))

    def test_m1_selection_fails_instead_of_head_truncation(self):
        rows = [{"dataset":"ds004940", "subject":f"s{s}", "linguistic_content_id":f"c{c}",
                 "trial_id":f"{s}-{c}", "tms_applied":False} for s in range(4) for c in range(27)]
        dataset = type("Dataset", (), {"frame": pd.DataFrame(rows)})()
        config = {"pilot":{"generalization_subjects_per_dataset":6,"generalization_contents_per_dataset":40,
                            "generalization_subjects_by_role":{"train":4,"validation":1,"test":1},
                            "generalization_contents_by_role":{"train":28,"validation":6,"test":6},
                            "max_train_trials_per_dataset":512,"max_validation_trials_per_dataset":128,
                            "max_test_trials_per_dataset":128,"primary_ds006104_tms":False}}
        with self.assertRaises(RuntimeError): pilot_indices(dataset, config, "generalization", "train")

    def test_audio_renderer_contract(self):
        renderer = AudioMFCCRenderer(hidden_dimension=24, layers=1, dropout=0.0)
        state = renderer(torch.randn(2,39,161))
        self.assertEqual(state.log_mel.shape,(2,80,161))
        self.assertEqual(state.rms.shape,(2,161))
        self.assertEqual(state.activity_logits.shape,(2,161))

    def test_content_dataset_fails_closed_without_speech_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{"trial_id":"trial-1","dataset":"ds004940","subject":"s1","task":"task",
                           "condition":"active","linguistic_content_id":"content-1","phoneme_label":"",
                           "build_status":"included","supervision_type":"paired_audio","pairing_level":"verified_exact",
                           "audio_id":"audio-missing","audio_sha256":"abc","audio_semantics":"presented_waveform",
                           "preprocess_config_sha256":"config","shard_path":"missing.h5","shard_row":0}]).to_csv(root/"manifest.csv",index=False)
            pd.DataFrame([{"trial_id":"trial-1","role":"train"}]).to_csv(root/"split.csv",index=False)
            with self.assertRaisesRegex(RuntimeError,"speech-target cache"):
                JointManifestDataset(root/"manifest.csv",root/"split.csv","train","ds004940",
                                     speech_targets=root/"absent.h5",normalizer_path=root/"absent.json")

    def test_collate_rejects_cross_dataset_batch(self):
        with self.assertRaises(ValueError):
            homogeneous_collate([{"dataset":"ds004940"},{"dataset":"ds006104"}])

    def test_alternating_iterator_is_round_robin(self):
        iterator = iter(AlternatingBatchIterator({"a":[1],"b":[2]},["a","b"]))
        self.assertEqual(next(iterator),("a",1))
        self.assertEqual(next(iterator),("b",2))
        self.assertEqual(next(iterator),("a",1))


if __name__ == "__main__":
    unittest.main()
