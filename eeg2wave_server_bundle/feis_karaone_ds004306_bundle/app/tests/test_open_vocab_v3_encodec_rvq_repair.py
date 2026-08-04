"""Small CPU-only contract tests for repair-v3 (no KaraOne WAVs required)."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import inspect
import numpy as np
import torch

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))
from src.open_vocab_v3.encodec_rvq_repair import (
    AudioCTeacher, DirectEEGMFCC, SequentialRVQBridge, diagonal_band_infonce,
    soft_dtw_token_clip,
)
from src.open_vocab_v3.runtime import load_config
from scripts.train_open_vocab_v3_encodec_rvq_repair import folds

class RepairV3ContractTest(unittest.TestCase):
    def test_repair_config_has_unique_schema(self):
        _,cfg=load_config(APP/'configs/open_vocab_v3_mfcc_encodec_rvq_repair_v3.yaml')
        self.assertEqual(cfg['version'],'openvoice-v3-mfcc-encodec-rvq-repair-v3')
        self.assertEqual(cfg['audio']['canonical_frames'],161)

    def test_sequential_rvq_shape_and_dependency(self):
        torch.manual_seed(3);model=SequentialRVQBridge();content=torch.randn(2,39,161);p=torch.randn(2,161,3);voice=torch.randn(2,192);target=torch.zeros(2,8,192,dtype=torch.long)
        logits=model(content,p,voice,torch.ones(2),targets=target,teacher_forcing=1.0)
        self.assertEqual(tuple(logits.shape),(2,8,1024,192))
        valid=torch.zeros(2,192,dtype=torch.bool);valid[:,:37]=True
        hard=model.hard_codes(logits,code_mask=valid)
        self.assertTrue(bool((hard[:,:,37:]==0).all()))
        sampled=model.sample_residual_codes(logits,code_mask=valid,generator=torch.Generator().manual_seed(8))
        self.assertTrue(torch.equal(sampled[:,:4],hard[:,:4]))
        self.assertTrue(bool((sampled[:,:,37:]==0).all()))
        changed=target.clone();changed[:,0]=1
        with torch.no_grad():later_a=model(content,p,voice,torch.ones(2),targets=target,teacher_forcing=1.0)[:,1:];later_b=model(content,p,voice,torch.ones(2),targets=changed,teacher_forcing=1.0)[:,1:]
        self.assertGreater(float((later_a-later_b).abs().sum()),0.0)

    def test_diagonal_loss_beats_temporal_scramble(self):
        torch.manual_seed(4);x=torch.randn(2,96,16);mask=torch.ones(2,96,dtype=torch.bool)
        same,_=diagonal_band_infonce(x,x,mask,mask)
        wrong,_=diagonal_band_infonce(x,x.flip(1),mask,mask)
        self.assertLess(float(same),float(wrong))
        soft_same,_=soft_dtw_token_clip(x,x,mask,mask);soft_wrong,_=soft_dtw_token_clip(x,x.flip(1),mask,mask)
        self.assertLess(float(soft_same),float(soft_wrong))

    def test_audio_and_eeg_grids(self):
        audio=AudioCTeacher();state=audio(torch.zeros(1,8,192,dtype=torch.long),torch.ones(1,192,dtype=torch.bool),torch.randn(1,80,768),torch.ones(1,80,dtype=torch.bool))
        self.assertEqual(tuple(state.local.shape),(1,96,256))
        eeg=DirectEEGMFCC();mfcc,local,mask=eeg(torch.randn(1,4,601),torch.randn(1,4,3),torch.ones(1,4,dtype=torch.bool),torch.ones(1,601,dtype=torch.bool))
        self.assertEqual(tuple(mfcc.shape),(1,39,161));self.assertEqual(tuple(local.shape),(1,96,256));self.assertTrue(bool(mask.all()))

    def test_labels_are_not_model_forward_inputs(self):
        for model in (SequentialRVQBridge,AudioCTeacher,DirectEEGMFCC):
            self.assertNotIn('label',inspect.signature(model.forward).parameters)

    def test_m1_outer_inner_folds_are_disjoint(self):
        indices=np.arange(50,dtype=np.int32);labels=np.asarray([f'label_{row//5}' for row in range(50)]);keys=np.asarray([f'k_{row:03d}' for row in range(50)])
        split=folds(indices,keys,labels);outer=[]
        self.assertEqual(len(split),5)
        for train,inner,held in split:
            self.assertFalse(set(train)&set(inner));self.assertFalse(set(train)&set(held));self.assertFalse(set(inner)&set(held));self.assertEqual(len(train),30);self.assertEqual(len(inner),10);self.assertEqual(len(held),10);outer.extend(held.tolist())
        self.assertEqual(sorted(outer),indices.tolist())

if __name__=='__main__':unittest.main()
