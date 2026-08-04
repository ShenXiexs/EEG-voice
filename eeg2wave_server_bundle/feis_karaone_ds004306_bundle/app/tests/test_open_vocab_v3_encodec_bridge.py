from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from scripts.train_open_vocab_v3_encodec_bridge import LabelGroupedBatchSampler, micro_generalization_folds
from src.open_vocab_v3.data import _bridge_content_target, _p_medoid_bank
from src.open_vocab_v3.encodec_bridge import (
    AudioCEncoder, ContinuousEnCodecBridge, EEGCEncoder,
    SharedContentMFCCDecoder, masked_token_infonce,
)
from src.open_vocab_v3.runtime import checkpoint_schema, load_config, output_path


APP=Path(__file__).resolve().parents[1]


def test_bridge_config_is_isolated_and_uses_161_active_content_frames():
    path,cfg=load_config(APP/"configs"/"open_vocab_v3_mfcc_encodec_bridge_v2.yaml")
    assert cfg["version"]=="openvoice-v3-mfcc-encodec-bridge-v2"
    assert cfg["audio"]["canonical_frames"]==161
    assert cfg["audio"]["content_tokens"]==96
    assert "open_vocab_v3_mfcc_encodec_bridge_v2" in str(output_path(path,cfg,"prepared_cache"))
    assert checkpoint_schema(cfg,"bridge").endswith("bridge-v2")
    assert checkpoint_schema(cfg,"audio_c").endswith("teacher-v2")


def test_active_cmvn_content_removes_c0_and_resamples_only_active_support():
    mfcc=np.arange(40*10,dtype=np.float32).reshape(40,10)
    active=np.zeros(10,dtype=bool);active[2:7]=True
    content,start,end=_bridge_content_target(mfcc,active,np.ones(10,dtype=bool),161)
    assert content.shape==(39,161)
    assert (start,end)==(2,6)
    assert np.allclose(content[:,0],mfcc[1:,2])
    assert np.allclose(content[:,-1],mfcc[1:,6])


def test_p_bank_contains_actual_trials_not_pointwise_median():
    p=np.stack([np.full((161,3),float(index),dtype=np.float32) for index in range(6)])
    bank,duration,keys=_p_medoid_bank(p,np.arange(6,dtype=np.float32),np.asarray([f"k{index}" for index in range(6)]),4)
    assert bank.shape==(4,161,3)
    assert all(any(np.array_equal(item,source) for source in p) for item in bank)
    assert duration.shape==(4,) and keys.shape==(4,)


def test_audio_c_keeps_codebook_order_and_has_no_parallel_code_heads():
    torch.manual_seed(3)
    model=AudioCEncoder(embedding_dimension=8,dimension=32,heads=4,stem_layers=1,local_layers=1,dropout=0,speakers=2).eval()
    codes=torch.randint(0,1024,(2,8,192));mask=torch.ones(2,192,dtype=torch.bool)
    one=model(codes,mask);two=model(codes[:,torch.tensor([1,0,2,3,4,5,6,7])],mask)
    assert one.local.shape==(2,96,32)
    assert not torch.allclose(one.local,two.local)
    assert not any("code_head" in name for name,_ in model.named_parameters())


def test_c_local_decoder_and_eeg_forward_have_the_fixed_public_contract():
    decoder=SharedContentMFCCDecoder(dimension=32,heads=4,layers=1,dropout=0)
    content,diagnostics=decoder(torch.randn(2,96,32),torch.ones(2,96,dtype=torch.bool))
    assert content.shape==(2,39,161)
    assert diagnostics["attention"].shape==(2,161,96)
    assert tuple(inspect.signature(EEGCEncoder.forward).parameters)==("self","eeg","channel_xyz","channel_mask","time_mask")
    eeg=EEGCEncoder(dimension=32,heads=4,layers=1,local_layers=1,dropout=0)
    state=eeg(torch.randn(2,4,128),torch.randn(2,4,3),torch.ones(2,4,dtype=torch.bool),torch.ones(2,128,dtype=torch.bool))
    assert state.local.shape==(2,96,32)


def test_same_label_trials_are_masked_from_local_contrastive_negatives():
    torch.manual_seed(4)
    value=torch.randn(3,6,8);mask=torch.ones(3,6,dtype=torch.bool)
    loss,scores=masked_token_infonce(value,value,mask,mask,["a","a","b"])
    assert torch.isfinite(loss)
    assert scores[0,1] < -1e3 and scores[1,0] < -1e3
    assert scores.diag().argmax().item() in (0,1,2)


def test_m1_folds_are_disjoint_and_hold_one_trial_per_label():
    indices=np.arange(50,dtype=np.int32);keys=np.asarray([f"k{index:02d}" for index in indices]);labels=np.asarray([f"l{index//5}" for index in indices])
    folds=micro_generalization_folds(indices,keys,labels)
    assert len(folds)==5
    all_held=[]
    for train,held in folds:
        assert len(train)==40 and len(held)==10
        assert not set(train.tolist())&set(held.tolist())
        assert len(set(labels[held].tolist()))==10
        all_held.extend(held.tolist())
    assert sorted(all_held)==indices.tolist()


def test_continuous_bridge_has_one_latent_output_and_no_code_logits():
    bridge=ContinuousEnCodecBridge(latent_dimension=128,dimension=32,blocks=2)
    latent=bridge(torch.randn(2,39,161),torch.randn(2,161,3),torch.randn(2,192),torch.ones(2))
    assert latent.shape==(2,128,192)
    assert not any("codebook" in name or "logit" in name for name,_ in bridge.named_parameters())


def test_label_grouped_batch_sampler_keeps_multiple_labels_and_repeats():
    class Items:
        def __init__(self):self.values=[{"label":str(index//3)} for index in range(12)]
        def __len__(self):return len(self.values)
        def __getitem__(self,index):return self.values[index]
    sampler=LabelGroupedBatchSampler(Items(),batch_size=6,seed=31)
    batch=next(iter(sampler));labels=[Items()[index]["label"] for index in batch]
    assert len(batch)==6 and len(set(labels))>=2
