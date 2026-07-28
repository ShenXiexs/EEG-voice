from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.open_vocab_0728.data import internal_split
from src.open_vocab_0728.metrics import fit_stss
from src.open_vocab_0728.model import DualLatentAudioModel, DualLatentEEGToSpeech, EEGEncoder
from src.open_vocab_0728.runtime import ensure_output_firewall
from scripts.validate_open_vocab_0728_metric import metric_gate


def test_four_input_label_free_generation_and_zero_silence() -> None:
    audio=DualLatentAudioModel(); eeg=EEGEncoder(); model=DualLatentEEGToSpeech(eeg,audio)
    with torch.no_grad():
        eeg.evidence[-1].bias.fill_(-30.0)
    output=model.generate(torch.zeros(2,14,1280),torch.zeros(2,14,3),torch.ones(2,14,dtype=torch.bool),torch.ones(2,1280,dtype=torch.bool))
    assert output.linguistic_latent.shape==(2,50,128)
    assert output.realization_latent.shape==(2,50,64)
    assert output.log_mel.shape==(2,80,400)
    assert torch.allclose(output.log_mel,torch.full_like(output.log_mel,-80),atol=1e-4)
    assert output.activity_mask.shape==(2,400)


def test_deterministic_15_12_11_internal_split() -> None:
    rows=[]
    for count in (15,12,11):
        subject=f"karaone:S{count}"
        for trial in range(count): rows.append({"subject_group_id":subject,"label":"/tiy/","sample_key":f"{subject}:{trial}"})
    result=internal_split(rows,seed=15,development_subjects=["karaone:S15","karaone:S12","karaone:S11"])
    assert list(result.values()).count("train")==25
    assert list(result.values()).count("validation")==6
    assert list(result.values()).count("locked_test")==7
    assert result==internal_split(rows,seed=15,development_subjects=["karaone:S15","karaone:S12","karaone:S11"])


def test_stss_prefers_tolerable_shift_to_silence() -> None:
    reference=np.full((80,400),-80,np.float32); reference[20:45,100:250]=-20
    shifted=np.full_like(reference,-80); shifted[:,20:]=reference[:,:-20]
    silence=np.full_like(reference,-80)
    stss,report=fit_stss([(reference,shifted)]*6,[(reference,silence)]*6)
    assert report["auc"]>.9
    assert stss.score(reference,shifted)>stss.score(reference,silence)


def test_metric_gate_allows_auc_ceiling_tie_but_records_it() -> None:
    config={"evaluation":{"metric_positive_auc_minimum":.90,"metric_gain_over_best_component":.02}}
    gate=metric_gate({"auc":1.0,"best_component_auc":1.0,"pairwise_accuracy":1.0},config)
    assert gate["passed"]
    assert gate["ceiling_limited"]
    assert gate["required_gain"]==0.0


def test_output_firewall_rejects_v0724() -> None:
    with tempfile.TemporaryDirectory() as directory:
        config=Path(directory)/"config.yaml"; config.write_text("x")
        try:
            ensure_output_firewall(config,{"paths":{"output_root":"../artifacts/open_vocab_0724_bad"}})
        except ValueError:
            return
    raise AssertionError("protected output namespace was accepted")
