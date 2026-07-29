from __future__ import annotations

import numpy as np
import torch

from src.open_vocab_0730.data import assign_roles, prosody_from_mel
from src.open_vocab_0730.model import CPMelRenderer, ContentProsodyEEG
from scripts.train_open_vocab_0730 import openai_style_token_clip


def test_fixed_split_roles_are_exhaustive_and_label_free() -> None:
    subjects = np.asarray(["karaone:MM05", "karaone:MM19", "karaone:MM20", "karaone:MM05"])
    labels = np.asarray(["pat", "pat", "pot", "pot"])
    roles = assign_roles(subjects, labels, subject_holdout=("karaone:MM19", "karaone:MM20"), unseen_label="pot")
    assert roles.tolist() == ["fit", "subject_holdout_seen", "subject_and_label_holdout", "label_holdout_seen_subject"]


def test_explicit_prosody_has_only_configured_fields() -> None:
    target = prosody_from_mel(np.full((80, 400), -40, dtype=np.float32), np.ones(400, dtype=bool), 1.5)
    assert target.shape == (66,)
    assert target[0] == 1.5


def test_cp_model_does_not_accept_labels_and_renders_expected_shape() -> None:
    encoder = ContentProsodyEEG(codebook_size=8, dimension=32, heads=4, layers=1)
    renderer = CPMelRenderer(codebook_size=8, dimension=32)
    state = encoder(torch.randn(2, 14, 1280), torch.randn(2, 14, 3), torch.ones(2, 14, dtype=torch.bool), torch.ones(2, 1280, dtype=torch.bool))
    assert state.content_logits.shape == (2, 16, 8)
    assert state.prosody.shape == (2, 66)
    assert renderer(state.content_logits, state.prosody).shape == (2, 80, 400)


def test_openai_style_token_clip_is_symmetric_and_tokenwise() -> None:
    eeg = torch.randn(3, 16, 64, requires_grad=True)
    audio = torch.randn(3, 16, 64)
    global_loss, token_loss = openai_style_token_clip(eeg, audio, torch.ones(3, 16, dtype=torch.bool), torch.tensor(2.0))
    (global_loss + token_loss).backward()
    assert global_loss.item() > 0 and token_loss.item() > 0
    assert eeg.grad is not None
