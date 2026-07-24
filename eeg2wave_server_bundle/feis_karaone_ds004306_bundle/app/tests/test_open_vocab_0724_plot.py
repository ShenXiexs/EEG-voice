from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_open_vocab_0724_pairs import (
    parse_modes,
    render_comparisons,
    stacked_energy,
)
from src.open_vocab_0722.audio_io import write_wav


def test_comparison_plot_uses_v0724_manifest_and_numerical_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthesis"
    stem = "karaone_mm05_001"
    mode = "correct_content_correct_realization"
    sample_rate = 16_000
    time = np.arange(3_200, dtype=np.float32) / sample_rate
    reference = 0.12 * np.sin(2.0 * np.pi * 220.0 * time)
    candidate = 0.10 * np.sin(2.0 * np.pi * 230.0 * time)

    write_wav(root / "reference" / f"{stem}.wav", reference, sample_rate)
    write_wav(root / mode / f"{stem}.wav", candidate, sample_rate)
    np.save(root / "reference" / f"{stem}.mel.npy", np.full((80, 20), -30.0))
    np.save(root / mode / f"{stem}.mel.npy", np.full((80, 16), -35.0))
    (root / "synthesis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "openvoice-0724-synthesis-v1",
                "dataset": "karaone",
                "split": "test",
                "records": [
                    {
                        "sample_key": "karaone:MM05:001",
                        "stem": stem,
                        "label": "/IY/",
                        "metrics": {
                            mode: {
                                "morphology_ssim": 0.61,
                                "soft_dtw_divergence": 0.23,
                                "native_log_mel_mae_db": 4.2,
                                "predicted_duration_error_seconds": 0.08,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "comparison_pairs"
    summary = render_comparisons(
        root,
        output,
        modes=(mode,),
        limit=-1,
        dpi=50,
    )

    assert summary["plots_written"] == 1
    assert summary["metrics_use_png_pixels"] is False
    assert summary["frequency_axis_scaled"] is False
    assert (output / f"{stem}.png").is_file()
    saved = json.loads((output / "comparison_manifest.json").read_text())
    assert saved["energy_panels"] == {
        "explicit": "reference cache log-mel / predicted condition log-mel",
        "decoded": "reference WAV log-mel / reconstructed WAV log-mel",
    }

    (root / mode / f"{stem}.mel.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete v0724 synthesis output"):
        render_comparisons(
            root,
            tmp_path / "incomplete_comparison_pairs",
            modes=(mode,),
            limit=-1,
            dpi=50,
        )


def test_plot_helper_preserves_frequency_axis_and_requires_modes() -> None:
    reference = np.full((80, 4), -20.0, dtype=np.float32)
    candidate = np.full((80, 6), -30.0, dtype=np.float32)
    stacked = stacked_energy(reference, candidate)
    assert stacked.shape == (160, 6)
    assert np.all(stacked[:80, 4:] == -80.0)
    assert parse_modes("content_only, zero_eeg") == ("content_only", "zero_eeg")
