#!/usr/bin/env python3
"""Create the raw-audio audit and a persistent human-editable denoise queue."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0724.audio_features import AcousticFeatureConfig, AudioPreparationConfig, extract_acoustic_features
from src.open_vocab_v3.data import _read_waveform, light_prepare_waveform
from src.open_vocab_v3.runtime import load_config, output_path, write_json


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle) if row.get("dataset") == "karaone"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_key"])
        writer.writeheader(); writer.writerows(rows)


def existing_selection(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row["sample_key"]): dict(row) for row in csv.DictReader(handle)}


def main() -> None:
    parser=argparse.ArgumentParser(description="Audit KaraOne raw audio before optional v3 denoising")
    parser.add_argument("--config",type=Path,required=True);args=parser.parse_args()
    config_path,cfg=load_config(args.config)
    rows=read_manifest(output_path(config_path,cfg,"unified_manifest"));root=output_path(config_path,cfg,"audio_root")
    prep_cfg=AudioPreparationConfig(sample_rate=int(cfg["audio"]["sample_rate"]),max_active_seconds=float(cfg["audio"]["max_active_seconds"]),target_rms=float(cfg["audio"]["target_rms"]))
    feature_cfg=AcousticFeatureConfig(sample_rate=int(cfg["audio"]["sample_rate"]),n_fft=int(cfg["audio"]["n_fft"]),mel_bins=int(cfg["audio"]["mel_bins"]),max_frames=int(cfg["audio"]["canonical_frames"]),min_db=float(cfg["audio"]["mel_db_min"]),max_db=float(cfg["audio"]["mel_db_max"]))
    manual=set(map(str,cfg["audio"].get("manual_review_sample_keys",())))
    audit=[]
    for row in tqdm(rows,desc="[v3 raw audit]",unit="trial",dynamic_ncols=True):
        key=str(row["sample_key"]);wave,rate=_read_waveform(root/row["audio_relpath"]);prepared,_=light_prepare_waveform(wave,rate,prep_cfg)
        features=extract_acoustic_features(prepared.waveform,valid_samples=prepared.valid_samples,config=feature_cfg)
        valid=features.frame_valid_mask.astype(bool);active=features.activity_mask.astype(bool)&valid;inactive=valid&~active
        contrast=float(np.median(features.log_rms_dbfs[active])-np.median(features.log_rms_dbfs[inactive])) if active.any() and inactive.any() else float("nan")
        low=bool(np.isfinite(contrast) and contrast<float(cfg["audio"]["low_contrast_db_threshold"]));named=key in manual
        audit.append({"sample_key":key,"audio_relpath":row["audio_relpath"],"subject":row.get("subject",""),"label":row.get("label",""),"native_sample_rate":rate,"active_duration_seconds":prepared.active_duration_seconds,"active_inactive_contrast_db":contrast,"low_contrast":low,"manual_review_required":named,"denoise_candidate":low or named})
    write_csv(output_path(config_path,cfg,"raw_audio_audit_csv"),audit)
    write_json(output_path(config_path,cfg,"raw_audio_audit"),{"schema_version":"openvoice-v3-raw-audio-audit-v1","n":len(audit),"candidate_count":sum(bool(row["denoise_candidate"]) for row in audit),"records":audit})
    selection_path=output_path(config_path,cfg,"denoise_selection");old=existing_selection(selection_path);selection=[]
    for row in audit:
        if not row["denoise_candidate"] and row["sample_key"] not in old:continue
        previous=old.get(str(row["sample_key"]),{})
        automatic=bool(cfg["denoise"].get("auto_select_low_contrast",False) and row["low_contrast"] and not row["manual_review_required"])
        selection.append({"sample_key":row["sample_key"],"apply":previous.get("apply","1" if automatic else "0"),"review_status":previous.get("review_status","pending"),"reason":previous.get("reason","manual_anomaly" if row["manual_review_required"] else "low_contrast"),"notes":previous.get("notes","")})
    write_csv(selection_path,selection)
    print(f"[v3 raw audit] n={len(audit)} candidates={len(selection)} selection={selection_path}",flush=True)


if __name__=="__main__":main()
