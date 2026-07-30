#!/usr/bin/env python3
"""Apply DeepFilterNet only to explicitly selected trials and gate preservation."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))

from src.open_vocab_v3.data import _read_waveform
from src.open_vocab_v3.denoise import DeepFilterNetEnhancer,envelope_lag_ms,truthy,vad_boundary_seconds,waveform_sha256
from src.open_vocab_v3.hubert import HubertMetric,dtw_cosine
from src.open_vocab_v3.runtime import default_device,load_config,output_path,write_json
from src.open_vocab_v3.speaker import ECAPAEncoder


def csv_rows(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return [dict(row) for row in csv.DictReader(handle)]


def main()->None:
    parser=argparse.ArgumentParser(description="Selective DeepFilterNet v3 branch")
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--device",default="cpu");args=parser.parse_args()
    config_path,cfg=load_config(args.config);selection_path=output_path(config_path,cfg,"denoise_selection")
    selected=[row for row in csv_rows(selection_path) if truthy(row.get("apply",False))]
    manifest_path=output_path(config_path,cfg,"denoise_manifest");destination=output_path(config_path,cfg,"denoise_root");destination.mkdir(parents=True,exist_ok=True)
    if not selected:
        write_json(manifest_path,{"schema_version":"openvoice-v3-selective-denoise-v1","backend":str(cfg["denoise"]["backend"]),"selected":0,"accepted":0,"records":[]})
        print("[v3 denoise] no trial has apply=1; raw audio remains active",flush=True);return
    manifest={row["sample_key"]:row for row in csv_rows(output_path(config_path,cfg,"unified_manifest")) if row.get("dataset")=="karaone"}
    root=output_path(config_path,cfg,"audio_root");device=default_device(args.device);enhancer=DeepFilterNetEnhancer(cfg)
    hubert=HubertMetric(output_path(config_path,cfg,"hubert_root"),layer=int(cfg["teachers"]["hubert_layer"]),device=device)
    ecapa=ECAPAEncoder(source=str(cfg["speaker"]["model_id"]),savedir=output_path(config_path,cfg,"speaker_model_root"),device=device)
    records=[]
    for item in selected:
        key=str(item["sample_key"])
        if key not in manifest:raise KeyError(f"denoise selection is not a KaraOne sample: {key}")
        raw,rate=_read_waveform(root/manifest[key]["audio_relpath"]);enhanced=enhancer.enhance(raw,rate)
        output=destination/f"{key.replace(':','__')}__deepfilternet.wav";wavfile.write(output,rate,(np.clip(enhanced,-1,1)*32767).astype(np.int16))
        raw_start,raw_end=vad_boundary_seconds(raw,rate);new_start,new_end=vad_boundary_seconds(enhanced,rate)
        lag=abs(envelope_lag_ms(raw,enhanced,rate));duration_change=abs(len(raw)-len(enhanced))*1000.0/rate
        boundary=max(abs(raw_start-new_start),abs(raw_end-new_end))*1000.0
        hubert_score=dtw_cosine(hubert.encode(raw,rate),hubert.encode(enhanced,rate))
        ecapa_score=float(ecapa.encode(raw)@ecapa.encode(enhanced))
        checks={"envelope_lag":lag<=float(cfg["denoise"]["max_envelope_lag_ms"]),"vad_boundary":boundary<=float(cfg["denoise"]["max_vad_boundary_shift_ms"]),"duration":duration_change<=float(cfg["denoise"]["max_duration_change_ms"]),"hubert":hubert_score>=float(cfg["denoise"]["min_hubert_dtw_cosine"]),"ecapa":ecapa_score>=float(cfg["denoise"]["min_ecapa_cosine"])}
        records.append({"sample_key":key,"source_wav":str(root/manifest[key]["audio_relpath"]),"enhanced_wav":str(output),"source_pcm_sha256":waveform_sha256(raw),"enhanced_pcm_sha256":waveform_sha256(enhanced),"envelope_lag_ms":lag,"vad_boundary_shift_ms":boundary,"duration_change_ms":duration_change,"hubert_dtw_cosine":hubert_score,"ecapa_cosine":ecapa_score,"checks":checks,"accepted":bool(all(checks.values())),"selection":item})
    try:version=importlib.metadata.version("deepfilternet")
    except importlib.metadata.PackageNotFoundError:version="unknown"
    write_json(manifest_path,{"schema_version":"openvoice-v3-selective-denoise-v1","backend":str(cfg["denoise"]["backend"]),"package_version":version,"model_identity":enhancer.model_identity,"processing_sample_rate":int(cfg["denoise"]["processing_sample_rate"]),"compensate_delay":bool(cfg["denoise"]["compensate_delay"]),"selected":len(records),"accepted":sum(bool(row["accepted"]) for row in records),"records":records})
    print(f"[v3 denoise] selected={len(records)} accepted={sum(bool(row['accepted']) for row in records)} manifest={manifest_path}",flush=True)


if __name__=="__main__":main()
