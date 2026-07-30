#!/usr/bin/env python3
"""Export training-pair WAVs and MFCC/Mel diagnostics after v3 validation passes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from scripts.train_open_vocab_v3 import load_audio, load_eeg
from src.open_vocab_v3.data import V3Dataset, _accepted_denoise_paths, channel_shuffled_eeg, collate, light_prepare_waveform, load_prepared, time_shuffled_eeg
from src.open_vocab_v3.runtime import capture_lineage, default_device, load_config, move_batch, output_path, read_json, require_passed_gate, write_json
from src.open_vocab_v3.vocoder import SpeechT5PowerDbHiFiGan, pcm16
from src.open_vocab_0724.audio_features import AudioPreparationConfig


def parse() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Export v3 training pair diagnostics")
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--device",default="cpu")
    parser.add_argument("--limit",type=int,default=0);parser.add_argument("--resume",action="store_true")
    return parser.parse_args()


def manifest_paths(path: Path) -> dict[str,str]:
    with path.open(newline="",encoding="utf-8") as handle:
        return {str(row["sample_key"]):str(row["audio_relpath"]) for row in csv.DictReader(handle) if row.get("dataset")=="karaone"}


def reference(path: Path) -> tuple[np.ndarray,int]:
    import soundfile as sf
    wave,rate=sf.read(path,always_2d=False,dtype="float32")
    return (wave.mean(1) if wave.ndim==2 else wave).astype(np.float32),int(rate)


def light_cleaned_reference(path: Path, cfg: dict) -> tuple[np.ndarray, int]:
    wave, rate = reference(path)
    prepared, _ = light_prepare_waveform(
        wave,
        rate,
        AudioPreparationConfig(
            sample_rate=int(cfg["audio"]["sample_rate"]),
            max_active_seconds=float(cfg["audio"]["max_active_seconds"]),
            target_rms=float(cfg["audio"]["target_rms"]),
        ),
    )
    return prepared.waveform[: max(1, prepared.valid_samples)], int(cfg["audio"]["sample_rate"])


def write_wave(path: Path,wave: np.ndarray,rate: int) -> None:
    wavfile.write(path,rate,(np.clip(wave,-1,1)*32767).astype(np.int16))


def heatmap(path: Path,target_mfcc: np.ndarray,predicted_mfcc: np.ndarray,target_mel: np.ndarray,predicted_mel: np.ndarray,title: str) -> None:
    figure,axes=plt.subplots(2,2,figsize=(12,6),constrained_layout=True)
    for axis,value,name in zip(axes.flat,(target_mfcc,predicted_mfcc,target_mel,predicted_mel),("Target MFCC","EEG MFCC","Target Mel","EEG Mel")):
        image=axis.imshow(value,aspect="auto",origin="lower",interpolation="nearest",cmap="magma")
        axis.set_title(name);axis.set_xlabel("canonical frame");figure.colorbar(image,ax=axis,fraction=.046,pad=.04)
    figure.suptitle(title);figure.savefig(path,dpi=120);plt.close(figure)


@torch.no_grad()
def main() -> None:
    args=parse();config_path,cfg=load_config(args.config);device=default_device(args.device);started=time.monotonic()
    validation_path=output_path(config_path,cfg,"validation_report")
    fit_gate_path=output_path(config_path,cfg,"fit_gate");fit_gate=require_passed_gate(config_path,cfg,"fit_gate",lineage_artifact_keys=("fit_checkpoint","micro_gate"))
    expected_report_lineage=capture_lineage(config_path,cfg,artifact_keys=("fit_checkpoint","fit_gate","training_review"))
    for report_key in ("validation_report","locked_report","locked_unseen_report"):
        report_path=output_path(config_path,cfg,report_key)
        if not report_path.is_file() or read_json(report_path).get("lineage")!=expected_report_lineage:
            raise RuntimeError(f"refusing pair export: missing or stale report {report_path}")
    records=load_prepared(output_path(config_path,cfg,"prepared_cache"));dataset=V3Dataset(records,("fit",),eligible_only=True)
    eeg,_=load_eeg(config_path,cfg,device,stage="fit");audio,_=load_audio(config_path,cfg,device);vocoder=SpeechT5PowerDbHiFiGan(output_path(config_path,cfg,"vocoder_root"),device=device)
    destination=output_path(config_path,cfg,"pair_root");destination.mkdir(parents=True,exist_ok=True)
    paths=manifest_paths(output_path(config_path,cfg,"unified_manifest"));audio_root=output_path(config_path,cfg,"audio_root");denoised=_accepted_denoise_paths(config_path,cfg)
    fit_keys=list(map(str,fit_gate.get("sample_keys",[])));fit_ranks=list(fit_gate.get("correct",{}).get("paired_rank_per_trial",[]))
    if len(fit_keys)!=len(fit_ranks):raise RuntimeError("fit gate lacks per-trial retrieval ranks")
    rank_by_key={key:int(rank) for key,rank in zip(fit_keys,fit_ranks)}
    rows=[];total=min(len(dataset),args.limit) if args.limit else len(dataset)
    for number,batch in enumerate(tqdm(DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate,num_workers=0),total=total,desc="[v3 pairs] WAV export",unit="pair",dynamic_ncols=True,mininterval=.5)):
        batch=move_batch(batch,device);key=batch["sample_key"][0];stem=destination/key;meta=stem.with_suffix(".json")
        names={name:stem.with_name(f"{stem.name}__{name}.wav") for name in ("cleaned_reference","v0_vocoder_oracle","analytic_mfcc_oracle","cvae_posterior_oracle","cvae_prior_mfcc_oracle","eeg","zero_eeg","time_shuffled","channel_shuffled")}
        figure=stem.with_name(f"{stem.name}__comparison.png")
        if args.resume and meta.is_file() and all(path.is_file() for path in names.values()) and figure.is_file():
            cached=json.loads(meta.read_text())
            if "within_label_trial_retrieval_rank" in cached:
                rows.append(cached);continue
        source=int(np.flatnonzero(records.arrays["sample_keys"].astype(str)==key)[0])
        target_mfcc=batch["mfcc"].cpu().numpy()[0];target_mel=batch["mel"].cpu().numpy()[0]
        kwargs=(batch["channel_xyz"].float(),batch["channel_mask"],batch["time_mask"])
        predicted={"eeg":eeg(batch["eeg"].float(),*kwargs)[0],"zero_eeg":eeg(torch.zeros_like(batch["eeg"]).float(),*kwargs)[0],"time_shuffled":eeg(time_shuffled_eeg(batch["eeg"].float(),batch["time_mask"]),*kwargs)[0],"channel_shuffled":eeg(channel_shuffled_eeg(batch["eeg"].float(),batch["channel_mask"]),*kwargs)[0]}
        voice=batch["canonical_voice"].float();mean=batch["canonical_mfcc_mean"].float();std=batch["canonical_mfcc_std"].float()
        prior=audio.generate(batch["mfcc"].float(),voice,mean,std,stochastic=False);posterior=audio.reconstruct(batch["mfcc"].float(),voice,mean,std,batch["mel"].float(),stochastic=False)
        mel={"analytic_mfcc_oracle":prior["analytic_mel"],"cvae_posterior_oracle":posterior["mel"],"cvae_prior_mfcc_oracle":prior["mel"]}
        for name,value in predicted.items():mel[name]=audio.generate(value,voice,mean,std,stochastic=False)["mel"]
        generated={"v0_vocoder_oracle":pcm16(vocoder.synthesize(torch.from_numpy(records.arrays["vocoder_mel"][source:source+1]).to(device))[0]),**{name:pcm16(vocoder.synthesize(value)[0]) for name,value in mel.items()}}
        ref,rate=light_cleaned_reference(denoised.get(key,audio_root/paths[key]),cfg);write_wave(names["cleaned_reference"],ref,rate)
        for name,wave in generated.items():write_wave(names[name],wave,int(cfg["vocoder"]["sample_rate"]))
        heatmap(figure,target_mfcc,predicted["eeg"].cpu().numpy()[0],target_mel,mel["eeg"].cpu().numpy()[0],key)
        record = {
            "sample_key": key,
            "label": batch["label"][0],
            "subject": batch["subject"][0],
            "role": batch["role"][0],
            "cleaned_reference_wav": str(names["cleaned_reference"]),
            **{f"{name}_wav": str(path) for name, path in names.items() if name != "cleaned_reference"},
            "comparison_png": str(figure),
            "mfcc_mae": float(np.mean(np.abs(predicted["eeg"].cpu().numpy()[0] - target_mfcc))),
            "within_label_trial_retrieval_rank": rank_by_key[key],
        }
        write_json(meta,record);rows.append(record)
        if args.limit and number+1>=args.limit:break
    with (destination/"manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ["sample_key"]);writer.writeheader();writer.writerows(rows)
    elapsed=time.monotonic()-started
    write_json(destination/"export_manifest.json",{"schema_version":"openvoice-v3-training-pairs-v3-cvae","eligible_training_pairs":len(dataset),"exported_pairs":len(rows),"complete":len(rows)==len(dataset) if not args.limit else False,"limit":int(args.limit),"elapsed_seconds":elapsed,"seconds_per_pair":elapsed/max(len(rows),1),"fit_gate":str(fit_gate_path),"validation_report":str(validation_path),"lineage":capture_lineage(config_path,cfg,artifact_keys=("fit_checkpoint","fit_gate","training_review","validation_report","locked_report","locked_unseen_report")),"records":rows})
    print(destination/"manifest.csv",flush=True)


if __name__=="__main__":main()
