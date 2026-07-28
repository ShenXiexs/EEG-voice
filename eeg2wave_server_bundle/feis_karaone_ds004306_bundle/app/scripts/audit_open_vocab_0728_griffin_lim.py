#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))

from src.open_vocab_0728.data import CacheV3
from src.open_vocab_0728.metrics import envelope_correlation, load_stss, ms_ssim
from src.open_vocab_0728.runtime import default_device, load_config, resolve_config_path, write_json
from src.open_vocab_0728.vocoder import griffin_lim_from_log_mel, mel_filterbank


def log_mel(wave:torch.Tensor,cfg:dict)->torch.Tensor:
    a=cfg["audio"]; window=torch.hann_window(int(a["win_length"]),device=wave.device); spec=torch.stft(wave,n_fft=int(a["n_fft"]),hop_length=int(a["hop_length"]),win_length=int(a["win_length"]),window=window,return_complex=True); mel=mel_filterbank(sample_rate=int(a["sample_rate"]),n_fft=int(a["n_fft"]),device=wave.device)@spec.abs().pow(2); return torch.clamp(10*torch.log10(F.interpolate(mel.unsqueeze(0),size=400,mode="linear",align_corners=False).squeeze(0).clamp_min(1e-10)),-80,0)


@torch.no_grad()
def main()->None:
    parser=argparse.ArgumentParser(description="Audit fixed Griffin–Lim ceiling before EEG training")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--limit",type=int,default=32); parser.add_argument("--device",default=None); args=parser.parse_args(); config,cfg=load_config(args.config); device=default_device(args.device); cache=CacheV3(resolve_config_path(config,cfg["paths"]["cache_root"]),"validation"); stss=load_stss(resolve_config_path(config,cfg["paths"]["metric_manifest"]))
    mae=[]; ssim=[]; corr=[]; score=[]
    for index in tqdm(range(min(len(cache),args.limit)),desc="[0728 Griffin-Lim ceiling]",unit="trial",mininterval=1.0,disable=False):
        ref=torch.from_numpy(np.asarray(cache.raw["mel"][index])).to(device); wav=griffin_lim_from_log_mel(ref,iterations=int(cfg["audio"]["griffin_lim_iterations"]),seed=index); generated=log_mel(wav,cfg).cpu().numpy(); target=ref.cpu().numpy(); mae.append(float(np.abs(generated-target).mean())); ssim.append(ms_ssim(generated,target)); corr.append(envelope_correlation(generated,target)); score.append(stss.score(generated,target))
    report={"schema_version":"openvoice-0728-griffin-lim-ceiling-v1","trials":len(mae),"median_log_mel_mae":float(np.median(mae)),"median_msssim":float(np.median(ssim)),"median_envelope_correlation":float(np.median(corr)),"median_stss":float(np.median(score)),"passed":bool(np.median(mae)<=12 and np.median(ssim)>=.60 and np.median(corr)>=.75)}
    output=resolve_config_path(config,cfg["paths"]["output_root"])/"audio"/"metrics"/"griffin_lim_ceiling.json"; write_json(output,report); print(report)
    if not report["passed"]: raise RuntimeError("Griffin–Lim ceiling failed")
if __name__=="__main__": main()
