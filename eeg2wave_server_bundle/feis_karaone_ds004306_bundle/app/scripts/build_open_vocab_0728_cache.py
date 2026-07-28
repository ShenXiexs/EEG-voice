#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import HubertModel

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0, str(APP))

from src.open_vocab_0728.data import Context, internal_split, load_context, normalize_label
from src.open_vocab_0728.lineage import build_lineage
from src.open_vocab_0728.runtime import default_device, load_config, resolve_config_path, sha256_file, write_json
from src.open_vocab_0728.vocoder import mel_filterbank


class HubertTeacher:
    def __init__(self, reference: Path, *, layer: int, device: torch.device):
        self.model = HubertModel.from_pretrained(str(reference), local_files_only=True, output_hidden_states=True).to(device).eval()
        self.layer = int(layer); self.device = device
    @torch.no_grad()
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        value = torch.from_numpy(waveform).to(self.device).unsqueeze(0)
        hidden = self.model(value, output_hidden_states=True).hidden_states[self.layer]
        token = F.interpolate(hidden.transpose(1,2), size=50, mode="linear", align_corners=False).transpose(1,2)
        return token.squeeze(0).detach().cpu().numpy().astype(np.float32)


def sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value,dtype=np.float32).tobytes()).hexdigest()


def active_bounds(wave: np.ndarray, cfg: dict[str, Any]) -> tuple[int,int]:
    audio=cfg["audio"]; window=int(audio["sample_rate"]*audio["active_window_ms"]/1000); hop=int(audio["sample_rate"]*audio["active_hop_ms"]/1000)
    if len(wave)<window: return 0,len(wave)
    rms=np.array([np.sqrt(np.mean(np.square(wave[i:i+window]))+1e-12) for i in range(0,len(wave)-window+1,hop)])
    db=20*np.log10(np.maximum(rms,1e-8)); threshold=max(np.percentile(db,10)+audio["active_noise_margin_db"],db.max()-audio["active_peak_margin_db"])
    active=np.flatnonzero(db>=threshold)
    if not len(active): return 0,len(wave)
    # Merge short silent gaps by returning the outer active interval; preserve context afterward.
    start=max(0,int(active[0]*hop-audio["active_context_ms"]*audio["sample_rate"]/1000)); end=min(len(wave),int(active[-1]*hop+window+audio["active_context_ms"]*audio["sample_rate"]/1000))
    return start,max(start+1,end)


def prepared_audio(path: Path, cfg: dict[str,Any]) -> tuple[np.ndarray,np.ndarray,float,dict[str,Any]]:
    sr,waveform=wavfile.read(path)
    if np.issubdtype(waveform.dtype,np.integer): waveform=waveform.astype(np.float32)/max(float(np.iinfo(waveform.dtype).max),1.0)
    else: waveform=waveform.astype(np.float32)
    if waveform.ndim>1: waveform=waveform.mean(-1)
    target=int(cfg["audio"]["sample_rate"])
    if sr!=target: waveform=resample_poly(waveform,target,int(sr)).astype(np.float32)
    begin,end=active_bounds(waveform,cfg); active=waveform[begin:end]
    maximum=int(cfg["audio"]["max_samples"]); clipped=active[:maximum]
    gain=min(10.0,float(cfg["audio"]["target_rms"])/(float(np.sqrt(np.mean(clipped**2)+1e-12))))
    normalized=clipped*gain; padded=np.zeros(maximum,dtype=np.float32); padded[:len(normalized)]=normalized
    activity=np.zeros(400,dtype=bool); active_frames=max(1,min(400,int(np.ceil(len(normalized)/160)))); activity[:active_frames]=True
    metadata={"source_audio_sha256":sha256_file(path),"segment_pcm_sha256":sha256_bytes(normalized),"active_start_sample":begin,"active_end_sample":end,"native_sample_count":len(waveform),"native_rms":float(np.sqrt(np.mean(waveform**2)+1e-12)),"normalization_gain":gain,"active_duration_seconds":len(normalized)/target}
    return padded,activity,metadata["active_duration_seconds"],metadata


def log_mel(waveform: np.ndarray, cfg: dict[str,Any]) -> np.ndarray:
    audio=cfg["audio"]; value=torch.from_numpy(waveform); window=torch.hann_window(int(audio["win_length"]))
    spec=torch.stft(value,n_fft=int(audio["n_fft"]),hop_length=int(audio["hop_length"]),win_length=int(audio["win_length"]),window=window,return_complex=True)
    power=spec.abs().pow(2); filt=mel_filterbank(sample_rate=int(audio["sample_rate"]),n_fft=int(audio["n_fft"])); mel=filt@power
    mel=F.interpolate(mel.unsqueeze(0),size=int(audio["mel_frames"]),mode="linear",align_corners=False).squeeze(0)
    return torch.clamp(10*torch.log10(mel.clamp_min(1e-10)),-80,0).numpy().astype(np.float32)


def cache_row(context: Context, row: dict[str,str], teacher: HubertTeacher, cfg: dict[str,Any]) -> dict[str,Any]:
    waveform,activity,duration,metadata=prepared_audio(context.eeg_root/row["audio_relpath"],cfg)
    mel=log_mel(waveform,cfg); hubert=teacher.encode(waveform)
    with np.load(context.eeg_root/row["eeg_relpath"],allow_pickle=False) as payload:
        eeg=np.asarray(payload["eeg"][int(row["eeg_row"])],dtype=np.float32); valid=int(payload["valid_lengths"][int(row["eeg_row"])])
    montage_id=context.recording_to_montage[row["eeg_relpath"]]; montage=context.montages[montage_id]
    if eeg.shape[0]!=len(montage.names) or not np.isfinite(eeg).all() or not np.isfinite(montage.xyz).all(): raise ValueError(f"non-finite or mismatched EEG: {row['sample_key']}")
    return {"sample_key":row["sample_key"],"audio_key":row["audio_key"],"label":normalize_label(row["label"]),"subject":row["subject_group_id"],"eeg":eeg,"channel_xyz":montage.xyz,"channel_mask":np.ones(eeg.shape[0],dtype=bool),"time_mask":np.arange(eeg.shape[1])<valid,"hubert":hubert,"hubert_mask":np.arange(50)<max(1,min(50,int(np.ceil(duration/4*50)))),"mel":mel,"activity":activity,"duration":np.float32(duration),"metadata":metadata}


def write_split(path: Path, values: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays={name:np.stack([value[name] for value in values]) for name in ("eeg","channel_xyz","channel_mask","time_mask","hubert","hubert_mask","mel","activity")}
    arrays.update({"sample_keys":np.asarray([v["sample_key"] for v in values]),"audio_keys":np.asarray([v["audio_key"] for v in values]),"labels":np.asarray([v["label"] for v in values]),"subjects":np.asarray([v["subject"] for v in values]),"duration":np.asarray([v["duration"] for v in values],dtype=np.float32)})
    np.savez_compressed(path,**arrays)


def main() -> None:
    parser=argparse.ArgumentParser(description="Build independent KaraOne-only v0728 cache v3")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--device",default=None); parser.add_argument("--force",action="store_true")
    args=parser.parse_args(); config_path,cfg=load_config(args.config); context=load_context(config_path,cfg); root=resolve_config_path(config_path,cfg["paths"]["cache_root"])
    marker=root/"index.json"
    if marker.exists() and not args.force:
        print(f"[0728 cache] already present: {root}"); return
    device=default_device(args.device); teacher=HubertTeacher(resolve_config_path(config_path,cfg["teachers"]["hubert_model"]),layer=int(cfg["teachers"]["hubert_layer"]),device=device)
    split_map=internal_split(context.rows,seed=int(cfg["data"]["internal_split_seed"]),development_subjects=context.development_subjects)
    output={"train":[],"validation":[],"locked_test":[],"diagnostic":[]}; records={}
    for row in tqdm(context.rows,desc="[0728 cache]",unit="trial"):
        split=split_map.get(row["sample_key"],"diagnostic")
        value=cache_row(context,row,teacher,cfg); output[split].append(value); records[row["sample_key"]]={**{k:value[k] for k in ("sample_key","audio_key","label","subject","duration")},**value["metadata"],"split":split}
    for split,values in output.items():
        if values: write_split(root/f"records_{split}.npz",values)
    lineage=build_lineage(config_path,cfg,manifest=context.manifest_path,split=context.split_path,montage=context.montage_path)
    index={"schema_version":"openvoice-0728-cache-v3","lineage":lineage.as_dict(),"records":records,"counts":{key:len(value) for key,value in output.items()},"labels":sorted({v["label"] for values in output.values() for v in values}),"locked_test_physically_isolated":True}
    write_json(marker,index); write_json(root/"audit.json",{"passed":True,"counts":index["counts"],"finite":True,"frequency_axis_resized":False})
    print(json.dumps(index["counts"],sort_keys=True))

if __name__=="__main__": main()
