#!/usr/bin/env python3
"""Create a fit-only cache using the original frozen EnCodec scale contract."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))
from src.open_vocab_0724.audio_features import AudioPreparationConfig
from src.open_vocab_v3.data import _accepted_denoise_paths,_read_waveform,light_prepare_waveform,load_prepared
from src.open_vocab_v3.encodec_rvq_repair import PREPARATION_SCHEMA,SCHEMA,FrozenEnCodecRVQ
from src.open_vocab_v3.runtime import default_device,load_config,output_path,sha256_file,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--force',action='store_true');a=p.parse_args()
    cp,cfg=load_config(a.config);destination=output_path(cp,cfg,'encodec_cache');manifest=output_path(cp,cfg,'encodec_cache_manifest')
    if destination.exists() and not a.force: raise RuntimeError(f'RVQ-repair cache exists: {destination}; use --force only in this timestamped run')
    records=load_prepared(output_path(cp,cfg,'prepared_cache'),expected_schema=PREPARATION_SCHEMA)
    select=(records.roles=='fit')&records.arrays['fit_eligible'].astype(bool);indices=np.flatnonzero(select)
    with output_path(cp,cfg,'unified_manifest').open(newline='',encoding='utf-8') as f: paths={str(x['sample_key']):str(x['audio_relpath']) for x in csv.DictReader(f) if x.get('dataset')=='karaone'}
    prep=AudioPreparationConfig(sample_rate=16000,max_active_seconds=float(cfg['audio']['max_active_seconds']),target_rms=float(cfg['audio']['target_rms']))
    renderer=FrozenEnCodecRVQ(output_path(cp,cfg,'encodec_root'),device=default_device(a.device),bandwidth=float(cfg['audio']['encodec_bandwidth']))
    root=output_path(cp,cfg,'audio_root');denoised=_accepted_denoise_paths(cp,cfg);max_samples=round(float(cfg['audio']['max_active_seconds'])*16000)
    codes=[];masks=[];scales=[];waves=[];wave_masks=[];counts=[]
    for source in tqdm(indices.tolist(),desc='[v3 rvq frozen EnCodec cache]',unit='trial',dynamic_ncols=True):
        key=str(records.arrays['sample_keys'][source]);wave,rate=_read_waveform(denoised.get(key,root/paths[key]));prepared,_=light_prepare_waveform(wave,rate,prep)
        active=torch.from_numpy(prepared.waveform[:prepared.valid_samples]).unsqueeze(0);code,mask,scale=renderer.encode_16k(active)
        if tuple(code.shape[1:2])!=(8,) or code.shape[-1]>192: raise RuntimeError(f'EnCodec contract failed for {key}: {tuple(code.shape)}')
        padded=torch.zeros((1,8,192),dtype=torch.long,device=code.device);code_mask=torch.zeros((1,192),dtype=torch.bool,device=code.device);padded[...,:code.shape[-1]]=code;code_mask[...,:mask.shape[-1]]=mask
        full=np.zeros(max_samples,np.float32);full[:prepared.valid_samples]=prepared.waveform[:prepared.valid_samples];valid=np.zeros(max_samples,bool);valid[:prepared.valid_samples]=True
        codes.append(padded[0].cpu().numpy().astype(np.int16));masks.append(code_mask[0].cpu().numpy());scales.append(scale[0].cpu().numpy().astype(np.float32));waves.append(full);wave_masks.append(valid);counts.append(int(prepared.valid_samples))
    destination.parent.mkdir(parents=True,exist_ok=True);prepared_path=output_path(cp,cfg,'prepared_cache')
    np.savez_compressed(destination,schema=np.asarray(SCHEMA),prepared_cache_sha256=np.asarray(sha256_file(prepared_path)),source_indices=indices.astype(np.int32),sample_keys=records.arrays['sample_keys'][indices].astype(str),encodec_codes=np.stack(codes),encodec_mask=np.stack(masks),audio_scales=np.stack(scales),waveform_16k=np.stack(waves),waveform_mask=np.stack(wave_masks),waveform_samples=np.asarray(counts,np.int32),normalize=np.asarray(renderer.normalize),tokenizer=np.asarray('frozen_encodec_original_config_24khz_6kbps_8x1024'))
    write_json(manifest,{'schema_version':SCHEMA,'n':int(len(indices)),'scope':'fit_only_including_internal_dev','prepared_cache_sha256':sha256_file(prepared_path),'cache':str(destination),'sha256':sha256_file(destination),'encodec_config_sha256':sha256_file(output_path(cp,cfg,'encodec_root')/'config.json'),'normalize':renderer.normalize,'audio_scales_saved':True,'valid_crop_only':True,'shapes':{'codes':[len(indices),8,192],'audio_scales':[len(indices),1],'waveform':[len(indices),max_samples]}})
    print(destination,flush=True)
if __name__=='__main__': main()
