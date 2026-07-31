#!/usr/bin/env python3
"""Build the fit-only EnCodec-ID cache after T0 passes.

This cache intentionally contains no held-out trial.  The sample-key index is
checked by every consumer so a cache cannot silently be paired with a rebuilt
or legacy prepared dataset.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP)) if str(APP) not in sys.path else None
from src.open_vocab_v3.data import _accepted_denoise_paths,_read_waveform,light_prepare_waveform,load_prepared
from src.open_vocab_0724.audio_features import AudioPreparationConfig
from src.open_vocab_v3.encodec_content import EnCodecGenerator,SCHEMA
from src.open_vocab_v3.runtime import default_device,load_config,output_path,sha256_file,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--force',action='store_true');a=p.parse_args();cp,cfg=load_config(a.config);dst=output_path(cp,cfg,'encodec_cache');manifest=output_path(cp,cfg,'encodec_cache_manifest')
 if dst.exists() and not a.force:raise RuntimeError('EnCodec cache exists; use --force only after deliberately rebuilding this schema')
 records=load_prepared(output_path(cp,cfg,'prepared_cache'));indices=np.flatnonzero((records.roles=='fit')&records.arrays['fit_eligible'].astype(bool));
 if len(indices)!=1016:raise RuntimeError(f'fit-only EnCodec cache expected 1016 eligible records, found {len(indices)}')
 with (cp.parent/cfg['data']['unified_manifest']).resolve().open(newline='',encoding='utf-8') as h: paths={str(x['sample_key']):str(x['audio_relpath']) for x in csv.DictReader(h) if x.get('dataset')=='karaone'}
 prep=AudioPreparationConfig(sample_rate=16000,max_active_seconds=float(cfg['audio']['max_active_seconds']),target_rms=float(cfg['audio']['target_rms']));denoised=_accepted_denoise_paths(cp,cfg);codec=EnCodecGenerator(output_path(cp,cfg,'encodec_adapted_root'),device=default_device(a.device),bandwidth=float(cfg['audio']['encodec_bandwidth']));codes=[];masks=[];root=(cp.parent/cfg['data']['audio_root']).resolve()
 for index in tqdm(indices.tolist(),desc='[v3 EnCodec cache]',unit='trial',dynamic_ncols=True):
  key=str(records.arrays['sample_keys'][index]);wave,rate=_read_waveform(denoised.get(key,root/paths[key]));prepared,_=light_prepare_waveform(wave,rate,prep);value=torch.from_numpy(prepared.waveform[:prepared.valid_samples]).unsqueeze(0);one,mask=codec.encode(value);one=one.cpu().numpy()[0];mask=mask.cpu().numpy()[0]
  if one.shape[0]!=8 or one.shape[1]>192:raise RuntimeError(f'bad EnCodec shape {one.shape} for {key}')
  padded=np.zeros((8,192),dtype=np.int16);padded[:,:one.shape[1]]=one;padded_mask=np.zeros(192,dtype=bool);padded_mask[:mask.size]=mask;codes.append(padded);masks.append(padded_mask)
 dst.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst,schema=np.asarray(SCHEMA),source_prepared_sha256=np.asarray(sha256_file(output_path(cp,cfg,'prepared_cache'))),source_indices=indices.astype(np.int32),sample_keys=records.arrays['sample_keys'][indices].astype(str),encodec_codes=np.stack(codes),encodec_mask=np.stack(masks))
 write_json(manifest,{'schema_version':SCHEMA,'scope':'fit_only','n':len(indices),'cache':str(dst),'sha256':sha256_file(dst),'prepared_cache_sha256':sha256_file(output_path(cp,cfg,'prepared_cache')),'shape':{'codes':[len(indices),8,192],'mask':[len(indices),192]}});print(dst,flush=True)
if __name__=='__main__':main()
