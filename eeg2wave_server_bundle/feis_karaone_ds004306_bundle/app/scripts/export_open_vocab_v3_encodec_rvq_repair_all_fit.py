#!/usr/bin/env python3
"""Infer and export every fit-eligible v3 RVQ pair.

This is deliberately separate from the original 50-pair preview exporter.  It
uses the existing M0b EEG checkpoint for inference over all fit rows; it never
opens validation/test rows and writes to a new output directory.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0,str(APP))
from scripts.train_open_vocab_v3_encodec_rvq_repair import (
    TokenDataset, base_subset, fit_indices, load_cache, load_checkpoint,
    make_models, token_collate,
)
from scripts.export_open_vocab_v3_encodec_rvq_repair_preview import plot
from src.open_vocab_v3.data import V3Dataset, load_prepared, time_shuffled_eeg, channel_shuffled_eeg
from src.open_vocab_v3.encodec_rvq_repair import PREPARATION_SCHEMA, SCHEMA, FrozenEnCodecRVQ
from src.open_vocab_v3.runtime import checkpoint_schema, default_device, load_config, move_batch, output_path, sha256_file, write_json
from torch.utils.data import DataLoader

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--output-root',type=Path,default=None); p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--resume',action='store_true'); p.add_argument('--explore',action='store_true')
    return p.parse_args()

def wav(path,x): sf.write(path,np.asarray(x,np.float32),16000,subtype='PCM_16')

def one_batch(records,cache,mapping,source,device):
    base=V3Dataset(records,('fit',),eligible_only=True); position={int(x):i for i,x in enumerate(base.indices)}[int(source)]
    item=base[position]; row=mapping[int(source)]
    for k in ('encodec_codes','encodec_mask','audio_scales','waveform_16k','waveform_mask','waveform_samples'): item[k]=cache[k][row]
    return move_batch(token_collate([item]),device)

@torch.inference_mode()
def infer_all(records,cache,mapping,model,indices,cfg,device,batch_size):
    dataset=TokenDataset(base_subset(records,indices),cache,mapping)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=token_collate,num_workers=0)
    out={}
    for b in tqdm(loader,desc='[v3 all-fit] EEG inference',unit='batch'):
        b=move_batch(b,device)
        def pred(signal): return model(signal.float(),b['channel_xyz'].float(),b['channel_mask'],b['time_mask'])[0]
        values={'prediction':pred(b['eeg']),'zero':pred(torch.zeros_like(b['eeg'])),'time':pred(time_shuffled_eeg(b['eeg'],b['time_mask'])),'channel':pred(channel_shuffled_eeg(b['eeg'],b['channel_mask']))}
        for i,source in enumerate(b['source_index'].cpu().tolist()):
            out[int(source)]={k:v[i].detach().cpu().numpy().astype(np.float32) for k,v in values.items()}
    return out

@torch.inference_mode()
def export_all(args):
    cp,cfg=load_config(args.config); device=default_device(args.device)
    records=load_prepared(output_path(cp,cfg,'prepared_cache'),expected_schema=PREPARATION_SCHEMA)
    cache,mapping=load_cache(cp,cfg); indices=fit_indices(records,False)
    root=(args.output_root or (output_path(cp,cfg,'preview_root').parent/'all_fit_m0b_1016_explore')).resolve()
    if root.exists() and not args.resume: raise RuntimeError(f'output already exists; choose another --output-root or use --resume: {root}')
    root.mkdir(parents=True,exist_ok=True)
    bridge,audio,decoder,eeg=make_models(cfg,device)
    load_checkpoint(output_path(cp,cfg,'rvq_bridge_checkpoint'),checkpoint_schema(cfg,'rvq_bridge'),{'bridge':bridge},device)
    load_checkpoint(output_path(cp,cfg,'audio_c_checkpoint'),checkpoint_schema(cfg,'audio_c'),{'audio':audio,'decoder':decoder},device)
    load_checkpoint(output_path(cp,cfg,'micro_m0b_checkpoint'),checkpoint_schema(cfg,'micro_m0b'),{'eeg':eeg},device)
    for module in (bridge,audio,decoder,eeg): module.eval()
    predictions=infer_all(records,cache,mapping,eeg,indices,cfg,device,int(args.batch_size))
    prediction_file=root/'all_fit_m0b_predictions.npz'
    ordered=[]
    for source in indices.tolist():
        ordered.append(predictions[int(source)])
    np.savez_compressed(prediction_file,schema=np.asarray(SCHEMA),prediction_source=np.asarray('m0b_checkpoint_all_fit'),source_indices=indices.astype(np.int32),prediction=np.stack([x['prediction'] for x in ordered]),zero=np.stack([x['zero'] for x in ordered]),time=np.stack([x['time'] for x in ordered]),channel=np.stack([x['channel'] for x in ordered]))
    prediction_sha=sha256_file(prediction_file)
    renderer=FrozenEnCodecRVQ(output_path(cp,cfg,'encodec_root'),device=device,bandwidth=float(cfg['audio']['encodec_bandwidth']))
    bank=records.arrays['canonical_p_bank']; bank_duration=records.arrays['canonical_p_bank_duration_fraction']; rows=[]
    for n,source in enumerate(tqdm(indices.tolist(),desc='[v3 all-fit] WAV export',unit='pair')):
        b=one_batch(records,cache,mapping,source,device); key=b['sample_key'][0]; folder=root/key.replace(':','_'); folder.mkdir(parents=True,exist_ok=True); metadata_path=folder/'metadata.json'
        expected=['00_reference.wav','01_frozen_encodec_oracle.wav','02_real_C_real_P_independent_voice.wav','03_zero_C_real_P.wav','04_shuffled_C_real_P.wav','05_real_C_duration_only_P.wav','06_pred_audio_C_real_P.wav','07_eeg_C_P0_canonical_voice.wav','08_zero_eeg.wav','09_time_shuffle_eeg.wav','10_channel_shuffle_eeg.wav','content_mfcc.png','rvq_codes.png']
        if args.resume and metadata_path.is_file() and all((folder/x).is_file() for x in expected):
            rows.append({'sample_key':key,'label':b['label'][0],'folder':str(folder),'metadata':str(metadata_path)}); continue
        n_samples=int(b['waveform_samples'][0]); steps=int(b['encodec_mask'][0].sum()); reference=b['waveform_16k'][0,:n_samples].cpu().numpy(); oracle=renderer.decode_codes(b['encodec_codes'][...,:steps],b['audio_scales'],target_samples=n_samples)[0].cpu().numpy()
        state=audio(b['encodec_codes'],b['encodec_mask'],b['hubert'].float(),b['hubert_mask']); audio_c,_=decoder(state.local,state.token_mask)
        pred={k:torch.from_numpy(v).to(device).unsqueeze(0) for k,v in predictions[int(source)].items()}; real=b['content_mfcc'].float(); p=b['p_base'].float(); voice=b['speaker_reference'].float(); duration=b['duration_fraction'].float(); canonical_voice=b['canonical_voice'].float(); p0=torch.from_numpy(bank[0]).to(device).unsqueeze(0); d0=torch.full_like(duration,float(bank_duration[0]))
        def render(content,prosody,v,d):
            codes=bridge.hard_codes(bridge(content,prosody,v,d),code_mask=b['encodec_mask'],duration_fraction=d); valid=int(b['encodec_mask'][0].sum()); return renderer.decode_codes(codes[...,:valid],b['audio_scales'],target_samples=n_samples)[0].cpu().numpy()
        files={'00_reference.wav':reference,'01_frozen_encodec_oracle.wav':oracle,'02_real_C_real_P_independent_voice.wav':render(real,p,voice,duration),'03_zero_C_real_P.wav':render(torch.zeros_like(real),p,voice,duration),'04_shuffled_C_real_P.wav':render(real.flip(-1),p,voice,duration),'05_real_C_duration_only_P.wav':render(real,torch.zeros_like(p),voice,duration),'06_pred_audio_C_real_P.wav':render(audio_c,p,voice,duration),'07_eeg_C_P0_canonical_voice.wav':render(pred['prediction'],p0,canonical_voice,d0),'08_zero_eeg.wav':render(pred['zero'],p0,canonical_voice,d0),'09_time_shuffle_eeg.wav':render(pred['time'],p0,canonical_voice,d0),'10_channel_shuffle_eeg.wav':render(pred['channel'],p0,canonical_voice,d0)}
        for name,value in files.items(): wav(folder/name,value)
        plot(folder/'content_mfcc.png',[('real C',real[0].cpu()),('Audio-C',audio_c[0].cpu()),('EEG-C',pred['prediction'][0].cpu()),('zero EEG',pred['zero'][0].cpu())],'all-fit content MFCC c1…c39')
        plot(folder/'rvq_codes.png',[('true q0…q7',b['encodec_codes'][0].cpu()),('pred q0…q7',bridge.hard_codes(bridge(real,p,voice,duration),code_mask=b['encodec_mask'])[0].cpu())],'all-fit sequential RVQ codes')
        meta={'schema_version':SCHEMA,'exploratory':True,'fit_only':True,'prediction_source':'m0b_checkpoint_all_fit','sample_key':key,'source_index':int(source),'label':b['label'][0],'prediction_cache_sha256':prediction_sha,'rvq_bridge_sha256':sha256_file(output_path(cp,cfg,'rvq_bridge_checkpoint')),'audio_c_sha256':sha256_file(output_path(cp,cfg,'audio_c_checkpoint')),'micro_m0b_sha256':sha256_file(output_path(cp,cfg,'micro_m0b_checkpoint')),'files':{name:sha256_file(folder/name) for name in files}|{'content_mfcc.png':sha256_file(folder/'content_mfcc.png'),'rvq_codes.png':sha256_file(folder/'rvq_codes.png')}}
        write_json(metadata_path,meta); rows.append({'sample_key':key,'label':b['label'][0],'folder':str(folder),'metadata':str(metadata_path)})
    with (root/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=('sample_key','label','folder','metadata')); writer.writeheader(); writer.writerows(rows)
    write_json(root/'manifest.json',{'schema_version':SCHEMA,'exploratory':True,'fit_only':True,'prediction_source':'m0b_checkpoint_all_fit','n':len(rows),'source_fit_n':int(len(indices)),'prediction_cache':str(prediction_file),'prediction_cache_sha256':prediction_sha,'rvq_bridge_checkpoint_sha256':sha256_file(output_path(cp,cfg,'rvq_bridge_checkpoint')),'audio_c_checkpoint_sha256':sha256_file(output_path(cp,cfg,'audio_c_checkpoint')),'micro_m0b_checkpoint_sha256':sha256_file(output_path(cp,cfg,'micro_m0b_checkpoint')),'heldout_accessed':False,'pairs':rows})
    print(f'[v3 all-fit] complete n={len(rows)} root={root}',flush=True)

if __name__=='__main__':
    a=parse(); export_all(a)
