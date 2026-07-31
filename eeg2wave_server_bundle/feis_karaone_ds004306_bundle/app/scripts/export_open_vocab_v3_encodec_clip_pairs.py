#!/usr/bin/env python3
"""Export the eight required train-pair WAVs and aligned diagnostic images."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP)) if str(APP) not in sys.path else None
from src.open_vocab_v3.data import V3Dataset,_accepted_denoise_paths,_read_waveform,channel_shuffled_eeg,light_prepare_waveform,time_shuffled_eeg
from src.open_vocab_0724.audio_features import AudioPreparationConfig
from src.open_vocab_v3.native_mel import native_speecht5_mel
from src.open_vocab_v3.runtime import capture_lineage,default_device,load_config,move_batch,output_path,read_json,sha256_file,write_json
from src.open_vocab_v3.encodec_content import EnCodecGenerator
from scripts.train_open_vocab_v3_encodec_clip import TokenDataset,attach_codes,micro_subset,modules,load,token_collate

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=('micro','fit','final'),default='fit');p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true');p.add_argument('--explore',action='store_true',help='allow final export without the training-WAV approval; label outputs exploratory');return p.parse_args()
def wav(path,x):sf.write(path,np.asarray(x,dtype=np.float32),16000,subtype='PCM_16')
def image(path,rows,title):
 fig,ax=plt.subplots(len(rows),1,figsize=(12,2.2*len(rows)),sharex=True)
 if len(rows)==1:ax=[ax]
 for a,(name,value) in zip(ax,rows):a.imshow(value,origin='lower',aspect='auto',cmap='magma');a.set_ylabel(name)
 fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
def main():
 a=parse();cp,cfg=load_config(a.config);device=default_device(a.device);from src.open_vocab_v3.data import load_prepared
 records=load_prepared(output_path(cp,cfg,'prepared_cache'));checkpoint_stage='micro' if a.stage=='micro' else 'fit';base=micro_subset(records,cfg) if a.stage=='micro' else V3Dataset(records,('fit',),eligible_only=True);cache,map_=attach_codes(records,cp,cfg);ds=TokenDataset(base,cache,map_);audio,decoder,eeg=modules(cfg,device);load(output_path(cp,cfg,'audio_content_checkpoint'),'openvoice-v3-audio-content-v1',{'audio':audio,'decoder':decoder},device);load(output_path(cp,cfg,f'{checkpoint_stage}_checkpoint'),f'openvoice-v3-eeg-encodec-clip-{checkpoint_stage}-v1',{'eeg':eeg},device);from transformers import SpeechT5HifiGan
 if a.stage=='final' and not a.explore:
  review=output_path(cp,cfg,'training_review');expected=capture_lineage(cp,cfg,artifact_keys=('fit_checkpoint','fit_gate','fit_preview_manifest'));payload=read_json(review) if review.is_file() else {}
  if not payload.get('passed',False) or payload.get('lineage')!=expected:raise RuntimeError('final export refused before exact, non-stale training preview approval')
 with output_path(cp,cfg,'unified_manifest').open(newline='',encoding='utf-8') as h: paths={str(row['sample_key']):str(row['audio_relpath']) for row in csv.DictReader(h) if row.get('dataset')=='karaone'}
 raw_root=output_path(cp,cfg,'audio_root'); denoised=_accepted_denoise_paths(cp,cfg); prep=AudioPreparationConfig(sample_rate=16000,max_active_seconds=float(cfg['audio']['max_active_seconds']),target_rms=float(cfg['audio']['target_rms']))
 vocoder=SpeechT5HifiGan.from_pretrained(str(output_path(cp,cfg,'vocoder_adapted_root')),local_files_only=True).to(device).eval();codec=EnCodecGenerator(output_path(cp,cfg,'encodec_adapted_root'),device=device,bandwidth=float(cfg['audio']['encodec_bandwidth']));fit=(records.roles=='fit')&records.arrays['fit_eligible'].astype(bool);fixed_frames=int(np.median(records.arrays['speech_t5_mel_mask'][fit].sum(1)));fixed_samples=fixed_frames*int(cfg['vocoder']['hop_length']);root=output_path(cp,cfg,'micro_preview_root') if a.stage=='micro' else output_path(cp,cfg,'pair_root') if a.stage=='final' else output_path(cp,cfg,'fit_preview_root');root.mkdir(parents=True,exist_ok=True);manifest=[]
 cvae_path=output_path(cp,cfg,'cvae_checkpoint')
 if not cvae_path.is_file():raise RuntimeError('cannot export training WAVs before T2 CVAE checkpoint exists')
 from src.open_vocab_v3.model import NativeSpeechT5MFCCMelCVAE
 cvae=NativeSpeechT5MFCCMelCVAE(mfcc_bins=40,mel_bins=80,dimension=int(cfg['model']['audio_dimension']),voice_dim=int(cfg['speaker']['embedding_dimension']),latent_dim=int(cfg['model']['audio_latent_dimension']),residual_limit_log10=float(cfg['model']['audio_residual_limit_log10'])).to(device);load(cvae_path,'openvoice-v3-native-mel-cvae-v1',{'cvae':cvae},device);cvae.eval()
 for batch in DataLoader(ds,batch_size=1,shuffle=False,collate_fn=token_collate,num_workers=0):
  b=move_batch(batch,device);key=b['sample_key'][0].replace(':','_');folder=root/key;folder.mkdir(parents=True,exist_ok=True)
  def render(mfcc,voice=None,mean=None,std=None):
   voice=b['canonical_voice'].float() if voice is None else voice;mean=b['canonical_mfcc_mean'].float() if mean is None else mean;std=b['canonical_mfcc_std'].float() if std is None else std;mel=cvae.generate(mfcc,voice,mean,std,stochastic=False)['mel'];mel=torch.nn.functional.interpolate(mel,size=int(cfg['audio']['native_mel_frames']),mode='linear',align_corners=False);return vocoder(mel.transpose(1,2))[0].detach().cpu().numpy(),mel[0].detach().cpu().numpy()
  atok=audio(b['encodec_codes'],b['encodec_mask']);oracle_mfcc=decoder(atok);correct=decoder(eeg(b['eeg'].float(),b['channel_xyz'].float(),b['channel_mask'],b['time_mask']));zero=decoder(eeg(torch.zeros_like(b['eeg']).float(),b['channel_xyz'].float(),b['channel_mask'],b['time_mask']));time=decoder(eeg(time_shuffled_eeg(b['eeg'].float(),b['time_mask']),b['channel_xyz'].float(),b['channel_mask'],b['time_mask']));channel=decoder(eeg(channel_shuffled_eeg(b['eeg'].float(),b['channel_mask']),b['channel_xyz'].float(),b['channel_mask'],b['time_mask']))
  raw,rate=_read_waveform(denoised.get(b['sample_key'][0],raw_root/paths[b['sample_key'][0]]));prepared,_=light_prepare_waveform(raw,rate,prep);reference=prepared.waveform[:prepared.valid_samples];ref=native_speecht5_mel(torch.from_numpy(reference).unsqueeze(0).to(device),cfg,frames=int(cfg['audio']['native_mel_frames']))[0].cpu().numpy()
  codes,code_mask=codec.encode(torch.from_numpy(reference).unsqueeze(0));valid=int(code_mask[0].sum());codec_oracle=codec.decode(codes[:,:,:valid],target_samples_16k=len(reference))[0].cpu().numpy();oracle,mel_oracle=render(oracle_mfcc);target,mel_target=render(b['mfcc'].float());target_voice,_=render(b['mfcc'].float(),b['speaker_reference'].float(),b['speaker_reference_mfcc_mean'].float(),b['speaker_reference_mfcc_std'].float());pred,mel_pred=render(correct);z,mel_z=render(zero);t,mel_t=render(time);c,mel_c=render(channel);oracle=oracle[:fixed_samples];target=target[:fixed_samples];pred=pred[:fixed_samples];z=z[:fixed_samples];t=t[:fixed_samples];c=c[:fixed_samples];target_voice=target_voice[:len(reference)]
  # Exact names required by the plan; 03 is explicitly audio-only side info.
  wav(folder/'00_cleaned_reference.wav',reference);wav(folder/'01_encodec_codec_oracle.wav',codec_oracle);wav(folder/'02_real_mfcc_canonical_voice.wav',target);wav(folder/'03_real_mfcc_target_voice_audio_only.wav',target_voice);wav(folder/'04_eeg_aligned_mfcc.wav',pred);wav(folder/'05_zero_eeg.wav',z);wav(folder/'06_time_shuffled_eeg.wav',t);wav(folder/'07_channel_shuffled_eeg.wav',c)
  image(folder/'mfcc_comparison.png',[('real',b['mfcc'][0].cpu().numpy()),('audio',oracle_mfcc[0].cpu().numpy()),('eeg',correct[0].cpu().numpy()),('zero',zero[0].cpu().numpy())],'CMVN MFCC');image(folder/'mel_comparison.png',[('real',ref),('audio',mel_oracle),('eeg',mel_pred),('zero',mel_z)],'native SpeechT5 Mel');image(folder/'token_similarity.png',[('similarity',(torch.nn.functional.normalize(eeg(b['eeg'].float(),b['channel_xyz'].float(),b['channel_mask'],b['time_mask'])[0],dim=-1)@torch.nn.functional.normalize(atok[0],dim=-1).T).detach().cpu().numpy())],'EEG/audio token cosine')
  source_path=denoised.get(b['sample_key'][0],raw_root/paths[b['sample_key'][0]]);meta={'schema_version':'openvoice-v3-encodec-clip-pair-v1','sample_key':b['sample_key'][0],'label':b['label'][0],'stage':a.stage,'exploratory_gate_bypass':bool(a.explore),'primary_fixed_duration_frames':fixed_frames,'checkpoint_sha256':sha256_file(output_path(cp,cfg,f'{checkpoint_stage}_checkpoint')),'audio_checkpoint_sha256':sha256_file(output_path(cp,cfg,'audio_content_checkpoint')),'cvae_checkpoint_sha256':sha256_file(cvae_path),'config_sha256':sha256_file(cp),'source_audio':str(source_path),'source_audio_sha256':sha256_file(source_path),'files':{x.name:sha256_file(x) for x in folder.iterdir() if x.is_file()}};write_json(folder/'metadata.json',meta);manifest.append({'sample_key':b['sample_key'][0],'folder':str(folder),'metadata':str(folder/'metadata.json')})
 name='export_manifest.json' if a.stage=='final' else 'preview_manifest.json';lineage=capture_lineage(cp,cfg,artifact_keys=(f'{checkpoint_stage}_checkpoint','fit_gate') if checkpoint_stage=='fit' else (f'{checkpoint_stage}_checkpoint','micro_gate'));write_json(root/name,{'schema_version':'openvoice-v3-encodec-clip-preview-v1','stage':a.stage,'exploratory_gate_bypass':bool(a.explore),'complete':True,'n':len(manifest),'lineage':lineage,'pairs':manifest});
 if a.stage=='final':
  import csv
  with (root/'manifest.csv').open('w',newline='',encoding='utf-8') as h:
   writer=csv.DictWriter(h,fieldnames=('sample_key','folder','metadata'));writer.writeheader();writer.writerows(manifest)
 print(root,flush=True)
if __name__=='__main__':main()
