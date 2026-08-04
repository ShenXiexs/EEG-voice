#!/usr/bin/env python3
"""Write repair-v3 fit-only lineage; no held-out paths are read here."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))
from src.open_vocab_v3.encodec_rvq_repair import SCHEMA
from src.open_vocab_v3.runtime import capture_lineage,load_config,output_path,read_json,write_json
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--explore',action='store_true');a=p.parse_args();cp,cfg=load_config(a.config)
 gates={k:str(output_path(cp,cfg,k)) for k in ('a0_gate','r0_gate','e1a_gate','e1b_gate','b0_gate','c1_gate','c2_gate','m0a_gate','m0b_gate','m1_gate')};payload={'schema_version':SCHEMA,'exploratory':bool(a.explore),'prepared_scope':'fit_only','heldout_audio_opened':False,'heldout_rows_cached':False,'heldout_used_for_training_or_evaluation':False,'source_container_note':'records_train/validation NPZ files are monolithic; excluded rows may be decompressed by NumPy but are filtered before the prepared cache and never exposed to a model','gates':gates,'lineage':capture_lineage(cp,cfg,artifact_keys=('rvq_micro_checkpoint','rvq_bridge_checkpoint','audio_c_checkpoint','micro_m0a_checkpoint','micro_m0b_checkpoint','micro_m1_checkpoint'))};write_json(output_path(cp,cfg,'run_manifest'),payload);print(output_path(cp,cfg,'run_manifest'),flush=True)
if __name__=='__main__':main()
