#!/usr/bin/env python3
"""Write a compact, hash-verified manifest for the bridge-v2 fit-only run."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))

from src.open_vocab_v3.encodec_bridge import SCHEMA
from src.open_vocab_v3.runtime import load_config, output_path, read_json, sha256_file, write_json


def identity(path):return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha256_file(path)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--explore",action="store_true");args=parser.parse_args();cp,cfg=load_config(args.config)
    keys=("prepared_cache","prepared_manifest","encodec_cache","encodec_cache_manifest","a0_gate","e0_gate","bridge_checkpoint","e1_gate","e2_gate","b0_gate","audio_c_checkpoint","c1_gate","c2_gate","micro_m0_checkpoint","micro_m0_predictions","m0_gate","micro_m1_checkpoint","micro_m1_predictions","m1_gate","preview_manifest","training_review")
    artifacts={key:identity(path) for key in keys if (path:=output_path(cp,cfg,key)).is_file()}
    errors=[];preview=output_path(cp,cfg,"preview_manifest")
    verified=0
    if preview.is_file():
        for row in read_json(preview).get("pairs",[]):
            metadata=Path(str(row.get("metadata","")))
            if not metadata.is_file():errors.append(f"missing metadata: {metadata}");continue
            payload=read_json(metadata);valid=True
            for name,expected in payload.get("files",{}).items():
                path=metadata.parent/name
                if not path.is_file() or sha256_file(path)!=expected:valid=False;errors.append(f"hash mismatch: {path}")
            verified+=int(valid)
    report={"schema_version":SCHEMA,"created_at_utc":datetime.now(timezone.utc).isoformat(),"exploratory":bool(args.explore),"scientific_status":"exploratory_fit_only" if args.explore else "strict_fit_only_waiting_for_human_review","artifacts":artifacts,"preview_pairs_verified":verified,"lineage_errors":errors,"lineage_valid":not errors,"heldout_accessed":False}
    write_json(output_path(cp,cfg,"run_manifest"),report)
    if errors:raise RuntimeError(f"bridge lineage failed with {len(errors)} errors")
    print(f"[v3 bridge lineage] complete: {output_path(cp,cfg,'run_manifest')}",flush=True)


if __name__=="__main__":main()
