#!/usr/bin/env python3
"""Bind a human listening decision to exact bridge-v2 preview hashes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))

from src.open_vocab_v3.encodec_bridge import SCHEMA
from src.open_vocab_v3.runtime import capture_lineage, load_config, output_path, read_json, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--approve",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    cp,cfg=load_config(args.config);preview=output_path(cp,cfg,"preview_manifest")
    if not preview.is_file():raise RuntimeError("bridge preview is missing; export fit-only WAVs first")
    expected=capture_lineage(cp,cfg,artifact_keys=("bridge_checkpoint","audio_c_checkpoint","micro_m0_checkpoint","micro_m1_checkpoint"))
    payload={"schema_version":SCHEMA,"passed":bool(args.approve),"approval_scope":"human listened to fit-only E1/E2/C2/M0/M1 WAVs; this does not approve held-out evaluation","preview_manifest":str(preview),"preview_manifest_sha256":__import__('hashlib').sha256(preview.read_bytes()).hexdigest(),"lineage":expected}
    destination=output_path(cp,cfg,"training_review")
    if args.approve:write_json(destination,payload);print(f"[v3 bridge review] approved: {destination}",flush=True);return
    if args.check:
        current=read_json(destination) if destination.is_file() else {}
        if current.get("passed") is not True or current.get("lineage")!=expected:raise RuntimeError("bridge training-WAV approval is missing or stale")
        print(f"[v3 bridge review] exact approval valid: {destination}",flush=True);return
    print(f"[v3 bridge review] preview ready: {preview}",flush=True)


if __name__=="__main__":main()
