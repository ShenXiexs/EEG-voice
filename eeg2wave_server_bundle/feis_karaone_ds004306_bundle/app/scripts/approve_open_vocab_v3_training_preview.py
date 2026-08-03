#!/usr/bin/env python3
"""Bind a human training-WAV decision to the exact full-fit preview lineage."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

APP=Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:sys.path.insert(0,str(APP))

from src.open_vocab_v3.runtime import capture_lineage,load_config,output_path,read_json,require_passed_gate,write_json


def expected_lineage(config_path,cfg):
    if str(cfg.get("experiment",{}).get("schema",""))=="openvoice-v3-cp-temporal-large-v1":
        require_passed_gate(config_path,cfg,"fit_gate",lineage_artifact_keys=("fit_checkpoint",))
        preview=read_json(output_path(config_path,cfg,"fit_preview_manifest"))
        if preview.get("stage")!="fit" or not preview.get("complete",False):raise RuntimeError("CP-temporal full-fit training preview is incomplete")
        expected_preview=capture_lineage(config_path,cfg,artifact_keys=("fit_checkpoint","content_checkpoint","cvae_checkpoint"))
        if preview.get("lineage")!=expected_preview:raise RuntimeError("CP-temporal full-fit preview lineage is stale")
        return capture_lineage(config_path,cfg,artifact_keys=("fit_checkpoint","fit_gate","fit_preview_manifest"))
    require_passed_gate(config_path,cfg,"fit_gate",lineage_artifact_keys=("fit_checkpoint","micro_gate"))
    preview=read_json(output_path(config_path,cfg,"fit_preview_manifest"))
    if preview.get("stage")!="fit" or not preview.get("complete",False):raise RuntimeError("full-fit training preview is incomplete")
    if preview.get("lineage")!=capture_lineage(config_path,cfg,artifact_keys=("fit_checkpoint","fit_gate")):raise RuntimeError("full-fit preview lineage is stale")
    return capture_lineage(config_path,cfg,artifact_keys=("fit_checkpoint","fit_gate","fit_preview_manifest"))


def main()->None:
    parser=argparse.ArgumentParser(description="Approve or check the v3 training-WAV listening gate")
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--approve",action="store_true");parser.add_argument("--reviewer",default="");parser.add_argument("--note",default="");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    config_path,cfg=load_config(args.config);path=output_path(config_path,cfg,"training_review");lineage=expected_lineage(config_path,cfg)
    if args.approve:
        if not args.reviewer.strip():raise SystemExit("--reviewer is required with --approve")
        write_json(path,{"schema_version":"openvoice-v3-training-wav-human-review-v1","passed":True,"decision":"approved_for_heldout_evaluation","reviewer":args.reviewer.strip(),"note":args.note,"reviewed_at_utc":datetime.now(timezone.utc).isoformat(),"lineage":lineage})
        print(f"[v3 training review] approved: {path}",flush=True);return
    if not path.is_file():raise SystemExit(f"training WAV review is missing; listen to the full-fit preview, then run this script with --approve --reviewer NAME")
    report=read_json(path)
    if not report.get("passed",False) or report.get("lineage")!=lineage:raise SystemExit("training WAV review is rejected or stale")
    print(f"[v3 training review] valid approval: {path}",flush=True)


if __name__=="__main__":main()
