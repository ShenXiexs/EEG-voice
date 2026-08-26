#!/usr/bin/env python3
"""Freeze the exact 4/1/1-subject x 28/6/6-content Stage-2 split."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from prepare_training_data import (ROOT, build as build_eeg_shards, fit_normalizer,
                                   load_config, output_root, sha256_bytes, stable_json)

sys.path.insert(0, str(ROOT / "app" / "src"))
from eeg2speech.gates import registered_m0_gate_status, require_registered_m0_gates
from cache_speech_targets import cache as cache_speech_targets


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def choose_subjects(frame: pd.DataFrame, count: int, contents: int, namespace: str) -> tuple[list[str], list[str]]:
    by_subject = {subject: set(group.linguistic_content_id) for subject, group in frame.groupby("subject")}
    best = None
    for combination in itertools.combinations(sorted(by_subject), count):
        common = set.intersection(*(by_subject[subject] for subject in combination))
        if len(common) < contents:
            continue
        score = (-len(common), stable(f"{namespace}|{'|'.join(combination)}"))
        if best is None or score < best[0]: best = (score, list(combination), sorted(common))
    if best is None:
        raise RuntimeError(f"{namespace}: no {count} subjects share {contents} contents")
    return best[1], best[2]


def assign(values: list[str], counts: dict[str, int], namespace: str) -> dict[str, str]:
    ordered = sorted(values, key=lambda value: stable(f"{namespace}|{value}"))
    if len(ordered) < sum(int(value) for value in counts.values()):
        raise RuntimeError(f"{namespace}: need {sum(counts.values())} groups, found {len(ordered)}")
    result = {}; offset = 0
    for role in ("train", "validation", "test"):
        for value in ordered[offset:offset + int(counts[role])]: result[value] = role
        offset += int(counts[role])
    return result


def build(data_config: Path, pilot_config: Path) -> Path:
    config, _ = load_config(data_config); pilot = yaml.safe_load(pilot_config.read_text())
    root = output_root(config); frame = pd.read_csv(root / "manifests" / "manifest_all.csv", keep_default_na=False, low_memory=False)
    lock = json.loads((root / "source_lock.json").read_text()); p = pilot["pilot"]
    subject_counts = {key:int(value) for key,value in p["generalization_subjects_by_role"].items()}
    content_counts = {key:int(value) for key,value in p["generalization_contents_by_role"].items()}
    records=[]; assignment={"algorithm":"exact-stage2-subject-content-v2-one-trial-per-cell","datasets":{},
                            "source_lock_sha256":lock["source_lock_sha256"],"preprocess_config_sha256":config["_config_sha256"]}
    eligible = frame[(frame.build_status == "included") & (frame.qc_pass.astype(str).str.lower() == "true")]
    for dataset in ("ds004940","ds006104"):
        content = eligible[(eligible.dataset==dataset)&eligible.supervision_type.isin(["paired_audio","weak_audio"])].copy()
        if dataset == "ds004940" and p.get("primary_ds004940_task"): content=content[content.task==p["primary_ds004940_task"]]
        if dataset == "ds006104" and not bool(p["primary_ds006104_tms"]):
            content=content[~content.tms_applied.astype(str).str.lower().isin(["true","1","yes"])]
        subjects, common = choose_subjects(content, sum(subject_counts.values()), sum(content_counts.values()), f"M1|{dataset}")
        subject_roles=assign(subjects,subject_counts,f"M1|{dataset}|subject")
        content_roles=assign(common,content_counts,f"M1|{dataset}|content")
        dataset_rows=eligible[eligible.dataset==dataset]
        label_roles={}
        if dataset=="ds006104":
            labels=dataset_rows[(dataset_rows.supervision_type=="label_only")&dataset_rows.subject.isin(subjects)].linguistic_content_id.unique().tolist()
            label_roles=assign(labels,{key:int(value) for key,value in p["generalization_label_contents_by_role"].items()},"M1|ds006104|label")
        for _,row in dataset_rows.iterrows():
            subject_role=subject_roles.get(str(row.subject)); group=str(row.linguistic_content_id)
            group_role=(label_roles if row.supervision_type=="label_only" else content_roles).get(group)
            primary_condition = not (dataset == "ds004940" and p.get("primary_ds004940_task") and row.task != p["primary_ds004940_task"])
            if dataset == "ds006104" and not bool(p["primary_ds006104_tms"]):
                primary_condition = primary_condition and str(row.tms_applied).lower() not in {"true","1","yes"}
            if not primary_condition:
                role,reason="excluded","outside_registered_primary_condition"
            elif subject_role is None or group_role is None:
                role,reason="excluded","outside_registered_stage2_groups"
            elif subject_role != group_role:
                role,reason="excluded","stage2_cross_quadrant"
            else:
                role,reason=subject_role,""
            records.append({"trial_id":row.trial_id,"protocol":"stage2_joint_ood","fold":0,"role":role,
                            "exclusion_reason":reason,"subject_group":f"{dataset}:{row.subject}",
                            "subject_fold":"","subject_group_trial_weight":"","subject_sort_position":"",
                            "audio_group":group,"linguistic_content_group":group,
                            "waveform_group":f"sha256:{row.audio_sha256}" if row.audio_sha256 else "",
                            "supervision_axis":"label" if row.supervision_type=="label_only" else "audio",
                            "audio_fold":"","audio_group_trial_weight":"","audio_sort_position":"",
                            "stage2_subject_role":subject_role or "","stage2_content_role":group_role or "",
                            "assignment_algorithm":"exact-stage2-subject-content-v2-one-trial-per-cell","assignment_seed":config["split_seed"],
                            "source_lock_sha256":lock["source_lock_sha256"],"preprocess_config_sha256":config["_config_sha256"],
                            "code_commit":row.code_commit,"code_diff_hash":row.code_diff_hash})
        assignment["datasets"][dataset]={"subjects":subject_roles,"contents":content_roles,"label_contents":label_roles}
    result=pd.DataFrame(records)
    # A split is a frozen experiment specification, not merely a set of
    # eligible groups.  Repeated presentations must not silently inflate one
    # dataset or one subject×content cell.  Keep exactly one deterministic
    # trial in every registered cell and mark all other realizations excluded.
    active = result.role.isin(["train", "validation", "test"])
    result["_cell_order"] = result.trial_id.map(lambda value: stable(f"stage2-cell|{value}"))
    cell_keys = ["subject_group", "linguistic_content_group", "supervision_axis", "role"]
    selected_indices = set(
        result[active].sort_values("_cell_order").drop_duplicates(cell_keys).index.tolist()
    )
    duplicates = active & ~result.index.isin(selected_indices)
    result.loc[duplicates, "role"] = "excluded"
    result.loc[duplicates, "exclusion_reason"] = "duplicate_stage2_subject_content_cell"
    result = result.drop(columns="_cell_order")
    for dataset in ("ds004940","ds006104"):
        selected=result[(result.trial_id.str.startswith(dataset))&result.role.isin(["train","validation","test"])]
        for left,right in (("train","validation"),("train","test"),("validation","test")):
            for column in ("subject_group","linguistic_content_group"):
                overlap=set(selected[selected.role==left][column])&set(selected[selected.role==right][column])
                if overlap: raise RuntimeError(f"Stage2 leakage {dataset} {column} {left}/{right}: {sorted(overlap)[:3]}")
        for role in ("train", "validation", "test"):
            audio = selected[(selected.role == role) & (selected.supervision_axis == "audio")]
            expected = subject_counts[role] * content_counts[role]
            if len(audio) != expected or audio.groupby(["subject_group", "linguistic_content_group"]).size().ne(1).any():
                raise RuntimeError(f"Stage2 {dataset} {role} audio grid is {len(audio)}, expected {expected}")
            if dataset == "ds006104":
                label = selected[(selected.role == role) & (selected.supervision_axis == "label")]
                expected_label = subject_counts[role] * int(p["generalization_label_contents_by_role"][role])
                if len(label) != expected_label or label.groupby(["subject_group", "linguistic_content_group"]).size().ne(1).any():
                    raise RuntimeError(f"Stage2 {dataset} {role} label grid is {len(label)}, expected {expected_label}")
    prehash=sha256_bytes(result.to_csv(index=False).encode()); result["split_csv_sha256"]=prehash
    target=root/"splits"/"stage2_joint_ood_fold-0.csv"; result.to_csv(target,index=False)
    assignment["split_csv_sha256"]=sha256_bytes(target.read_bytes()); assignment["assignment_sha256"]=sha256_bytes(stable_json(assignment))
    (root/"splits"/"stage2_assignment.json").write_text(json.dumps(assignment,indent=2,sort_keys=True)+"\n")
    summary=result.groupby([result.trial_id.str.split("-").str[0],"supervision_axis","role"]).size().to_dict()
    report={"status":"pass","split":str(target),"counts":{"|".join(key):int(value) for key,value in summary.items()},"assignment_sha256":assignment["assignment_sha256"]}
    (root/"qc"/"stage2_split.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return target


def artifact_status(config: dict, pilot: dict) -> dict:
    root = output_root(config)
    split_path = root / "splits" / "stage2_joint_ood_fold-0.csv"
    manifest_path = root / "manifests" / "manifest_stage2.csv"
    normalizer_path = root / "normalizers" / "stage2_joint_ood_fold-0.json"
    target_path = root / "speech_targets" / "speech_targets_stage2.h5"
    missing_paths = [str(path) for path in (split_path, manifest_path, normalizer_path, target_path) if not path.exists()]
    missing_trials: list[str] = []
    unexpected_trials: list[str] = []
    if split_path.exists() and manifest_path.exists():
        split = pd.read_csv(split_path, keep_default_na=False)
        expected = set(split[split.role.isin(["train", "validation", "test"])].trial_id)
        manifest = pd.read_csv(manifest_path, keep_default_na=False, low_memory=False)
        actual = set(manifest[manifest.build_status == "included"].trial_id)
        missing_trials = sorted(expected - actual)
        unexpected_trials = sorted(actual - expected)
    gates = registered_m0_gate_status(ROOT, pilot)
    ready = not missing_paths and not missing_trials and not unexpected_trials and not gates["missing"] and not gates["failed"]
    return {"status": "pass" if ready else "blocked", "m0_gates": gates,
            "missing_paths": missing_paths, "missing_trials": missing_trials,
            "unexpected_trials": unexpected_trials}


def materialize(config: dict, pilot: dict, hubert_local_path: Path) -> dict:
    """Build Stage-2 artifacts without overwriting the frozen M0 artifact set."""
    require_registered_m0_gates(ROOT, pilot)
    split_path = output_root(config) / "splits" / "stage2_joint_ood_fold-0.csv"
    for role in ("train", "validation", "test"):
        for dataset in ("ds004940", "ds006104"):
            build_eeg_shards(
                config, dataset, "all", "all", None, None, None, "any", role,
                "stage2_joint_ood", 0, True, False, "stage2",
            )
    fit_normalizer(config, split_path, 0, False, "stage2")
    config["audio"]["content"]["hubert_local_path"] = str(hubert_local_path.resolve())
    cache_speech_targets(config, "all", None, True, False, "stage2", "speech_targets_stage2")
    status = artifact_status(config, pilot)
    target = output_root(config) / "qc" / "stage2_artifacts.json"
    target.write_text(json.dumps(status, indent=2) + "\n")
    if status["status"] != "pass":
        raise RuntimeError(f"Stage2 artifact materialization did not pass readiness: {status}")
    return status


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config",type=Path,default=ROOT/"configs"/"training_data_v3.yaml")
    parser.add_argument("--pilot-config",type=Path,default=ROOT/"configs"/"joint_pilot_v1.yaml")
    parser.add_argument("--materialize",action="store_true",help="after all M0 gates pass, build isolated Stage-2 shards/normalizer/targets")
    parser.add_argument("--hubert-local-path",type=Path)
    parser.add_argument("--check-readiness",action="store_true")
    args=parser.parse_args(); build(args.data_config,args.pilot_config)
    config,_=load_config(args.data_config); pilot=yaml.safe_load(args.pilot_config.read_text())
    if args.materialize:
        if args.hubert_local_path is None:
            parser.error("--materialize requires --hubert-local-path; implicit model download is forbidden")
        print(json.dumps(materialize(config,pilot,args.hubert_local_path),indent=2))
    if args.check_readiness:
        status=artifact_status(config,pilot); print(json.dumps(status,indent=2))
        return 0 if status["status"]=="pass" else 2
    return 0


if __name__=="__main__": raise SystemExit(main())
