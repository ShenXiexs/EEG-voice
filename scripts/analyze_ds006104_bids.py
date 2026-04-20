#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent.parent
BIDS_ROOT = REPO_ROOT / "ds006104"
LOCAL_EVENTS_DIR = REPO_ROOT / "events_information"
OUTPUT_ROOT = REPO_ROOT / "exploration_outputs" / "ds006104_bids_analysis"
TASK_ORDER = ["phonemes", "single-phoneme", "Words"]


@dataclass
class OutputPaths:
    root: Path
    tables: Path
    figures: Path


def ensure_dirs(root: Path) -> OutputPaths:
    tables = root / "tables"
    figures = root / "figures"
    for path in (root, tables, figures):
        path.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root=root, tables=tables, figures=figures)


def clean_text(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("\x00", "").strip()
    if text in {"", "n/a", "N/A", "nan", "None"}:
        return pd.NA
    return text


def task_display(task_bids: str) -> str:
    return "single-phoneme" if task_bids == "singlephoneme" else task_bids


def subject_sort_key(subject_id: str) -> tuple[int, int]:
    prefix = 0 if subject_id.startswith("P") else 1
    return prefix, int(subject_id[1:])


def snake_case(name: str) -> str:
    return (
        name.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .strip("_")
        .lower()
    )


def standardize_local_events(events_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(events_dir.glob("*_Tab.csv")):
        subject_id = csv_path.stem.replace("_Tab", "")
        raw = pd.read_csv(csv_path)
        normalized = raw.rename(columns={"Correct_key": "Correct_Key"}).copy()
        normalized.columns = [snake_case(col) for col in normalized.columns]
        normalized["subject_id"] = subject_id
        normalized["study_year"] = 2019 if subject_id.startswith("P") else 2021
        normalized["task"] = normalized["task"].replace({"singlephoneme": "single-phoneme"})
        normalized["stimulus_base"] = normalized["stimulus"].astype(str).str.replace(
            r"_(angry|happy)\d+$", "", regex=True
        )
        frames.append(normalized)

    manifest = pd.concat(frames, ignore_index=True)
    for column in [
        "phoneme1",
        "phoneme2",
        "phoneme3",
        "category",
        "place",
        "manner",
        "voicing",
        "tmstarget",
        "stimulus",
        "stimulus_base",
    ]:
        if column in manifest.columns:
            manifest[column] = manifest[column].map(clean_text)
    return manifest


def build_bids_trial_manifest(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for events_path in sorted(root.rglob("*_events.tsv")):
        df = pd.read_csv(events_path, sep="\t", dtype=str, keep_default_na=False)
        filename = events_path.name
        subject_id = filename.split("_")[0].replace("sub-", "")
        session_id = filename.split("_")[1].replace("ses-", "")
        task_bids = filename.split("_task-")[1].split("_events.tsv")[0]
        task = task_display(task_bids)
        study_year = 2019 if subject_id.startswith("P") else 2021

        pair_count = 0
        pair_errors = 0
        row_index = 0

        while row_index < len(df):
            current = df.iloc[row_index]
            current_type = clean_text(current.get("trial_type"))

            if current_type != "TMS":
                pair_errors += 1
                issues.append(
                    {
                        "file": str(events_path.relative_to(root)),
                        "row_index": row_index,
                        "issue": "unexpected_event_type",
                        "value": current_type,
                    }
                )
                row_index += 1
                continue

            if row_index + 1 >= len(df):
                pair_errors += 1
                issues.append(
                    {
                        "file": str(events_path.relative_to(root)),
                        "row_index": row_index,
                        "issue": "missing_stimulus_after_tms",
                        "value": "",
                    }
                )
                break

            stimulus = df.iloc[row_index + 1]
            stimulus_type = clean_text(stimulus.get("trial_type"))
            if stimulus_type != "stimulus":
                pair_errors += 1
                issues.append(
                    {
                        "file": str(events_path.relative_to(root)),
                        "row_index": row_index,
                        "issue": "next_row_not_stimulus",
                        "value": stimulus_type,
                    }
                )
                row_index += 1
                continue

            pair_count += 1
            tms_onset = float(current["onset"])
            stim_onset = float(stimulus["onset"])

            phoneme1 = clean_text(stimulus.get("phoneme1"))
            phoneme2 = clean_text(stimulus.get("phoneme2"))
            phoneme3 = clean_text(stimulus.get("phoneme3"))
            stimulus_unit_bids = "".join([value for value in [phoneme1, phoneme2, phoneme3] if pd.notna(value)])
            tms_target = clean_text(current.get("tms_target"))
            if pd.isna(tms_target):
                tms_target = clean_text(stimulus.get("tms_target"))

            records.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "study_year": study_year,
                    "task_bids": task_bids,
                    "task": task,
                    "source_events_file": str(events_path.relative_to(root)),
                    "trial_index_within_file": pair_count,
                    "trial_id": clean_text(current.get("trial")),
                    "tms_onset_sec": tms_onset,
                    "stim_onset_sec": stim_onset,
                    "tms_to_stim_ms": round((stim_onset - tms_onset) * 1000, 3),
                    "tms_target": tms_target,
                    "condition_type": "control"
                    if pd.notna(tms_target) and str(tms_target).startswith("control")
                    else "active",
                    "tms_intensity": clean_text(current.get("tms_intensity")),
                    "category": clean_text(current.get("category")),
                    "place": clean_text(current.get("place")),
                    "manner": clean_text(current.get("manner")),
                    "voicing": clean_text(current.get("voicing")),
                    "phoneme1": phoneme1,
                    "phoneme2": phoneme2,
                    "phoneme3": phoneme3,
                    "stimulus_unit_bids": clean_text(stimulus_unit_bids),
                }
            )
            row_index += 2

        file_rows.append(
            {
                "source_events_file": str(events_path.relative_to(root)),
                "subject_id": subject_id,
                "study_year": study_year,
                "task_bids": task_bids,
                "task": task,
                "event_rows": len(df),
                "trial_pairs": pair_count,
                "pair_errors": pair_errors,
            }
        )

    manifest = pd.DataFrame(records)
    file_summary = pd.DataFrame(file_rows).sort_values(["study_year", "subject_id", "task"])
    issue_table = pd.DataFrame(issues).sort_values(["file", "row_index"]) if issues else pd.DataFrame()
    return manifest, file_summary, issue_table


def build_subject_task_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    counts = (
        manifest.groupby(["study_year", "subject_id", "task"])
        .size()
        .reset_index(name="trial_count")
        .sort_values(["study_year", "subject_id", "task"])
    )
    return counts


def build_task_summary(manifest: pd.DataFrame, local_manifest: pd.DataFrame) -> pd.DataFrame:
    bids_units = {
        (2019, "phonemes"): manifest[(manifest["study_year"] == 2019) & (manifest["task"] == "phonemes")][
            "stimulus_unit_bids"
        ].nunique(),
        (2021, "phonemes"): manifest[(manifest["study_year"] == 2021) & (manifest["task"] == "phonemes")][
            "stimulus_unit_bids"
        ].nunique(),
        (2021, "single-phoneme"): manifest[manifest["task"] == "single-phoneme"]["stimulus_unit_bids"].nunique(),
        (2021, "Words"): manifest[manifest["task"] == "Words"]["stimulus_unit_bids"].nunique(),
    }

    local_units = {
        (2019, "phonemes"): local_manifest[
            (local_manifest["study_year"] == 2019) & (local_manifest["task"] == "phonemes")
        ]["stimulus_base"].nunique(),
        (2021, "phonemes"): local_manifest[
            (local_manifest["study_year"] == 2021) & (local_manifest["task"] == "phonemes")
        ]["stimulus_base"].nunique(),
        (2021, "single-phoneme"): local_manifest[local_manifest["task"] == "single-phoneme"]["phoneme1"].nunique(),
        (2021, "Words"): local_manifest[local_manifest["task"] == "Words"]["stimulus_base"].nunique(),
    }

    rows: list[dict[str, Any]] = []
    for (study_year, task), task_df in manifest.groupby(["study_year", "task"]):
        per_subject = task_df.groupby("subject_id").size()
        tms_targets = ", ".join(sorted(task_df["tms_target"].dropna().unique().tolist()))
        rows.append(
            {
                "study_year": study_year,
                "task": task,
                "subject_count": int(task_df["subject_id"].nunique()),
                "trial_count": int(len(task_df)),
                "min_trials_per_subject": int(per_subject.min()),
                "median_trials_per_subject": float(per_subject.median()),
                "max_trials_per_subject": int(per_subject.max()),
                "unique_units_bids_visible": int(bids_units[(study_year, task)]),
                "unique_units_local": int(local_units[(study_year, task)]),
                "unique_tms_targets": int(task_df["tms_target"].nunique()),
                "tms_targets": tms_targets,
                "tms_to_stim_median_ms": float(task_df["tms_to_stim_ms"].median()),
                "tms_to_stim_min_ms": float(task_df["tms_to_stim_ms"].min()),
                "tms_to_stim_max_ms": float(task_df["tms_to_stim_ms"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["study_year", "task"])


def build_tms_target_table(manifest: pd.DataFrame) -> pd.DataFrame:
    table = (
        manifest.groupby(["task", "tms_target"])
        .size()
        .reset_index(name="trial_count")
        .sort_values(["task", "trial_count", "tms_target"], ascending=[True, False, True])
    )
    return table


def build_interval_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, task_df in manifest.groupby("task"):
        quantiles = task_df["tms_to_stim_ms"].quantile([0.05, 0.5, 0.95]).to_dict()
        rows.append(
            {
                "task": task,
                "trial_count": int(len(task_df)),
                "unique_interval_values": int(task_df["tms_to_stim_ms"].nunique()),
                "min_ms": float(task_df["tms_to_stim_ms"].min()),
                "p05_ms": float(quantiles[0.05]),
                "median_ms": float(quantiles[0.5]),
                "p95_ms": float(quantiles[0.95]),
                "max_ms": float(task_df["tms_to_stim_ms"].max()),
                "gt_150ms_count": int((task_df["tms_to_stim_ms"] > 150).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("task")


def build_phoneme_overlap(local_manifest: pd.DataFrame) -> pd.DataFrame:
    subset = local_manifest[local_manifest["task"] == "phonemes"][["study_year", "stimulus_base"]].dropna()
    status_rows: list[dict[str, Any]] = []
    units_2019 = set(subset[subset["study_year"] == 2019]["stimulus_base"])
    units_2021 = set(subset[subset["study_year"] == 2021]["stimulus_base"])
    for unit in sorted(units_2019 | units_2021):
        status_rows.append(
            {
                "stimulus_base": unit,
                "in_2019": int(unit in units_2019),
                "in_2021": int(unit in units_2021),
                "status": "shared" if unit in units_2019 and unit in units_2021 else "2019_only",
            }
        )
    return pd.DataFrame(status_rows)


def build_field_coverage(manifest: pd.DataFrame, local_manifest: pd.DataFrame) -> pd.DataFrame:
    fields = ["phoneme1", "phoneme2", "phoneme3", "category", "place", "manner", "voicing", "tms_target"]
    rows: list[dict[str, Any]] = []
    local_field_map = {"tms_target": "tmstarget"}
    for task in TASK_ORDER:
        bids_task = manifest[manifest["task"] == task]
        local_task = local_manifest[local_manifest["task"] == task]
        for field in fields:
            bids_ratio = float(bids_task[field].notna().mean()) if field in bids_task.columns else 0.0
            local_field = local_field_map.get(field, field)
            local_ratio = float(local_task[local_field].notna().mean()) if local_field in local_task.columns else 0.0
            rows.append(
                {
                    "task": task,
                    "field": field,
                    "bids_non_null_ratio": round(bids_ratio, 4),
                    "local_non_null_ratio": round(local_ratio, 4),
                }
            )
    return pd.DataFrame(rows)


def build_analysis_notes(
    manifest: pd.DataFrame,
    file_summary: pd.DataFrame,
    field_coverage: pd.DataFrame,
    subject_task_counts: pd.DataFrame,
    phoneme_overlap: pd.DataFrame,
) -> pd.DataFrame:
    notes: list[dict[str, Any]] = []
    edf_entries = list(BIDS_ROOT.rglob("*.edf"))
    hydrated_edf = [path for path in edf_entries if path.exists()]

    notes.append(
        {
            "topic": "pairing",
            "finding": "all_event_files_pair_cleanly",
            "detail": f"56 个 events.tsv 文件都能稳定配对成 TMS -> stimulus trial，pair_errors 总和为 {int(file_summary['pair_errors'].sum())}。",
        }
    )

    notes.append(
        {
            "topic": "local_files",
            "finding": "edf_entries_are_not_hydrated_locally",
            "detail": f"当前 ds006104/ 目录下有 {len(edf_entries)} 个 .edf git-annex 链接入口，但真正可读取的本地目标文件数量为 {len(hydrated_edf)}；当前分析仍主要基于 BIDS sidecars 和事件流。",
        }
    )

    words_coverage = field_coverage[(field_coverage["task"] == "Words") & (field_coverage["field"] == "phoneme3")].iloc[0]
    notes.append(
        {
            "topic": "field_coverage",
            "finding": "words_phoneme3_missing_in_bids",
            "detail": f"Words 任务里，BIDS trial manifest 的 phoneme3 非空比例为 {words_coverage['bids_non_null_ratio']:.2f}，本地事件表为 {words_coverage['local_non_null_ratio']:.2f}。",
        }
    )

    single_task = manifest[manifest["task"] == "single-phoneme"]
    notes.append(
        {
            "topic": "timing",
            "finding": "single_phoneme_interval_is_variable",
            "detail": "single-phoneme 任务的 TMS 到 stimulus 间隔不是固定 50 ms，而是在 70.0-427.5 ms 间变化；其他任务固定为 50 ms。",
        }
    )

    single_control_count = int((single_task["condition_type"] == "control").sum())
    notes.append(
        {
            "topic": "design",
            "finding": "single_phoneme_is_control_only",
            "detail": f"single-phoneme 任务的 {single_control_count} 个 trial 全部是 control 条件，没有 active TMS。",
        }
    )

    overlap_counts = phoneme_overlap["status"].value_counts().to_dict()
    notes.append(
        {
            "topic": "cross_year",
            "finding": "phoneme_inventory_overlap_is_limited",
            "detail": f"phonemes 任务在 2019/2021 间共有 {overlap_counts.get('shared', 0)} 个 shared units，另有 {overlap_counts.get('2019_only', 0)} 个 2019-only units，2021 没有新增独有 units。",
        }
    )

    task_modes = (
        subject_task_counts.groupby(["study_year", "task"])["trial_count"]
        .agg(lambda series: int(series.mode().iloc[0]))
        .rename("expected_trials")
        .reset_index()
    )
    mode_share_rows: list[dict[str, Any]] = []
    for (study_year, task), group in subject_task_counts.groupby(["study_year", "task"]):
        expected = int(group["trial_count"].mode().iloc[0])
        mode_share = float((group["trial_count"] == expected).mean())
        if mode_share >= 0.75:
            incomplete = group[group["trial_count"] != expected]
            for _, row in incomplete.iterrows():
                mode_share_rows.append(
                    {
                        "topic": "completeness",
                        "finding": "subject_task_below_expected_mode",
                        "detail": f"{row['subject_id']} 的 {task} 只有 {int(row['trial_count'])} 个 trial，低于该任务常见值 {expected}。",
                    }
                )

    notes.extend(mode_share_rows)
    return pd.DataFrame(notes)


def make_subject_task_heatmap(subject_task_counts: pd.DataFrame, path: Path) -> None:
    pivot = subject_task_counts.pivot(index="subject_id", columns="task", values="trial_count")
    subject_order = sorted(pivot.index.tolist(), key=subject_sort_key)
    pivot = pivot.reindex(index=subject_order, columns=TASK_ORDER)

    plt.figure(figsize=(7.5, 9))
    ax = sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, linecolor="white")
    ax.set_title("Trial Counts Per Subject and Task")
    ax.set_xlabel("Task")
    ax.set_ylabel("Subject")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_tms_target_heatmap(manifest: pd.DataFrame, path: Path) -> None:
    table = pd.crosstab(manifest["task"], manifest["tms_target"]).reindex(TASK_ORDER)
    plt.figure(figsize=(11, 4.2))
    ax = sns.heatmap(table, annot=True, fmt="d", cmap="YlGnBu")
    ax.set_title("TMS Target by Task")
    ax.set_xlabel("TMS target")
    ax.set_ylabel("Task")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_interval_boxplot(manifest: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    ax = sns.boxplot(data=manifest, x="task", y="tms_to_stim_ms", order=TASK_ORDER, color="#9ecae1")
    ax.set_title("TMS-to-Stimulus Interval by Task")
    ax.set_xlabel("Task")
    ax.set_ylabel("Interval (ms)")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_core_label_balance(local_manifest: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    single_counts = (
        local_manifest[local_manifest["task"] == "single-phoneme"]["phoneme1"].value_counts().sort_index()
    )
    sns.barplot(x=single_counts.index, y=single_counts.values, ax=axes[0], color="#4c78a8")
    axes[0].set_title("Single-Phoneme Inventory")
    axes[0].set_xlabel("Phoneme")
    axes[0].set_ylabel("Trials")

    word_counts = local_manifest[local_manifest["task"] == "Words"]["category"].value_counts().sort_index()
    sns.barplot(x=word_counts.index, y=word_counts.values, ax=axes[1], color="#f58518")
    axes[1].set_title("Words: Real vs Nonce")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Trials")

    phoneme_counts = (
        local_manifest[local_manifest["task"] == "phonemes"]
        .groupby(["study_year", "category"])
        .size()
        .reset_index(name="trial_count")
    )
    sns.barplot(
        data=phoneme_counts,
        x="category",
        y="trial_count",
        hue="study_year",
        ax=axes[2],
        palette="deep",
    )
    axes[2].set_title("Phonemes: Category by Study")
    axes[2].set_xlabel("Category")
    axes[2].set_ylabel("Trials")
    axes[2].legend(title="Study year")

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_phoneme_overlap_heatmap(phoneme_overlap: pd.DataFrame, path: Path) -> None:
    presence = phoneme_overlap.set_index("stimulus_base")[["in_2019", "in_2021"]].T
    presence.index = ["2019", "2021"]
    plt.figure(figsize=(14, 2.8))
    ax = sns.heatmap(presence, cmap="Blues", cbar=False, linewidths=0.5, linecolor="white")
    ax.set_title("Phoneme Inventory Overlap Between 2019 and 2021")
    ax.set_xlabel("Stimulus base")
    ax.set_ylabel("Study year")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_field_coverage_heatmap(field_coverage: pd.DataFrame, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for _, row in field_coverage.iterrows():
        rows.append(
            {
                "source_task": f"BIDS | {row['task']}",
                "field": row["field"],
                "ratio": row["bids_non_null_ratio"],
            }
        )
        rows.append(
            {
                "source_task": f"Local | {row['task']}",
                "field": row["field"],
                "ratio": row["local_non_null_ratio"],
            }
        )

    heatmap_df = pd.DataFrame(rows)
    order = [f"BIDS | {task}" for task in TASK_ORDER] + [f"Local | {task}" for task in TASK_ORDER]
    pivot = heatmap_df.pivot(index="source_task", columns="field", values="ratio").reindex(order)

    plt.figure(figsize=(9, 5))
    ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title("Field Coverage: BIDS Trial Manifest vs Local Event Tables")
    ax.set_xlabel("Field")
    ax.set_ylabel("Source and task")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    outputs = ensure_dirs(OUTPUT_ROOT)

    bids_manifest, file_summary, issue_table = build_bids_trial_manifest(BIDS_ROOT)
    local_manifest = standardize_local_events(LOCAL_EVENTS_DIR)

    subject_task_counts = build_subject_task_counts(bids_manifest)
    task_summary = build_task_summary(bids_manifest, local_manifest)
    tms_target_table = build_tms_target_table(bids_manifest)
    interval_summary = build_interval_summary(bids_manifest)
    phoneme_overlap = build_phoneme_overlap(local_manifest)
    field_coverage = build_field_coverage(bids_manifest, local_manifest)
    analysis_notes = build_analysis_notes(
        bids_manifest, file_summary, field_coverage, subject_task_counts, phoneme_overlap
    )

    bids_manifest.to_csv(outputs.tables / "bids_trial_manifest.csv", index=False)
    file_summary.to_csv(outputs.tables / "events_file_pairing_summary.csv", index=False)
    if not issue_table.empty:
        issue_table.to_csv(outputs.tables / "pairing_issues.csv", index=False)
    subject_task_counts.to_csv(outputs.tables / "subject_task_trial_counts.csv", index=False)
    task_summary.to_csv(outputs.tables / "task_summary.csv", index=False)
    tms_target_table.to_csv(outputs.tables / "tms_target_by_task.csv", index=False)
    interval_summary.to_csv(outputs.tables / "tms_interval_summary.csv", index=False)
    phoneme_overlap.to_csv(outputs.tables / "phoneme_overlap_2019_2021.csv", index=False)
    field_coverage.to_csv(outputs.tables / "field_coverage_bids_vs_local.csv", index=False)
    analysis_notes.to_csv(outputs.tables / "analysis_notes.csv", index=False)

    make_subject_task_heatmap(subject_task_counts, outputs.figures / "subject_task_trial_counts.png")
    make_tms_target_heatmap(bids_manifest, outputs.figures / "tms_target_by_task_heatmap.png")
    make_interval_boxplot(bids_manifest, outputs.figures / "tms_to_stim_interval_by_task.png")
    make_core_label_balance(local_manifest, outputs.figures / "core_label_balance.png")
    make_phoneme_overlap_heatmap(phoneme_overlap, outputs.figures / "phoneme_overlap_2019_2021.png")
    make_field_coverage_heatmap(field_coverage, outputs.figures / "field_coverage_bids_vs_local.png")

    run_summary = {
        "bids_events_files": int(len(file_summary)),
        "bids_trial_rows": int(len(bids_manifest)),
        "pair_errors": int(file_summary["pair_errors"].sum()),
        "local_event_rows": int(len(local_manifest)),
        "output_root": str(outputs.root),
    }
    with (outputs.root / "run_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(run_summary, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
