# EEG-Voice: Moreira Dataset Exploration

本仓库用于整理 `Moreira et al., 2025` 公开 EEG 语音解码数据集的前期研究材料，重点是数据结构梳理、标签说明、条件关系、描述性统计和后续 `EEG token / speech token` 对齐研究所需的数据边界判断。

当前阶段仅处理数据理解与简单统计，不包含模型训练，也不直接开展 `EEG -> voice reconstruction`。

## 项目范围

当前工作集中在以下内容：

- 整理 `Moreira 2025` 数据集的任务结构、标签字段和条件变量
- 标准化本地事件表，生成统一 manifest
- 验证 `OpenNeuro ds006104 v1.0.1` 的 BIDS 组织与样本文件
- 给出面向后续研究的数据可用性结论

当前不包括：

- 复杂模型训练
- `EEG -> mel/voice` 重建实验
- 基于 imagined speech 的 paired-audio 监督设置

## 核心文档

- [数据探索报告](reports/moreira_dataset_exploration_cn.md)
- [论文解析与概念说明](reports/moreira_paper_data_concepts_cn.md)
- [ds006104 本地 BIDS 数据集说明](reports/ds006104_bids_dataset_note_cn.md)
- [ds006104 事件级数据分析](reports/ds006104_bids_analysis_cn.md)

## 当前主要结论

- 本地 `24` 份事件表已汇总为统一 manifest：
  - [exploration_outputs/tables/local_events_manifest.csv](exploration_outputs/tables/local_events_manifest.csv)
- 数据公开主源已确认对应 `OpenNeuro ds006104 v1.0.1`
- 当前工作区已加入本地 `ds006104/` 目录，可直接查看 `participants.tsv`、`events.tsv/json`、`channels.tsv`、`eeg.json`
- 当前公开结构支持 `trial-level` 与离散 `unit-level` 对齐准备
- 当前公开结构不支持直接的 `frame-level` 语音监督
- 后续更适合从 `speech-unit EEG token` 准备出发，而不是直接进入语音重建

## 重点输出

### 报告

- [reports/moreira_dataset_exploration_cn.md](reports/moreira_dataset_exploration_cn.md)
- [reports/moreira_paper_data_concepts_cn.md](reports/moreira_paper_data_concepts_cn.md)
- [reports/ds006104_bids_dataset_note_cn.md](reports/ds006104_bids_dataset_note_cn.md)
- [reports/ds006104_bids_analysis_cn.md](reports/ds006104_bids_analysis_cn.md)

### 表格

- [标准化 manifest](exploration_outputs/tables/local_events_manifest.csv)
- [候选子集优先级](exploration_outputs/data_priority/tables/candidate_subset_priority.csv)
- [监督信号完备性](exploration_outputs/data_priority/tables/supervision_signal_matrix.csv)
- [对齐粒度说明](exploration_outputs/data_priority/tables/alignment_granularity.csv)
- [OpenNeuro 验证结果](exploration_outputs/validation/openneuro_validation.json)

### 图表

- [受试者 trial 数](exploration_outputs/figures/subject_trial_counts.png)
- [任务分布](exploration_outputs/figures/task_distribution.png)
- [任务与 TMS 条件关系](exploration_outputs/figures/tms_target_by_task_heatmap.png)
- [字段缺失统计](exploration_outputs/figures/field_missingness.png)
- [任务规模与 unique stimuli](exploration_outputs/data_priority/figures/task_size_and_unique_stimuli.png)
- [候选子集类别平衡](exploration_outputs/data_priority/figures/candidate_subset_balance.png)
- [2019 vs 2021 phoneme overlap](exploration_outputs/data_priority/figures/phoneme_base_overlap.png)
- [对齐粒度示意图](exploration_outputs/data_priority/figures/alignment_granularity.png)

## 数据使用判断

现阶段可以明确的判断如下：

- `2021 single-phoneme` 适合做最基础的离散 speech-unit 分析
- `2021 Words` 可用于较高层级的 lexical unit 观察
- `control-only phonemes` 是当前最适合做跨条件整理的子集
- `2019/2021` 共有的 `20` 个 phoneme base units 可用于有限的跨年份比较

从公开结构看，`Moreira` 更适合作为后续 `EEG token` 数据准备的基础，不足以单独作为 `EEG -> voice` 主监督数据。

## 结果再生成

核心脚本：

- [scripts/generate_moreira_exploration.py](scripts/generate_moreira_exploration.py)
- [scripts/generate_moreira_data_priority.py](scripts/generate_moreira_data_priority.py)

运行方式：

```bash
python3 scripts/generate_moreira_exploration.py
python3 scripts/generate_moreira_data_priority.py
```

输出目录：

- `exploration_outputs/`
- `exploration_outputs/data_priority/`

## 仓库结构

```text
.
├── events_information/        # 本地事件表 CSV
├── exploration_outputs/       # manifest、表格、图表、验证结果
├── figures/                   # 原仓库附带图片
├── matlab_code/               # 原作者 MATLAB 技术验证代码
├── reports/                   # 中文报告与概念说明
├── scripts/                   # 数据整理与图表生成脚本
└── sensors_layout/            # 电极布局信息
```

## References

- Moreira, J. P. C., Carvalho, V. R., Mendes, E. M. A. M., et al. (2025). *An open-access EEG dataset for speech decoding: Exploring the role of articulation and coarticulation*. Scientific Data.
- OpenNeuro dataset: `ds006104`
- Lee et al. (2023). *Towards Voice Reconstruction from EEG during Imagined Speech*.
