# eeg-recon-0809

EEG-to-audio reconstruction experiment bundle. 两个数据集统一放在
`data/` 下；原始 EEG、WAV 和内部音频均保留在本地，不进入 Git。

## 数据布局

```text
eeg-recon-0809/
├── data/
│   ├── ds004940/              # 英语句子听觉 EEG；默认 Active 试跑
│   └── ds006104/              # 2021 / ses-02；S01-S16
│       └── audio_internal/    # 用户已有的 DS006104 内部音频和映射表
├── scripts/
├── app/
├── reports/
└── bundle_manifest.json
```

## 数据集入口

### DS004940

首轮建议下载 N400Active 的 4 位受试者，验证 EEG、事件和 WAV 对齐；确认流程后
再扩展到全部 Active。Passive 是可选控制条件。

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/eeg-recon-0809

SUBJECTS="001 002 003 004" ./scripts/download_ds004940.sh
SUBJECTS="001 002 003 004" ./scripts/verify_ds004940.sh

# 最终跨受试者模型
SUBJECTS=all ./scripts/download_ds004940.sh
SUBJECTS=all ./scripts/verify_ds004940.sh

# 可选：追加 Passive 控制任务
SUBJECTS=all MODE=active_passive ./scripts/download_ds004940.sh
```

### DS006104（仅 2021 部分）

下载范围固定为 `sub-S01`–`sub-S16` 的 `ses-02`，不包含 2019 年的 P01–P08。
公开部分下载的是 EEG/events；内部 trial-level 音频应放在
`data/ds006104/audio_internal/`，并保留音频映射 manifest。

```bash
./scripts/download_ds006104_2021.sh
./scripts/verify_ds006104_2021.sh
```

两个下载脚本都使用 OpenNeuro 公共 S3 的 include/exclude 过滤方式并支持断点续传。
下载前请确认 AWS CLI 可用；脚本不会自动开始下载。

## Harmonized v3 / joint content pilot

当前模型是 content-first EEG→speech pilot，不是已经验证的 waveform
reconstruction 系统。DS004940 为精确 presented-WAV 配对；DS006104 的
Words/phonemes 只按 `candidate_filename_timing` 使用。S15 官方 auxiliary 已
从固定 commit 获取并通过预登记 SHA256，single-phoneme 仅提供 label supervision。

### 推荐运行入口

运行脚本沿用参考 bundle 的阶段化、日志化和 fail-closed 约定。它们自动选择
`PYTHON_BIN`（优先项目 venv，其次 `/opt/anaconda3/envs/eegvoice`），日志写入
`outputs/joint_pilot_v1/logs/`，并默认复用参考 bundle 中锁定的本地 HuBERT
snapshot。可用同名环境变量覆盖这些路径。

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/eeg-recon-0809

# 只读查看当前 gate / M0 / renderer / Stage-2 状态
./app/run_joint_status.sh

# 已构建 artifact 上的 forward/backward、renderer dry-run 和 5-step smoke
# 不会计入正式 M0
./app/run_joint_smoke.sh

# 从审计、split、M0 artifact 开始；人工听音未通过时会安全地 exit 3
./app/run_joint_pilot_all.sh

# 人工逐条听音、核对转录并真实填写下列 CSV 后，再继续正式流程
# artifacts/training_data/v3/qc/ds004940_pair_review_20.csv
./app/run_joint_after_review.sh
```

也可以单独运行某一阶段：

```bash
./app/run_joint_stage0.sh
./app/run_joint_audio_oracle.sh
./app/run_joint_m0.sh ds006104   # DS006 content-only M0 不受 DS004 人工 gate 阻断
./app/run_joint_m0.sh all        # 3 modes × 3 registered seeds
./app/run_joint_stage2.sh        # 仅在全部 12 个 M0 evaluation gate 通过后执行
```

### Explore mode（不通过 scientific gate 也完整运行）

`explore` 用于检查端到端可运行性、损失曲线和潜在负迁移，不替代 preregistered
pilot。它仍保留数据完整性、source lock、target/normalizer provenance、NaN 和
shape 检查；它只绕过人工 pair-review、M0 成功标准及 Stage-2 的 M0 先决条件。

```bash
# 完整探索流程：audio oracle、M0 三模式、独立 explore Stage 2、single-vs-joint 评估
./app/run_joint_explore.sh

# 先用最小预算验证完整 runner；仍会构建 explore Stage-2 EEG artifacts
EXPLORE_SEEDS="31" EXPLORE_MAX_STEPS=20 ./app/run_joint_explore.sh

# 需要重建既有 exploration artifacts（而非 resume）时使用
REBUILD_M0=1 REBUILD_EXPLORE=1 ./app/run_joint_explore.sh

# audit/split 已完成但 M0 artifact 构建中断时，从 M0 继续
EXPLORE_FROM=m0 REBUILD_M0=1 ./app/run_joint_explore.sh

# M0 artifact、normalizer 和 speech target 已完成，但模型训练中断时继续
EXPLORE_FROM=overfit ./app/run_joint_explore.sh
```

Explore outputs 永远隔离在：

```text
outputs/joint_pilot_v1/explore/
artifacts/training_data/v3/manifests/manifest_explore_m0.csv
artifacts/training_data/v3/shards/explore_m0/
artifacts/training_data/v3/normalizers/explore_m0_joint_ood_fold-0.json
artifacts/training_data/v3/speech_targets/speech_targets_explore_m0.h5
artifacts/training_data/v3/manifests/manifest_explore_stage2.csv
artifacts/training_data/v3/shards/explore_stage2/
artifacts/training_data/v3/normalizers/explore_stage2_joint_ood_fold-0.json
artifacts/training_data/v3/speech_targets/speech_targets_explore_stage2.h5
```

因此 explore 的 checkpoint、metrics 和 comparison 文件不能被 M0 registry 或
`run_joint_stage2.sh` 当作正式实验结果使用。

常用安全覆盖参数：

```bash
PYTHON_BIN=/absolute/path/to/python \
HUBERT_LOCAL_PATH=/absolute/path/to/hubert-snapshot \
./app/run_joint_stage0.sh

# 仅用于重新构建已注册 M0 shards；默认是 provenance-compatible resume
REBUILD_M0=1 ./app/run_joint_stage0.sh

# 调试 runner 时可缩短步数，但此结果不能冒充预注册正式结果
SEEDS="31" MAX_STEPS=20 ./app/run_joint_m0.sh ds006104
```

`run_joint_stage2.sh` 会先物化隔离的 `manifest_stage2.csv`、`shards/stage2/`、
Stage-2 normalizer 和 `speech_targets_stage2.h5`，再运行 single-dataset/joint 的
validation/test 评估与 paired comparison。Stage 2 不会覆盖 M0 artifacts。

### 等价的底层命令（调试用）

```bash
# 1. 审计与冻结 subject × linguistic-content split
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_training_data.py \
  --config configs/training_data_v3.yaml audit --fetch-aux
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_training_data.py \
  --config configs/training_data_v3.yaml make-splits

# 2. tiny build；先指定 subjects/tasks，不要直接全量 build
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_training_data.py \
  --config configs/training_data_v3.yaml build --dataset ds004940 \
  --subjects sub-001,sub-002 --tasks N400Active \
  --limit-trials-per-group 4 --allow-audit-warnings

# 3. normalizer 只拟合 frozen train fold
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_training_data.py \
  --config configs/training_data_v3.yaml fit-normalizer \
  --split-csv artifacts/training_data/v3/splits/joint_ood_fold-0.csv --fold 0

# 3b. raw→harmonized PSD gate（只抽固定少量 trial，不改原始数据）
/opt/anaconda3/envs/eegvoice/bin/python scripts/eeg_preprocessing_qc.py \
  --config configs/training_data_v3.yaml

# 4. speech targets；HuBERT 可使用已锁定的本地 snapshot，禁止隐式下载
/opt/anaconda3/envs/eegvoice/bin/python scripts/cache_speech_targets.py \
  --config configs/training_data_v3.yaml --manifest built --include-hubert \
  --hubert-local-path /absolute/path/to/hubert-base-ls960-snapshot

# 5. 两种 channel space 的联合 forward/backward gate
/opt/anaconda3/envs/eegvoice/bin/python app/train_joint.py \
  --config configs/joint_pilot_v1.yaml --mode joint --dry-run

# 输出中必须同时出现 ds004940、ds006104 和 ds006104_label_only

# 6. 工程 smoke，仅验证优化器；不构成 Stage 1 科学结果
/opt/anaconda3/envs/eegvoice/bin/python app/train_joint.py \
  --config configs/joint_pilot_v1.yaml --mode joint \
  --smoke-model --max-steps 4

# 7. 冻结精确的 Stage-2 4/1/1 subject × 28/6/6 content split。
# 只生成 split；M0 全部通过前不要 build/train Stage 2。
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_stage2_split.py \
  --data-config configs/training_data_v3.yaml \
  --pilot-config configs/joint_pilot_v1.yaml

# 查看 Stage-2 readiness；M0 或独立 artifact 未完成时返回非零状态。
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_stage2_split.py \
  --data-config configs/training_data_v3.yaml \
  --pilot-config configs/joint_pilot_v1.yaml --check-readiness

# 仅在所有正式 M0 evaluation gate 通过后执行。Stage-2 使用独立的
# manifest_stage2、shards/stage2 和 speech_targets_stage2，不覆盖 M0 artifacts。
/opt/anaconda3/envs/eegvoice/bin/python scripts/prepare_stage2_split.py \
  --data-config configs/training_data_v3.yaml \
  --pilot-config configs/joint_pilot_v1.yaml --materialize \
  --hubert-local-path /absolute/path/to/hubert-base-ls960-snapshot

# 8. audio-only MFCC→log-mel/RMS/activity renderer 工程 gate
/opt/anaconda3/envs/eegvoice/bin/python app/train_audio_renderer.py \
  --config configs/joint_pilot_v1.yaml --dry-run
```

训练 checkpoint 固定 source lock、manifest、split、speech target、normalizer
和模型 runtime code 的 SHA256；评估时任一 artifact 改变都会被拒绝。
缺失 speech target 会直接报错，绝不回退为全零监督。

完整 gate、文件职责和当前限制见 [docs/joint_pilot_v1.md](docs/joint_pilot_v1.md)。
