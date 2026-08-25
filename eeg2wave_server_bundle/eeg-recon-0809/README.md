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

# 8. audio-only MFCC→log-mel/RMS/activity renderer 工程 gate
/opt/anaconda3/envs/eegvoice/bin/python app/train_audio_renderer.py \
  --config configs/joint_pilot_v1.yaml --dry-run
```

训练 checkpoint 固定 source lock、manifest、split、speech target、normalizer
和模型 runtime code 的 SHA256；评估时任一 artifact 改变都会被拒绝。
缺失 speech target 会直接报错，绝不回退为全零监督。

完整 gate、文件职责和当前限制见 [docs/joint_pilot_v1.md](docs/joint_pilot_v1.md)。
