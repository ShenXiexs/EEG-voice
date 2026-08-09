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
