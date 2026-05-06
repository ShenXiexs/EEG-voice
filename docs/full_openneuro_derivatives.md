# Full OpenNeuro Derivatives

`outputs/downloaded_openneuro_artifacts/` 只保存快速检查用的 preview 图和 inventory。完整训练数据应生成到 `data/derived/openneuro_full/`。

## 生成完整派生数据

```bash
python3 scripts/build_full_openneuro_derivatives.py \
  --out-dir data/derived/openneuro_full \
  --resample-hz 250
```

该命令读取：

```text
data/raw/openneuro/ds006104_datalad
data/raw/openneuro/ds005345_datalad
```

并输出：

```text
data/derived/openneuro_full/
  README.md
  recordings_manifest.csv
  recordings_manifest.json
  audio_manifest.csv
  audio_manifest.json
  ds006104/
    sub-*/ses-*/..._full_eeg.npz
    sub-*/ses-*/*_events.tsv
    sub-*/ses-*/*_channels.tsv
  ds005345/
    audio/*_audio_features.json
    annotation/*.csv
    sub-*/run-*/..._full_eeg.npz
```

## NPZ schema

```text
eeg: float32 [channels, time] for raw recordings
eeg: float32 [epochs, channels, time] for MNE Epochs FIF files
sfreq: float32 scalar
ch_names: object array [channels]
eeg_kind: raw 或 epochs
source: 原始文件路径
```

`ds006104` 的 EDF 导出为 full-length raw array。`ds005345` 的 derivative FIF 在当前下载中是 epochs 文件，因此导出为 `[epochs, channels, time]`。

## 只处理单个数据集

```bash
python3 scripts/build_full_openneuro_derivatives.py \
  --datasets ds006104 \
  --out-dir data/derived/openneuro_full_ds006104 \
  --resample-hz 250
```

```bash
python3 scripts/build_full_openneuro_derivatives.py \
  --datasets ds005345 \
  --out-dir data/derived/openneuro_full_ds005345 \
  --resample-hz 250
```

## Git

`data/`、`data/derived/`、`*.npz`、`*.wav`、`*.edf`、`*.fif` 已在 `.gitignore` 中忽略。完整派生数据不会进入 git commit。
