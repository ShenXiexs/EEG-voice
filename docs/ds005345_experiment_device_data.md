# ds005345 实验-设备-数据说明

## 定位

`ds005345` 是 Le Petit Prince Multi-talker 自然语音数据集，包含 EEG、fMRI、MRI、刺激音频和语言/声学 annotation。当前项目中，它用于：

```text
single female / single male / mix
-> speaker stream tracking
-> attended voice / target speaker retrieval
-> EEG token 与声音内容、pitch、timbre 表征对齐
```

它不是幻听数据，但比单说话人自然语音更接近“多个声音中选择目标声音”的问题。

## 实验设计

README 说明数据来自 25 名母语普通话被试，均参加 EEG 和 fMRI 实验。被试听中文《小王子》节选。

核心条件：

| 条件 | 音频 |
| --- | --- |
| single female | `stimuli/single_female.wav` |
| single male | `stimuli/single_male.wav` |
| mixed speech | `stimuli/mix.wav` |
| mix attend female | mixed condition 中 attend female stream |
| mix attend male | mixed condition 中 attend male stream |

每个 run 后有理解测验，用于确认被试听懂故事。

## 设备

本地 `sub-01_task-multitalker_eeg.json` 显示：

| 字段 | 值 |
| --- | --- |
| EEG 系统 | Brain Products BrainAmp DC |
| Cap | EasyCap M10 |
| Montage | 10-20 system |
| EEGChannelCount | 64 |
| SamplingFrequency | 500 Hz |
| Reference | Cz |
| Ground | AFz |
| PowerLineFrequency | 50 Hz |
| RecordingType | continuous |
| Software high-pass | 0.02 Hz Butterworth |
| Software low-pass | 1000 Hz Butterworth |

README 还说明 EEG 采集使用 64-channel actiCAP，采样率 500 Hz。

## 数据结构

本地路径：

```text
data/raw/openneuro/ds005345_datalad/
```

当前最需要的文件：

```text
stimuli/single_female.wav
stimuli/single_male.wav
stimuli/mix.wav
annotation/single_female_acoustic.csv
annotation/single_male_acoustic.csv
annotation/mix_acoustic.csv
annotation/single_female_word_information.csv
annotation/single_male_word_information.csv
quiz/multitalker_quiz_question.csv
sub-01/eeg/sub-01_task-multitalker_eeg.vhdr
sub-01/eeg/sub-01_task-multitalker_eeg.vmrk
sub-01/eeg/sub-01_task-multitalker_eeg.eeg
derivatives/sub-01/eeg/sub-01_task-multitalker_run-*_eeg_preprocessed.fif
```

## Annotation

声学 CSV：

```text
annotation/single_female_acoustic.csv
annotation/single_male_acoustic.csv
annotation/mix_acoustic.csv
```

用途：

- `f0`: pitch contour。
- `intensity`: RMS/intensity contour。
- 与 EEG run 时间轴对齐后，可构造 speech envelope / F0 / prosody regression 或 contrastive target。

词级 CSV：

```text
annotation/single_female_word_information.csv
annotation/single_male_word_information.csv
```

用途：

- word boundary。
- POS / frequency / parsing predictors。
- 构造 content-level retrieval target。

mixed condition 没有独立 word information；需要用 single female/male stream 的 timeline 作为两个候选语音流。

## 当前处理输出

处理脚本：

```text
scripts/prepare_downloaded_openneuro_artifacts.py
```

运行：

```bash
python3 scripts/prepare_downloaded_openneuro_artifacts.py \
  --ds005345-root data/raw/openneuro/ds005345_datalad \
  --out-dir outputs/downloaded_openneuro_artifacts
```

输出：

```text
outputs/downloaded_openneuro_artifacts/ds005345/inventory.json
outputs/downloaded_openneuro_artifacts/ds005345/*.wav.waveform.svg
outputs/downloaded_openneuro_artifacts/ds005345/*acoustic.csv.acoustic.svg
outputs/downloaded_openneuro_artifacts/ds005345/*.mne_needed.txt
```

如果当前环境没有 MNE，BrainVision/FIF EEG 只生成占位说明。安装 MNE 后可扩展脚本直接读：

```bash
python -m pip install mne matplotlib numpy scipy
```

## 当前项目使用方式

第一版 speaker stream retrieval：

```text
single_female EEG -> single_female audio embedding positive
single_male EEG   -> single_male audio embedding positive
mix attend female EEG -> female stream positive, male stream negative
mix attend male EEG   -> male stream positive, female stream negative
```

建议先用 derivatives 的预处理 FIF 调试，因为每个 run 已被切分：

```text
derivatives/sub-01/eeg/sub-01_task-multitalker_run-1_eeg_preprocessed.fif
...
derivatives/sub-01/eeg/sub-01_task-multitalker_run-4_eeg_preprocessed.fif
```

之后再回到 raw BrainVision：

```text
sub-01/eeg/sub-01_task-multitalker_eeg.vhdr/.vmrk/.eeg
```

## 注意事项

- 不要先下载 `sub-*/anat`、`sub-*/func` 或完整 `derivatives`，当前任务不需要 MRI/fMRI。
- `mix_acoustic.csv` 描述的是混合音频的整体声学，不能替代 attended speaker 的词级/语义时间轴。
- 做 target speaker retrieval 时，必须明确每个 run 的 condition 和 attended target。
- 数据是外部真实语音注意任务，不是幻听；它用于训练声音流选择和 timbre/pitch 对齐的代理能力。
