# ds006104 实验-设备-数据说明

## 定位

`ds006104` 是 speech decoding 的受控 phoneme discrimination + TMS-EEG 数据集。当前项目中，它用于验证：

```text
EEG token 是否保留 phoneme / CV / VC / CVC、articulatory feature、F0/timbre/style 的可解码线索
```

它不是自然幻听数据，也不是自然连续语音数据。它适合做 AVH Voice Image Dataset 之前的受控 probe。

## 实验设计

数据集包含两个相关研究：

| Study | Session | Subjects | 任务 |
| --- | --- | --- | --- |
| Study 1 | `ses-01` | `sub-P01` 到 `sub-P08` | CV / VC phoneme pairs |
| Study 2 | `ses-02` | `sub-S01` 到 `sub-S16` | single phonemes、CV pairs、real words、pseudowords |

被试听 speech sounds，并通过 button press 识别刺激。刺激包括：

- consonants: `/b/`, `/p/`, `/d/`, `/t/`, `/s/`, `/z/`
- vowels: `/i/`, `/E/`, `/A/`, `/u/`, `/oU/`
- CV / VC combinations
- CVC real words / pseudowords

实验同时包含 TMS：

- Study 1: LipM1、TongueM1
- Study 2: Broca's area BA44、verbal memory region BA6 等
- TMS 为 paired pulses，50 ms interpulse interval

建模时必须把 `tms_target`、`tms_intensity` 和 TMS/stimulus onset timing 作为 nuisance 或 control condition。

## 设备

本地 `sub-P01_ses-01_task-phonemes_eeg.json` 显示：

| 字段 | 值 |
| --- | --- |
| EEG 系统 | eego mylab system |
| Cap | ANT Neuro WaveGuard 64-channel EEG cap |
| Montage | extended 10-20 system |
| SamplingFrequency | 2000 Hz |
| Reference | CPz |
| Ground | AFz |
| PowerLineFrequency | 60 Hz |
| Hardware high-pass | 0.1 Hz |
| Hardware low-pass | 350 Hz |

## 数据结构

本地路径：

```text
data/raw/openneuro/ds006104_datalad/
```

关键文件：

```text
sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_eeg.edf
sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_events.tsv
sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_channels.tsv
sub-S01/ses-02/eeg/sub-S01_ses-02_task-singlephoneme_eeg.edf
sub-S01/ses-02/eeg/sub-S01_ses-02_task-phonemes_eeg.edf
sub-S01/ses-02/eeg/sub-S01_ses-02_task-Words_eeg.edf
```

`events.tsv` 核心列：

```text
onset
duration
trial_type
category
manner
phoneme1
phoneme2
phoneme3
place
tms_intensity
tms_target
trial
voicing
```

其中：

- `trial_type=TMS` 表示 TMS pulse。
- `trial_type=stimulus` 表示语音刺激 onset。
- `category/manner/place/voicing` 可作为 articulatory probe 标签。
- `phoneme1/phoneme2/phoneme3` 可构造 single phoneme、CV、VC、CVC 标签。

## 当前处理输出

处理脚本：

```text
scripts/prepare_downloaded_openneuro_artifacts.py
```

运行：

```bash
python3 scripts/prepare_downloaded_openneuro_artifacts.py \
  --ds006104-root data/raw/openneuro/ds006104_datalad \
  --out-dir outputs/downloaded_openneuro_artifacts
```

输出：

```text
outputs/downloaded_openneuro_artifacts/ds006104/inventory.json
outputs/downloaded_openneuro_artifacts/ds006104/*events.tsv.summary.md
outputs/downloaded_openneuro_artifacts/ds006104/*.edf.preview.svg
```

EDF preview 图只展示前几秒的多个 EEG 通道，用于快速检查：

- 文件是否真实下载，不只是 annex 指针。
- 信号是否非空。
- 通道尺度是否异常。
- 是否存在明显饱和或大伪迹。

## 当前项目使用方式

建议第一步做：

```text
EDF + events.tsv
-> epoch around stimulus onset
-> EEG tokenizer / shallow encoder
-> probe labels:
   phoneme
   CV / VC / CVC
   category / manner / place / voicing
   TMS target as control variable
```

如果要连接声音形象目标，应结合本地 `data/meeting_examples/ds006104/stimuli/` 的音频特征表，把 phoneme/content probe 扩展到 F0、brightness、style。

## 注意事项

- `derivatives/eeglab/*.set/.fdt` 体积较大，第一版不需要。
- 数据包含 TMS，不能把所有刺激响应都解释为纯听觉响应。
- 不是 AVH 患者数据，不能直接证明幻听声音重构。
