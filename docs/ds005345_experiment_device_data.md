# ds005345 实验-设备-数据说明

## 1. 数据定位

`ds005345` 是 Le Petit Prince Multi-talker 自然语音 EEG/fMRI 数据集。它在当前项目中的角色是自然语音阶段的 speaker stream retrieval 和 voice embedding 对齐数据。

核心问题：

```text
EEG token 是否能在 single speaker 和 mixed speaker 场景中对齐目标声音流
```

与 `ds006104` 不同，`ds005345` 不再是短音素 probe，而是约 10 分钟自然普通话叙事。它提供 single female、single male、mixed speech 和注意目标条件，因此适合检验模型在真实连续语音中是否能追踪 speaker stream、prosody、F0/intensity 和词级内容。

## 2. 数据已经做到什么

### 2.1 实验任务

README 说明该数据集包含 25 名母语普通话成人受试者，均参加 EEG 和 fMRI 实验。被试听中文《小王子》节选，在 single-talker 和 multi-talker 条件下完成理解任务。

核心音频条件：

| 条件 | 音频 | 对当前项目的含义 |
| --- | --- | --- |
| single female | `stimuli/single_female.wav` | 单说话人 female stream |
| single male | `stimuli/single_male.wav` | 单说话人 male stream |
| mixed speech | `stimuli/mix.wav` | male + female 单通道混合 |
| mix attend female | mixed condition 中 attend female stream | target stream retrieval |
| mix attend male | mixed condition 中 attend male stream | target stream retrieval |

本项目使用的 run 映射在：

```text
configs/ds005345_runs.yaml
```

当前配置：

| run | condition | positive stream | negative stream |
| --- | --- | --- | --- |
| run-1 | single_female | single_female | single_male |
| run-2 | single_male | single_male | single_female |
| run-3 | mix_attend_female | single_female | single_male |
| run-4 | mix_attend_male | single_male | single_female |

### 2.2 EEG 设备

本地 `sub-01_task-multitalker_eeg.json` 显示：

| 字段 | 值 |
| --- | --- |
| EEG 系统 | Brain Products BrainAmp DC |
| Cap | EasyCap M10 / actiCAP |
| Montage | 10-20 system |
| EEGChannelCount | 64 |
| SamplingFrequency | 500 Hz |
| Reference | Cz |
| Ground | AFz |
| PowerLineFrequency | 50 Hz |
| RecordingType | continuous |
| Software high-pass | 0.02 Hz Butterworth |
| Software low-pass | 1000 Hz Butterworth |

当前完整派生数据已经重采样到 250 Hz，保存为 `data/derived/openneuro_full/ds005345/sub-01/run-*/*_full_eeg.npz`。

### 2.3 音频与 annotation

本地音频：

| 音频 | sample rate | channels | duration |
| --- | ---: | ---: | ---: |
| `single_female.wav` | 44100 Hz | 1 | 603.000 s |
| `single_male.wav` | 44100 Hz | 1 | 599.818 s |
| `mix.wav` | 44100 Hz | 1 | 603.000 s |

声学 CSV：

```text
annotation/single_female_acoustic.csv
annotation/single_male_acoustic.csv
annotation/mix_acoustic.csv
```

列：

```text
time
f0
intensity
```

词级 CSV：

```text
annotation/single_female_word_information.csv
annotation/single_male_word_information.csv
```

列：

```text
word
onset
offset
duration
logfreq
pos
td
bu
lc
```

## 3. 能支持什么实验

### 3.1 Single-stream voice embedding alignment

任务：

```text
single_female EEG -> single_female audio / acoustic / text embedding
single_male EEG   -> single_male audio / acoustic / text embedding
```

用途：

- 建立自然语音中 EEG token 与 voice stream 的基础对齐。
- 检查 EEG representation 是否追踪 F0、intensity、envelope 和词级边界。
- 建立 speaker stream contrastive baseline。

### 3.2 Multi-talker target stream retrieval

任务：

```text
mix_attend_female EEG -> female stream positive, male stream negative
mix_attend_male EEG   -> male stream positive, female stream negative
```

用途：

- 检验 EEG token 是否能在混合语音中偏向 attended stream。
- 为 `AudioContrastiveHead` 提供正负样本结构。
- 对 `Voice Image EEG Dataset` 中的“从多个候选声音中检索目标声音形象”提供代理任务。

### 3.3 Prosody / pitch / intensity 对齐

任务：

```text
EEG token -> f0 contour / intensity contour / envelope-like target
```

来源：

```text
annotation/*_acoustic.csv
```

用途：

- 验证 token 是否携带连续 pitch tracking 信息。
- 验证 token 是否携带 intensity / rhythm 信息。
- 为 voice image reconstruction 中的音调和响度属性头提供自然语音目标。

### 3.4 Word-level content retrieval

任务：

```text
EEG segment -> corresponding word window / text embedding / parsing features
```

来源：

```text
annotation/single_female_word_information.csv
annotation/single_male_word_information.csv
```

用途：

- 在自然语音中测试 content-level alignment。
- 与 ds006104 的 phoneme/CV/VC probe 形成层级关系：`phoneme -> word -> stream`。
- 构造 EEG-to-text/audio segment retrieval 的弱监督标签。

## 4. 对当前模型的帮助

| 模型组件 | ds005345 的作用 |
| --- | --- |
| EEG tokenizer | 在自然连续普通话语音上测试长时序 token 稳定性 |
| `AudioContrastiveHead` | single vs mix、attended vs unattended stream retrieval |
| `VoiceAttributeHead` | F0、intensity、speaker stream、prosody target |
| text/audio alignment branch | word boundary、word embedding、HuBERT/wav2vec/audio embedding 对齐 |
| run-condition config | 将 run-1 到 run-4 映射为 positive/negative stream |

最小模型闭环：

```text
preprocessed FIF epoch + run mapping + wav/acoustic/word CSV
-> EEG tokenizer
-> pooled EEG token embedding
-> positive stream embedding / negative stream embedding
-> InfoNCE retrieval loss
```

该数据集是 `Voice Image EEG Dataset` 的自然语音代理任务。它不能直接训练“无外部声波的声音形象重构”，但能训练模型在真实听觉条件下对齐 speaker stream、pitch、intensity 和 word-level content。

## 5. 当前本地数据

原始数据：

```text
data/raw/openneuro/ds005345_datalad/
```

关键文件：

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

完整派生数据：

```text
data/derived/openneuro_full/ds005345/
```

当前完整派生文件：

```text
data/derived/openneuro_full/ds005345/sub-01/run-1/sub-01_task-multitalker_run-1_full_eeg.npz
data/derived/openneuro_full/ds005345/sub-01/run-2/sub-01_task-multitalker_run-2_full_eeg.npz
data/derived/openneuro_full/ds005345/sub-01/run-3/sub-01_task-multitalker_run-3_full_eeg.npz
data/derived/openneuro_full/ds005345/sub-01/run-4/sub-01_task-multitalker_run-4_full_eeg.npz
```

派生 annotation 和 audio features：

```text
data/derived/openneuro_full/ds005345/annotation/
data/derived/openneuro_full/ds005345/audio/
data/derived/openneuro_full/audio_manifest.json
data/derived/openneuro_full/recordings_manifest.csv
```

NPZ schema：

```text
eeg: float32 [epochs, channels, time]
sfreq: 250.0
ch_names: channel names
eeg_kind: epochs
source: original FIF path
```

当前 `sub-01` 的每个 run 是一个完整 10 分钟左右 epoch：

| run | n_epochs | channels | time points | duration |
| --- | ---: | ---: | ---: | ---: |
| run-1 | 1 | 64 | 150251 | 601.004 s |
| run-2 | 1 | 64 | 150251 | 601.004 s |
| run-3 | 1 | 64 | 150251 | 601.004 s |
| run-4 | 1 | 64 | 150251 | 601.004 s |

## 6. 能做到什么与不能做到什么

### 能做到

| 能力 | 说明 |
| --- | --- |
| 自然语音 EEG-token 对齐 | 10 分钟连续普通话故事 |
| single speaker retrieval | female / male 单流条件 |
| target stream retrieval | mix attend female / mix attend male |
| F0 / intensity tracking | acoustic CSV 直接提供 |
| word-level content alignment | single female / male word information |
| speaker stream contrastive learning | positive/negative stream 明确 |

### 不能做到

| 限制 | 影响 |
| --- | --- |
| 不是同一句话由男女各读一遍 | speaker 与 content 存在部分耦合，不能作为纯 timbre disentanglement 数据 |
| mixed condition 是单通道混合 | 无法直接分离左右耳或空间来源 |
| `mix_acoustic.csv` 是混合声学 | 不能替代 attended speaker 的 word-level semantic timeline |
| 不是声音想象任务 | 不能直接证明无外部声波 voice image reconstruction |
| 当前只派生了 sub-01 | 适合模型调试，跨受试者结论需要继续派生更多 subject |

## 7. 在项目路线中的位置

```text
ds006104
-> content / articulation / F0 / timbre / style controlled probe
-> ds005345
-> natural speech single-stream and multi-stream retrieval
-> Voice Image EEG Dataset
-> imagined / recalled voice image retrieval and reconstruction
```

`ds005345` 的结论用于回答“模型是否能在自然连续语音中追踪并检索目标声音流”。它是从受控 phoneme probe 走向真实声音形象重构的关键中间层。
