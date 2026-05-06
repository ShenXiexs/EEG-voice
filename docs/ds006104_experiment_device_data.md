# ds006104 实验-设备-数据说明

## 1. 数据定位

`ds006104` 是 controlled speech decoding + TMS-EEG 数据集。它在当前项目中的角色不是训练自然声音重构模型，而是作为 `Voice Image EEG Dataset` 之前的受控验证集，用来回答一个更基础的问题：

```text
EEG token 是否保留声音内容、音素结构、发音结构、音调/音色线索和情绪风格线索
```

该数据集的价值在于控制程度高。它把语音材料拆成 single phoneme、CV、VC、CVC real word / pseudoword，并在 `events.tsv` 中给出 phoneme、manner、place、voicing、category、TMS target 等标签。项目本地同时保存了对应的真实音频刺激，可直接提取 F0、duration、spectral centroid、brightness、energy envelope 等声音形象特征。

## 2. 数据已经做到什么

### 2.1 实验任务

数据集包含两个相关研究：

| Study | Session | Subjects | 任务结构 |
| --- | --- | --- | --- |
| Study 1 | `ses-01` | `sub-P01` 到 `sub-P08` | CV / VC phoneme pairs |
| Study 2 | `ses-02` | `sub-S01` 到 `sub-S16` | single phonemes、CV pairs、real words、pseudowords |

被试听 speech sounds，并通过 button press 识别刺激。刺激覆盖：

| 类型 | 内容 |
| --- | --- |
| consonants | `/b/`, `/p/`, `/d/`, `/t/`, `/s/`, `/z/` |
| vowels | `/i/`, `/E/`, `/A/`, `/u/`, `/oU/` |
| phoneme pairs | CV、VC |
| phoneme triplets | CVC real words、CVC pseudowords |

实验同时包含 TMS：

| Study | TMS target |
| --- | --- |
| Study 1 | LipM1、TongueM1 |
| Study 2 | Broca's area BA44、verbal memory region BA6、motor cortex targets |

TMS 采用 paired pulses，50 ms interpulse interval。建模时 `tms_target`、`tms_intensity`、TMS onset 与 stimulus onset 的相对时间必须作为 control / nuisance 信息进入数据表。

### 2.2 EEG 设备

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

当前完整派生数据已经重采样到 250 Hz，保存为 `data/derived/openneuro_full/ds006104/.../*_full_eeg.npz`。

### 2.3 EEG 事件标签

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

标签含义：

| 字段 | 用途 |
| --- | --- |
| `trial_type=TMS` | TMS pulse event |
| `trial_type=stimulus` | speech stimulus onset |
| `phoneme1/2/3` | single phoneme、CV、VC、CVC content unit |
| `category` | real / nonce 或 articulatory category |
| `manner` | stop 等发音方式 |
| `place` | bilabial、alveolar 等发音部位 |
| `voicing` | voiced / unvoiced |
| `tms_target` | control / nuisance condition |

## 3. 本地音频刺激

当前项目中，`ds006104` 的音频刺激位于：

```text
data/raw/openneuro/ds006104/stimuli/
```

该目录是本项目做声音形象建模时最重要的补充材料。它不是只给事件标签，而是给出了可直接分析的 wav 文件。

本地统计：

| 项目 | 数值 |
| --- | ---: |
| wav 总数 | 1273 |
| sample rate | 44100 Hz |
| sample width | 16 bit |
| mono 文件 | 1271 |
| stereo 文件 | 2 |
| median duration | 0.573 s |
| min duration | 0.163 s |
| max duration | 10.000 s |

目录结构：

| 目录 | wav 数 | 含义 |
| --- | ---: | --- |
| root | 482 | 扁平化复制的 CV/VC/CVC 音频 |
| `CV/` | 160 | consonant-vowel stimuli |
| `VC/` | 160 | vowel-consonant stimuli |
| `Words/` | 311 | real / nonce CVC words |
| `Controls/` | 160 | vowel/control/noise-like controls |

文件名直接携带 emotion/style 标签：

| 标签 | wav 数 |
| --- | ---: |
| angry | 639 |
| happy | 632 |
| other / noise | 2 |

示例：

```text
CV/CV i/Bi_angry1.wav
CV/CV i/Bi_happy1.wav
VC/VC i/cleaned/iB_angry1.wav
Words/Nonce i/Dit_happy1.wav
Controls/a_angry1_control.wav
```

这些音频允许构造以下声音形象标签：

| 标签 | 构造方式 |
| --- | --- |
| content unit | filename + `phoneme1/2/3` |
| CV / VC / CVC | directory + filename + event columns |
| real vs nonce | `category` 或 `Words/` 子目录 |
| emotion style | filename 中的 `happy` / `angry` |
| F0 | wav 提取 |
| duration | wav header |
| intensity / RMS | wav 提取 |
| timbre brightness | spectral centroid / bandwidth |
| voicing | `events.tsv` + pitch voiced ratio |
| articulatory features | `manner`、`place`、`voicing` |

## 4. 能支持什么实验

### 4.1 EEG token 内容保真度

任务：

```text
EEG window around speech onset
-> EEG tokenizer
-> probe phoneme / CV / VC / CVC / real-vs-nonce
```

可检验问题：

- token 是否区分 `/b/`、`/p/`、`/d/`、`/t/` 等 consonant。
- token 是否保留 vowel 信息。
- token 是否区分 CV 与 VC。
- token 是否区分 real word 与 pseudoword。

### 4.2 发音结构 probe

任务：

```text
EEG token -> manner / place / voicing
```

用途：

- 检验 tokenizer 是否学到 articulatory structure。
- 检查模型是否只利用低层声音能量，而没有捕捉 phonemic organization。
- 与 `Voice Image EEG Dataset` 中的 phoneme / syllable / word 维度对应。

### 4.3 音调、音色、情绪风格 probe

任务：

```text
EEG token -> F0 bin / brightness bin / happy-vs-angry / duration / energy
```

来源：

```text
data/raw/openneuro/ds006104/stimuli/*.wav
```

用途：

- 验证 EEG token 是否携带 pitch 线索。
- 验证 EEG token 是否携带 timbre brightness 线索。
- 验证 happy / angry 风格是否能从 EEG 中被线性或浅层模型读出。
- 为后续 voice image reconstruction 的 attribute heads 提供受控评估。

### 4.4 TMS confound / robustness

任务：

```text
EEG token -> speech label
conditioned on tms_target / tms_intensity
```

用途：

- 分离 speech-evoked response 与 TMS-evoked response。
- 检查模型是否错误利用 TMS target 预测语音类别。
- 在报告中给出 control-condition 结果。

## 5. 对当前模型的帮助

| 模型组件 | ds006104 的作用 |
| --- | --- |
| EEG tokenizer | 训练或验证短窗 speech-evoked EEG token |
| `ProbeHead` | phoneme、CV/VC/CVC、manner、place、voicing、happy/angry |
| `VoiceAttributeHead` | F0 high/low、brightness high/low、duration、RMS |
| nuisance/control branch | tms_target、tms_intensity、button response timing |
| evaluation | 检查 token 是否保留 content + voice attribute，而不是只重构 EEG |

最小模型闭环：

```text
EDF EEG + events.tsv + stimuli wav
-> speech-onset epochs
-> EEG tokenizer
-> discrete EEG tokens
-> content/articulation/style/acoustic attribute probes
```

这一步通过后，才具备进入 `ds005345` 自然语音 stream retrieval 和自建 `Voice Image EEG Dataset` 的基础。

## 6. 当前本地数据

原始 EEG / events：

```text
data/raw/openneuro/ds006104_datalad/
```

本地音频：

```text
data/raw/openneuro/ds006104/stimuli/
```

完整派生 EEG：

```text
data/derived/openneuro_full/ds006104/
```

当前完整派生文件：

```text
data/derived/openneuro_full/ds006104/sub-P01/ses-01/sub-P01_ses-01_task-phonemes_full_eeg.npz
data/derived/openneuro_full/ds006104/sub-S01/ses-02/sub-S01_ses-02_task-Words_full_eeg.npz
data/derived/openneuro_full/ds006104/sub-S01/ses-02/sub-S01_ses-02_task-phonemes_full_eeg.npz
data/derived/openneuro_full/ds006104/sub-S01/ses-02/sub-S01_ses-02_task-singlephoneme_full_eeg.npz
```

派生 manifest：

```text
data/derived/openneuro_full/recordings_manifest.csv
```

NPZ schema：

```text
eeg: float32 [channels, time]
sfreq: 250.0
ch_names: channel names
eeg_kind: raw
source: original EDF path
```

## 7. 不能支持什么

| 限制 | 影响 |
| --- | --- |
| 不是自然连续语音 | 不适合作为自然 voice reconstruction 主训练集 |
| 包含 TMS | speech label probe 必须控制 TMS 条件 |
| 音频多为短刺激 | 适合 phoneme / attribute probe，不适合长语义检索 |
| 没有同一完整句子的多说话人版本 | 不能单独解决 speaker identity 与 content 解耦 |
| 不是目标声音形象任务 | 不能直接证明无外部声波时的 voice image reconstruction |

## 8. 在项目路线中的位置

```text
ds006104
-> controlled speech unit + acoustic attribute probe
-> 验证 EEG token 是否有 content / pitch / timbre / style 信息
-> ds005345 single/mix stream retrieval
-> 自建 Voice Image EEG Dataset 的声音形象检索与重构
```

`ds006104` 的结论用于回答“模型是否具备读出声音基本属性的能力”。它不提供最终重构任务，但为最终任务提供最干净的低层验证。
