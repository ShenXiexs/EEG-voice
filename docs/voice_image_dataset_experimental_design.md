# Voice Image EEG Dataset 实验方案

## 1. 研究目标

Voice Image EEG Dataset 用于建立声音形象的 EEG 表征、检索与重构数据基础。研究对象为具备正常听力和正常语言理解能力的成人受试者；核心目标不是疾病分型，也不是单纯 `EEG -> text`，而是建立 EEG token 与声音内容、音调、音色、说话人属性、情绪风格和空间感之间的可检验映射。

核心链路：

```text
听到或想象某个 voice 时的 EEG
-> 离散 EEG token
-> 声音内容、音调、音色、说话人属性、情绪风格、空间感表征
-> voice bank 中的候选声音检索
-> 条件式声音形象重构
```

“声音形象”定义为一个可听见或可想象的 voice profile，而非单一文本内容。

| 维度 | 目标变量 |
| --- | --- |
| 内容 | phoneme、syllable、word、short phrase、speech rhythm |
| 音调 | F0、pitch contour、voicing、语调、声调 |
| 音色 | spectral envelope、formant、MFCC/mel、brightness、roughness |
| 说话人属性 | 性别感、年龄感、自我/他人、熟悉/陌生 |
| 情绪风格 | neutral、angry、commanding、comforting、mocking、whisper |
| 空间感 | inside-head-like、left/right/front、distance、externalization |
| 主观声音形象 | vividness、voice-image-likeness、confidence、controllability |

外部听觉任务存在客观 ground-truth audio；声音想象和声音记忆任务没有外部声波，标签由 cue、voice-bank item、受试者评分和 forced-choice matching 共同定义。所有重构结果以候选声音形象、声学属性和置信度形式报告。

## 2. 科学问题

### Q1. 外部声音感知

外部语音感知时，EEG token 对以下声音维度的可分性：

- 同一内容的不同说话人。
- 同一说话人的不同 F0、formant、timbre。
- 同一内容的 neutral、angry、commanding、whisper 风格。
- inside-head-like 与 outside-head-like spatialization。

### Q2. 声音想象 / 内语音

无外部声波条件下，声音想象 EEG token 与对应 voice-bank item 的内容、音调、音色和风格表征之间的相似性。

### Q3. 声音记忆与声音形象检索

受试者在延迟后重现或回忆某个目标声音时，EEG token 对以下标签的预测能力：

- voice-bank Top-K 匹配项。
- pitch、timbre、emotion style、speaker attributes。
- inside-head-like / outside-head-like 空间感。
- vividness、confidence、controllability。

### Q4. 重构有效性

模型检索的 Top-K 候选声音相似度高于随机候选、声学-only baseline、文本-only baseline 和 speaker-only baseline。

## 3. 受试者

### 3.1 主样本

| 阶段 | 样本量 | 目的 |
| --- | --- | --- |
| Pilot | 12-20 名成人受试者 | 任务可行性、trigger 精度、评分可靠性、声音想象任务稳定性 |
| Main | 60-100 名成人受试者 | 跨受试者模型、个体化模型、speaker/content/style 泛化 |

纳入标准：

- 年龄 18-65 岁。
- 正常或矫正正常听力。
- 能完成语音听觉、声音想象和评分任务。
- 具备知情同意能力并完成书面知情同意。
- 实验语言与受试者语言背景匹配。

排除标准：

- 严重听力障碍。
- 近期癫痫发作或 EEG 采集禁忌。
- 严重物质 intoxication 或 withdrawal。
- 基线评估显示无法稳定完成听觉或想象任务。
- 情绪化声音刺激显著增加不适或实验中止风险。

### 3.2 分层变量

| 分层变量 | 记录方式 | 用途 |
| --- | --- | --- |
| language background | 母语、熟练语言、方言背景 | 语音内容和声调建模 |
| voice imagery vividness | 0-100 评分或标准问卷 | 声音想象可解码性分层 |
| music/speech training | 年限、类型 | pitch/timbre 敏感性协变量 |
| familiarity with voices | self / familiar / unfamiliar | 说话人熟悉度建模 |
| sleep/caffeine/nicotine state | session 当日记录 | EEG 状态协变量 |

## 4. 实验结构

实验包含一个筛查环节和四个采集 session：

```text
Session 0: screening + 声音形象画像 + 基线问卷
Session 1: 个体化 voice bank 构建与评分
Session 2: 外部声音感知 EEG
Session 3: 声音想象 / 内语音 EEG
Session 4: 声音记忆与声音形象检索 EEG
```

Session 0 与 Session 1 按疲劳负荷安排在同一天或不同天。EEG session 分 2-3 天完成。每个 EEG session 包含中途休息、结束后状态评分和不适记录。

## 5. Session 0：筛查与声音形象画像

### 5.1 基线记录

记录项目：

- 年龄、性别、语言背景、利手。
- 听力筛查。
- 语音/音乐训练经历。
- 睡眠、咖啡因、尼古丁、酒精、近期应激状态。
- 声音想象 vividness baseline。
- 自我声音、熟悉声音和陌生声音的主观可想象程度。

### 5.2 声音形象画像

结构化字段：

| 字段 | 值域 |
| --- | --- |
| target_voice_type | self / familiar / unfamiliar / synthetic / mixed |
| perceived_gender | male / female / child / unknown / mixed |
| perceived_age | child / young / adult / elderly / unknown |
| familiarity | self / familiar person / public voice / stranger / synthetic |
| location_style | inside-head-like / externalized / left / right / front |
| loudness | whisper / soft / normal / loud / shouting |
| pitch | 0-100 visual analog scale |
| timbre | bright / dark / rough / breathy / metallic / hoarse |
| emotion_style | neutral / angry / commanding / mocking / comforting / whisper |
| speech_rate | slow / normal / fast |
| vividness | 0-100 |
| controllability | 0-100 |
| confidence | 0-100 |

该画像输出用于 voice bank 参数化、个体化候选声音选择和协变量建模。

## 6. Session 1：个体化 Voice Bank

### 6.1 刺激集合

voice bank 覆盖内容、音调、音色、说话人属性、情绪风格和空间感六类维度。

| 类型 | 内容 | 目的 |
| --- | --- | --- |
| Neutral phoneme/word | vowel、CV、VC、短词 | 低层音素、F0、formant 对齐 |
| Standard phrases | 1-3 秒中性短句 | 同内容多说话人比较 |
| Style phrases | neutral / angry / commanding / whisper | 情绪和风格编码 |
| Self/familiar voice phrases | 自我声音或授权熟悉声音 | 熟悉度和身份编码 |
| Synthetic controlled voices | 参数化 TTS / voice conversion | F0、formant、timbre 控制 |

高唤醒度或攻击性强的语义内容不进入默认刺激库。情绪风格通过语气、音高、强度和节奏操控表达，文本内容保持低风险和中性。

### 6.2 声音来源

声音库来源：

- 8-12 名真实说话人，覆盖性别、年龄层、音色差异。
- 受试者本人声音，在单独授权下采集。
- 熟悉声音，在声音提供者书面授权、去识别处理和伦理批准下采集。
- TTS / voice conversion 生成的参数化声音，用于控制 F0、formant、speaking rate、timbre。

每条语音保存字段：

```text
speaker_id
speaker_consent_type
speaker_gender
speaker_age_bin
speaker_identity_class
content_id
transcript
language
emotion_style
f0_shift_semitone
formant_shift_ratio
speaking_rate
intensity_db
spatial_azimuth
spatial_externalization
loudness_normalization_level
stim_file
deidentification_status
```

### 6.3 参数操控

标准短句包含以下操控水平：

| 操控 | 水平 |
| --- | --- |
| F0 | original、-4 semitones、+4 semitones |
| Formant | original、0.9x、1.1x |
| Emotion | neutral、angry、commanding、whisper |
| Spatialization | dry/inside-head-like、left 60°、right 60°、externalized front |

刺激库采用部分因子设计。每名受试者 200-500 条 voice-bank items。随机化约束包括：同一 speaker、同一 emotion style、同一 spatialization 连续出现不超过 2 次。

### 6.4 受试者评分

评分字段：

| 评分 | 范围 |
| --- | --- |
| voice_image_likeness | 0-100 |
| pitch_similarity | 0-100 |
| timbre_similarity | 0-100 |
| emotion_similarity | 0-100 |
| speaker_similarity | 0-100 |
| spatial_similarity | 0-100 |
| vividness | 0-100 |
| confidence | 0-100 |
| comfort | 0-100 |

评分流程：

```text
pairwise selection
-> Top-K candidate shortlist
-> 细粒度 0-100 评分
```

Pairwise selection 题目文本：

```text
请在两条声音中选择更接近目标声音形象的一条。
```

## 7. Session 2：外部声音感知 EEG

### 7.1 目的

建立外部语音感知条件下的监督对齐：

```text
外部语音 EEG token -> voice embedding
```

### 7.2 EEG 与音频采集参数

| 参数 | 方案 |
| --- | --- |
| EEG 系统 | 64 或 128 channel |
| 采样率 | 1000 Hz；最低 500 Hz |
| 参考 | 在线 Cz 或 mastoid；离线 average/mastoids |
| 阻抗 | active electrodes < 20 kΩ；passive electrodes < 5-10 kΩ |
| 辅助通道 | EOG、ECG、jaw/neck EMG、trigger、audio loopback |
| 音频呈现 | 插入式耳机或封闭耳机 |
| 声压级 | 个体听阈校准后约 60-70 dB SPL |
| 同步 | TTL trigger + audio loopback |

采集系统同时记录 TTL trigger 和 audio loopback。软件 trigger 不作为唯一同步依据。

### 7.3 Trial 设计

单 trial：

```text
fixation: 500-800 ms jitter
audio: 1-3 s
post-audio blank: 500 ms
rating cue: 1-3 个维度
ITI: 800-1500 ms jitter
```

Block 结构：

| Block | Trials | 内容 |
| --- | --- | --- |
| Passive listening | 80-120 | 只听；少量 catch question |
| Voice discrimination | 80-120 | 同内容不同 voice |
| Voice-image rating | 60-100 | Top candidate 声音细评分 |

总时长 45-70 分钟，包含休息。

### 7.4 对照条件

| 对照 | 用途 |
| --- | --- |
| same content, different speaker | 内容与音色分离 |
| same speaker, different content | 音色与内容分离 |
| same content/speaker, F0 shifted | 音调编码 |
| same content/speaker, formant shifted | 音色编码 |
| neutral vs angry/commanding | 风格编码 |
| speech-shaped noise | 低层声学控制 |
| silence / fixation | baseline |

## 8. Session 3：声音想象 / 内语音 EEG

### 8.1 目的

该 session 测量无外部声波条件下的声音表象 EEG，用于检验 EEG token 是否仍能指向目标 voice embedding 和声音属性。

### 8.2 Trial 设计

```text
cue voice_id + content_id: 1000 ms
imagery preparation: 500 ms
imagine voice speaking: 3000-5000 ms
button press when vivid voice starts, release when ends
rating: vividness / controllability / voice-image-likeness / confidence
ITI: 1500-2500 ms
```

任务条件：

| 条件 | 内容 |
| --- | --- |
| self voice | 想象自己说短句 |
| selected target voice | 想象 voice-bank 中的目标声音 |
| non-target voice | 想象明确不同于目标的声音 |
| pitch manipulation | 想象更高/更低的同一声音 |
| style manipulation | 想象 neutral / angry / commanding / whisper |

每名受试者 80-120 trials，分 4-6 个 block。

### 8.3 控制任务

非语音想象条件：

- 纯音想象。
- 环境声想象。
- 视觉图像想象。
- 静息 fixation。

这些条件用于区分 auditory voice imagery 与一般想象、注意和任务准备。

## 9. Session 4：声音记忆与声音形象检索 EEG

### 9.1 目的

该 session 用于建立更接近“声音形象重构”的无外部声波任务：受试者在听过目标声音后进行延迟回忆、silent replay 和候选声音匹配。

```text
target voice exposure
-> delay
-> silent voice replay / recall
-> EEG token
-> voice-bank Top-K matching
```

### 9.2 Trial 设计

```text
target cue or brief exposure: 1000-2000 ms
delay: 3000-8000 ms
silent replay: 3000-5000 ms
forced-choice voice matching: 2-4 candidates
attribute rating: pitch / timbre / speaker / emotion / spatiality
ITI: 1500-2500 ms
```

任务条件：

| 条件 | 内容 |
| --- | --- |
| immediate recall | 短延迟后重现目标声音 |
| delayed recall | 长延迟后重现目标声音 |
| content-fixed retrieval | 同内容多说话人候选 |
| speaker-fixed retrieval | 同说话人多内容候选 |
| attribute probe | F0、formant、emotion、spatialization 单维操控 |

### 9.3 声音形象事件标记

声音想象和回忆 episode 的 onset 和 offset 来自受试者实时标记，时间边界带有主观反应延迟。每个 episode 记录边界置信度和报告延迟。

| 字段 | 范围 |
| --- | --- |
| imagery_onset_confidence | 0-100 |
| imagery_offset_confidence | 0-100 |
| report_delay | 秒 |
| vividness_trace_file | slider 或 repeated-tap vividness 轨迹文件 |
| rating_time | 评分完成时间戳 |

### 9.4 按键运动混杂控制

声音想象 onset/offset 标记与按键运动同步出现，按键相关运动电位和肌电进入混杂变量。

控制数据：

```text
motor-only button task: 40-60 trials
random cue button press without imagery
slider movement calibration
```

分析标记：

```text
button_press_time
button_release_time
motor_control_condition
exclude_window_around_button: -500 to +1000 ms
```

### 9.5 Episode 数据切片

每个声音想象或回忆事件保存：

```text
pre_onset: -10 to 0 s
episode: onset to offset
post_offset: 0 to +10 s
matched_rest: duration-matched quiet segment
motor_exclusion_window: -500 to +1000 ms around button events
```

训练标签：

- target voice vs non-target voice。
- high-likeness vs low-likeness voice-bank retrieval。
- pitch / timbre / emotion / spatiality regression。
- content-preserving vs speaker-preserving retrieval。

## 10. 数据格式：BIDS + Phenotype + Stimuli

目录结构：

```text
VoiceImageEEG/
  dataset_description.json
  participants.tsv
  participants.json
  phenotype/
    baseline_voice_profile.tsv
    hearing_screening.tsv
    language_background.tsv
    session_state.tsv
  stimuli/
    voice_bank/
      sub-001/
        voice_0001.wav
        voice_0002.wav
    voice_bank_metadata.tsv
  sub-001/
    ses-screening/
    ses-voicebank/
      beh/
    ses-perception/
      eeg/
      beh/
    ses-imagery/
      eeg/
      beh/
    ses-retrieval/
      eeg/
      beh/
  derivatives/
    audio_features/
    eeg_preproc/
    voice_embeddings/
    eeg_tokens/
```

### 10.1 EEG `events.tsv`

每个 EEG task 的 `events.tsv` 包含以下核心列：

```text
onset
duration
trial_type
stim_file
content_id
speaker_id
voice_id
target_voice_id
candidate_voice_ids
correct_candidate_id
emotion_style
f0_shift_semitone
formant_shift_ratio
spatial_azimuth
spatial_externalization
response
response_time
button_press_time
button_release_time
voice_image_episode_id
imagery_onset
imagery_offset
imagery_onset_confidence
imagery_offset_confidence
report_delay
vividness_trace_file
rating_time
voice_image_likeness
pitch_rating
timbre_rating
emotion_rating
speaker_rating
spatial_rating
vividness_rating
confidence_rating
comfort_rating
motor_control_condition
```

### 10.2 EEG metadata

BIDS EEG metadata 包括：

```text
*_eeg.json
*_channels.tsv
*_electrodes.tsv
*_coordsystem.json
*_events.tsv
```

`channels.tsv` 记录 channel type、units、status、status_description。`electrodes.tsv` 与 `coordsystem.json` 记录电极空间坐标系统。`events.tsv` 记录所有刺激、反应、评分和声音形象 episode 事件。

### 10.3 Audio derivatives

每个 voice-bank item 提取：

```text
f0_mean
f0_median
f0_std
voiced_ratio
intensity_rms
speaking_rate
mfcc_1..mfcc_20
spectral_centroid
spectral_bandwidth
spectral_flatness
formant_f1_f2_f3
speaker_embedding
emotion_embedding
hubert_or_wav2vec_content_embedding
```

## 11. 预处理

### 11.1 EEG

原始数据保留在 raw BIDS 路径。预处理版本保存到 `derivatives/eeg_preproc/`。

流程：

```text
raw EEG
-> bad channel inspection
-> 0.1-40 Hz bandpass
-> 50/60 Hz notch
-> re-reference: average 或 mastoids
-> ICA / SSP 去眼动和心电
-> resample 到 250 Hz 或 200 Hz
-> epoch by audio onset / imagery cue / recall onset
-> reject high-amplitude muscle artifacts
```

保留文件：

- raw。
- cleaned continuous。
- epochs。
- bad channel list。
- ICA components。
- preprocessing log。
- trigger/audio-loopback delay report。

### 11.2 音频

```text
resample 到 16 kHz 或 24 kHz
loudness normalization
trim silence
保存原始版本和 normalized 版本
提取 F0 / mel / MFCC / speaker embedding / HuBERT-wav2vec embedding
```

## 12. 模型终点

### 12.1 外部听觉解码

任务：

```text
EEG segment -> retrieve correct voice-bank item among N candidates
```

指标：

- Top-1 / Top-5 retrieval accuracy。
- Mean reciprocal rank。
- Same-content different-speaker retrieval。
- Same-speaker different-content retrieval。

### 12.2 声音属性解码

任务：

```text
EEG token -> pitch_bin / timbre_bin / emotion_style / speaker_identity / spatial_location
```

指标：

- balanced accuracy。
- macro F1。
- Pearson/Spearman correlation for continuous ratings。
- permutation baseline。

### 12.3 声音形象重构

任务：

```text
imagery or recall EEG -> retrieve Top-K voice-bank candidates
```

主要检验：

```text
model Top-K similarity > random Top-K similarity
model Top-K similarity > acoustic-only baseline
model Top-K similarity > content-only baseline
model Top-K similarity > speaker-only baseline
```

## 13. 统计计划

### 13.1 单受试者模型

- Trial-level cross-validation。
- Episode-level leave-one-episode-out。
- Leave-one-content-out。
- Leave-one-speaker-out。
- Permutation test 建立 chance distribution。
- 按键运动混杂作为 nuisance regressor 或排除窗口。

### 13.2 跨受试者模型

```text
leave-one-subject-out
train on perception + imagery + retrieval data
adapt on subject-specific voice-bank perception data
test on subject-specific imagery/retrieval EEG
```

报告：

- effect size。
- confidence interval。
- permutation-corrected p value。
- calibration curve。

### 13.3 EEG 时频/源分析

解释性分析网络：

- bilateral superior temporal gyrus / auditory cortex。
- temporoparietal junction。
- inferior frontal gyrus。
- supplementary motor area。
- anterior cingulate / salience network。

脑区结果只作为与语音感知、声音想象、内语音、监控和显著性处理相关的解释性证据；不作为声音内容或身份的直接证明。

## 14. 安全与伦理

安全设置：

- 明确 stop rule。
- 任意刺激跳过权。
- 高威胁或辱骂文本排除在默认刺激库之外。
- 每个 session 后进行状态评分和不适记录。
- adverse events 记录。
- 自我声音、熟人声音和自由文本内容严格去识别。
- 含身份信息的 voice samples 不进入公开共享版本；共享版本包含匿名 metadata、派生特征和合成替代刺激。
- 模型输出仅作为研究终点，不作为个体身份识别、诊断、治疗决策或风险评估依据。

停止标准：

```text
comfort < 20/100 且持续
受试者要求停止
实验人员判断不适或疲劳明显上升
明显惊恐、激越或持续不适
```

## 15. Pilot 版本

| 模块 | 数量 |
| --- | --- |
| 成人受试者 | 12-20 |
| voice-bank items / subject | 200 |
| perception EEG trials | 180-240 |
| imagery EEG trials | 80-120 |
| recall/retrieval EEG trials | 80-120 |
| session state ratings | 每个 EEG block 前后 |

最小数据闭环：

```text
voice bank 声音 + 受试者相似度评分
外部听觉 EEG
声音想象 EEG
声音记忆 / retrieval EEG
voice-bank Top-K matching labels
```

无稳定声音想象报告的受试者保留在外部听觉解码和声音评分分析中，不进入 imagery/retrieval EEG 主分析。

## 16. 公开数据衔接

| 公开数据 | 在本 protocol 中的作用 |
| --- | --- |
| `ds006104` | phoneme、CV/VC、happy/angry、F0/timbre probe 预实验 |
| `ds005345` | single male/female/mix，说话人 stream retrieval 预实验 |
| Kara-One | imagined/vocalized speech proxy |
| Dryad-Speech / cocktail-party | 自然语音和竞争语音 tracking |
| Imagined Emotion | voice-induced emotion / imagery proxy |

公开数据用于预训练和任务完整性验证；Voice Image EEG Dataset 用于目标声音形象重构验证。

## 17. 参考

- EEG-BIDS 规范说明 `events.tsv`、`channels.tsv`、`electrodes.tsv` 和 `coordsystem.json` 对 EEG 数据复现的重要性。https://www.nature.com/articles/s41597-019-0104-8
- BIDS EEG specification 1.11.1。https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
- `ds006104` 提供 controlled phoneme、CV/VC、emotion-style speech stimuli 和 EEG 事件标签，用于声音内容、音调和音色 probe。
- `ds005345` 提供 single male、single female、mixed speech 条件，用于说话人 stream retrieval 和多说话人对齐。
