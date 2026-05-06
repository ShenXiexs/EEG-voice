# AVH Voice Image Dataset 实验方案

## 1. 研究目标

AVH Voice Image Dataset 用于建立幻听声音形象的 EEG 表征、检索与重构数据基础。研究对象为存在 auditory verbal hallucinations 的患者，以及健康对照和非 AVH 精神病对照。

核心链路：

```text
幻听相关 EEG
-> 离散 EEG token
-> 声音内容、音调、音色、说话人属性、情绪风格、空间感表征
-> 个体化 voice bank 中的候选声音检索
-> 条件式声音形象重构
```

“声音形象”定义为多维语音体验，而非单一文本内容。

| 维度 | 目标变量 |
| --- | --- |
| 内容 | phoneme、syllable、word、short phrase、speech rhythm |
| 音调 | F0、pitch contour、voicing、语调、声调 |
| 音色 | spectral envelope、formant、MFCC/mel、brightness、roughness |
| 说话人属性 | 性别感、年龄感、熟悉/陌生、自我/他人 |
| 情绪风格 | neutral、angry、threatening、commanding、comforting、mocking |
| 空间感 | inside head / outside head、left/right/front/back、distance |
| 临床体验 | reality、loudness、distress、controllability、hallucination-likeness |

本数据集的主要标签不是客观 ground-truth audio。自然幻听没有外部声波，患者评分和 voice-bank matching 构成主观相似度标签。所有“重构”结果均以候选声音形象和置信度形式报告。

## 2. 科学问题

### Q1. 外部声音感知

外部语音感知时，EEG token 对以下声音维度的可分性：

- 同一内容的不同说话人。
- 同一说话人的不同 F0、formant、timbre。
- 同一内容的 neutral、angry、commanding、whisper 风格。
- inside-head-like 与 outside-head-like spatialization。

### Q2. 声音想象 / 内语音

无外部声波条件下，声音想象 EEG token 与对应 voice-bank item 的内容、音调、音色、风格表征之间的相似性。

### Q3. 幻听自然捕获

自然发生幻听时，EEG token 对患者事后报告的声音形象属性的预测能力：

- voice-bank Top-K 匹配项。
- pitch、timbre、emotion style、speaker attributes。
- inside / outside head、方位和距离感。
- distress、reality、controllability。

### Q4. 临床有效性

模型检索的 Top-K 候选声音相似度高于随机候选、声学-only baseline 和文本-only baseline。

## 3. 受试者

### 3.1 AVH 患者组

| 阶段 | 样本量 | 目的 |
| --- | --- | --- |
| Pilot | 8-12 名 AVH 患者 | 任务可行性、触发精度、评分可靠性、患者耐受性 |
| Main | 40-60 名 AVH 患者 | 跨患者模型、个体化模型、临床表型分析 |

纳入标准：

- 年龄 18-65 岁。
- 过去 1 个月存在反复 auditory verbal hallucinations。
- 幻听频率达到每周多次；每日或接近每日出现者记录为高频 AVH 亚组。
- 具备知情同意能力并完成书面知情同意。
- 抗精神病药物状态稳定，基线前 2 周无重大剂量或药物类别调整。
- 听力筛查达到语音刺激实验要求。

排除标准：

- 基线评估时存在急性自杀或他伤风险。
- 严重物质 intoxication 或 withdrawal。
- 严重听力障碍。
- 近期癫痫发作或 EEG 采集禁忌。
- 临床评估显示任务显著增加精神症状风险。

### 3.2 对照组

| 对照组 | 样本量 | 用途 |
| --- | --- | --- |
| 健康对照 | 20-40 | 外部声音感知和声音想象 baseline |
| 非 AVH 精神病对照 | 20-40 | psychosis-general 与 AVH-specific 表征区分 |

对照组完成 screening、voice bank rating、外部声音感知和声音想象任务；不进入幻听自然捕获主分析。

## 4. 实验结构

实验包含一个筛查环节和四个采集 session：

```text
Session 0: screening + 临床量表 + 幻听声音访谈
Session 1: 个体化 voice bank 构建
Session 2: 外部声音感知 EEG
Session 3: 声音想象 / 内语音 EEG
Session 4: 幻听自然捕获 EEG
```

Session 0 与 Session 1 按临床评估和疲劳负荷安排在同一天或不同天。EEG session 按患者耐受性拆分为 2-3 天。每个 EEG session 包含中途休息和结束后 distress check。

## 5. Session 0：筛查与临床表型

### 5.1 临床量表

记录项目：

- PSYRATS-AH 或 Auditory Hallucination Rating Scale。
- PANSS positive items 或 SAPS hallucination/delusion items。
- PHQ-9、GAD-7。
- Medication log：药物名、剂量、最近调整时间、服药时间。
- 睡眠、咖啡因、尼古丁、酒精、近期应激事件。
- Hearing screening。
- Handedness 与语言背景。

### 5.2 幻听声音形象访谈

结构化字段：

| 字段 | 值域 |
| --- | --- |
| voice_count | 1 / 2 / 3 / many |
| dominant_voice | 最常见或最困扰的声音 |
| perceived_gender | male / female / child / unknown / mixed |
| perceived_age | child / young / adult / elderly / unknown |
| familiarity | self / familiar person / stranger / unknown |
| location | inside head / outside / both / left / right / front / back |
| loudness | whisper / soft / normal / loud / shouting |
| pitch | 0-100 visual analog scale |
| timbre | bright / dark / rough / breathy / metallic / hoarse |
| emotion_style | neutral / angry / threatening / commanding / mocking / comforting |
| content_type | commentary / command / conversation / insult / warning / other |
| onset_predictability | 0-100 |
| controllability | 0-100 |
| distress | 0-100 |
| reality | 0-100 |

该访谈输出用于 voice bank 参数化和临床协变量建模。

## 6. Session 1：个体化 Voice Bank

### 6.1 刺激集合

voice bank 覆盖内容、音调、音色、说话人属性、情绪风格和空间感六类维度。

| 类型 | 内容 | 目的 |
| --- | --- | --- |
| Neutral phoneme/word | vowel、CV、VC、短词 | 低层音素、F0、formant 对齐 |
| Standard phrases | 1-3 秒中性短句 | 同内容多说话人比较 |
| Style phrases | neutral / angry / commanding / whisper | 情绪和威胁风格 |
| Patient-tailored safe phrases | 患者常见幻听语义类别的低强度版本 | 生态效度 |

高创伤性、辱骂性或命令性原句不进入默认刺激库。患者特异内容经临床人员审核后以低强度改写版本进入刺激库。

### 6.2 声音来源

声音库来源：

- 8-12 名真实说话人，覆盖性别、年龄层、音色差异。
- 患者本人声音，仅在单独授权下采集。
- 熟悉声音，仅在声音提供者书面授权、去识别处理和伦理批准下采集。
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
| Spatialization | inside-head dry、left 60°、right 60°、externalized front |

刺激库采用部分因子设计，每名患者 200-500 条 voice-bank items。随机化约束包括：同一 speaker、同一 emotion style、同一 spatialization 连续出现不超过 2 次。

### 6.4 患者评分

评分字段：

| 评分 | 范围 |
| --- | --- |
| hallucination_likeness | 0-100 |
| pitch_similarity | 0-100 |
| timbre_similarity | 0-100 |
| emotion_similarity | 0-100 |
| speaker_similarity | 0-100 |
| spatial_similarity | 0-100 |
| distress | 0-100 |
| reality | 0-100 |

评分流程：

```text
pairwise selection
-> Top-K candidate shortlist
-> 细粒度 0-100 评分
```

Pairwise selection 题目文本：

```text
请在两条声音中选择更接近本人幻听声音的一条。
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
| Hallucination-likeness rating | 60-100 | Top candidate 声音细评分 |

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

该 session 测量无外部声波条件下的声音表象 EEG，与幻听体验在感知来源上更接近。

### 8.2 Trial 设计

```text
cue voice_id + content_id: 1000 ms
imagery preparation: 500 ms
imagine voice speaking: 3000-5000 ms
button press when vivid voice starts, release when ends
rating: vividness / control / hallucination-likeness / distress
ITI: 1500-2500 ms
```

任务条件：

| 条件 | 内容 |
| --- | --- |
| self voice | 想象自己说短句 |
| selected AVH-like voice | 想象最像幻听的 voice-bank 声音 |
| non-AVH voice | 想象明确不像幻听的声音 |
| pitch manipulation | 想象更高/更低的同一声音 |
| style manipulation | 想象 neutral / angry / commanding |

每名受试者 80-120 trials，分 4-6 个 block。

### 8.3 控制任务

非语音想象条件：

- 纯音想象。
- 环境声想象。
- 视觉图像想象。
- 静息 fixation。

这些条件用于区分 auditory voice imagery 与一般想象、注意和任务准备。

## 9. Session 4：幻听自然捕获 EEG

### 9.1 采集原则

采集采用自然捕获范式，不采用强诱发或加重症状的刺激范式。

```text
自然出现幻听
-> 按键/滑条标记 onset、offset、强度
-> 事后完成声音形象评分和 voice-bank Top-K 匹配
```

### 9.2 实验环境

| 环境 | 时长 | 数据 |
| --- | --- | --- |
| Lab resting capture | 20-40 min | EEG、button/slider、episode rating |
| Ambulatory / EMA extension | 3-14 days | 真实生活中的幻听声音形象报告 |
| Mobile EEG sub-study | 1-3 days | 低密度移动 EEG + EMA，作为扩展数据 |

EMA 不等同于 EEG 采集；没有移动 EEG 的 EMA 数据只用于生态表型和 voice-bank matching 分析。

### 9.3 Lab capture 任务

```text
eyes-open fixation: 5 min
quiet rest: 20-30 min
low-demand task: 10-15 min, 由伦理批准版本指定
```

Response box 映射：

| 操作 | 含义 |
| --- | --- |
| press onset | 幻听开始 |
| release offset | 幻听结束 |
| slider / repeated taps | 实时强度 |
| stop button | 停止实验 |

每次 episode 后记录简短评分：

| 字段 | 范围 |
| --- | --- |
| clarity | 0-100 |
| loudness | 0-100 |
| reality | 0-100 |
| distress | 0-100 |
| inside_outside | inside / outside / both |
| left_right_front_back | categorical |
| number_of_voices | 1 / 2 / 3 / many |
| content_summary | free text if reported |
| top_voice_match | voice_bank item |
| top_voice_similarity | 0-100 |

长评分在 block 结束后完成。

### 9.4 Episode 时间边界

自然幻听 episode 的 onset 和 offset 来自受试者实时标记，时间边界带有主观反应延迟。每个 episode 记录边界置信度和报告延迟。

| 字段 | 范围 |
| --- | --- |
| onset_confidence | 0-100 |
| offset_confidence | 0-100 |
| report_delay | 秒 |
| intensity_trace_file | slider 或 repeated-tap 强度轨迹文件 |
| episode_rating_time | 评分完成时间戳 |

### 9.5 按键运动混杂控制

幻听 onset/offset 标记与按键运动同步出现，按键相关运动电位和肌电进入混杂变量。

控制数据：

```text
motor-only button task: 40-60 trials
random cue button press without hallucination
slider movement calibration
```

分析标记：

```text
button_press_time
button_release_time
motor_control_condition
exclude_window_around_button: -500 to +1000 ms
```

### 9.6 Episode 数据切片

每个幻听事件保存：

```text
pre_onset: -10 to 0 s
episode: onset to offset
post_offset: 0 to +10 s
matched_non_avh_rest: duration-matched quiet segment
motor_exclusion_window: -500 to +1000 ms around button events
```

训练标签：

- episode vs matched rest。
- high-likeness vs low-likeness voice-bank retrieval。
- pitch / timbre / emotion / spatiality regression。

## 10. 数据格式：BIDS + Phenotype + Stimuli

目录结构：

```text
AVHVoiceImage/
  dataset_description.json
  participants.tsv
  participants.json
  phenotype/
    clinical_scales.tsv
    avh_voice_interview.tsv
    medication.tsv
    sleep_substance_state.tsv
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
    ses-capture/
      eeg/
      beh/
  derivatives/
    audio_features/
    eeg_preproc/
    voice_embeddings/
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
emotion_style
f0_shift_semitone
formant_shift_ratio
spatial_azimuth
spatial_externalization
response
response_time
button_press_time
button_release_time
hallucination_episode_id
hallucination_onset
hallucination_offset
onset_confidence
offset_confidence
report_delay
intensity_trace_file
episode_rating_time
hallucination_likeness
pitch_rating
timbre_rating
emotion_rating
spatial_rating
distress_rating
reality_rating
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

`channels.tsv` 记录 channel type、units、status、status_description。`electrodes.tsv` 与 `coordsystem.json` 记录电极空间坐标系统。`events.tsv` 记录所有刺激、反应、评分和幻听 episode 事件。

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
-> epoch by audio onset / imagery cue / AVH onset
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
EEG token -> pitch_bin / timbre_bin / emotion_style / spatial_location
```

指标：

- balanced accuracy。
- macro F1。
- Pearson/Spearman correlation for continuous ratings。
- permutation baseline。

### 12.3 幻听声音匹配

任务：

```text
AVH episode EEG -> retrieve Top-K voice-bank candidates
```

主要检验：

```text
model Top-K similarity > random Top-K similarity
model Top-K similarity > acoustic-only baseline
model Top-K similarity > content-only baseline
```

## 13. 统计计划

### 13.1 单患者模型

- Trial-level cross-validation。
- Episode-level leave-one-episode-out。
- Permutation test 建立 chance distribution。
- 按键运动混杂作为 nuisance regressor 或排除窗口。

### 13.2 跨患者模型

```text
leave-one-subject-out
train on patients + controls
adapt on patient voice-bank perception data
test on patient AVH capture episodes
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

脑区结果只作为与语音感知、内语音、监控和显著性处理相关的解释性证据；不作为幻听内容的直接证明。

## 14. 安全与伦理

安全设置：

- 精神科医生或受训临床人员在场；远程值守时设置即时联系渠道。
- 明确 stop rule。
- 任意刺激跳过权。
- 高威胁/辱骂内容排除在默认刺激库之外。
- 每个 session 后进行 distress check。
- adverse events 记录。
- 患者声音、熟人声音、自由文本幻听内容严格去识别。
- 含身份信息的 voice samples 不进入公开共享版本；共享版本包含匿名 metadata、派生特征和合成替代刺激。
- 模型输出仅作为研究终点，不作为诊断、治疗决策或个体风险评估依据。

停止标准：

```text
distress > 80/100 且持续
受试者要求停止
临床人员判断风险上升
明显解离、激越、惊恐或自伤/他伤风险
```

## 15. Pilot 版本

| 模块 | 数量 |
| --- | --- |
| AVH 患者 | 8-12 |
| 对照 | 8-12 |
| voice-bank items / patient | 200 |
| perception EEG trials | 180 |
| imagery EEG trials | 80 |
| lab capture | 30 min |
| follow-up EMA | 7 days |

最小数据闭环：

```text
voice bank 声音 + 患者相似度评分
外部听觉 EEG
声音想象 EEG
自然幻听 onset/offset EEG episodes
EMA voice-bank matching reports
```

无 lab AVH episode 的受试者保留在外部听觉、声音想象和 EMA 表型分析中，不进入 episode-level EEG 解码主分析。

## 16. 公开数据衔接

| 公开数据 | 在本 protocol 中的作用 |
| --- | --- |
| `ds006104` | phoneme、CV/VC、happy/angry、F0/timbre probe 预实验 |
| `ds005345` | single male/female/mix，说话人 stream retrieval 预实验 |
| Kara-One | imagined/vocalized speech proxy |
| Dryad-Speech / cocktail-party | 自然语音和竞争语音 tracking |
| Imagined Emotion | voice-induced emotion / imagery proxy |

公开数据用于预训练和任务完整性验证；AVH Voice Image Dataset 用于临床目标验证。

## 17. 参考

- EEG-BIDS 规范说明 `events.tsv`、`channels.tsv`、`electrodes.tsv` 和 `coordsystem.json` 对 EEG 数据复现的重要性。https://www.nature.com/articles/s41597-019-0104-8
- BIDS EEG specification 1.11.1。https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
- PSYRATS auditory hallucination subscale 用于测量幻听的多维症状严重度。https://pubmed.ncbi.nlm.nih.gov/10473315/
- Auditory Hallucination Rating Scale 包含 frequency、reality、loudness、number of voices、content length 等维度。https://jamanetwork.com/journals/jamapsychiatry/fullarticle/207057
- AVH 现象学研究显示 voices 的 number、duration、location、loudness、rhythm、inside/outside location 等维度存在显著个体差异。https://pmc.ncbi.nlm.nih.gov/articles/PMC3885292/
- AVH 临床和非临床群体比较显示 negative content、distress、daily disruption、controllability 等临床维度属于独立记录维度。https://pmc.ncbi.nlm.nih.gov/articles/PMC3406519/
- Smartphone EMA/I hearing voices 研究支持在日常环境中捕获 voice-hearing experience。https://pubmed.ncbi.nlm.nih.gov/31890641/
