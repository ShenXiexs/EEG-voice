# 幻听声音形象恢复：数据集需求与初级实验路线

## 1. 目标重新定义

当前目标不是普通的 `EEG -> text`，也不是只恢复“说了什么”。幻听症患者真正重要的问题是：

```text
患者正在经历幻听
-> 记录 EEG/行为/主观报告
-> 解码幻听声音的内容、音调、音色、说话人属性和情绪风格
-> 重构一个可听的“声音形象”
```

因此主链路应写成：

```text
EEG -> token -> 与声音内容/音色/音调/说话人风格表征对齐 -> 幻听声音形象重构
```

其中“声音形象”至少包含：

- **内容**：音素、音节、词、短句、语音节奏。
- **音调**：F0、pitch contour、声调、语调、voicing。
- **音色**：谱包络、formant、MFCC/mel、粗糙度、明亮度。
- **说话人属性**：性别感、年龄感、熟悉/陌生、自我/他人、内外部定位。
- **情绪/威胁风格**：angry、neutral、comforting、commanding、mocking 等。
- **空间感**：inside head / outside head、left/right/front/back、距离感。

只做内容识别不能回答临床问题。对幻听来说，**“谁在说、用什么声音说、声音是否威胁、声音像不像真实外部声音”**往往比文本内容更关键。

## 2. 公开数据能解决什么，不能解决什么

公开 EEG 数据集基本不能直接提供“幻听时的真实声音标签”。幻听没有外部声波，因此不存在天然 ground-truth audio。公开数据只能先解决三个代理问题：

1. **听到真实声音时，EEG token 是否能对齐声音内容/音色/音调。**
2. **想象或内语音时，EEG token 是否能保留声音形象线索。**
3. **同一内容在不同说话人、不同情绪、不同音调条件下，EEG 是否能区分声音形象。**

最终临床目标必须补一个自采数据层：

```text
患者幻听实时标记 + 事后声音形象评分/匹配 + 个体化 voice bank
```

没有这层数据，模型只能叫“幻听声音形象的预训练/代理解码模型”，不能声称已经恢复了患者真实幻听声音。

## 3. 现有数据集定位

### A. `ds006104`：受控声音内容/音调/音色 probe

本地已有：

```text
data/meeting_examples/ds006104/stimuli/
```

本地扫描结果：

- 1273 个 WAV。
- happy 条件 632 个，angry 条件 639 个。
- control 条件 158 个。
- unit 类型包含 vowel、CV、VC、word_or_cvc。
- 大多数为 44.1 kHz 单声道短语音。

适合做：

- EEG token 是否区分 phoneme / CV / VC / CVC。
- EEG token 是否对齐 F0、voicing、spectral centroid、brightness、mel/mfcc。
- happy vs angry、original vs control 的 voice style probe。

限制：

- OpenNeuro events 里只有 `phoneme1/phoneme2/category/manner/place/tms_target` 等字段，没有精确到 `a_happy1.wav` 这样的 stimulus filename。
- 数据含 TMS，TMS pulse 和任务结构会成为强 confound。
- 不是幻听、不是自然内语音、不是多说话人身份库。

结论：

```text
ds006104 适合证明 EEG token 是否含声音内容/音调/音色线索；
不适合单独作为“幻听声音形象恢复”的最终数据集。
```

### B. Kara-One：想象/发声语音的关键代理集

GitHub EEG-Datasets 列表中提到的 **Kara-One** 很重要。它包含 EEG、face tracking、audio，任务是 imagined and vocalized phonemic and single-word prompts，14 名被试。

用途：

- 用 vocalized speech 的真实音频训练 `EEG/audio/content` 对齐。
- 用 imagined speech 测试没有外部声波时，EEG token 是否还能接近对应的语音内容或声音形象。
- face tracking 和 audio 能帮助剥离发声运动成分。

对幻听目标的价值：

- 它比纯听觉数据更接近“没有外部声音但脑内有语音表象”的问题。
- 但它是主动想象/发声，不是被动幻听；不能直接替代患者数据。

### C. Dryad-Speech：自然语音理解和 cocktail-party 代理集

GitHub EEG-Datasets 列表中的 **Dryad-Speech** 包括 Natural Speech、Reverse Speech、Cocktail Party、Speech in Noise、N400、multisensory 等实验。

用途：

- 训练连续自然语音 EEG encoder。
- 做 speech envelope / phoneme / word / semantic control。
- Cocktail Party 和 Speech in Noise 可用于声音竞争情境，接近患者幻听与外界声音竞争的临床现象。

对幻听目标的价值：

- 适合训练“听到外部声音”的语音跟踪能力。
- 不够支持幻听声音形象，因为说话人音色/音调多样性和主观声音身份标签不足。

### D. Imagined Emotion / Imagined Emotion with Continuous Data

GitHub EEG-Datasets 列表中提到的 **Imagined Emotion** 来自 HeadIT。描述为：被试听 voice recordings，这些声音诱发某种情绪，然后被试想象情绪场景或回忆相关经历。相关研究描述了 31 名被试、250-channel BioSemi EEG、256 Hz。

用途：

- 建模声音诱发的情绪/内部想象状态。
- 训练 voice emotion/style 的 EEG 表征。
- 作为幻听中“威胁性、责备性、命令性声音风格”的弱代理。

限制：

- 不是语音内容重构数据。
- 不是幻听患者。
- 声音材料和主观报告需要核对访问权限。

### E. `ds005345`：多说话人听觉选择与男/女声音色

虽然不在 EEG-Datasets GitHub 列表的核心条目中，但它对当前目标非常有用。

OpenNeuro 文件中有：

- `stimuli/single_female.wav`
- `stimuli/single_male.wav`
- `stimuli/mix.wav`
- `annotation/single_female_acoustic.csv`
- `annotation/single_male_acoustic.csv`
- `annotation/mix_acoustic.csv`
- word information CSV

用途：

- 男声/女声的 pitch range 和 timbre 对齐。
- mixed condition 中测试 attended voice reconstruction。
- 做从 EEG token 到 target speaker representation 的 contrastive retrieval。

限制：

- 只有一个男声和一个女声；不是丰富 speaker bank。
- multi-talker 条件更适合选择性注意，不足以训练细粒度 speaker identity。

### F. `ds006434`：自然双说话人 attention 和高精度 timing

该数据集有两个 audiobook narrator：A Wrinkle in Time female narrator 和 The Alchemyst male narrator；包括 diotic/dichotic attention、64 s trials、cortical/subcortical EEG。

用途：

- 高精度语音同步。
- attend male/female voice 的选择性注意。
- 训练 voice stream tracking，而不是只做文本内容。

限制：

- 仍然主要是两个 narrator，不是多说话人音色库。
- 目标更偏 auditory attention / brainstem response，不是幻听。

## 4. 真正需要采集的数据集长什么样

最终需要一个 **AVH Voice Image Dataset**。建议分四个 session。

### Session 1：个体化 voice bank 校准

给患者听一组可控语音：

- 同一内容由多个说话人朗读。
- 同一说话人用 neutral / angry / whisper / commanding / comforting 等风格朗读。
- 同一内容覆盖不同 F0、语速、音量、空间化位置。
- 内容包括中性短句、命令句、评价句、患者常见幻听语义类别。

每条刺激记录：

```text
speaker_id
content_id
transcript
emotion_style
F0 contour
mel / MFCC / speaker embedding
spatialization
patient rating: 像不像幻听声音
```

目的：

- 建立患者自己的“幻听声音候选空间”。
- 让模型以后不是凭空生成，而是在一个可解释的 voice manifold 中定位幻听声音。

### Session 2：外部声音感知 EEG

患者听 voice bank。

记录：

- EEG。
- EOG/EMG。
- 音频 onset/offset trigger。
- 每段后患者评分：像不像幻听、熟悉度、威胁性、内外部定位。

训练目标：

```text
EEG token -> content embedding
EEG token -> F0/pitch embedding
EEG token -> timbre/speaker embedding
EEG token -> hallucination-likeness score
```

### Session 3：声音想象 / 内语音 EEG

患者根据提示想象某个声音说某句话：

- 想象自己的声音。
- 想象陌生男性/女性声音。
- 想象最像幻听的声音。
- 想象 angry / whisper / commanding 风格。

这一步连接 Kara-One 和 Imagined Emotion 的代理逻辑。

训练目标：

```text
imagined EEG token -> 与对应 voice bank 表征对齐
```

### Session 4：幻听 symptom capture

患者在静息或轻任务中按键/滑条标记幻听：

- onset / offset。
- 是否清楚。
- 是否像外部声音。
- 几个说话人。
- 性别感。
- 年龄感。
- 熟悉/陌生。
- 情绪风格。
- 空间位置。
- 内容简述。
- 在 voice bank 中选择最像的声音，或对 Top-K 候选声音打分。

训练/评估目标：

```text
hallucination EEG segment
-> retrieve Top-K closest voice bank items
-> predict pitch/timbre/style ratings
-> generate candidate voice image
```

## 5. 初级模型设计

### 5.1 EEG tokenizer：参考 BrainOmni / LUNA

使用 BrainOmni 风格的 tokenizer：

```text
EEG x channels x time
-> temporal Conv/SEANet encoder
-> sensor/layout encoder
-> cross-attention channel compression
-> RVQ / VQ discrete EEG tokens
```

为什么需要这个设计：

- 患者 EEG 通道数和电极布局可能不一致。
- 幻听数据量小，需要先用公开数据预训练 tokenizer。
- token 比连续 EEG latent 更适合作为跨被试、跨任务的中间表示。

建议先做简化版：

```text
EEG 64/128ch -> temporal CNN -> 16 latent sources -> VQ codebook -> q_t
```

预训练损失：

```text
L_token = L_time + L_freq + L_pcc + L_vq
```

其中 `L_freq` 很重要，因为音色/音调重构依赖频域细节。

### 5.2 声音表征编码器

声音侧不要只用文本模型。至少拆成四个头：

```text
audio waveform
-> content encoder: HuBERT / wav2vec / phoneme posterior
-> pitch encoder: F0, log-F0, voicing
-> timbre encoder: speaker embedding, MFCC, mel spectral envelope
-> style encoder: emotion / threat / familiarity / hallucination-likeness
```

### 5.3 EEG-token 到声音表征对齐

参考 Défossez 等的 contrastive speech decoding：

```text
z_eeg = EEGTokenEncoder(q_t)
z_voice = VoiceEncoder(audio)
L_contrast = InfoNCE(z_eeg, z_voice)
```

但这里不能只对齐 wav2vec，因为 wav2vec 更偏内容。需要多目标对齐：

```text
L = lambda_content * L_content_contrast
  + lambda_pitch * L_f0
  + lambda_timbre * L_timbre_contrast
  + lambda_style * L_style
  + lambda_token * L_token
```

关键点：

- content 对齐解决“说了什么”。
- pitch/timbre/style 对齐解决“幻听声音像谁、什么音调、什么风格”。
- 幻听重构阶段主要依赖 pitch/timbre/style，而不是只依赖 transcript。

### 5.4 重构方式

第一阶段不要直接生成 waveform。先做检索式重构：

```text
hallucination EEG token
-> predicted content/pitch/timbre/style embedding
-> retrieve Top-K voice bank samples
-> patient rates which one最像
```

第二阶段再做生成式重构：

```text
content_hat + f0_hat + timbre_hat + style_hat
-> neural vocoder / voice conversion model
-> candidate hallucinated voice audio
```

## 6. 实验路线

### Phase 0：用现有公开数据做可行性

1. `ds006104`
   - 做 content / pitch / emotion-style probe。
   - 检查 EEG token 是否能预测 phoneme、CV/VC、F0 bin、happy/angry。

2. Kara-One
   - 做 imagined vs vocalized speech。
   - 训练没有外部声音时的 voice representation proxy。

3. Dryad-Speech / `ds005345` / `ds006434`
   - 做自然听觉语音对齐。
   - 训练连续 speech tracking、speaker stream tracking。

4. Imagined Emotion
   - 做声音诱发情绪与内部想象状态建模。
   - 用作幻听情绪风格的弱代理。

### Phase 1：患者个体化 voice bank

采集少量患者外部听觉数据：

```text
30-60 min EEG + 200-500 条可控 voice stimuli + patient similarity rating
```

输出：

- 患者的 hallucination-likeness voice manifold。
- 每个候选声音的 content/pitch/timbre/style embedding。

### Phase 2：幻听实时捕获

患者在 EEG 中实时标记幻听 onset/offset。

模型输出：

- Top-K 最像的 voice bank sample。
- 预测的 speaker/style/pitch 属性。
- 不输出单一“真相音频”，而输出带置信度的候选声音形象。

### Phase 3：闭环验证

每次模型生成/检索候选声音后，让患者评分：

```text
像不像：0-100
内容是否相似
音色是否相似
音调是否相似
情绪/威胁性是否相似
空间感是否相似
```

这一步是幻听声音形象恢复的核心评估，不是 BLEU、WER 或单纯分类准确率。

## 7. 数据集选择结论

当前已有公开数据不够直接完成幻听声音形象恢复，但足够搭建预训练和代理验证链路。

优先级：

| 优先级 | 数据集 | 作用 | 是否足够最终目标 |
| --- | --- | --- | --- |
| 1 | 自采 AVH symptom-capture + voice bank | 真正目标数据 | 必需 |
| 2 | `ds006104` | 内容/音调/情绪音色 probe | 不够，适合验证 token |
| 3 | Kara-One | imagined/vocalized speech proxy | 不够，但非常关键 |
| 4 | `ds005345` | 男/女声、混合说话人、attended voice | 不够，适合 speaker/timbre 代理 |
| 5 | `ds006434` | 双 narrator、attention、timing | 不够，适合 voice stream tracking |
| 6 | Dryad-Speech | 自然语音理解/cocktail party | 不够，适合连续 speech encoder |
| 7 | Imagined Emotion | 声音诱发情绪想象 | 不够，适合 style/affect proxy |

一句话结论：

```text
ds006104 + Kara-One + 自然多说话人听觉数据
可以训练“声音形象 EEG token 对齐模型”；
真正恢复幻听声音形象，必须新增患者 voice bank + symptom-capture 数据。
```

## 8. 下一步最小可行版本

一周内可做：

1. 用 `ds006104` 本地音频生成 content/pitch/timbre/style labels。
2. 训练一个 EEG-token probe：
   - phoneme/CV/VC。
   - happy vs angry。
   - F0 high/low。
   - centroid/brightness high/low。
3. 用 `ds005345` 的 single male/female/mix 做 speaker stream contrastive retrieval。
4. 写出患者采集 protocol：
   - voice bank 刺激表。
   - 幻听 onset/offset 标记界面。
   - 事后声音形象评分表。

如果这四步完成，讨论会可以清楚地说明：

```text
已有公开数据负责预训练和代理验证；
临床幻听声音恢复需要专门采集 patient-specific voice-image dataset。
```

## 参考来源

- EEG-Datasets GitHub list: https://github.com/meagmohit/EEG-Datasets
- Kara-One database: https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html
- Dryad-Speech dataset: https://datadryad.org/dataset/doi:10.5061/dryad.070jc
- HeadIT studies / Imagined Emotion: https://headit.ucsd.edu/studies
- OpenNeuro `ds006104`: https://openneuro.org/datasets/ds006104
- OpenNeuro `ds005345`: https://openneuro.org/datasets/ds005345
- OpenNeuro `ds006434`: https://openneuro.org/datasets/ds006434
- Défossez et al. speech decoding model: local `paper-ref/Défossez 等 - 2022 - Decoding speech perception from non-invasive brain recordings.pdf`
- BrainOmni tokenizer design: local `paper-ref/Xiao 等 - 2025 - BrainOmni A Brain Foundation Model for Unified EEG and MEG Signals.pdf`
