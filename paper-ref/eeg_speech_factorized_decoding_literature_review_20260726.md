# EEG 到文本与语音重建：内容—韵律—音色分解的文献综述与研究定位

> 整理日期：2026-07-26
> 适用项目：KaraOne、FEIS、ds004306 等非侵入式 EEG→语音/文本研究
> CCF 分级依据：用户提供的 `/Users/samxie/Desktop/ccf_2022.json`
> 文档性质：面向研究设计的叙述性综述，不是 PRISMA 系统综述或临床证据指南

## 1. 核心结论

“一个文本 label 对应多种声音实现”是 EEG-to-speech 中非常关键的建模问题。文本内容通常可以表示为相对离散的词、音素或子词序列；声音却同时包含内容、节奏、时长、能量、基频、说话人音色和难以从 EEG 唯一确定的细节。因此，更合理的生成关系是：

\[A = G(C, P, V, R)]

其中：

- \(A\)：最终音频、频谱或 neural codec 表征；
- \(C\)：linguistic/phonetic content，如词、音素、HuBERT units；
- \(P\)：prosody/event structure，如有声区间、起止点、时长、能量包络、voiced/unvoiced、可选 F0；
- \(V\)：speaker/voice/timbre；
- \(R\)：EEG 很可能无法可靠确定的高频声学细节与随机残差。

现有文献已经分别证明：语音可以被分解为内容与风格表征 [S02, S03, S04]；神经信号经过发音运动或可解释声学参数等中间变量后可以合成语音 [S11, S12]；非侵入 EEG/MEG 更适合与预训练语音或语义表征进行对比对齐，而不是直接回归完整 Mel 频谱 [S10, S19]。但当前证据尚未充分解决以下联合任务：

> 在严格 subject-disjoint、无 label/audio 泄漏的条件下，由 scalp EEG 同时恢复内容 \(C\) 与 trial-specific 韵律 \(P\)，并证明二者对最终生成音频的贡献都来自 EEG。

这可以成为本项目最清楚的研究定位。

## 2. 为什么直接回归完整 Mel/波形容易失败

对相同内容（例如 “Hi”），训练集中可能存在不同说话人、语速、响度、基频和发音时长。如果模型以 MSE、SmoothL1 等点估计损失直接学习 `EEG → Mel`，而 EEG 又不足以唯一确定全部声学细节，模型的风险最小解往往是条件均值：

- 输出接近平均频谱或 label prototype；
- 不同 trial 的预测高度相关；
- `pred_std_ratio` 偏低；
- 生成音频听起来“像一个模糊模板”，但不包含 trial-specific 信息；
- 强 vocoder、语言模型或检索先验可能补出合理声音，从而掩盖 EEG 条件无效。

因此，“先生成能量结构图”是一个合理的降维方向，但必须区分两类目标：

1. **低带宽韵律目标**：active mask、onset/offset、duration、RMS/多频带 envelope。这些可以作为 \(P\) 分支的主要目标。
2. **完整 Mel 频谱**：除能量外还混合了音素、共振峰、音色和微观声学细节，仍是高维、多模态目标，不应被等同于“纯韵律”。

## 3. 检索范围与纳入标准

### 3.1 数据源

- 官方会议或论文库：NeurIPS Proceedings、PMLR、ACL Anthology、IEEE/ACM、ICASSP；
- 生物医学与综合期刊：Nature、Nature Neuroscience、Nature Machine Intelligence、Nature Communications、NEJM、PLOS Biology；
- 预印本：arXiv，仅用于识别最新方向，单独标记为“未完成同行评议”；
- CCF 分级：仅依据论文实际发表 venue 与 `ccf_2022.json` 的匹配结果，不把 arXiv 预印本按作者或主题推定为 CCF 论文。

### 3.2 纳入标准

- 直接研究 EEG/MEG/ECoG/sEEG 到文本、语音、频谱或可解释声学参数；
- 提供内容—风格分解、跨模态 speech-text alignment、speech SSL units 或 neural codec 的关键方法；
- 对跨被试划分、泄漏、生成评测或临床可行性具有直接启示；
- 优先 peer-reviewed 原始研究，并保留少量具有较高时效性的预印本。

### 3.3 主要排除项

- 仅做通用 EEG 分类、与语言/语音无直接关系的工作；
- 仅有二手报道、无法核对题名与出版信息的材料；
- 把听觉感知、overt speech、imagined speech 与临床 attempted speech 混为同一任务的结论；
- 只报告生成样例或 ASR 结果，却没有 zero/shuffled/prior 等条件有效性对照的强结论。

## 4. CCF-A 相关论文

| ID  | 论文                                                                                                                                                                                                                                                     | Venue/等级          | 核心方法                                                                          | 对本项目的直接启示                                                           | 主要边界                                                    |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------- |
| S01 | [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://proceedings.neurips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf)                                                                  | NeurIPS 2020，CCF-A | 在原始波形 latent 上进行 masking、quantization 与 contrastive learning            | 可把中间层序列作为 EEG 对齐目标，替代直接 Mel 回归                           | wav2vec 2.0 表征混合声学与语言信息，不是天然“纯内容”      |
| S02 | [AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://proceedings.mlr.press/v97/qian19c.html)                                                                                                                                       | ICML 2019，CCF-A    | 通过信息瓶颈与 speaker embedding 分离内容和说话人风格                             | 支持\(C/V\) 分解；可借鉴 bottleneck 与 counterfactual voice swapping         | 分离并不等于 EEG 能预测对应变量；需要独立 probe             |
| S03 | [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://aclanthology.org/2022.acl-long.393/)                                                                                                                        | ACL 2022，CCF-A     | speech/text modal-specific pre/post-net + shared encoder-decoder + cross-modal VQ | 可建立 text、speech unit 和 EEG 的共同语义空间；适合“先内容、后生成”       | 模型本身不是神经解码模型，迁移到 EEG 需要额外对齐与数据控制 |
| S04 | [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html) | NeurIPS 2023，CCF-A | 将 style 建模为 latent random variable，并使用 diffusion 生成                     | 支持把同 label 下的多种发声视为条件分布；style prior 可补全 EEG 不可辨识细节 | 适合作为生成后端，不应据此声称 EEG 解出了 style             |

### CCF-A 证据综合

CCF-A 论文提供的最强启示不是“直接把更大的语音模型接在 EEG 后面”，而是重新定义监督目标：用 speech SSL 表征和 speech-text shared space 表示内容，用受控潜变量表示 style，再让生成模型从条件分布中采样。对小样本 EEG 而言，语音大模型应主要作为冻结 teacher、表征器、vocoder 或 prior，而不是从头联合训练的主体。

## 5. CCF-B 相关论文

| ID  | 论文                                                                                                                                     | Venue/等级                 | 核心方法                                                 | 对本项目的直接启示                                                                                                                              | 主要边界                                                                      |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| S05 | [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://doi.org/10.1109/TASLP.2021.3122291) | IEEE/ACM TASLP 2021，CCF-B | masked prediction + offline clustering 获得离散语音单元  | 可用 middle-layer features、k-means units、unit-CTC 表示\(C\)                                                                                   | HuBERT units 同样混合声学和上下文信息，应做层选择与内容/说话人 probe          |
| S06 | [SoundStream: An End-to-End Neural Audio Codec](https://doi.org/10.1109/TASLP.2021.3129994)                                               | IEEE/ACM TASLP 2022，CCF-B | fully convolutional codec + residual vector quantization | 适合作为冻结的最终 waveform decoder 或\(A\) 空间                                                                                                | 完整 codec token 含大量细节，直接让小样本 EEG 全量预测容易塌缩                |
| S07 | [Speech Synthesis Using EEG](https://doi.org/10.1109/ICASSP40776.2020.9053340)                                                            | ICASSP 2020，CCF-B         | GRU 从 EEG 特征回归声学特征；包含 spoken/listening EEG   | 是直接`EEG → acoustic feature → speech` 的早期基线                                                                                          | 仅四名被试的 subject-specific 结果；overt speech 可能混入肌电和运动伪迹       |
| S08 | [Rethinking Cross-Subject Data Splitting for Brain-to-Text Decoding](https://aclanthology.org/2025.emnlp-main.289/)                       | EMNLP 2025，CCF-B          | 重新审计 brain-to-text 的 cross-subject split 与数据泄漏 | 应把 subject、trial、刺激、label 和派生缓存的边界同时锁定；validation/test 不得进入 prototype、retrieval bank、normalization 或 teacher fitting | 主要针对 brain-to-text 数据设置，需将规则具体化到音频配对与生成链路           |
| S09 | [NeuroIncept Decoder for High-Fidelity Speech Reconstruction from Neural Activity](https://doi.org/10.1109/ICASSP49660.2025.10888547)     | ICASSP 2025，CCF-B         | CNN+GRU 从高伽马侵入式神经记录重建音频频谱               | 可借鉴时频特征与 spectrogram decoder 结构                                                                                                       | 研究对象是 invasive EEG/high-gamma，不应与 scalp EEG 的信噪比和空间分辨率类比 |

### CCF-B 证据综合

CCF-B 文献中，S05–S06 更适合作为预训练语音 teacher 与生成后端；S07、S09 是直接神经到语音的工程参照；S08 则决定实验结论是否可信。对当前项目，S08 的重要性可能高于再增加一个更复杂的 decoder，因为错误划分或隐性检索泄漏会让任何生成指标失去解释性。

## 6. Nature、医学与关键神经解码论文

| ID  | 论文                                                                                                                                                | 信号/任务                                            | 输出                                           | 关键发现与项目启示                                                                                      | 证据边界                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| S10 | [Decoding speech perception from non-invasive brain recordings](https://www.nature.com/articles/s42256-023-00714-5)                                  | EEG/MEG；被动听语音                                  | speech representation retrieval                | 预训练 wav2vec 2.0 表征、对比目标和多被试训练优于直接 Mel 回归；支持 sequence-level shared space        | 解码的是 speech perception；不能直接外推到 imagined/attempted production |
| S11 | [Speech synthesis from neural decoding of spoken sentences](https://www.nature.com/articles/s41586-019-1119-1)                                       | ECoG；spoken/silently mimed speech                   | articulatory kinematics→acoustics→speech     | 显式发音运动中间变量提高有限数据下的合成效果；支持“先可解释中间表征，再合成”                          | 侵入式记录且依赖精确任务条件，是 scalp EEG 的方法上界而非性能基线        |
| S12 | [A neural speech decoding framework leveraging deep learning and speech synthesis](https://www.nature.com/articles/s42256-024-00824-8)               | ECoG；48 名临床受试者、多种 speech task              | pitch、formant、loudness 等参数→spectrogram   | 与本项目的\(P\) 分支最接近：先学习可解释 speech parameters，再使用 differentiable synthesizer           | 仍是 ECoG；论文的参数可预测性不能直接视为 scalp EEG 可预测性             |
| S13 | [Reconstructing Speech from Human Auditory Cortex](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001251)                    | ECoG；听觉感知                                       | continuous auditory/spectrogram representation | 证明时频/声谱表征可以作为神经重建目标，尤其是与 intelligibility 相关的频谱成分                          | 感知任务与侵入式信号；不能证明 imagined EEG 可重建同等细节               |
| S14 | [Semantic reconstruction of continuous language from non-invasive brain recordings](https://www.nature.com/articles/s41593-023-01304-9)              | fMRI；感知/想象语言                                  | 连续语义文本                                   | 证明“先恢复意义而非逐字复制”可行，为\(C\) 分支提供概念依据                                            | fMRI 时空特性与 EEG 不同，且需要大量个体化训练数据                       |
| S15 | [Neuroprosthesis for Decoding Speech in a Paralyzed Person with Anarthria](https://www.nejm.org/doi/full/10.1056/NEJMoa2027540)                      | 植入式 ECoG；attempted speech                        | 50 词 vocabulary 的文本/句子                   | 证明 neural decoder 与语言模型组合可以先恢复内容；临床上文本路径有独立价值                              | 单名参与者、小词表、侵入式；语言模型增益不能归因于脑信号本身             |
| S16 | [A high-performance speech neuroprosthesis](https://www.nature.com/articles/s41586-023-06377-x)                                                      | intracortical arrays；attempted speech               | 高速文本                                       | 展示 phoneme/word-level 内容解码和语言模型整合的临床上界                                                | 侵入式、少数参与者；不构成非侵入 EEG 的性能预期                          |
| S17 | [A high-performance neuroprosthesis for speech decoding and avatar control](https://www.nature.com/articles/s41586-023-06443-4)                      | 高密度 cortical surface recordings；attempted speech | text、speech audio、facial avatar              | 证明内容、声音与面部运动可作为互补输出，而非单一 waveform 任务                                          | 临床植入式系统；多模态输出依赖高质量局部皮层信号                         |
| S18 | [A Method To Map EEG Signals To Spoken Speech Using Gaussian Process Modeling](https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0001140.pdf) | scalp EEG；普通话 overt speech                       | 多频带 temporal envelopes→vocoder             | 与“先能量结构再生成声音”最直接；支持把低维 envelope 作为\(P\) 目标                                    | 样本小、任务受限、overt EMG 风险高，未证明 imagined 或强跨被试泛化       |
| S19 | [Towards decoding individual words from non-invasive brain recordings](https://doi.org/10.1038/s41467-025-65499-0)                                   | EEG/MEG；阅读与听语音，723 名参与者                  | individual-word semantic embeddings            | 大规模结果支持开放词汇语义表征方向，同时显示 MEG 优于 EEG、reading 优于 listening，数据量与协议影响显著 | 仍依赖已知 word onset，且跨数据集联合训练没有自动带来提升                |

### 关于 Science 论文的处理

当前纳入文献中，Nature 系列、NEJM、PLOS 和 IEEE/ACL/NeurIPS 的论文比检索到的 Science 论文更直接对应“神经信号→文本/语音”问题。因此本综述没有为了期刊名称而纳入关系较弱的 Science 论文。后续如扩展到 speech motor cortex、auditory cortical entrainment 或临床 BCI 机制综述，可再单独补充 Science/Science Translational Medicine 的相关机制工作。

## 7. 最新预印本与监测项

### S20：NeuroSonic（2026）

[NeuroSonic: Conditional Flow Matching for EEG-to-Speech Reconstruction](https://arxiv.org/abs/2606.24087) 提出 shared EEG/audio token space、time-conditioned gated Transformer 与 conditional flow matching，并报告跨被试 EEG-to-speech 结果。它与当前计划中的 conditional transport 很接近。

但该工作截至本综述整理时仍是 2026 年 arXiv 预印本，尚不能作为“问题已经解决”的决定性证据。建议重点核查：

- 数据集中的 EEG 是 listening、overt、imagined 还是其他任务；
- cross-subject 是否同时隔离 stimulus、trial 与派生 audio prior；
- EEG-conditioned 是否显著优于 zero EEG、shuffled EEG、same-label shuffled 和 dataset-only prior；
- 生成质量增益是否仍包含 trial-specific 内容或韵律，而非更强的生成器带来的无条件改善；
- 是否存在面部/眼动/肌电伪迹驱动语音结构的可能。

预印本后 LLM 时代的引用污染风险较高，因此本条只作为前沿监测项，不承担核心论点。

## 8. 主题综合

### 8.1 内容与声音实现必须分开评价

AutoVC、SpeechT5 和 StyleTTS 2 从语音建模侧表明，content、speaker/style 与 acoustic realization 可以使用不同条件或 latent 表示 [S02–S04]。Moses、Willett 等临床工作则说明，内容解码本身就是独立且有价值的 BCI 输出 [S15–S16]。两条证据共同支持：EEG→文本和 EEG→声音不应只用一个最终 waveform 指标混合评价。

### 8.2 “能量结构”应被定义为低维、可证伪的韵律变量

Wu 等人的多频带 envelope 重建 [S18] 和 Chen 等人的可解释 speech-parameter synthesizer [S12] 支持先预测低维声学参数。最稳妥的顺序是 active mask、onset/offset、duration、energy envelope，再根据 voiced 片段的信号质量决定是否加入 F0。完整 Mel、formant trajectory、speaker timbre 与 codec residual 应位于更晚阶段。

### 8.3 预训练表征优于从 EEG 学习全部声学空间

Défossez 等人的结果明确比较了直接 Mel、端到端 Deep Mel 与预训练 speech representation，支持使用 pretrained representation 与 contrastive learning [S10]。wav2vec 2.0、HuBERT 和 SpeechT5 提供了可冻结的 teacher 空间 [S01, S03, S05]。这与 KaraOne 小样本条件高度一致。

### 8.4 非侵入式 EEG 与侵入式神经记录必须分层解释

ECoG/sEEG/intracortical 工作提供了“脑中确实存在可解码内容、发音运动和声学参数”的机制与工程上界 [S11–S17]；scalp EEG 工作更接近当前可行性边界 [S07, S10, S18, S19]。不能因为侵入式系统能恢复 speech audio，就推断 scalp EEG 也携带相同带宽的信息。

### 8.5 生成模型会产生“看似成功但条件无效”的风险

neural codec、style diffusion 和语言模型能够从弱条件生成自然语音 [S04, S06, S15]。因此，生成自然、ASR 识别正确或频谱看起来合理，只能说明整个系统会生成语音，不能单独证明 EEG 提供了内容或 trial-specific 韵律。条件有效性必须通过反事实对照建立。

## 9. 证据—论点映射

| 论点 ID | 可写入论文的论点                                                                      | 主要证据                                        | 建议引用位置                                                        | 风险/限定                                                    |
| ------- | ------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------ |
| C1      | 相同 linguistic content 对应多种声学实现，适合用 factorized latent 或条件分布建模。   | S02, S03, S04                                   | Introduction：问题定义；Related Work：speech factorization          | 语音模型中的可分解性不保证 EEG 中存在全部因子                |
| C2      | 非侵入 EEG/MEG 与预训练 speech representation 的对比对齐优于直接回归低层 Mel。        | S10，方法基础由 S01 支撑                        | Related Work：non-invasive decoding；Method：target representation  | S10 主要是 perception，且 MEG 表现通常优于 EEG               |
| C3      | 发音运动或可解释声学参数可作为 neural-to-speech 的有效中间变量。                      | S11, S12                                        | Related Work：invasive upper bound；Method：prosody/parameter heads | 两项均为 ECoG，不能直接外推性能                              |
| C4      | 低维 speech envelope 能从部分 scalp EEG speech-production 设置中估计并用于 vocoding。 | S18，早期神经网络基线 S07                       | Related Work：direct EEG-to-speech；Method：P branch                | 小样本、overt EMG、subject-specific 等问题限制结论           |
| C5      | Brain-to-text 的跨被试划分可能产生数据泄漏和性能高估。                                | S08                                             | Method：data split；Experiments：leakage audit                      | 需将其规则扩展到 audio cache、retrieval bank 和生成 prior    |
| C6      | 生成语音的自然度或 ASR 正确率不能单独证明 EEG 条件有效。                              | S04, S06, S15 与生成链路的综合推论              | Evaluation：negative controls；Discussion：interpretation           | 属于跨文献方法论推论，应明确说明是本研究的识别设计           |
| C7      | 当前尚缺少严格验证的 scalp EEG、subject-general、内容+trial-specific 韵律联合恢复。   | S07, S10, S18, S19 与侵入式 S11–S17 的边界比较 | Introduction：research gap；Discussion：contribution                | 这是基于当前纳入语料的结论，不应写成穷尽全球文献后的绝对断言 |

## 10. 建议的研究问题与假设

### RQ1：内容可解码性

在严格 subject-disjoint 的条件下，EEG 是否包含高于 zero/shuffled/prior 基线的内容信息 \(C\)？

- H1a：EEG→prompt/phoneme classification 高于 label-prior 与 shuffled EEG；
- H1b：EEG→HuBERT/wav2vec2 semantic sequence 的检索或 CTC 指标高于 mean-query；
- H1c：模型的内容增益不能由 subject identity、trial order 或 audio bank 泄漏解释。

### RQ2：条件于内容后的韵律可解码性

给定真实或可靠预测的内容 \(C\)，EEG 是否额外包含 trial-specific \(P\)？

- H2a：active/onset/duration/energy 指标优于 same-label shuffled EEG；
- H2b：EEG-predicted \(P\) 比 label-average \(P\) 更接近同一 trial 的音频；
- H2c：F0 仅在 voiced 区间、信号质量与控制实验通过后作为次级终点。

### RQ3：组合生成的条件有效性

`predicted C + predicted P + fixed/neutral V + frozen decoder` 是否比以下对照更好：

- `predicted C + label-average P`；
- `true C + zero/shuffled EEG P`；
- `dataset prior only`；
- `same-label random audio/prosody`；
- `train-bank nearest-neighbor retrieval`。

## 11. 建议的模型结构

```text
EEG sequence
  -> subject-robust EEG encoder
  -> C head: phoneme/text/speech-SSL semantic units
  -> P head: active + onset/offset + duration + energy (+ optional F0)
  -> uncertainty/gating

[C, P, fixed or externally controlled V]
  -> conditional flow / factorized codec generator
  -> frozen neural codec decoder or vocoder
  -> waveform
```

关键限制：

- \(V\) 默认固定为 neutral voice，或由外部用户选择；不使用 subject ID 作为偷懒路径；
- \(A\) 的不可辨识细节由生成 prior 补全，不能宣称全部来自 EEG；
- decoder 先在纯音频上训练并冻结，EEG encoder 必须先通过 \(C/P\) 的独立 gate；
- thinking EEG 不承担 frame-accurate waveform loss，优先学习 \(C\) 与低带宽 \(P\)；
- overt EEG 必须额外进行口面肌电、声学串扰和运动伪迹审计。

## 12. 训练与评测建议

### 12.1 数据划分

- 主结论采用 subject-holdout 或 LOSO；
- validation subject 用于模型选择，test subject 只在最终冻结后使用；
- normalization、prototype、k-means、retrieval bank、audio teacher 微调和 codec statistics 均只使用 train split；
- 若做 unseen-label，必须同时避免相同 audio/stimulus 的派生数据出现在训练集；
- 每个 split 保存 manifest 和 hash，避免预处理重跑后边界漂移。

### 12.2 语义指标

- prompt/phoneme top-k accuracy；
- semantic-unit edit distance 或 CTC loss；
- HuBERT/wav2vec2 sequence retrieval R@1/R@5/MRR；
- EEG→text 与 EEG→speech 的内容一致性；
- same-label/different-subject 与 different-label/same-subject 分层评价。

### 12.3 韵律指标

- active mask AUROC/F1；
- onset/offset error、duration MAE；
- energy envelope correlation/MAE，避免只报告 post-hoc best shift；
- voiced/unvoiced F1；
- F0 RMSE/correlation 仅在 voiced 区间报告；
- 与 label-average、same-label shuffled、zero EEG 比较 paired gain。

### 12.4 生成指标

- MR-STFT、MCD、STOI 等声学指标；
- ASR CER/WER 作为 intelligibility 辅助指标；
- speaker similarity 只用于检查 neutral/fixed voice 是否稳定，不作为 EEG 成功证据；
- 人类听辨时分别评价“说了什么”“节奏/语调是否匹配”“自然度”；
- 必须同时报告 oracle C/P、predicted C/oracle P、oracle C/predicted P、predicted C/P 四种条件，以分离误差来源。

### 12.5 反事实与负对照

- zero EEG；
- within-trial time shuffle；
- channel shuffle/dropout；
- same-label different-trial EEG；
- same-subject different-label EEG；
- text-only/label-only；
- dataset/subject prior；
- train-bank retrieval；
- overt 条件下的高频/口面 EMG proxy 与去伪迹前后对照。

## 13. 当前研究空白

1. **经验空白**：缺少在 scalp EEG 上同时通过内容与 trial-specific 韵律 gate 的 subject-general 证据。
2. **方法空白**：多数工作在 text、retrieval、envelope 或 waveform 中选择单一输出，较少显式建模 \(C/P/V/R\) 的可辨识边界。
3. **评测空白**：生成质量与条件有效性常被混合；强 decoder 可在 EEG 无效时仍生成合理语音。
4. **数据空白**：现有 imagined-speech 数据通常 label 少、重复次数有限、EEG/audio 并非严格 frame-aligned。
5. **泛化空白**：MEG、ECoG、sEEG 与 scalp EEG 之间存在显著信噪比和空间分辨率差异，跨模态迁移尚无稳定结论。
6. **临床空白**：健康被试的 listening/overt/imagined 结果不能直接代表失语或瘫痪患者的 attempted speech。

## 14. 建议阅读顺序

### 第一优先级：直接决定研究设计

1. S10：非侵入式 brain–speech 对齐为什么应使用预训练表征与对比学习；
2. S12：如何把声音拆为可解释、低维 speech parameters；
3. S08：怎样避免 cross-subject 数据泄漏；
4. S18：scalp EEG→多频带 envelope 的直接历史基线；
5. S03：speech/text shared representation；
6. S11：中间 articulatory representation 的价值。

### 第二优先级：实现 teacher 与生成后端

1. S01、S05：wav2vec 2.0/HuBERT semantic teacher；
2. S06：neural codec；
3. S02、S04：content/style factorization 与条件生成。

### 第三优先级：临床上界与结果表述

1. S15–S17：侵入式 speech neuroprosthesis 的临床上界；
2. S14、S19：非侵入式语义内容解码；
3. S20：最新 conditional flow EEG-to-speech 方向，仅作待验证前沿。

## 15. 文献质量与使用建议

本综述采用面向技术主张的证据等级，而不是把工程论文机械套入临床 RCT 金字塔：

- **A级使用**：经过同行评议、方法与目标直接支持当前主张，可作为主要证据；
- **B级使用**：经过同行评议但存在信号模态、任务或样本外推问题，可作为方法/机制证据；
- **C级使用**：预印本、小样本或评测控制不足，仅用于提出假设或追踪前沿。

| 来源组        | 建议等级 | 使用方式                                                                    |
| ------------- | -------- | --------------------------------------------------------------------------- |
| S01–S06、S08 | A/B      | 方法学主要证据；注意它们本身不是 EEG 临床证据                               |
| S10–S19      | A/B      | 按信号模态分层引用；侵入式工作只作机制和上界                                |
| S07、S18      | C/B      | 直接 scalp EEG 证据，但需显式报告小样本、overt EMG 和 subject-specific 限制 |
| S09           | B        | peer-reviewed 但为 invasive EEG，不能作为 scalp EEG 性能基线                |
| S20           | C        | 2026 arXiv 预印本，只作前沿监测和复现实验候选                               |

## 16. 参考文献

1. Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). [wav2vec 2.0: A framework for self-supervised learning of speech representations](https://proceedings.neurips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf). *NeurIPS 33*.
2. Qian, K., Zhang, Y., Chang, S., Yang, X., & Hasegawa-Johnson, M. (2019). [AutoVC: Zero-shot voice style transfer with only autoencoder loss](https://proceedings.mlr.press/v97/qian19c.html). *Proceedings of ICML 36*, 5210–5219.
3. Ao, J., Wang, R., Zhou, L., et al. (2022). [SpeechT5: Unified-modal encoder-decoder pre-training for spoken language processing](https://aclanthology.org/2022.acl-long.393/). *Proceedings of ACL 60*, 5723–5738. https://doi.org/10.18653/v1/2022.acl-long.393
4. Li, Y. A., Han, C., Raghavan, V., Mischler, G., & Mesgarani, N. (2023). [StyleTTS 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html). *NeurIPS 36*.
5. Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). [HuBERT: Self-supervised speech representation learning by masked prediction of hidden units](https://doi.org/10.1109/TASLP.2021.3122291). *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29*, 3451–3460.
6. Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., & Tagliasacchi, M. (2022). [SoundStream: An end-to-end neural audio codec](https://doi.org/10.1109/TASLP.2021.3129994). *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30*, 495–507.
7. Krishna, G., Tran, C., Han, Y., Carnahan, M., & Tewfik, A. H. (2020). [Speech synthesis using EEG](https://doi.org/10.1109/ICASSP40776.2020.9053340). *ICASSP 2020*, 1235–1238.
8. Yin, C., Yu, Q., Fang, Z., Peng, C., & Li, P. (2025). [Rethinking cross-subject data splitting for brain-to-text decoding](https://aclanthology.org/2025.emnlp-main.289/). *EMNLP 2025*, 5675–5689. https://doi.org/10.18653/v1/2025.emnlp-main.289
9. Khanday, O. M., Pérez-Córdoba, J. L., Mir, M. Y., Najar, A. A., & Gonzalez-Lopez, J. A. (2025). [NeuroIncept decoder for high-fidelity speech reconstruction from neural activity](https://doi.org/10.1109/ICASSP49660.2025.10888547). *ICASSP 2025*, 1–5.
10. Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J.-R. (2023). [Decoding speech perception from non-invasive brain recordings](https://www.nature.com/articles/s42256-023-00714-5). *Nature Machine Intelligence, 5*, 1097–1107. https://doi.org/10.1038/s42256-023-00714-5
11. Anumanchipalli, G. K., Chartier, J., & Chang, E. F. (2019). [Speech synthesis from neural decoding of spoken sentences](https://www.nature.com/articles/s41586-019-1119-1). *Nature, 568*, 493–498. https://doi.org/10.1038/s41586-019-1119-1
12. Chen, X., Wang, R., Khalilian-Gourtani, A., et al. (2024). [A neural speech decoding framework leveraging deep learning and speech synthesis](https://www.nature.com/articles/s42256-024-00824-8). *Nature Machine Intelligence, 6*(4), 467–480. https://doi.org/10.1038/s42256-024-00824-8
13. Pasley, B. N., David, S. V., Mesgarani, N., et al. (2012). [Reconstructing speech from human auditory cortex](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001251). *PLOS Biology, 10*(1), e1001251. https://doi.org/10.1371/journal.pbio.1001251
14. Tang, J., LeBel, A., Jain, S., & Huth, A. G. (2023). [Semantic reconstruction of continuous language from non-invasive brain recordings](https://www.nature.com/articles/s41593-023-01304-9). *Nature Neuroscience, 26*, 858–866. https://doi.org/10.1038/s41593-023-01304-9
15. Moses, D. A., Metzger, S. L., Liu, J. R., et al. (2021). [Neuroprosthesis for decoding speech in a paralyzed person with anarthria](https://www.nejm.org/doi/full/10.1056/NEJMoa2027540). *New England Journal of Medicine, 385*, 217–227. https://doi.org/10.1056/NEJMoa2027540
16. Willett, F. R., Kunz, E. M., Fan, C., et al. (2023). [A high-performance speech neuroprosthesis](https://www.nature.com/articles/s41586-023-06377-x). *Nature, 620*, 1031–1036. https://doi.org/10.1038/s41586-023-06377-x
17. Metzger, S. L., Littlejohn, K. T., Silva, A. B., et al. (2023). [A high-performance neuroprosthesis for speech decoding and avatar control](https://www.nature.com/articles/s41586-023-06443-4). *Nature, 620*, 1037–1046. https://doi.org/10.1038/s41586-023-06443-4
18. Wu, H., Pan, C., Li, M., & Chen, F. (2021). [A method to map EEG signals to spoken speech using Gaussian process modeling](https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0001140.pdf). *EUSIPCO 2021*, 1140–1144.
19. d’Ascoli, S., Bel, C., Rapin, J., et al. (2025). [Towards decoding individual words from non-invasive brain recordings](https://doi.org/10.1038/s41467-025-65499-0). *Nature Communications, 16*, 10521.
20. Gao, W., Wang, Y., Ma, Y., Yang, C., Li, W., & You, C. (2026). [NeuroSonic: Conditional flow matching for EEG-to-speech reconstruction](https://arxiv.org/abs/2606.24087). *arXiv preprint arXiv:2606.24087*.

## 17. 综述限制与使用声明

- 本文是围绕当前 KaraOne/FEIS 研究问题构建的定向叙述性综述，没有声称穷尽所有 EEG speech decoding 文献。
- 不同论文的任务包括 listening、reading、overt、imagined、silently mimed 和 clinical attempted speech；结果不可直接横向比较。
- EEG、MEG、ECoG、sEEG 和 intracortical arrays 的信号质量差异很大，文中已尽量分层标注。
- CCF 等级只描述 venue 分类，不等于研究与本项目的相关性或证据强度。
- arXiv 预印本已明确单列，未按同行评议论文处理。
- 本文由 AI 辅助进行文献检索、元数据核对、结构化整理与研究综合；正式投稿前，应由研究者阅读全文、核查实验设置，并使用 Crossref/出版方页面再次确认参考文献格式。
