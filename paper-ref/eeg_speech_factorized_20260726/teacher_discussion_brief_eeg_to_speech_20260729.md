# EEG 语音解码的“内容—韵律”

## 1. 一分钟说明白我在想什么

同一句文本并不对应唯一声音。以 “Hi” 为例，发音可有不同的时长、停顿、响度、节奏、基频和音色；因此，直接让 EEG 回归整段波形或 Mel 频谱，面对“一对多”的目标很容易学成一个平均化的模板。

我想把目标拆成：

\[\text{audio}=G(C,P,V,R)]

- **C：内容**——词、音素或语义/语音 SSL units；
- **P：韵律/事件结构**——何时发声、起止、时长、能量包络，条件成熟后再加 F0；
- **V：音色/说话人**——先固定为一个中性或用户指定的 voice，不让模型从 EEG 猜；
- **R：细节残差**——由冻结的语音生成器补全，但不声称它来自 EEG。

所以第一步不是“生成一张漂亮声谱图”，而是问：**EEG 是否真的提供了内容 C，以及同一 label 内、逐 trial 的韵律 P？** 这两个问题成立后，再把它们送入成熟语音模型生成声音。

## 2. 我认为最值得做、也最容易讲清楚的研究问题

> 在严格跨被试划分下，scalp EEG 能否分别恢复语音内容 C 和同一文本 label 内的 trial-specific 韵律 P；并且二者是否都比“不使用 EEG”的对照更好？

这比“从 EEG 生成语音”更有可检验性，也保留了未来生成语音的出口。

## 3. 最小可行路线：先证明什么，再做什么

| 阶段         | 要证明的事                              | 输出/指标                                            | 成功标准                                               |
| ------------ | --------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| A. 内容 gate | EEG 含内容信息 C                        | label/音素分类；HuBERT/wav2vec2 表征 retrieval       | 显著优于 zero EEG、time-shuffle EEG、label prior       |
| B. 韵律 gate | 给定相同内容，EEG 仍含 trial-specific P | active mask、onset/offset、duration、energy envelope | 优于 label-average P 和 same-label shuffled EEG        |
| C. 组合生成  | 生成器确实用了 C/P                      | `pred C + pred P + fixed V` 生成音频               | 比`pred C + average P`、`zero EEG P`、检索基线更好 |

**建议的第一版 P：**只做 active mask、起止点、时长和 energy envelope。F0、共振峰、完整 Mel、音色都先不作为主结论。

## 4. 这和“先生成能量结构图”有什么关系？

这个直觉是对的，但应把“能量结构”说得更严格：它是低维韵律变量 P 的一部分，而不是完整声谱图。

- 合理：预测什么时候有声、持续多久、能量怎样随时间变化；这些目标与语速、重音和节奏有关。
- 不应混同：Mel 频谱还包含音素、共振峰、音色等；直接预测 Mel 仍然是高维、多模态问题，容易得到平均频谱。

因此可以表述为：**先尝试 EEG→内容 + 能量/时序结构；再由固定 voice 的语音生成器合成音频。**

## 5. 为什么我不直接主张“EEG 能还原个人声音”

scalp EEG 的空间分辨率和信噪比远低于 ECoG、sEEG 或 intracortical recording。侵入式研究已经能解码发音运动、pitch/formant/loudness，甚至文本、语音和头像；这说明分解路线在机制上合理，但它们只能作**上界和方法参照**，不能当作 scalp EEG 应达到的性能预期。

对当前数据，更可信的目标是：

1. 先做 subject-general 或 LOSO 的内容/低带宽韵律证据；
2. 音色固定；
3. 把语音生成视为展示接口，而不是 EEG 解码成功的唯一证据。

## 6. 最需要避免的“看起来成功”

- **平均模板塌缩**：不同 trial 输出几乎一样，只像该 label 的平均声音。
- **生成器替 EEG 补答案**：强语言模型、vocoder 或检索库在 EEG 为零时也能生成合理语音。
- **数据泄漏**：同一被试、stimulus、trial、音频派生特征或 retrieval bank 跨越 train/test。
- **overt speech 伪迹**：口面肌电、运动或声学串扰被误当为脑信号。

所以每个阶段都要有 zero EEG、time-shuffle、same-label shuffled、label-average 和 train-bank retrieval 对照；自然度或 ASR 正确率本身不能证明 EEG 条件有效。

## 7. 希望老师帮我拍板的四件事

1. **任务优先级**：先把“内容 C”做扎实，还是在现有数据条件允许时并行做低带宽 P？
2. **数据条件**：KaraOne/FEIS/ds004306 中，哪一个任务有真实、可对齐的 audio，且最适合作为第一主线？是否先限定 overt/reading，还是优先 imagined speech？
3. **泛化主张**：第一篇工作应坚持 LOSO/subject-disjoint，还是先做 subject-specific 的机制验证，再做跨被试？
4. **贡献边界**：老师是否认同把“内容 + trial-specific 韵律的可证伪证据”作为主贡献，而不是把“生成最自然的声音”作为主目标？

## 8. 可直接对老师说的 30 秒版本

“我现在觉得 EEG-to-speech 不能把文字和声音当作一样的 label 问题。文字内容相对唯一，但同一句话有很多声学实现；如果直接回归 Mel 或波形，很可能只能学到平均模板。所以我想先把声音拆成内容 C 和低维韵律 P：先证明 EEG 能否恢复内容，再在同一个文本下证明 EEG 是否还预测逐 trial 的发声时序和能量结构，最后用固定音色的成熟语音模型把 C 和 P 合成出来。关键不是让音频听起来像语音，而是通过 zero/shuffle/label-average 对照证明信息确实来自 EEG。您觉得这个问题应该先在哪个数据集、哪个任务上做最小验证？”

## 9. 讨论时可引用的 8 篇文献

| 作用                 | 文献                                                          | 一句话用途                                                                                |
| -------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 非侵入式对齐主线     | Défossez et al., 2023,*Nature Machine Intelligence*（S10） | EEG/MEG 对齐预训练 speech representation；支持先做表征/retrieval 而非完整 Mel。           |
| 可解释声学中间变量   | Chen et al., 2024,*Nature Machine Intelligence*（S12）      | ECoG→pitch/formant/loudness 等参数→speech；是 P 分支的重要方法参照。                    |
| 内容→发音→声音     | Anumanchipalli et al., 2019,*Nature*（S11）                 | articulatory intermediate 表明“先中间表征、再生成”是合理谱系。                          |
| scalp EEG 能量基线   | Wu et al., 2021,*EUSIPCO*（S18）                            | scalp EEG→多频带 envelope→vocoder；最接近“先能量结构”的直接证据，但样本与伪迹限制强。 |
| 大样本非侵入内容证据 | d’Ascoli et al., 2025,*Nature Communications*（S19）       | 非侵入 word/semantic embedding 解码可行，但 EEG 仍弱于 MEG。                              |
| 内容/音色可分        | Qian et al., 2019,*ICML*（S02）                             | 声音建模中的 content—voice factorization。                                               |
| 跨被试有效性         | Yin et al., 2025,*EMNLP*（S08）                             | 强调 cross-subject split 与派生数据<br />泄漏审计。                                       |
| 临床上界             | Metzger et al., 2023,*Nature*（S17）                        | 内容、声音与头像可作为不同输出；但为高质量侵入式 ECoG，不能直接外推。                     |

完整引用、PDF 与链接见同级目录中的 `eeg_speech_factorized_decoding_literature_review_20260726.md` 与 `MANIFEST.md`。
