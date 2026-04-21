# EEG Token 对齐研究执行手册

这份文档是给当前仓库状态写的，不是泛泛谈方法。

你的工作已经跨过了最麻烦的第一步：原始 `EDF` 已经全部 hydrated 并完成了批量基础分析，当前记录是 `56` 份 EDF 全部下载并分析成功，失败数为 `0`：

- [EDF 批量运行摘要](../exploration_outputs/edf_full_analysis/batch_run_summary.json)
- [示例 EDF 摘要：P01 phonemes](../exploration_outputs/edf_full_analysis/sub-P01_ses-01_task-phonemes_eeg/sub-P01_ses-01_task-phonemes_eeg_summary.json)

这意味着你现在不再处于“能不能读数据”的阶段，而是正式进入：

**`EDF -> epoch -> EEG 表征 -> EEG token -> speech unit 对齐 -> 训练 -> 论文`**

## 1. 先把研究目标定对

当前最稳的目标不是：

`EEG -> mel / waveform reconstruction`

而是：

`弱监督 speech-unit EEG token 学习`

原因已经被你现有分析说明得很清楚：

- `trial-level` 可用
- `unit-level` 部分可用
- `frame-level` 当前不可用

见：

- [alignment_granularity.csv](../exploration_outputs/data_priority/tables/alignment_granularity.csv)
- [supervision_signal_matrix.csv](../exploration_outputs/data_priority/tables/supervision_signal_matrix.csv)

更直白地说：

1. 你已经有足够信息做 `stimulus onset` 级切窗。
2. 你已经有足够标签做 `phoneme / phoneme pair / word class` 级监督。
3. 你没有公开语音波形、音素边界、mel target，所以不适合把第一篇文章做成 `EEG-to-voice reconstruction`。

因此，这个项目最好的论文定位应该是：

`从离散 speech unit 出发，学习可泛化的 EEG token，并研究它与 speech token 的弱监督对齐关系。`

## 2. 这篇文章应该回答什么问题

建议把论文核心问题压缩成下面 3 个。

### Q1. EEG 里能不能学出稳定的 speech-unit token

也就是：

- 单音时，EEG 表征能不能稳定区分 `11` 个 single phoneme
- 双音时，EEG 表征能不能保留 `CV / VC` 的序列结构
- 词时，EEG 表征能不能保留 `real / nonce` 或 `CVC` 结构

### Q2. 这些 token 是否能和 speech-side token 对齐

这里的 `speech-side token` 在当前阶段先不要定义成音频 token，而应该定义成：

- `single-phoneme`: `phoneme1`
- `phonemes`: `[phoneme1, phoneme2]`
- `Words`: `[phoneme1, phoneme2, phoneme3]` 或 `real/nonce` 或 lexical base

### Q3. 这些 token 是否能跨被试、跨年份泛化

最有文章价值的不是单被试高分，而是：

- leave-one-subject-out 是否还能成立
- `2019 -> 2021` 的 shared `20` 个 `CV` units 是否可迁移
- `single-phoneme -> phoneme pairs -> words` 是否能形成层级迁移

## 3. 用哪些子集做，顺序怎么排

不要三个任务一起上。

按照你现有数据优先级，建议严格按下面顺序推进：

### 第一阶段：`2021 single-phoneme`

用途：

- 做最干净的 EEG token 起点
- 做表征学习、聚类、token purity、基础分类
- 做最早一版 leave-one-subject-out

优点：

- `16` 名被试
- `3355` trials
- `11` 类完全均衡
- 全部是 `control*` 条件

见：

- [candidate_subset_priority.csv](../exploration_outputs/data_priority/tables/candidate_subset_priority.csv)
- [ds006104_bids_analysis_cn.md](./ds006104_bids_analysis_cn.md)

### 第二阶段：`control-only phonemes`

用途：

- 做双音序列对齐
- 做 articulatory label 辅助任务
- 做 `2019 vs 2021` 的 shared `20` 个 `CV` units 跨年迁移

关键限制：

- 不是全 inventory 跨年对齐
- 只能落在 shared `20` 个 `CV` units

### 第三阶段：`2021 Words`

用途：

- 做更高层的 lexical / CVC token
- 做 `real vs nonce` 高层判别
- 做层级迁移的最后一层

关键限制：

- 不能只靠 BIDS `events.tsv` 恢复完整 `phoneme3`
- 词级 `CVC` 需要结合本地事件表

## 4. 你现在的数据资产到底够做什么

从当前仓库状态看，你已经具备下面 4 类关键资产。

### 4.1 连续 EEG 波形

这一块已经完成：

- 原始 `EDF` 全部下载成功
- 批量分析成功
- 每个文件都有基础摘要与统计图

这意味着你可以直接开展：

- 重采样
- 滤波
- rereference
- trial 切窗
- token 序列表征学习

### 4.2 事件时间

你已经有：

- `stimulus onset`
- `TMS onset`
- 任务、trial、条件标签

这足够支持：

- stimulus-locked epoch
- 条件分层采样
- 基于真实 onset 的 trial 对齐

注意：

`single-phoneme` 的 `TMS -> stimulus` 间隔不是固定值，所以不能用固定偏移代替真实 onset。

### 4.3 speech-side 离散标签

你已经有：

- `phoneme1`
- `phoneme2`
- `phoneme3`
- `place / manner / voicing`
- `real / nonce`
- lexical base

这足够支撑：

- 单标签分类
- 序列标签学习
- 多任务学习
- CTC 式弱监督对齐

### 4.4 目前还缺的监督

当前仍缺：

- 公开 stimulus audio 波形
- phoneme boundaries
- mel / vocoder target
- 被试自己发声的 paired audio

因此，第一篇文章不建议写成：

- `EEG -> waveform`
- `EEG -> mel`
- `EEG forced alignment to acoustic frames`

## 5. 研究总体路线

建议按下面 6 步走。

### Step 1. 先做统一的 epoch 数据集

目标：

把连续 EDF 变成统一格式的 `trial dataset`。

建议输出字段：

- `subject_id`
- `study_year`
- `task`
- `session_id`
- `trial_id`
- `tmstarget`
- `stimulus_onset_sec`
- `eeg_epoch`
- `sampling_rate`
- `phoneme1`
- `phoneme2`
- `phoneme3`
- `place`
- `manner`
- `voicing`
- `category`

epoch 建议：

- baseline: `-0.2s ~ 0s`
- analysis window: `0.05s ~ 0.45s` 相对 `stimulus onset`

如果要保守一点：

- 用 `50-450 ms` 做主分析
- 用 `0-600 ms` 做补充分析

理由：

- 避开 stimulus 前基线
- 留出早期 auditory / phonological processing
- 减少 TMS 脉冲附近污染

### Step 2. 做最小但严谨的 EEG 预处理

不要一开始上很复杂的手工 pipeline。

首版建议：

1. 只保留 `EEG` 通道
2. 重采样到 `250 Hz`
3. band-pass：`0.5-40 Hz` 或 `1-45 Hz`
4. notch：`50/60 Hz` 视电源噪声而定
5. common average reference 或保持现有参考后做对照
6. 按 session 或 subject 做 robust z-score
7. 记录坏 trial，而不是一上来过度删除

一定要留两个版本：

- `minimal-clean`
- `artifact-suppressed`

因为文章里常见风险是：

“模型只学到预处理副产物，而不是 speech-related EEG structure”

### Step 3. 先学连续 EEG embedding，不要一开始就硬量化

推荐顺序是：

1. `EEG encoder -> continuous embedding`
2. 用聚类把 embedding 量化成 token
3. 再做 token 和 speech unit 的对齐

不要一开始就上复杂 VQ-VAE，原因有两个：

1. 你先要知道 EEG 表征本身是否可分
2. 你先要知道合适的时间粒度和 token 数

#### 3.1 encoder 输入粒度

建议把一个 trial 的 EEG epoch 再切成短片段：

- window: `40 ms`
- hop: `20 ms`

如果采样率是 `250 Hz`：

- `40 ms = 10` 个采样点
- `20 ms = 5` 个采样点

对 `400 ms` 分析窗，大致能得到 `19` 个时间步，适合后续 CTC。

#### 3.2 encoder 结构

首版建议用简单但稳的结构：

- temporal conv
- spatial mixing
- transformer encoder 或轻量 BiLSTM

不要一开始堆太深。

第一版你需要的是：

- 能学出可聚类的 latent
- 能跨被试训练
- 能输出固定步长序列

### Step 4. 再把 EEG embedding 量化成 EEG token

最稳妥的量化策略不是 end-to-end VQ，而是：

`offline k-means`

具体做法：

1. 用 encoder 提取所有训练 trial 的 frame embeddings
2. 在训练集上做 `k-means`
3. 每个时间步映射到一个 cluster id
4. cluster id 序列就是 `EEG token sequence`

建议试的 `K`：

- `K = 16`
- `K = 32`
- `K = 64`
- `K = 128`

你需要比较：

- token purity
- NMI / ARI
- 下游分类性能
- 跨被试泛化

论文里不要把 cluster id 说成“真实神经音素”。

更严谨的说法是：

`discrete EEG latent units`

### Step 5. 定义 speech-side token

当前阶段的 speech-side token 应该分三层。

#### 层 1：single phoneme token

- `a`
- `i`
- `e`
- `u`
- `o`
- `s`
- `z`
- `p`
- `b`
- `t`
- `d`

#### 层 2：phoneme pair token sequence

例如：

- `[b, a]`
- `[a, b]`
- `[t, i]`

#### 层 3：word-level token sequence

例如：

- `[b, a, d]`
- `[p, e, t]`

或者更粗粒度：

- `real`
- `nonce`

当前阶段不要强行引入 acoustic token，除非你后面补到了 stimulus audio。

### Step 6. 用弱监督对齐，而不是逐帧强对齐

当前最合适的方法是：

`CTC`

因为你的输入天然是：

- 长度为 `T` 的 EEG token / EEG embedding 序列

输出天然是：

- 长度为 `U` 的 speech token 序列

并且：

- `T > U`
- 对齐未知
- 对齐单调

这正是 CTC 最适合的设置。

## 6. 推荐的训练范式

建议采用 3 阶段训练。

### 阶段 A：表征预训练

目标：

只学 EEG encoder，不做序列输出。

可选任务：

- masked reconstruction
- contrastive learning
- subject-invariant representation learning
- temporal order prediction

首版建议：

- `time-mask reconstruction`
- `contrastive trial discrimination`

因为实现成本低，且适合你的数据规模。

### 阶段 B：speech-unit 监督学习

先做简单监督任务：

- single-phoneme classification
- place / manner / voicing classification
- `real / nonce` classification

这一步的意义不是追最好分数，而是验证：

- encoder 学到的东西是否真的和 speech unit 有关

### 阶段 C：序列对齐训练

正式模型：

- 输入：EEG embedding sequence
- 输出：speech token sequence
- loss：`CTC + classification + contrastive`

推荐的总损失：

`L = L_ctc + lambda_1 * L_cls + lambda_2 * L_contrast + lambda_3 * L_domain`

其中：

- `L_ctc`: EEG 序列到 speech token 序列
- `L_cls`: 试次级 label 辅助监督
- `L_contrast`: 同 token 拉近、异 token 拉远
- `L_domain`: 减少 subject / year 信息泄露

## 7. 论文里的核心 baseline 应该怎么放

别把 baseline 设计成一堆很杂的模型。

建议 4 组 baseline 就够了。

### Baseline A. 传统判别式

- bandpower + SVM
- CSP/FBCSP + logistic regression

目的：

- 证明不是任何简单统计特征都能达到你的结果

### Baseline B. trial-level 深度分类

- EEGNet
- shallow convnet

目的：

- 给 single-phoneme / real-nonce 一个强但常见的下界

### Baseline C. 不量化的连续表征

- encoder + linear probe
- encoder + MLP classifier

目的：

- 检查离散 token 是否真的有增益

### Baseline D. 序列模型

- encoder + CTC
- encoder + attention pooling + classification

目的：

- 证明序列建模优于单一 trial pooling

## 8. 评价指标怎么选

建议分 4 层评价。

### 8.1 trial-level

- accuracy
- macro-F1
- balanced accuracy

### 8.2 token-level

- purity
- NMI
- ARI

### 8.3 sequence-level

- token error rate
- edit distance
- CTC decoding accuracy

### 8.4 retrieval-level

- EEG-to-token retrieval `R@1 / R@5 / MRR`

这类指标很重要，因为它能说明：

模型是否真的学到了共享表征空间，而不只是分类边界。

## 9. 数据划分和泛化实验必须怎么做

论文里最值钱的是泛化，不是随机划分高分。

至少做下面 4 套。

### Exp 1. subject-dependent

同被试内 train/test。

作用：

- 验证上限

### Exp 2. leave-one-subject-out

作用：

- 验证跨被试泛化

### Exp 3. cross-year on shared CV units

- train on `2019`
- test on `2021`

以及反过来

作用：

- 验证跨年份稳定性

### Exp 4. task curriculum transfer

- pretrain on `single-phoneme`
- finetune on `phonemes`
- evaluate on `Words`

作用：

- 证明层级 speech unit 学习路线是成立的

## 10. 文章最可能出的图表

建议至少准备下面这些。

1. 数据流程图：`EDF -> epoch -> EEG encoder -> EEG token -> CTC -> speech token`
2. 三个任务的标签层级图
3. token embedding 的 UMAP/t-SNE
4. token purity 随 `K` 变化图
5. leave-one-subject-out 主结果表
6. `2019 vs 2021` shared CV 迁移结果表
7. `single -> pair -> word` curriculum 迁移图
8. confusion matrix：single phoneme
9. articulatory feature 解码结果图
10. case study：单个 trial 的 EEG token 与目标 token 序列

## 11. 这篇文章最可能的标题方向

下面这些方向都比 “EEG-to-voice reconstruction” 更稳。

### 方向 A

`Weakly Supervised Alignment of EEG Tokens and Speech Units from Non-Invasive Speech Perception EEG`

### 方向 B

`Learning Discrete EEG Tokens for Hierarchical Speech-Unit Decoding`

### 方向 C

`From Single Phonemes to Words: Hierarchical EEG Token Learning for Speech Decoding`

## 12. 最容易踩的坑

### 坑 1

把 `speech token` 理解成必须来自音频 tokenizer。

当前不是。

当前更稳的是符号 speech unit token。

### 坑 2

把 `single-phoneme` 当成所有任务的代表。

它只是干净，不代表词级结构。

### 坑 3

忽视 subject leakage。

一定要明确：

- train subjects
- val subjects
- test subjects

### 坑 4

只看 accuracy，不看 token structure。

这会让文章退化成普通分类论文。

### 坑 5

强行往 `EEG->audio` 写。

当前公开监督不支持。

## 13. 你现在就该做的事

建议按下面顺序推进。

### 第 1 周

1. 统一 epoch 导出格式
2. 先完成 `2021 single-phoneme`
3. 做一个最小 trial dataset

### 第 2 周

1. 跑 `single-phoneme` 的线性 probe 和 EEGNet baseline
2. 训练连续 EEG encoder
3. 做 first-pass `k-means` tokenization

### 第 3 周

1. 跑 token purity / NMI / ARI
2. 跑 leave-one-subject-out
3. 做 `phonemes` 的双音序列任务

### 第 4 周

1. 上 `CTC`
2. 做 `2019/2021` shared CV 迁移
3. 做文章主图和主表

## 14. 一个足够稳的首版实现

如果你现在只想做第一版可运行系统，我建议如下。

### 数据

- 任务：`2021 single-phoneme`
- 输入：`50-450 ms` stimulus-locked EEG
- 采样率：`250 Hz`
- 通道：全部 EEG 通道

### 模型

- encoder：`Conv -> Transformer`
- output 1：trial classifier
- output 2：frame embedding
- tokenization：训练集 `k-means`
- sequence head：`CTC`

### loss

- `single-phoneme`: cross-entropy
- `phonemes`: CTC + auxiliary articulatory CE

### 结果

至少拿到：

- single-phoneme leave-one-subject-out
- token purity
- shared CV cross-year transfer

如果这三项成立，这篇文章就已经有骨架了。

## 15. 何时再去碰语音侧 token 或 audio

只有在下面条件满足时再往前走：

1. 你已经证明 EEG token 在当前数据上是稳定的
2. 你已经补齐 stimulus audio 或可复现实验音频
3. 你已经有逐 token 的更细粒度 speech-side 表征

那时可以升级到：

- HuBERT / wav2vec speech features
- EEG-to-speech retrieval
- EEG-to-acoustic embedding regression

但这应该是第二篇文章，不是第一篇。

## 16. 一句话版行动建议

你现在最正确的研究主线不是：

`从 EDF 直接做语音重建`

而是：

`先把 EDF 变成刺激锁定的 trial/sequence 数据集，先证明 EEG token 能稳定表征 single phoneme、phoneme pair 和 word-level speech units，再用 CTC 和对比学习做弱监督对齐。`
