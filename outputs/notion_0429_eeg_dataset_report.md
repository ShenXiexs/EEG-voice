**Last updated:** 2026-04-28

## 0429 讨论会目标

围绕 EEG 信号与音频/语音/音乐信号的跨模态解码，确定第一阶段可落地的数据集组合、EEG tokenization 方案、时间/语义对齐评估方式，以及一周内可执行的最小实验路线。

本页采用数据可用性、标注密度、对齐粒度、下载/解析可复现性四个标准进行筛选。所有数据集判断均附带本地探测脚本和证据文件，便于后续复核。

## 最终建议

第一阶段不建议把所有公开 EEG-audio 数据直接合并训练。推荐采用“主线自然语音 + 受控音位验证 + 同步/AAD/音乐扩展”的三层结构。

**核心训练与评估集合：**

1. `ds004718 LPPHK`：自然粤语语音，词级 timing、POS、词频、f0/intensity、trigger-sentence 映射齐全，最适合词边界/韵律/语义对齐。
2. `ds005345 LPP Multi-talker`：普通话单说话人与多说话人条件，适合 speaker selection、multi-talker attention、语义检索。
3. `ds004408 continuous naturalistic speech`：英语自然语音，有 TextGrid 音素/词级标注，适合自然语音 EEG encoder/tokenizer 预训练。
4. `ds006104 EEG dataset for speech decoding`：受控 phoneme/articulation 数据，适合验证 token 是否编码音位、发音方式、发音部位，但不适合作为自然连续语音主训练集。

**同步和注意解码基线：**

- `ds006434`：用于高精度 timing、attention trigger、64 s epoch、皮层/亚皮层速率差异验证。
- KUL `4004271`、DTU `1199011`、255ch `4518754`：用于 envelope/AAD baseline、跨房间泛化、高密度空间编码验证。

**音乐扩展：**

- OpenMIIR：优先用于 beat/downbeat/tempo/meter 对齐。
- `ds003774 MUSIN-G`：用于自然音乐 EEG 自监督预训练和弱监督情感/熟悉度分析。
- MAD-EEG `4537751`：用于 target instrument attention，小规模补充。

## 本地可复现资产

代码、原始探测结果、详细证据和下载片段均已保存在仓库内。

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 探测脚本 | `scripts/probe_eeg_audio_datasets.py` | 远端元数据探测、WAV header 解析、CSV/TSV/XLSX/TextGrid 预览、raw EEG byte-range 采样、artifact 保存 |
| 机器可读结果 | `outputs/eeg_audio_dataset_probe_results.json` | 13 个数据集的完整探测 JSON，含 URL、解析字段、artifact path、SHA256、partial 标记 |
| 简表报告 | `outputs/eeg_audio_dataset_probe_results.md` | 数据集级摘要，适合快速查看 |
| 详细报告 | `outputs/eeg_audio_dataset_probe_detailed.md` | target 级证据表，含每个下载目标的本地 artifact 路径和解析摘要 |
| 原始证据目录 | `outputs/probe_artifacts/` | 61 个实际保存的远端小文件/片段；完整小文件直接保存，大文件保存 header 或 byte-range 片段 |
| 既有研究报告 | `eeg-datasets.md` | 前一版数据集研究报告，含更长的方法论说明 |
| 论文目录 | `paper-ref/` | 本地参考论文 PDF，包括 Defossez、DeWave、DELTA、LUNA、BrainOmni、Moreira 等 |

复现命令：

```bash
python3 scripts/probe_eeg_audio_datasets.py   --json-out outputs/eeg_audio_dataset_probe_results.json   --md-out outputs/eeg_audio_dataset_probe_results.md   --detail-md-out outputs/eeg_audio_dataset_probe_detailed.md   --artifact-dir outputs/probe_artifacts
```

脚本设计原则：不默认下载完整 EEG 大文件；对 `.eeg/.edf/.set/.fif/.wav` 等大型文件只读取 header 或 byte-range；对 CSV/TSV/XLSX/TextGrid/README/脚本等小文件直接保存；所有保存内容带 SHA256，便于复核。

本次探测状态：13 个数据集全部完成，61 个 artifact 已落盘，target 级错误数为 0。

## 数据集实测证据总表

| Dataset | Source | Priority | 实测证据 |
| --- | --- | --- | --- |
| `ds004408` | OpenNeuro | core | `audio01.TextGrid` 有 2465 个非空标签；WAV header 显示 44.1 kHz、双声道、约 177.6 s；run01 EEG sidecar 显示 512 Hz；BrainVision header 显示 128 channels |
| `ds005345` | OpenNeuro | core | word information CSV 有 9 列；acoustic CSV 有 time/f0/intensity；single-female WAV 44.1 kHz、单声道、603 s；raw EEG sidecar 显示 500 Hz、Cz reference、Brain Products |
| `ds004718` | OpenNeuro | core | word timing xlsx 为 4474 x 12；trigger-sentence xlsx 为 556 x 2；acoustic CSV 有 time/f0/intensity；句子 WAV 44.1 kHz；raw EEG sidecar 显示 1000 Hz、average reference |
| `ds006104` | OpenNeuro | controlled | events TSV 有 12 列，含 trial_type、phoneme1、phoneme2、manner、place、tms_target；raw EDF byte-range 可读；EEG sidecar 显示 2000 Hz、CPz reference |
| `ds006434` | OpenNeuro | alignment | events TSV 有 9 列，含 chapter_ind、att_story、att_side、EEG_trigger；stim WAV 约 64 s；cortex EEG sidecar 显示 500 Hz、TP9/TP10 reference |
| `ds003774` | OpenNeuro | music | MUSIN-G events/channels/eeg sidecar 可读；EEG sidecar 显示 250 Hz、Cz reference；ESong WAV header 可读 |
| `ds007591` | OpenNeuro | secondary | participants/events/channels/EDF header 可读；events 含 session_type、task_condition；EEG sidecar 显示 256 Hz、g.tec system |
| KUL `4004271` | Zenodo | baseline | README 和 preprocessing script 已保存；README 明确提示随机切片交叉验证会产生 trial-specific overfitting |
| DTU `1199011` | Zenodo | baseline | `preproc_data.m` 已保存，展示 EEG/audio alignment 入口；Zenodo 文件包括 AUDIO、EEG、DATA_preproc |
| 255ch `4518754` | Zenodo | spatial | `misc.zip` 与 `scripts.zip` 已保存；包含 255ch channel layout 和加载脚本 |
| ESAA `7078451` | Zenodo | secondary | readme、preprocess zip、CNN baseline zip 已保存；readme 显示 17 subjects、32 trials/subject、每 trial 55-90 s |
| OpenMIIR | GitHub/Figshare | music | stimuli metadata/electrode info xlsx 可读；beat annotation 文本可读；metadata 含 song/cue length、bpm 等字段 |
| MAD-EEG `4537751` | Zenodo | music-secondary | behavioral xlsx、raw yaml、sequence yaml 已保存；适合 target-instrument attention 补充验证 |

## 数据集分级

### A. 第一优先级：自然语音 tokenization 与语义对齐

#### 1. ds004718 LPPHK

推荐定位：词级/韵律级对齐主数据集。

优势：

- 自然粤语语音，具备完整句子文本、词级 onset/offset、POS、词频、f0/intensity。
- 受试者数量较大，适合跨被试 tokenizer 和 subject generalization。
- EEG 采样率 1000 Hz，便于构建 128/256 Hz 皮层分支，也保留较高时间分辨率。

适合实验：

- EEG token 与 word boundary/prosody 的对齐。
- 词级 EEG-to-text retrieval。
- prosody-conditioned contrastive learning。
- Cantonese-specific speech representation 与跨语种迁移测试。

风险：

- 粤语分词、词频和普通话/英语 NLP 表征之间存在语言特异性。
- 若作为金标准语义对齐集，需要抽查词切分一致性。

#### 2. ds005345 LPP Multi-talker

推荐定位：多说话人注意与语义检索主数据集。

优势：

- 同一故事包含 single female、single male、mixed speech 条件。
- word information、f0/intensity acoustic CSV、stimuli WAV、预处理 EEG FIF 均可定位。
- 适合比较单说话人听觉理解与多说话人选择性注意。

适合实验：

- attended speaker / attended stream 表征学习。
- multi-talker EEG token disentanglement。
- EEG-to-audio/text segment retrieval。
- single-to-mix transfer。

风险：

- 参与者数和 run 数需要以下载后的实际 BIDS 文件为准。
- 多说话人条件需要明确 attended target，否则容易把任务退化为声学混合条件分类。

#### 3. ds004408 continuous naturalistic speech

推荐定位：英语自然语音 EEG encoder/tokenizer 预训练集。

优势：

- 128 channel、512 Hz、自然有声书听觉范式。
- TextGrid 中有可用的音素/词级标签，适合 phoneme/onset supervision。
- 数据结构为 BIDS + BrainVision，解析成本低。

适合实验：

- 英语自然语音低层同步。
- phoneme auxiliary branch。
- cross-subject masked EEG modeling。
- 与 ds004718/ds005345 的跨语言迁移。

风险：

- 语义级标注不如 LPP 系列丰富，高层语义需要额外文本或 forced alignment。

### B. 第二优先级：受控音位与同步工程

#### 4. ds006104

推荐定位：音位/发音结构 controlled probe。

优势：

- events 字段直接包含 phoneme、manner、place、voicing、TMS target。
- 适合验证 tokenizer 是否学到 articulatory/phonemic structure。

限制：

- 不是自然连续语音。
- TMS 与任务结构可能引入 confound。

建议用法：只作为模块验证集，不作为主训练集。

#### 5. ds006434

推荐定位：时间同步与 attention event 工程基准。

优势：

- events 中含 attention story/side 和 EEG trigger。
- stimulus WAV、HDF5 regressor/stimulus 文件、cortex/subcortex EEG 均可定位。
- 适合检验 lag estimation、clock drift correction、64 s epoch slicing。

限制：

- 语义标注不如 LPP 系列。
- 部分高精度 stimulus/regressor 文件较大，正式使用前需规划存储。

### C. AAD 与空间编码基线

#### 6. KUL AAD

推荐定位：经典 AAD envelope baseline。

关键注意：KUL README 明确指出，深度模型在随机切片交叉验证下容易学习 trial-specific patterns，导致注意解码准确率虚高。因此必须使用 leave-one-trial-out、leave-one-story-out 或 leave-one-subject-out。

#### 7. DTU AAD

推荐定位：不同房间声学条件下的 robustness baseline。

适合验证 envelope/onset 跟踪模型在混响和声学条件变化下是否稳定。

#### 8. 255ch AAD

推荐定位：高密度空间编码和 sensor-ablation benchmark。

适合回答：EEG tokenizer 是否真正利用空间拓扑，或者只是利用少数通道的低维特征。

### D. 音乐扩展

#### 9. OpenMIIR

推荐定位：音乐 beat/downbeat/tempo/meter 对齐的第一数据集。

优势：

- metadata 中含 cue/song duration、BPM 等字段。
- beat annotation 可直接读取。
- 适合先验证音乐事件 token 与 EEG 的锁相关系。

#### 10. MUSIN-G

推荐定位：自然音乐 EEG 自监督预训练。

优势：

- 自然音乐片段较长。
- 可用熟悉度、愉悦度等弱监督信号。

限制：

- beat/乐句/歌词级标注不如 OpenMIIR。
- 更适合预训练和弱监督，而不是精确事件对齐主基准。

#### 11. MAD-EEG

推荐定位：target instrument attention 补充数据。

适合作为音乐方向第二阶段实验，不建议进入第一阶段主线。

### E. 补充数据

#### 12. ds007591

推荐定位：speech production/covert speech sanity check。

该数据更偏 overt/minimally overt/covert speech production，可以作为 tokenizer 迁移测试，不应替代听觉语音主数据。

#### 13. ESAA

推荐定位：普通话 AAD 补充。

可用于 tonal-language competing speech baseline，但优先级低于 LPP Multi-talker，因为语义/词级注释密度不足。

## 模型与实验路线

### 阶段 1：低层同步基线

目标：确认 EEG 与音频低层特征的同步关系，避免后续 tokenization 建在错误 lag 上。

建议特征：

- Speech audio：broadband envelope、sub-band envelope、onset envelope、log-mel、f0、voicing。
- Music audio：onset strength、tempogram、beat/downbeat、chroma、harmonic/percussive features。
- EEG：0.5-45 Hz 皮层分支，统一到 128/256 Hz。

建议方法：

- ridge-TRF。
- backward envelope reconstruction。
- CCA / DCCA。
- cross-correlation lag scan。

输出指标：

- 每个数据集、被试、run 的最佳 lag 分布。
- 低层 audio feature 与 EEG 的可预测性。
- 严格切分下的 subject/run/story generalization。

### 阶段 2：EEG tokenization

推荐起点：TCN/Conformer encoder + RVQ。

默认参数：

- latent rate：25 Hz 或 50 Hz。
- latent dimension：128-256。
- RVQ 层数：2-4。
- 每层 codebook：256 或 512。
- 损失：masked EEG reconstruction + CPC/InfoNCE + codebook usage regularization。

关键约束：

- tokenizer 不能只编码被试身份、设备、参考方式或 trial fingerprint。
- 需要报告 codebook perplexity、dead codes、subject leakage、run/story generalization。
- 多数据集训练前应加入 channel-set adaptor 或 topology-agnostic latent。

### 阶段 3：事件级与语义级对齐

推荐数据组合：

- `ds006104`：phoneme/articulation auxiliary branch。
- `ds004408`：phoneme/TextGrid auxiliary branch。
- `ds004718`：word/prosody supervision。
- `ds005345`：single/mix speaker semantic retrieval。

推荐方法：

- constrained monotonic attention。
- Soft-DTW。
- local contrastive alignment。
- 0-500 ms neural lag prior，禁止模型使用未来音频信息。

核心指标：

- audio segment retrieval accuracy。
- word/phoneme boundary prediction。
- token-to-prosody mutual information。
- attended stream retrieval。
- leave-subject / leave-story / leave-run generalization。

## 参考论文对方案的影响

- Defossez et al.：支持先对齐 self-supervised speech representation，而不是直接重建 waveform。
- DeWave：说明 EEG 离散化能够减少连续脑电到语言序列之间的表示断层，但其阅读 EEG 设置不能直接替代连续听觉语音。
- DELTA：RVQ + diffusion decoding 对小规模 EEG-text 有启发，但当前项目应先完成听觉连续 EEG token 与 audio/text 对齐。
- LUNA：强调电极拓扑异质性，需要 topology-agnostic latent 或 channel adaptor。
- BrainOmni：支持跨 EEG/MEG/多数据集的统一 foundation model 方向，但第一阶段应先建立高质量 benchmark。
- Lee 2025：parallel phoneme prediction 可作为辅助任务，适合 `ds006104` 和 `ds004408`。
- Moreira 2025：`ds006104` 适合 articulation/coarticulation probe，同时需要控制 TMS 与任务结构 confound。

## 明天讨论需要拍板的问题

1. 第一版 benchmark 是否固定为 `ds004718 + ds005345 + ds004408 + ds006104`？
2. EEG token rate 选择 25 Hz 还是 50 Hz？是否单独保留 100 Hz event/prosody branch？
3. 所有结果是否强制报告 leave-subject、leave-run、leave-story 三种切分？
4. 第一篇工作的主张是“EEG tokenizer + alignment benchmark”，还是“EEG-to-audio/text retrieval system”？
5. 音乐数据是否进入第一篇主线，还是作为第二阶段扩展？
6. 是否先实现非深度 baseline（TRF/CCA）作为所有深度结果的 sanity check？

## 一周执行计划

**Day 1-2：数据索引与最小下载**

- 下载核心四个数据集的最小子集。
- 建立统一 manifest：subject、run、task、EEG file、event file、stimulus file、annotation file。
- 明确每个 run 的 audio/stimulus 对应关系。

**Day 2-3：音频特征与事件表**

- 抽取 envelope/onset/log-mel/f0/voicing。
- 解析 TextGrid、word timing、phoneme events。
- 统一到 10 ms、20 ms、40 ms 三种时间网格。

**Day 3-4：EEG 预处理**

- 统一滤波、重参考、resample。
- 输出 bad channel、missing trigger、duration mismatch 报告。
- 保存 `EEG_cortex@128Hz` 和必要的高采样分支。

**Day 4-5：同步 baseline**

- TRF/CCA/cross-correlation lag scan。
- 输出被试级最佳 lag 分布和 sanity plots。

**Day 5-7：第一版 tokenizer**

- 训练 continuous EEG encoder + RVQ。
- 报告 reconstruction、codebook usage、audio-token retrieval。
- 在 `ds006104` 上跑 phoneme/articulation probe。

## 参考链接

- [OpenNeuro ds004408](https://openneuro.org/datasets/ds004408)
- [OpenNeuro ds005345](https://openneuro.org/datasets/ds005345)
- [OpenNeuro ds004718](https://openneuro.org/datasets/ds004718)
- [OpenNeuro ds006104](https://openneuro.org/datasets/ds006104)
- [OpenNeuro ds006434](https://openneuro.org/datasets/ds006434)
- [OpenNeuro ds003774](https://openneuro.org/datasets/ds003774)
- [OpenNeuro ds007591](https://openneuro.org/datasets/ds007591)
- [KUL AAD Zenodo 4004271](https://zenodo.org/records/4004271)
- [DTU AAD Zenodo 1199011](https://zenodo.org/records/1199011)
- [255ch AAD Zenodo 4518754](https://zenodo.org/records/4518754)
- [ESAA Zenodo 7078451](https://zenodo.org/records/7078451)
- [MAD-EEG Zenodo 4537751](https://zenodo.org/records/4537751)
- [OpenMIIR GitHub](https://github.com/sstober/openmiir)
