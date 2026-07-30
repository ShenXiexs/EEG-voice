# 20 篇 EEG/神经语音论文到底用了什么数据集？

> 整理日期：2026-07-30  
> 对应文献包：`eeg_speech_factorized_20260726`  
> 目的：区分论文真正使用的脑信号数据、语音刺激/语料、预训练语料，以及这些数据能否用于当前 scalp EEG→speech 项目。

## 先说结论

1. **这 20 篇论文里没有一篇使用 KaraOne 或 FEIS。** KaraOne、FEIS 是当前项目自己的实验数据来源，不能把这 20 篇论文的结果当成它们的既有基线。
2. 与“scalp EEG + 同步自发声 audio”最接近的是 **S07 Krishna 2020** 和 **S18 Wu 2021**，但二者都是作者自采的小样本 overt-speech 数据，论文没有给出可直接下载的数据集。
3. **S20 NeuroSonic 并未使用专门的 EEG 自发声语料**：它使用 CineBrain（看电视剧时的 EEG/fMRI）和 EAV（情绪对话 EEG/audio/video）。其中 CineBrain 的音频主要是外部视听刺激，不是参与者自己的发声。
4. 真正公开、同时具有侵入式神经信号和自发声录音、最适合作为“机制上界”的是 **S09 使用的 SingleWordProductionDutch sEEG 数据**。
5. S10、S19 虽然规模很大，但任务是**听语音或阅读**，输出也是检索/语义表征，不是 EEG→自发声波形。
6. S01–S06 完全没有脑信号数据；它们提供的是 speech representation、content/style factorization、TTS 或 neural codec 后端。

## 图例

- ✅：原始数据公开或论文给出公开数据入口。
- 🟡：部分公开、申请后提供，或仅公开一名参与者/派生数据。
- ❌：论文未给出可直接下载的原始神经数据。
- “自采”表示论文作者为该研究采集的数据，不等于公开数据集。

## 总表

| ID | 论文简称 | 真正使用的数据集/语料 | 脑信号与任务 | 规模 | 原始数据可用性 | 对当前项目的意义 |
|---|---|---|---|---|---|---|
| S01 | wav2vec 2.0 | LibriSpeech、Libri-Light/LibriVox、TIMIT | 无脑信号；ASR/音素识别 | 960 h、53.2k h、TIMIT 约 5 h | ✅ | speech teacher；不是 EEG 证据 |
| S02 | AutoVC | VCTK；speaker encoder 另用 VoxCeleb1+LibriSpeech | 无脑信号；voice conversion | VCTK 44 h/109 speakers；speaker encoder 共 3,549 speakers | ✅/依许可 | 内容—说话人分解参考 |
| S03 | SpeechT5 | LibriSpeech/LibriSpeech-LM、LibriTTS、MuST-C、CMU Arctic、WHAM!、VoxCeleb1 | 无脑信号；统一 speech/text 预训练和六类下游任务 | 960 h speech+400M text sentences 等 | ✅/依许可 | 共享 speech-text 空间；不是 EEG 数据 |
| S04 | StyleTTS 2 | LJSpeech、VCTK、LibriTTS train-clean-460 | 无脑信号；单/多说话人 TTS、zero-shot adaptation | 24 h；约 44k clips/109 speakers；245 h/1,151 speakers | ✅ | 同 label 多种 style 的生成后端 |
| S05 | HuBERT | LibriSpeech、Libri-Light | 无脑信号；自监督语音表示/ASR | 960 h、60k h | ✅ | 内容/声学 unit teacher |
| S06 | SoundStream | LibriTTS、Freesound、MagnaTagATune、自采真实噪声/混响语音；VCTK 仅用于额外评估 | 无脑信号；音频 codec | 论文未统一报告总训练小时数；主观测试 200 clips | 🟡 | waveform/codec 后端；部分评测数据私有 |
| S07 | EEG Speech Synthesis | 作者自采四条英文命令 | 32-channel scalp EEG；听和 overt speech | 4人；每人4句×70次=280 trials | ❌ | 最接近 paired scalp EEG→own speech，但极小且有 EMG 风险 |
| S08 | Cross-subject split audit | ZuCo EEG、Narratives fMRI | 阅读 EEG；听故事 fMRI；brain-to-text | 论文沿用两个数据集的处理版本，未在正文重报完整样本量 | ✅ | 数据划分/泄漏审计，不是语音重建数据 |
| S09 | NeuroIncept | SingleWordProductionDutch-iBIDS；词来自 Dutch IFA corpus | sEEG+同步 overt speech | 10人、100 Dutch words、1,103 electrodes、约5 min/人 | ✅ | 公开侵入式 paired neural-audio 上界 |
| S10 | Défossez 2023 | Broderick2019、Brennan2019、Schoffelen2019/MOUS、Gwilliams2022/MEG-MASC | 2 EEG+2 MEG；被动听语音 | 175人、约163 h | ✅ | 非侵入 speech-perception 表征检索；不是 production |
| S11 | Anumanchipalli 2019 | 自采 ECoG；主要句料为 MOCHA-TIMIT，另含 Alice、场景描述和自由访谈 | ECoG；overt 和部分 silent mime | 5名临床参与者；各人句料量不同 | 🟡 | articulatory→acoustic 分解上界 |
| S12 | Chen 2024 | 作者自采 48 人多任务 ECoG speech 数据 | ECoG+同步 speech；5类产词任务 | 48人；50 unique words；400 trials/人 | 🟡 | pitch/formant/loudness 等中间参数上界 |
| S13 | Pasley 2012 | 作者自采 ECoG；孤立词+TIMIT sentences | ECoG；听语音/词检测/复述 | 15名临床参与者 | ❌ | auditory spectrogram reconstruction 上界 |
| S14 | Tang 2023 | Huth 实验室自然故事 fMRI 数据；OpenNeuro ds003020、ds004510 | fMRI；听故事、想象讲故事、无声电影 | 3名核心参与者；最高约16 h训练/人 | ✅（主要数据） | semantic content 路线参考；不是 EEG/音频生成 |
| S15 | Moses 2021 | 单一临床参与者自采 attempted-speech ECoG | 128-channel ECoG；50词 attempted speech | 1人；48 sessions、22 h、50-word vocabulary | ❌/受隐私限制 | 小词表文本解码临床上界 |
| S16 | Willett 2023 | BrainGate T12 attempted-speech 数据；另分析 USC-TIMIT/Haskins EMA | 4×64 intracortical arrays；attempted speech | 1人；10,850 sentences | ✅ Dryad | 大词表 phoneme/text 解码临床上界 |
| S17 | Metzger 2023 | 单一临床参与者；50-phrase-AAC、529-phrase-AAC、1024-word-General | 253-channel high-density ECoG；attempted speech | 1人；最大语料9,655句/1,024词 | 🟡 restricted access | text+audio+avatar 多分支上界 |
| S18 | Wu 2021 | 作者自采普通话单音节 EEG-speech 数据；刺激来自 Mandarin Speech Perception Test | 64-channel scalp EEG；论文只用 overt-speaking stage | 11人采集、7人纳入；350 trials/人 | ❌ | “先预测多频带 envelope 再 vocode”最直接参考 |
| S19 | d’Ascoli 2025 | 7个公共 EEG/MEG 数据集+2个自采 Little Prince MEG 数据集 | 阅读或听语音；word semantic retrieval | 723人、772 h、约487.7万词 | 🟡（7 public，2 request） | 大规模非侵入语义解码；没有自发声重建 |
| S20 | NeuroSonic | CineBrain、EAV | scalp EEG；视听感知/情绪对话的同步音频 | 48人、约60 h、49,200 segments | ✅（源数据公开） | 任务混合且非纯 self-speech；预印本结论需独立审计 |

## 一、S01–S06：它们用的是语音语料，不是脑数据

### S01 — wav2vec 2.0

- **无标注预训练**：
  - LibriSpeech `LS-960`：960 h；
  - LibriVox 经 Libri-Light 预处理得到的 `LV-60k`：实际使用 53.2k h。
- **有标注微调**：Libri-Light 10 min、1 h、10 h，LibriSpeech train-clean-100 和完整 960 h。
- **音素识别**：TIMIT，约 5 h。
- **语言模型文本**：LibriSpeech LM corpus。
- 这篇论文只能支持“用预训练 speech representation 当 EEG target/teacher”，不能支持任何 EEG 可解码性结论。

原文：[NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf)

### S02 — AutoVC

- 主训练和评测语料是 **VCTK**：44 h、109名说话人、非平行句料；每名说话人的数据按 9:1 分 train/test。
- speaker encoder 预先在 **VoxCeleb1 + LibriSpeech** 上训练，共 3,549 名说话人。
- WaveNet vocoder 也在 VCTK 上训练。
- 因此 AutoVC 的“content”是由语音模型信息瓶颈学出的 speaker-independent representation，不是文本 label，也不是从 EEG 学出的 content。

原文：[PMLR paper](https://proceedings.mlr.press/v97/qian19c.html)

### S03 — SpeechT5

预训练：

- **LibriSpeech 960 h audio**；
- **LibriSpeech-LM 400M text sentences**。

下游实验：

| 任务 | 数据集 | 论文使用方式 |
|---|---|---|
| ASR | LibriSpeech | 100 h 与 960 h 微调 |
| TTS | LibriTTS clean | 460 h 多说话人朗读语音 |
| Speech translation | MuST-C | EN-DE 408 h/234k translated sentences；EN-FR 492 h/280k |
| Voice conversion | CMU Arctic | 4名说话人，speaker conversion |
| Speech enhancement | WHAM! | noisy speech enhancement |
| Speaker identification | VoxCeleb1 | 官方划分 |

原文：[ACL Anthology](https://aclanthology.org/2022.acl-long.393/)

### S04 — StyleTTS 2

- **LJSpeech**：13,100 clips、约24 h、单说话人；12,500 train/100 validation/500 test。
- **VCTK**：约44,000 clips、109名母语说话人，用于多说话人模型。
- **LibriTTS train-clean-460**：论文筛选后约245 h、1,151 speakers，用于 zero-shot speaker adaptation。
- 另使用预训练 WavLM 作为 speech-language-model discriminator；这不表示 StyleTTS 2 自己重新采集或训练了 WavLM 的94k h数据。

原文：[NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eaad2a0b62b5ed7a2e66c2188bb1449-Paper-Conference.pdf)

### S05 — HuBERT

- **LibriSpeech 960 h** 和 **Libri-Light 60,000 h** 用于自监督预训练。
- 有标注微调设置为 10 min、1 h、10 h、100 h 和 960 h。
- 使用 LibriSpeech LM data 进行 ASR decoding。
- 对当前项目最有价值的是不同层 hidden state 和 k-means unit，不是该论文的数据划分本身。

原文：[IEEE/ACM TASLP DOI](https://doi.org/10.1109/TASLP.2021.3122291)

### S06 — SoundStream

- clean speech：**LibriTTS**；
- noisy speech：LibriTTS speech 与 **Freesound** noise 合成；
- music：**MagnaTagATune**；
- real-world noisy/reverberant speech：作者另外自采，未命名、未公开；
- 主观评价：200个2–4 s clips，四类数据各50个；
- VCTK 只用于后续未见训练数据的 speaker/phonetic/pitch probe。

因此 SoundStream 不是一个 EEG 数据来源，只是潜在的冻结 audio codec。

原文：[IEEE/ACM TASLP DOI](https://doi.org/10.1109/TASLP.2021.3129994)

## 二、与非侵入 EEG/MEG 最相关的数据

### S07 — Krishna 2020：作者自采四命令 EEG-speech

- 4名 UT Austin 本科生，3女1男，均20岁出头。
- 4条固定命令：`Hi Bixby`、`Call Mom`、`Open Camera`、`What's the weather`。
- 每名参与者对每句话采70次：每人280 trials，合计1,120 participant-trials。
- 每个 trial 同时包含：听刺激时的 EEG、刺激音频、随后出声复述时的 EEG、参与者自己的语音。
- 32个湿电极（含 ground），EEG 1,000 Hz；speech 16 kHz。
- 论文按每个被试内部 80/10/10 划分；不是 cross-subject generalization。
- **论文没有提供公开下载入口。**

这篇最接近当前 paired scalp EEG→speech，但只有4个 label，而且 overt speech 的面肌/发音运动伪迹很强。

原文：[ICASSP DOI](https://doi.org/10.1109/ICASSP40776.2020.9053340)

### S08 — Yin 2025：ZuCo EEG + Narratives fMRI

- **ZuCo 1.0**：自然阅读任务的 EEG+eye-tracking 数据；该论文用它评估 EEG2Text 的跨被试划分。
- **Narratives**：多个自然故事听觉 fMRI 数据的集合；该论文用它评估 UniCoRN。
- 这篇论文的重点是重新构造 subject/story/text 的 train-validation-test 图划分。正文没有把两个源数据集的完整招募人数和所有 trial 数重新报告，因此不应从其结果表反推原始数据规模。
- 它不包含 EEG→同步自发声的任务。

原文：[EMNLP 2025](https://aclanthology.org/2025.emnlp-main.289/)

### S10 — Défossez 2023：四个公共 M/EEG speech-perception 数据集

| 数据集 | 语言/模态 | sensors | subjects | 本文训练时长 | 刺激 |
|---|---:|---:|---:|---:|---|
| Broderick2019 | English EEG | 128 | 19 | 19.2 h | 听《The Old Man and the Sea》片段 |
| Brennan2019 | English EEG | 60 | 33 | 6.7 h | 听《Alice in Wonderland》一章 |
| Schoffelen2019 / MOUS | Dutch MEG | 273 | 96 | 80.9 h | 听孤立荷兰语句子和打乱词序列表 |
| Gwilliams2022 / MEG-MASC | English MEG | 208 | 27 | 56.2 h | 听4个故事，两次约1 h sessions |
| **合计** |  |  | **175** | **约163 h** | 全部为被动听语音 |

- 四个数据集都公开，但许可不同：Schoffelen 需在 Donders 注册；Gwilliams 和 Broderick 为 CC0；Brennan 为 CC BY 4.0。
- 音频是参与者听到的外部刺激，并非参与者自己发出的声音。
- 论文做3 s brain/audio representation retrieval，不直接生成波形。

原文：[Nature Machine Intelligence](https://www.nature.com/articles/s42256-023-00714-5)

### S18 — Wu 2021：作者自采普通话 overt-speech EEG

- 招募11名普通话母语者（7男4女，20–24岁）；4人因未专注或语音未成功记录被排除，最终分析7人。
- 64-channel Neuroscan Quick-cap，EEG 1,000 Hz；speech 16 kHz。
- 18个普通话单音节，来自 **Mandarin Speech Perception Test corpus**；覆盖不同声母—韵母组合。
- 每个音节通常4个声调，两个缺少第一声；每个有调音节重复5次，共350 trials/人。
- 每个 trial 有 rest、listening、imagined、silent intended、overt speaking 五阶段；**论文的 speech reconstruction 只用了 overt-speaking stage**。
- 最终数据量为7×350=2,450个纳入分析的 trials；每名参与者按280 train/70 test，再做五折验证。
- **论文没有给出原始 EEG/audio 的公开下载入口。**

这篇与“先生成能量结构”最接近，因为目标是24个频带的 temporal envelopes，再用 vocoder 合成语音；但它不能证明 imagined EEG 也能完成同样任务。

原文：[EUSIPCO proceedings PDF](https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0001140.pdf)

### S19 — d’Ascoli 2025：9个非侵入语言数据集

论文整合7个公共数据集，并自采2个 Little Prince MEG 数据集：

| 数据集 | subjects | hours | 语言 | task | device | 公开情况 |
|---|---:|---:|---|---|---|---|
| Nieuwland | 295（原334，本文排除一个站点后） | 171 | English | RSVP reading | 63-ch EEG | public |
| Broderick | 19 | 20 | English | narrative listening | 128-ch EEG | public |
| Accou / SparrKULee single-speaker stories | 80（85中5人的记录不可公开） | 150 | Dutch | narrative listening | 64-ch EEG | public subset |
| Gwilliams / MEG-MASC | 27 | 57 | English | narrative listening | 208-sensor MEG | public |
| Schoffelen Listen / MOUS | 96 | 81 | Dutch | sentence listening | 273-sensor MEG | public/registration |
| Schoffelen Read / MOUS | 99 | 106 | Dutch | sentence reading | 273-sensor MEG | public/registration |
| LittlePrinceListen | 58 | 94 | French | narrative listening | 306-sensor MEG | request |
| LittlePrinceRead | 46 | 59 | French | RSVP reading | 306-sensor MEG | request |
| Armeni | 3 | 34 | English | 10 h/人 narrative listening | 298-sensor MEG | public |
| **合计** | **723** | **772** | 3 languages | reading/listening | EEG+MEG | mixed |

- 论文表1报告约4.877 million word tokens。
- 模型已知 word onset，并从固定词汇表检索 semantic embedding。
- 没有 overt/attempted speech，也没有参与者自己的同步语音作为重建目标。

原文：[Nature Communications](https://doi.org/10.1038/s41467-025-65499-0)

### S20 — NeuroSonic 2026：CineBrain + EAV，不是 KaraOne/FEIS

论文称预处理后共48人、约60 h同步 EEG/audio、49,200 paired segments：

| 数据集 | subjects | 原始任务 | 同步模态 | 对 EEG→speech 的关键限制 |
|---|---:|---|---|---|
| CineBrain | 6 | 看《The Big Bang Theory》连续视听材料 | EEG+fMRI+ECG+外部视听刺激 | audio 含对白和环境声，主要是**感知刺激**，不是参与者自发声 |
| EAV | 42 | cue-based conversation，诱发 neutral/anger/happiness/sadness/calmness | 30-ch EEG+audio+video | 含 listen/speak 互动，但原数据设计目标是情绪识别，不是纯净的 speech-production BCI |

需要特别注意：

- NeuroSonic 论文把两个数据都称为 EEG–audio benchmark，但两者的任务、声源和神经过程不同。
- 论文只报告 combined corpus 总量，没有分别报告每个数据集最终保留的 segment 数，也没有在正文列出具体 test subject IDs。
- CineBrain 的结果不能写成“从说话者的 EEG 重建其自己的声音”。更准确的表述是“从观看自然视听材料时的 EEG，预测/生成时间对齐的 audio target”。
- 该文截至整理日仍为 arXiv 预印本；DNSMOS、FAD 等结果尤其需要 zero-EEG、shuffled-EEG、audio-prior 和 train-bank retrieval 对照后才能判断条件有效性。

原文：[arXiv:2606.24087](https://arxiv.org/abs/2606.24087)；源数据：[CineBrain](https://huggingface.co/datasets/Fudan-fMRI/CineBrain)、[EAV dataset description](https://nur.nu.edu.kz/server/api/core/bitstreams/244c86c8-4716-4e58-afa1-b10fa194f30c/content)

## 三、侵入式数据：只能作为上界和方法参照

### S09 — NeuroIncept / SingleWordProductionDutch-iBIDS

- 10名药物难治性癫痫患者，5男5女，平均32岁；临床植入 sEEG。
- 共1,103个电极接触点，位置因临床需要而异。
- 参与者朗读 Dutch IFA corpus 中的100个词；sEEG 1,024/2,048 Hz，audio 48 kHz。
- NeuroIncept 使用 high-gamma 70–170 Hz；论文称每人约5 min数据，subject-specific 10-fold CV。
- 原始 neural、audio、stimulus markers、electrode locations 和 anatomy 均公开，音频为保护身份做 pitch shift。

数据：[OSF DOI](https://doi.org/10.17605/OSF.IO/NRGX6)；数据说明：[Scientific Data](https://doi.org/10.1038/s41597-022-01542-9)

### S11 — Anumanchipalli 2019

- 5名因癫痫临床治疗而植入高密度 subdural ECoG 的英语参与者。
- 核心读句语料是 **MOCHA-TIMIT 460 sentences**，但每个参与者的数据组成不同：
  - P1 读两套 MOCHA-TIMIT，并读部分 Alice in Wonderland；另有 audible/silent-mime 配对；
  - P2 读一套 MOCHA-TIMIT，并多次重复50句话；
  - P3/P4/P5 还包含 scene descriptions 或 free-response interviews。
- 论文对 P1 的听辨实验使用101个 test synthesized sentences；silent-mime 分析使用58个 held-out sentences。
- 神经数据并不是“MOCHA-TIMIT 公共数据”：公开的只是句子语料，ECoG 是作者自采。
- 原始数据和代码需联系作者，论文表述为 upon request / non-commercial use。

原文：[Nature](https://www.nature.com/articles/s41586-019-1119-1)

### S12 — Chen 2024

- 48名 refractory epilepsy 临床参与者，26女22男；32人左半球、16人右半球电极覆盖。
- 43人使用64-contact low-density 8×8 grid；5人另有64个 interspersed contacts 的 hybrid-density grid。
- 所有人完成5种任务：auditory repetition、auditory naming、sentence completion、visual word reading、picture naming。
- 5种任务使用相同的50个 target words，不同任务通过不同刺激引出；每人400 trials，平均每个 spoken response 约500 ms。
- 主实验每人约350 train/50 test；模型按参与者分别训练。
- 仅一名明确同意公开 neural+audio 的参与者完整数据放在 Mendeley；其余需向作者申请并满足 IRB/语音隐私条件。

数据：[Mendeley Data](https://data.mendeley.com/datasets/fp4bv9gtwk/2)；原文：[Nature Machine Intelligence](https://www.nature.com/articles/s42256-024-00824-8)

### S13 — Pasley 2012

- 15名癫痫或脑肿瘤手术患者，subdural ECoG；4人4 mm grid、11人10 mm grid。
- 任务分为 passive listening 5人、target-word detection 5人、word/sentence repetition 5人。
- 10人听单一女性说话人的孤立词；5人听多个男女说话人的 **TIMIT sentences**。
- 一项词识别分析使用47个候选词，比较 single trial 与3–5次平均。
- TIMIT 是公开语音刺激语料，但患者 ECoG 并不是公开 TIMIT 数据的一部分；论文未给出可直接下载的完整神经数据入口。

原文：[PLOS Biology](https://doi.org/10.1371/journal.pbio.1001251)

### S15 — Moses 2021

- 1名脑干卒中后 anarthria、spastic quadriparesis 的男性临床参与者。
- 左侧 speech sensorimotor cortex 上覆盖128-channel high-density ECoG array。
- 48 sessions、22 h cortical recordings、历时81周。
- attempted speech vocabulary 为50词；使用单词检测、分类和语言模型组合成句。
- 这是自采临床个案数据，论文没有提供可直接公开下载的原始 neural/audio 数据。

原文：[NEJM DOI](https://doi.org/10.1056/NEJMoa2027540)

### S16 — Willett 2023

- BrainGate2 参与者 T12，1名 ALS 女性。
- 4个 Utah arrays，每个64 electrodes，共256 intracortical electrodes；两组位于 ventral premotor area 6v，两组位于 Broca area 44。
- 训练集最终包含10,850个 attempted-speech sentences。
- 大词表评估使用 Switchboard corpus 句子和125,000词 vocabulary；另复用 Moses 50-word vocabulary。
- 所有复现实验所需 neural data 已公开在 Dryad；另分析 USC-TIMIT 和 Haskins 的公开 electromagnetic articulography 数据。

数据：[Dryad](https://doi.org/10.5061/dryad.x69p8czpq)；代码：[speechBCI](https://github.com/fwillett/speechBCI)

### S17 — Metzger 2023

- 1名脑干卒中后 severe paralysis/anarthria 的女性参与者。
- 253-channel high-density ECoG，覆盖 speech sensorimotor cortex 和 superior temporal gyrus。
- 三套自定义句料：
  - `50-phrase-AAC`：50个高实用句、119 unique words；
  - `529-phrase-AAC`：529句、372 unique words；
  - `1024-word-General`：9,655句、1,024 unique words，取自 Twitter 和电影转录。
- 参与者 silent attempted speech；模型分别输出 text、HuBERT speech units/synthesized audio 和 avatar articulator movements。
- 神经数据按临床协议 restricted access，可合理申请，但不能公开再分发。

数据申请：[Zenodo access record](https://doi.org/10.5281/zenodo.8200782)；原文：[Nature](https://www.nature.com/articles/s41586-023-06443-4)

## 四、fMRI 语义解码

### S14 — Tang 2023

- 核心实验为3名参与者，每人进行大量个体化 fMRI 训练。
- training stimulus 是自然叙事 podcast/stories；最多15 sessions、约16 h/人。扩展分析显示约7.5 h后增益开始平台化，但最终模型仍用最多约16 h。
- 测试包括：听未见故事、想象讲5个约1 min故事、观看无声电影片段。
- 输出是语义相近的文字序列，不重建说话人的音色、F0、formant 或 waveform。
- 除 decoder-resistance 实验外，主要数据公开在 OpenNeuro；抵抗实验因 mental-privacy 风险仅按申请提供。

数据：[OpenNeuro ds003020](https://openneuro.org/datasets/ds003020)、[OpenNeuro ds004510](https://openneuro.org/datasets/ds004510)；原文：[Nature Neuroscience](https://www.nature.com/articles/s41593-023-01304-9)

## 五、哪些数据能直接帮助当前实验？

| 优先级 | 数据 | 可以做什么 | 不能证明什么 |
|---|---|---|---|
| A | KaraOne、FEIS（当前项目） | paired scalp EEG→content/prosody/audio；严格 subject/label 抽样和重建 | 不能借侵入式结果预设可达到的音质 |
| A | S09 SingleWordProductionDutch | 在公开 sEEG 上验证 content/prosody/spectrogram 分解和评测代码 | 不能代表 scalp EEG 性能 |
| B | S10 四个公开 M/EEG 数据 | 复现 pretrained speech representation alignment、negative retrieval controls | 不能验证 self-produced speech reconstruction |
| B | S19 九数据集框架 | 参考跨数据集统一预处理、subject layer、known-onset word semantics | 不提供同步自发声 target |
| B | S12 单参与者公开数据 | 测试 speech-parameter intermediate targets | 单一 ECoG 参与者不能支撑跨被试 EEG 结论 |
| C | S20 CineBrain/EAV | 审计 conditional flow 在弱 EEG 条件下是否依赖 audio prior | 不能把视听刺激音频统一称为“参与者声音” |
| 方法后端 | S01–S06 的公共语音语料 | 训练/冻结 HuBERT、wav2vec2、TTS、codec、vocoder | 不能增加 EEG 中本来不存在的信息 |

## 六、和老师讨论时最简洁的表述

> 现有论文的数据其实分成三层。第一层是 LibriSpeech、VCTK、LJSpeech 这类纯语音语料，只能训练表示和生成后端。第二层是 scalp EEG/MEG 的听觉或阅读数据，规模可以很大，但多数没有参与者自己的发声，因此主要证明语义或语音表征能被检索。第三层是 ECoG、sEEG、intracortical 的临床数据，它们已经能解码内容、pitch/formant/loudness、语音甚至 avatar，但只能作为机制和方法上界。真正与我们“EEG→自己发出的声音”一致的公开证据仍然很少；纳入的两篇 scalp overt-speech 工作都是小样本自采且未公开。因此，KaraOne/FEIS 上严格验证 content 与 trial-specific prosody 是否真的来自 EEG，仍然是一个没有被这些论文直接解决的问题。

## 核对说明

- 样本量、任务和数据可用性均核对了文献包内的论文全文或论文官方 Data Availability。
- “公开”不等于无条件自由使用；VoxCeleb、MOUS、临床数据等仍需遵守各自许可、注册或伦理限制。
- S20 为2026年预印本；其数据描述按当前版本记录，后续版本可能变化。
- 对 S08，本文只记录该论文实际调用的 ZuCo 与 Narratives，不把源数据论文的所有参与者都误算成该文实际训练样本。
