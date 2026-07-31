# v3：内容优先 EEG → MFCC、CVAE 与 training-first

v3 独立于 v1/v2。EEG 不直接预测 waveform、speaker、F0、能量或 duration。主链为：

`EEG → 40×256 CMVN MFCC → fixed inverse-DCT Mel baseline → conditional VAE Mel residual → KaraOne-finetuned SpeechT5 HiFi-GAN → WAV`

MFCC 是 VAD active crop、utterance CMVN、c0 置零后的内容优先低维声学表示，并非纯文本内容。固定 inverse-DCT 后端与 `librosa.feature.inverse.mfcc_to_mel` 做数值一致性 gate；CVAE 只学习它无法解释的 bounded Mel residual。

CVAE posterior 只能用于 audio-only training 和 posterior oracle。EEG、validation 和 locked test 只能使用不读取目标 Mel 的 conditional prior mean/sample。

## 首次依赖

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle/app
./bootstrap_open_vocab_v3.sh
```

该环境增加 SpeechBrain ECAPA 和 librosa。默认选择性去噪器是无预训练参数的 conservative spectral gate；这样不会把一个未在 KaraOne 适配的外部模型偷偷放进主链。

## 外部 audio 模型微调协议

主流程在 V0 之前真正更新两个生成侧外部模型：

- SpeechT5 HiFi-GAN 的全部 generator parameters；
- SpeechBrain ECAPA 中实际用于 voice conditioning 的 feature/embedding backbone parameters。

SpeechT5 前增加一个可学习的 KaraOne Mel adapter：把本项目 10 ms、power-dB Mel 按真实 hop 比例转换到 SpeechT5 的 256-sample hop，并和 vocoder 一起反向传播。ECAPA 使用 fit subjects 的 speaker-classification objective 微调；原 VoxCeleb classifier 被丢弃，不进入生成。

主实验只用 `fit + eligible` WAV 微调，随后冻结 adapted checkpoints。A0 gate 要求训练 loss 实际下降、至少 50% 的预训练参数 tensor 确实改变，并把每条训练 WAV 的路径、role 与 SHA256 写入 lineage。HuBERT 与原始 ECAPA 是外部评价裁判，不属于生成模型；它们保持原始权重，避免“微调后给自己打分”。

如需满足“audio backend 确实见过全部 eligible WAV”的展示需求，单独运行：

```bash
./run_open_vocab_v3_all_wav_audio_adaptation_demo.sh
```

输出写入 `audio_adaptation/transductive_all/`，明确标记 `heldout_eeg_claims_allowed=false`，不会被 validation/test 或最终论文主结果加载。

## 两阶段运行：held-out 前必须试听 training WAV

推荐的一次性、交互式完整入口是：

```bash
./run_open_vocab_v3_complete.sh
```

它会自动补安装缺失依赖、运行全部 training stages、在当前终端暂停供试听，并且只有输入大写 `YES` 后才继续 held-out evaluation 和最终全量 pair export。最终所有 WAV、PNG、逐 trial JSON 与 manifest 位于同一个目录：

`../artifacts/open_vocab_v3_mfcc_training_first/pairs/training_fit_eligible/`

下面的分阶段命令保留用于中断恢复和调试。

第一阶段：审计、可选去噪、audio oracles、50-pair overfit、full fit 和 training preview。

```bash
./run_open_vocab_v3_all.sh
```

脚本会在以下目录生成两组 training WAV：

- `../artifacts/open_vocab_v3_mfcc_training_first/pairs/micro_50_preview/`
- `../artifacts/open_vocab_v3_mfcc_training_first/pairs/full_fit_preview/`

随后脚本以退出码 3 正常暂停，不会读取 validation/locked role。每个 preview trial 包含 cleaned reference、V0、fixed analytic MFCC oracle、CVAE posterior oracle、CVAE prior oracle、EEG prior mean、多个 prior samples 和 zero/time/channel controls。

试听 full-fit preview 后批准绑定到当前 preview/checkpoint/cache hash 的人工 gate：

```bash
./approve_open_vocab_v3_training_preview.sh samxie "内容可辨、未见统一模板塌缩"
./run_open_vocab_v3_after_review.sh
```

如果不满意，不运行批准命令；修改模型或数据后旧批准会因 lineage 不一致自动失效。

训练总预算默认 20 小时，SpeechT5/ECAPA domain fine-tuning、audio CVAE、micro EEG 和 full-fit EEG 共用一个绝对 deadline。去噪、特征准备、评估和 WAV 导出不计入训练预算。evaluation/export 默认 CPU，避免此前 MPS counterfactual forward 的 buffer assertion。需要临时调整时可在运行前设置 `BUDGET_HOURS`。

## 选择性去噪

原 WAV 始终只读。第一阶段先生成：

`../artifacts/open_vocab_v3_mfcc_training_first/denoise/selection.csv`

默认只列出低 active/inactive contrast 或命名异常 trial，且 `apply=0`。确认属于加性噪声后，把对应行改成 `apply=1` 再重新运行第一阶段。默认使用内置 deterministic spectral gate；只有同时通过以下 preservation checks 的输出才进入 prepared cache：

- envelope lag；
- VAD start/end shift；
- duration change；
- pre/post DTW-HuBERT cosine；
- pre/post ECAPA cosine。

未通过的 enhanced WAV 留作审计，但 `accepted=false`，训练继续使用原音频。`MM18:94`、`MM14:36`、`MM16:56` 在未获得 accepted enhancement 前仍排除出 fit。低对比度样本不会自动去噪。

## Fail-closed 顺序

1. raw audio audit → explicit selective deterministic denoising → post-denoise feature audit；
2. base audio cache → SpeechT5/ECAPA fit-only fine-tuning → A0 parameter-change/data-lineage gate；
3. 用 adapted ECAPA 重建 voice conditions，同时保留 base ECAPA audit embeddings；
4. V0：v0728 power-dB Mel → learned Mel adapter → adapted SpeechT5 HiFi-GAN；
5. audio CVAE training；
6. V1：fixed analytic、posterior oracle、prior-mean content oracle；
7. V2：target/canonical speaker swap 与 independent base-ECAPA content-preservation gate；
8. MM05 × 10 labels × 5 trials 的 direct EEG→MFCC overfit gate及 WAV preview；
9. all eligible fit records full-fit gate及 50 个 WAV preview；
10. exact-preview human approval；
11. subject-holdout validation、locked seen-label、`pot` exploratory；
12. 全部 eligible training-pair export。

A0/V0/V1/V2/micro/full-fit 或人工 listening gate 任一未通过，后续阶段停止。prepared manifest 绑定 raw/post audio audit、denoise selection/manifest、adapted vocoder/ECAPA manifests、speaker manifest 和 cache；每个模型 gate 继续绑定配置、上游 gate 和 checkpoint hash。

## 内容与音色证据

- V1 primary 使用 conditional prior mean，不允许目标 Mel；posterior 只作为 audio upper bound。
- V1 同时报告 fixed librosa-equivalent analytic backend、prior、posterior，以及 prior–posterior Mel gap。
- V2 的 target voice 来自同 subject 的 non-target fit trials。
- V2 先估计 KaraOne real-audio same/different-speaker 分布，再要求至少 80% target-voice generations 超过 different-speaker/same-label P90、speaker swap bootstrap CI 大于零，并保持 content retrieval。
- primary EEG WAV 永远使用 fit-only canonical speaker medoid，不使用 validation/test subject reference audio。

## 主要输出

- `audit/raw_audio_audit.*`、`denoise/selection.csv`、`denoise/manifest.json`；
- `gates/A0_*`、`V0_*`、`V1_*`、`V2_*`、`C_50_pair_*`、`D_full_fit_*`；
- `gates/E_training_wav_human_review.json`；
- `pairs/micro_50_preview/`、`pairs/full_fit_preview/`；
- `evaluation/*.json`；
- `pairs/training_fit_eligible/`；
- `run_manifest.json`。

所有 v3 写入均位于 `../artifacts/open_vocab_v3_mfcc_training_first/`。v0724/v0730 只作为只读来源，不会被覆盖。
