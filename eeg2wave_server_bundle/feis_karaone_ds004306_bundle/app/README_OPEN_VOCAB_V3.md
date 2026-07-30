# v3：内容优先 EEG → MFCC、CVAE 与 training-first

v3 独立于 v1/v2。EEG 不直接预测 waveform、speaker、F0、能量或 duration。主链为：

`EEG → 40×256 CMVN MFCC → fixed inverse-DCT Mel baseline → conditional VAE Mel residual → frozen SpeechT5 HiFi-GAN → WAV`

MFCC 是 VAD active crop、utterance CMVN、c0 置零后的内容优先低维声学表示，并非纯文本内容。固定 inverse-DCT 后端与 `librosa.feature.inverse.mfcc_to_mel` 做数值一致性 gate；CVAE 只学习它无法解释的 bounded Mel residual。

CVAE posterior 只能用于 audio-only training 和 posterior oracle。EEG、validation 和 locked test 只能使用不读取目标 Mel 的 conditional prior mean/sample。

## 首次依赖

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle/app
./bootstrap_open_vocab_v3.sh
```

该环境增加 SpeechBrain ECAPA、librosa 和官方 `deepfilternet` Python backend。DeepFilterNet 按官方接口在 48 kHz 运行并启用 delay compensation。

## 两阶段运行：held-out 前必须试听 training WAV

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

训练总预算默认 9.5 小时，audio CVAE、micro EEG 和 full-fit EEG 共用一个绝对 deadline。去噪、特征准备、评估和 WAV 导出不计入训练预算。evaluation/export 默认 CPU，避免此前 MPS counterfactual forward 的 buffer assertion。

## 选择性去噪

原 WAV 始终只读。第一阶段先生成：

`../artifacts/open_vocab_v3_mfcc_training_first/denoise/selection.csv`

默认只列出低 active/inactive contrast 或命名异常 trial，且 `apply=0`。确认属于加性噪声后，把对应行改成 `apply=1` 再重新运行第一阶段。只有同时通过以下 preservation checks 的 DeepFilterNet 输出才进入 prepared cache：

- envelope lag；
- VAD start/end shift；
- duration change；
- pre/post DTW-HuBERT cosine；
- pre/post ECAPA cosine。

未通过的 enhanced WAV 留作审计，但 `accepted=false`，训练继续使用原音频。`MM18:94`、`MM14:36`、`MM16:56` 在未获得 accepted enhancement 前仍排除出 fit。低对比度样本不会自动去噪。

## Fail-closed 顺序

1. raw audio audit → explicit selective DeepFilterNet → post-denoise feature audit；
2. V0：v0728 power-dB Mel → frozen SpeechT5 HiFi-GAN；
3. audio CVAE training；
4. V1：fixed analytic、posterior oracle、prior-mean content oracle；
5. V2：target/canonical speaker swap 与 content-preservation gate；
6. MM05 × 10 labels × 5 trials 的 direct EEG→MFCC overfit gate及 WAV preview；
7. all eligible fit records full-fit gate及 50 个 WAV preview；
8. exact-preview human approval；
9. subject-holdout validation、locked seen-label、`pot` exploratory；
10. 全部 eligible training-pair export。

V0/V1/V2/micro/full-fit 或人工 listening gate 任一未通过，后续阶段停止。prepared manifest 绑定 raw/post audio audit、denoise selection/manifest、speaker manifest 和 cache；每个模型 gate 继续绑定配置、上游 gate 和 checkpoint hash。

## 内容与音色证据

- V1 primary 使用 conditional prior mean，不允许目标 Mel；posterior 只作为 audio upper bound。
- V1 同时报告 fixed librosa-equivalent analytic backend、prior、posterior，以及 prior–posterior Mel gap。
- V2 的 target voice 来自同 subject 的 non-target fit trials。
- V2 先估计 KaraOne real-audio same/different-speaker 分布，再要求至少 80% target-voice generations 超过 different-speaker/same-label P90、speaker swap bootstrap CI 大于零，并保持 content retrieval。
- primary EEG WAV 永远使用 fit-only canonical speaker medoid，不使用 validation/test subject reference audio。

## 主要输出

- `audit/raw_audio_audit.*`、`denoise/selection.csv`、`denoise/manifest.json`；
- `gates/V0_*`、`V1_*`、`V2_*`、`C_50_pair_*`、`D_full_fit_*`；
- `gates/E_training_wav_human_review.json`；
- `pairs/micro_50_preview/`、`pairs/full_fit_preview/`；
- `evaluation/*.json`；
- `pairs/training_fit_eligible/`；
- `run_manifest.json`。

所有 v3 写入均位于 `../artifacts/open_vocab_v3_mfcc_training_first/`。v0724/v0730 只作为只读来源，不会被覆盖。
