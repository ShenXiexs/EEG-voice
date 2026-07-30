# v3：内容优先 EEG → MFCC training-first

v3 是独立于 v1/v2 的最小闭环，不训练 EEG 音色、F0、能量、时长、prosody、CVAE residual 或文本锚点。EEG 主链仅为：

`EEG → 40×256 content-normalized MFCC → deterministic MFCC-to-Mel audio oracle → frozen SpeechT5 HiFi-GAN → WAV`

MFCC 是 VAD active crop 后的 utterance-level CMVN 表征，且 c0 被置零；它是内容优先的低维声学目标，不是“纯内容”或完整音频。

## 一次性运行

在 `app` 目录执行：

```bash
./bootstrap_open_vocab_v3.sh       # 首次一次：安装 SpeechBrain 的 ECAPA 依赖
./run_open_vocab_v3_all.sh
```

默认总训练时间上限为 9.5 小时（audio decoder、50-pair EEG、full fit 共用一个绝对 deadline）；批量 WAV 导出不计入该预算。可选环境变量：`BUDGET_HOURS=9.5`、`TRAIN_DEVICE=auto`、`EVAL_DEVICE=cpu`、`EXPORT_DEVICE=cpu`。CPU 是 evaluation/export 默认值，以规避此前 PyTorch MPS 多次 counterfactual forward 的已知崩溃。

所有 v3 写入均在：

`../artifacts/open_vocab_v3_mfcc_training_first/`

v0724/v0730 仅作为只读源缓存，入口内置输出路径防火墙。

## Fail-closed 顺序

1. light audio audit（DC removal、VAD crop、RMS normalization、长度与 active/inactive contrast 审核；不自动 DeepFilterNet）
2. V0：immutable v0728 power-dB Mel `/10` → frozen SpeechT5 vocoder
3. audio-only MFCC→Mel training，随后 V1 内容 oracle、V2 音色 oracle
4. MM05 × 10 labels × 5 trials 的 direct EEG→MFCC 50-pair gate
5. all eligible fit records 的 full-fit gate
6. 仅当第 5 步通过，运行 subject-holdout、locked seen-label、`pot` locked-unseen exploratory report 和 training-pair WAV export

任一 V0/V1/V2/50-pair/full-fit gate 未通过，shell 会以非零状态停止，后续阶段不会运行。超过 2.56 秒 active duration 的 trial 会出现在 `audit/audio_audit.*`，但不会进入 fit；`MM18:94`、`MM14:36`、`MM16:56` 在人工确认前直接排除出 fit。低对比度样本进入审计队列但不自动排除或去噪。

Micro 与 full-fit 使用两个独立 checkpoint。prepared manifest 同时绑定 cache、音频审计和 speaker manifest；每个 gate 再绑定配置、上游 gate 和对应 checkpoint hash。旧 gate 或不完整审计不能授权新的 cache/checkpoint 继续运行。

## 主要输出

- `gates/V0_vocoder_oracle.json`：完整 Mel 参数和数值范围、vocoder manifest/hash、WAV hash、label retrieval、DTW-HuBERT。
- `gates/C_50_pair_overfit.json`：label retrieval、strict same-label trial R@1、zero/time/channel controls。
- `gates/D_full_fit.json`：full-fit gate；strict one-to-one R@1 的 bootstrap CI 必须高于 chance。
- `evaluation/*.json`：held-out MFCC 指标、locked-unseen `pot` exploratory，以及以 fit-only canonical voice 合成的 pooled-HuBERT retrieval、paired DTW-HuBERT 与 correct-minus-control bootstrap CI。
- `pairs/training_fit_eligible/`：每个合格 fit trial 的 cleaned reference、V0/V1、correct EEG/zero/time/channel WAV、MFCC/Mel comparison PNG、trial retrieval rank 和 manifest；manifest 独立记录导出耗时。
- `run_manifest.json`：audio/micro/full-fit 三阶段耗时、checkpoint hash 与累计训练耗时。

`target reference voice` 只在 V2 audio-only oracle 中使用，且总是同 subject 的 non-target trial reference；primary EEG validation/test WAV 始终使用 fit-only canonical voice medoid。
