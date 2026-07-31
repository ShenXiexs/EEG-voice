# v3：EnCodec token + CLIP 对齐 + MFCC 内容解码

当前 v3 schema 为 `openvoice-eeg-v3-encodec-clip-mfcc-v1`。旧 direct-EEG→MFCC v3 产物不删除，但新 loader 会拒绝旧 cache/checkpoint。

主链：

```text
WAV → fit-only KaraOne-finetuned EnCodec → 8×192 codec IDs
    → AudioContentEncoder → 32×128 A_C → SharedMFCCDecoder → 40×256 CMVN-MFCC

EEG → EEGContentEncoder → 32×128 E_C
    ↔ strict trial-diagonal token CLIP / same-label global CLIP ↔ A_C
    → 同一个 SharedMFCCDecoder → 40×256 CMVN-MFCC

MFCC + fit-only canonical voice + fit-only median duration
    → native-SpeechT5-Mel residual CVAE prior mean
    → fit-only KaraOne-finetuned SpeechT5 HiFi-GAN → WAV
```

EEG forward 不接收 label、文本、speaker、目标时长或目标音频。文本只占 full-fit loss 的 5%，作为 deterministic training-only anchor。

## 音频特征与模型

- EnCodec：24 kHz、6 kbps、8 codebooks、1024 vocabulary、最多 192 steps；encoder/decoder 参数和 quantizer EMA codebook state 均参与适配并检查变化。
- MFCC：VAD active crop、utterance CMVN、相对时间重采样至 256 帧、c0 固定为 0。
- SpeechT5 Mel：严格调用官方 `SpeechT5FeatureExtractor` audio-target 路径；64 ms Hann、16 ms hop、80–7600 Hz、Slaney、log10。不存在旧 power-dB `/10` 或 learned Mel adapter。
- CVAE：analytic inverse-DCT/log10 Mel 为基线，CVAE 只学习 residual；posterior 仅用于 audio oracle，EEG 主结果只用 prior mean。
- 生成侧 EnCodec、SpeechT5 HiFi-GAN、ECAPA 仅用 1,016 个 fit-eligible WAV 微调。独立 HuBERT 和独立 ECAPA 只负责评价并保持原始权重。

## 完整 fail-closed 顺序

```text
A0 audit / selective denoise / split lineage
→ fit-only EnCodec + SpeechT5 + ECAPA adaptation
→ T0 EnCodec round-trip
→ T0b native SpeechT5 Mel round-trip
→ EnCodec token cache
→ audio token→MFCC training
→ T1 MFCC retrieval / T1d content-speaker probes
→ native-Mel CVAE
→ T2 prior/posterior content
→ T2v posterior improvement + prior diversity/retention
→ T3 target/canonical speaker swap
→ MM05 50-pair EEG overfit → C
→ full-fit EEG → D
→ training WAV/PNG preview
→ exact hash-bound human approval
→ validation / locked seen-label / pot exploratory
→ final all-training-pair WAV/PNG export
```

所有 gate 都计算真实指标。任何一个失败都会以非零状态停止；不会生成“占位通过”报告。人工批准前 held-out evaluator 会校验审批文件、full-fit checkpoint、D gate 和 preview manifest 的 SHA-256 lineage，任一变化都会拒绝访问。

## 一次性 20 小时入口

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle/app
./run_open_vocab_v3_complete.sh
```

20 小时是所有训练阶段共享的 absolute deadline，不会给每个阶段重新计时。审计、gate 评价和 WAV/PNG 导出不计入训练时长。

## Explore：记录失败但继续跑完整链路

若目标是诊断所有后续阶段而非得到可报告的主结果，可运行：

```bash
./run_open_vocab_v3_explore_20h.sh
```

它同样使用 20 小时的共享训练上限，但会记录每个失败 gate 后继续执行，包括跳过人工试听审批。所有输出会独立写入 `../artifacts/open_vocab_v3_mfcc_training_first_explore/`；gate JSON、held-out report、pair metadata 和 manifest 都带有 `exploratory_gate_bypass: true`。这些结果只能用于排错、试听和 ablation，不能替代 fail-closed 主实验的 held-out 结论。

脚本在 full-fit training preview 生成后暂停。试听：

```text
../artifacts/open_vocab_v3_mfcc_training_first/pairs/full_fit_preview/
```

只有在终端输入大写 `YES` 后才运行 validation/locked/pot。也可分两次：

```bash
./run_open_vocab_v3_all.sh
./approve_open_vocab_v3_training_preview.sh samxie "内容清楚且 controls 明显更差"
./run_open_vocab_v3_after_review.sh
```

## 每个展示 trial

```text
00_cleaned_reference.wav
01_encodec_codec_oracle.wav
02_real_mfcc_canonical_voice.wav
03_real_mfcc_target_voice_audio_only.wav
04_eeg_aligned_mfcc.wav
05_zero_eeg.wav
06_time_shuffled_eeg.wav
07_channel_shuffled_eeg.wav
mfcc_comparison.png
mel_comparison.png
token_similarity.png
metadata.json
```

最终目录：

```text
../artifacts/open_vocab_v3_mfcc_training_first/pairs/encodec_clip_mfcc_training_fit_v1/
```

其中 `manifest.csv` 与 `export_manifest.json` 汇总全部 fit-eligible training pairs。

## 选择性去噪

原 WAV 始终只读。`denoise/selection.csv` 默认 `apply=0`；只有人工确认是加性噪声并改为 `apply=1` 后才处理。enhanced WAV 必须同时通过时延、VAD 边界、duration、HuBERT 和独立 ECAPA preservation gate，否则仍使用原音频。异常长 trial 在人工确认前排除出 fit。

## Transductive audio demo

如需让生成侧 audio 模型见过全部 eligible WAV，可单独运行：

```bash
./run_open_vocab_v3_all_wav_audio_adaptation_demo.sh
```

它写入 `audio_adaptation/transductive_all_encodec_clip_v1/`，明确标记为 transductive，不会被主 validation/test runner 加载。
