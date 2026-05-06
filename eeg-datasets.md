# EEG Token 与声音内容/音色/音调对齐到说话形象重构

## 当前目标

本项目当前不再把“EEG token 与文本语义对齐”作为第一主线。新的核心问题是：

**被试听到语音后，连续 EEG 是否能先被编码成稳定 token，再与声音内容、音色、音调表征对齐，最后用于说话形象重构。**

主链路固定为：

```text
EEG -> token -> 与声音内容/音色/音调表征对齐 -> 说话形象重构
```

这里的“说话形象”优先落到两个可操作的声学目标：

- **声音内容 / content**：音素、音节、CV/VC/word、voicing、onset、粗粒度发音结构。
- **音调 / pitch**：F0、log-F0、voicing、pitch contour、局部上升/下降趋势。
- **音色 / timbre**：谱包络、formant 相关结构、MFCC / mel 频谱统计、speaker / emotion style embedding。

文本、词边界、语义标签可以作为辅助控制变量，但不应该作为主目标。讨论会中应把问题表述为 **EEG token 与声音内容/音色/音调表征对齐后重构说话形象**，而不是 EEG -> text。

## 数据集重新分工

### 1. ds006104：主控实验数据

本地已放入：

`data/meeting_examples/ds006104/stimuli/`

这个目录包含大量短语音刺激，文件名中直接标出 `happy`、`angry`、`control` 等条件。当前本地扫描结果：

- WAV 数量：1273
- happy 条件：632
- angry 条件：639
- control 条件：158
- 采样率：绝大多数为 44.1 kHz 单声道
- 时长：约 0.16 s 到 10 s，中位数约 0.57 s

它最适合承担三个任务：

1. **声音内容对齐**：EEG token 与 CV/VC/word、phoneme、voicing、onset 表征对齐。
2. **受控音调对齐**：EEG token 与 F0 / log-F0 / pitch contour 表征对齐。
3. **受控音色对齐**：EEG token 与 mel / MFCC / 谱质心 / 谱带宽 / 谱包络表征对齐。
4. **说话形象 probe**：happy vs angry、original vs control、CV/VC/word 条件下的声音属性是否可由 EEG token 线性读出。

这个数据集的优点不是自然连续语音，而是刺激短、条件密、声学差异清楚。它适合验证 tokenizer 是否真的保留了声音形象的低层线索。

### 2. ds005345：单男声、单女声、多人说话扩展

已准备讨论会音频：

- `data/meeting_examples/ds005345/ds005345_single_female_mandarin_audio_10s.wav`
- `data/meeting_examples/ds005345/ds005345_single_male_mandarin_audio_10s.wav`
- `data/meeting_examples/ds005345/ds005345_mix_two_talker_mandarin_audio_10s.wav`

这个数据集仍然重要，但用途需要调整：

- 单女声 / 单男声：用于 speaker timbre 和 pitch range 的粗粒度重构。
- mixed condition：用于测试 EEG token 是否能偏向 attended voice，避免只恢复混合声学平均值。
- word / prosody annotation：现在作为辅助定位，不作为主监督目标。

### 3. ds004408 与 ds004718：自然语音辅助集

这两个数据集仍可保留，但不再是第一阶段主线：

- `ds004408`：英语有声书，可用于自然连续 speech envelope、pitch contour、mel reconstruction 的预训练。
- `ds004718`：粤语《小王子》，有 f0/intensity 和词级 timing，适合检验自然语音中的 pitch / intensity tracking。

它们的贡献在于连续性和生态效度，不在于受控音色条件。

## 模型路线

### EEG tokenizer

输入连续 EEG：

```text
EEG x(t, channel) -> temporal encoder -> latent z(t) -> vector quantizer -> EEG token q_t
```

建议保留两个时间分支：

- 100-128 Hz 分支：保留 onset、voicing、短时 pitch 变化。
- 20-50 Hz 分支：保留较慢的 prosody、timbre envelope 和听觉皮层响应。

### 声音表征目标

每个语音片段提取：

- `content_unit[t]`
- `log_mel[t, f]`
- `f0[t]`
- `voicing[t]`
- `rms[t]`
- `spectral_centroid[t]`
- `spectral_bandwidth[t]`
- 可选：speaker / emotion embedding

第一阶段不必直接生成 waveform。先做 EEG token 与声音内容/音色/音调表征的对齐，再用声码器或检索式方式重建说话形象，会更稳。

### 损失函数

推荐总损失：

```text
L = lambda_mel * L_mel
  + lambda_content * L_content
  + lambda_f0 * L_f0
  + lambda_vuv * L_voicing
  + lambda_timbre * L_timbre
  + lambda_ctr * L_contrast
  + lambda_vq * L_vq
```

其中：

```text
L_mel = || mel_hat - mel ||_1
L_content = CE(content_hat, content_unit)
L_f0 = mean(|log_f0_hat - log_f0| over voiced frames)
L_voicing = BCE(vuv_hat, vuv)
L_timbre = 1 - cosine(e_timbre_hat, e_timbre)
L_contrast = InfoNCE(EEG_segment, matched_audio_segment)
```

对 `ds006104`，可以额外加入条件分类损失：

```text
L_style = CE(happy_angry_hat, happy_angry)
L_control = CE(control_hat, original_or_control)
```

这两个损失不是最终目标，但能快速验证 EEG token 是否保留了声音内容、音色、音调相关信息。

## 最小实验顺序

### Step 1：只看音频

先用本地 `ds006104/stimuli` 提取声学特征，确认 happy/angry/control 在 F0、谱质心、能量、时长上是否可分。

运行：

```bash
python3 scripts/analyze_ds006104_voice_features.py
```

输出：

- `data/meeting_examples/ds006104/voice_timbre_pitch_manifest.csv`
- `data/meeting_examples/ds006104/voice_timbre_pitch_manifest.json`
- `data/meeting_examples/ds006104/ds006104_voice_timbre_pitch_examples.svg`
- `data/meeting_examples/ds006104/voice_timbre_pitch_summary.md`

### Step 2：EEG 事件对齐

使用 `data/meeting_examples/ds006104/raw/sub-P01_ses-01_task-phonemes_events.tsv` 建立：

```text
event onset -> stimulus filename / phoneme / condition -> EEG window
```

先做 0-800 ms 或 0-1200 ms epoch。音频本身很短，EEG 响应需要保留神经延迟。

### Step 3：线性 probe

冻结 EEG tokenizer，测试：

- EEG token -> happy / angry
- EEG token -> control / original
- EEG token -> high / low pitch bin
- EEG token -> spectral centroid bin

通过这一步确认 EEG token 与声音内容/音色/音调表征确实可对齐，再进入说话形象重构；不要一开始直接做 waveform。

### Step 4：声音表征对齐

从 EEG token 预测 frame-level voice target：

```text
q_1 ... q_T -> content_hat, mel_hat, f0_hat, voicing_hat, timbre_hat
```

评估指标：

- content unit accuracy / retrieval recall。
- F0 RMSE / correlation，只在 voiced frames 上算。
- mel L1 / multi-resolution STFT loss。
- timbre embedding cosine similarity。
- happy/angry 重构后分类一致性。

### Step 5：说话形象重构

输入已对齐的声音内容/音色/音调表征，做两类重构：

- 检索式重构：从候选音频库中检索最匹配的 voice segment。
- 生成式重构：用声码器根据 `content_hat + f0_hat + timbre_hat` 生成可听声音。

### Step 6：跨数据集验证

在 `ds006104` 训练受控音色/音调 probe 后，再测：

- `ds005345`：男声、女声、混合双人语音。
- `ds004718`：粤语自然语音的 f0/intensity tracking。
- `ds004408`：英语自然语音的 pitch/envelope tracking。

## 讨论会建议表述

当前项目的卖点应该改成：

> 与其先追 EEG-to-text，不如先回答 EEG token 能否与声音内容、音色、音调表征对齐，并据此重构“说话形象”：内容轮廓、pitch contour、timbre envelope、speaker/style cues。这个目标更贴近听觉皮层的时间分辨率，也更适合用 ds006104 的受控语音刺激做第一轮强验证。

因此，报告中应减少“词级语义检索”篇幅，把重点放在：

- EEG token 是否能对齐声音内容单元、F0 / voicing。
- EEG token 是否能对齐谱包络和音色特征。
- happy/angry/control 条件是否能作为 voice style probe。
- 单男声、单女声、多人语音是否能测试 speaker timbre 和 attended voice。
