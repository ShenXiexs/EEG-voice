# FEIS + KARA ONE + ds004306 preprocessing bundle

## OpenVoice-EEG v0724（内容—发声实现解耦）

v0724 位于 `app/src/open_vocab_0724/`，与 0721/0722 checkpoint、cache 和
artifact 完全隔离。模型从 EEG 分别预测 content tokens 与 realization/timbre
tokens，显式输出 80-bin log-mel 能量结构和可变长度 EnCodec codes。正式
`encode`/`generate` 接口只接收 EEG、通道坐标、通道 mask 和时间 mask；label、
subject、dataset 与真实音频均不能进入推理路径。

默认配置采用固定三 seed（15/31/47）和所有 locked-train KaraOne subject 的
LOSO。`audit-audio --strict` 未通过时，EEG 预训练和配对训练都会被 audio
freeze gate 阻止；validation gate 未通过时，locked test 不可访问。

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
bash app/run_open_vocab_0724_v1.sh cache
bash app/run_open_vocab_0724_v1.sh train-audio
bash app/run_open_vocab_0724_v1.sh audit-audio --strict
bash app/run_open_vocab_0724_v1.sh seeds
bash app/run_open_vocab_0724_v1.sh loso-all
bash app/run_open_vocab_0724_v1.sh synthesize karaone validation
bash app/run_open_vocab_0724_v1.sh gate --strict
```

`synthesize` 默认只处理 reconstruction-eligible 的 trial；超过 4 秒主动发声的
KaraOne trial 保留在 content 学习/latent evaluation 中，不会进入 exact
energy/code gate。正式 gate 会拒绝 `--limit`、缺失反事实、跳过 decoded-content
metric、非主 g1 run，或不完整的三-seed/全-subject LOSO artifact。它不会接受
latent cosine 替代解码波形的 HuBERT frame matching/SpeechBERTScore。

通过严格 validation gate 后，使用下面唯一入口运行最终 test；该入口原子地占用
一次 final-test session，依次执行 combined latent evaluation、KaraOne 与 FEIS
reconstruction。不要分别直接调用脚本的 `--split test`：一次 session 不能重放。

```bash
bash app/run_open_vocab_0724_v1.sh test
```

若需要从 cache v2 开始一口运行正式训练、全部 validation/LOSO gate、唯一的
locked test，以及最终 KaraOne/FEIS 重建语音对比图，使用总控脚本：

```bash
DEVICE=mps bash app/run_open_vocab_0724_full.sh
```

它固定使用 `15/31/47` 三个 seed，先完成严格 validation gate，才在最后一步原子地
执行 latent test、KaraOne test synthesis 和 FEIS test synthesis。随后输出每个 test
trial 的波形、包络、显式预测能量图、以及从最终解码 WAV 重新计算的 log-mel 对比图到：

```text
artifacts/open_vocab_0724_factorized_v1/synthesis/karaone/test/comparison_pairs/
artifacts/open_vocab_0724_factorized_v1/synthesis/feis/test/comparison_pairs/
```

对比图只用于展示；所有数值指标仍读取 float `.mel.npy` 或 WAV 张量，且频率轴从不
缩放。总控脚本会在长训练开始前、以及 locked test 前做只读 preflight；若 test 已被
占用或绘图中断，不要重新运行总控脚本。完成 test 后可安全地单独重画图：

```bash
bash app/run_open_vocab_0724_v1.sh plot karaone test
bash app/run_open_vocab_0724_v1.sh plot feis test
```

完整架构、监督边界、输出格式、指标和反事实条件见
[`reports/open_vocab_eeg_to_speech_factorized_0724.md`](reports/open_vocab_eeg_to_speech_factorized_0724.md)。
KaraOne 是唯一进入 trial-specific realization gate 的数据集；FEIS 只提供
subject-label content/弱 timbre prototype，ds004306 不参与音频重建 gate。
`synthesize` 因此只接受 `karaone|feis`，不会为 ds004306 生成伪音频。

主 seed 通过后，可用隔离 artifact 路径运行另外两个 seed、训练集内
subject-LOSO 和 held-label 开发实验，不会覆盖主 checkpoint：

```bash
bash app/run_open_vocab_0724_v1.sh seeds
bash app/run_open_vocab_0724_v1.sh loso karaone:MM05
bash app/run_open_vocab_0724_v1.sh loso-all
bash app/run_open_vocab_0724_v1.sh held-label /IY/
```

`all` 是上述完整正式顺序的便捷入口（包含三 seed 与 `loso-all`，但绝不触发
locked test）：

```bash
bash app/run_open_vocab_0724_v1.sh all
```

正式消融配置由同参数量开关生成；除 ContentVec 外均复用已经通过 gate 的冻结
audio prior，只重新训练 EEG。ContentVec 需要显式提供本地模型或模型 ID，并建立
独立 cache/audio oracle：

```bash
bash app/run_open_vocab_0724_v1.sh ablation-config content_only \
  app/configs/open_vocab_0724_ablation_content_only.yaml
bash app/run_open_vocab_0724_v1.sh ablation-config full_contentvec \
  app/configs/open_vocab_0724_ablation_contentvec.yaml \
  --contentvec-model /absolute/path/to/contentvec
```

## OpenVoice-EEG 0722 V1（新版本）

开放词汇、label-free EEG→语音版本已隔离实现于
`app/src/open_vocab_0722/`，输出固定写到 `artifacts/open_vocab_0722_v1/`。
主推理 API 只有 `generate(eeg, channel_xyz, channel_mask, time_mask)`；label、dataset、
subject 和真实音频均不能进入主生成路径。完整方法、证据边界、gate 和运行顺序见
[`reports/open_vocab_eeg_to_speech_moe_0722_plan.md`](reports/open_vocab_eeg_to_speech_moe_0722_plan.md)。

有进度条的一键 Track A G1 运行：

```bash
LIBRITTS_ROOT=/absolute/path/to/LibriTTS \
AISHELL_ROOT=/absolute/path/to/AISHELL-1 \
DEVICE=mps bash app/run_open_vocab_0722_v1.sh all
```

`PROJECT_ONLY=1` 仅用于小规模 smoke，不构成开放词汇实验。Track B 通过
`bash app/run_open_vocab_0722_v1.sh track-b` 单独生成，不覆盖14通道数据。旧0715/0721
checkpoint 与0722 schema不兼容。`data/`、`eeg_output/`、`artifacts/`、checkpoint、WAV、
teacher/cache 均继续由本 bundle 的 `.gitignore` 排除。

This bundle creates a unified imagined-speech EEG dataset without modifying
anything under `data/`.

The output is written to `eeg_output/` and contains one compressed subject
bundle per recording, a trial manifest, audio cache, subject-disjoint split
manifest, and quality-control reports.

## What is harmonised

- EEG channel order: `F3 FC5 AF3 F7 T7 P7 O1 O2 P8 T8 F8 AF4 FC6 F4`
- EEG shape: 14 channels x 1280 samples (5 seconds at 256 Hz)
- FEIS: existing `thinking` stage, normalized against its `resting` stage
- KARA ONE: existing `thinking` stage, normalized against its `clearing` stage
- ds004306: raw 1024-Hz EEGLAB data, temporary `.set/.fdt` staging, 50-Hz
  notch, 1--40-Hz bandpass, average reference, 256-Hz resampling, then
  `Imagination_*` event epoching
- Audio: lossless mono 16-kHz WAV cache; variable durations are retained and
  recorded in `audio_valid_samples`

The default uses ds004306 **auditory-cued imagination** only.  Its published
audio files are stored by category rather than unambiguously by trial.  The
manifest therefore marks them `weak_category_level`; do not evaluate direct
waveform reconstruction on ds004306 as though every trial had a unique,
confirmed waveform target.

## Run

First validate paths without writing output:

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
/opt/anaconda3/bin/python scripts/preprocess_combined_eeg.py --dry-run
```

Then launch preprocessing with visible per-subject progress:

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
bash run_preprocess.sh
```

The default excludes FEIS subject `05`, which the source manifest marks as
anomalous.  To include it, or to also preprocess ds004306 text/image prompts:

```bash
/opt/anaconda3/bin/python scripts/preprocess_combined_eeg.py \
  --data-root data \
  --output-root eeg_output \
  --ds-modalities auditory text image \
  --include-feis-subject-05
```

The run processes ds004306 one continuous recording at a time.  It normally
needs several hours, roughly 4--8 GB peak RAM, and a few GB for outputs.  If a
run is interrupted, simply run the same command again: already completed
subject files are reused and only missing recordings are processed.  Use
`--overwrite` only when deliberately regenerating every output NPZ/CSV.

## Training invariants

- Mask all samples at or after `eeg_valid_samples`.
- Split only on `subject_group_id`; never randomly split trials.
- The **only authoritative training split** is
  `app/configs/split/combined_0715_v1_split.yaml`.
- `eeg_output/manifests/subject_holdout_splits.json` is generated only for
  preprocessing QC (`purpose=preprocessing_qc_only`,
  `authoritative_for_training=false`) and must never select training,
  validation, or locked-test rows.
- Preserve `dataset`, `modality`, `audio_pairing`, and `pairing_confidence` as
  model covariates/weights.  In particular, ds004306 is weaker audio
  supervision than KARA ONE.

## Combined 0715 V1 training

The first combined training implementation is documented in
[`reports/combined_0715_v1_training_plan.md`](reports/combined_0715_v1_training_plan.md).
It adapts the latest KaraOne 0715 codec-token method to 14-channel imagined EEG,
uses a locked cross-dataset subject split, and treats FEIS/ds004306 pairing as
weaker than KaraOne trial-level pairing.

Before training for the first time, rebuild the KaraOne normalized output with
valid clearing lengths:

```bash
bash run_preprocess.sh --datasets karaone --overwrite
```

After preprocessing exists, use the following order.  `--verify-only` scans
the existing manifest/NPZ files, validates the locked YAML split and writes a
real pass/fail QC record bound to the exact split, manifest, EEG NPZ and QC
hashes; it does not rebuild EEG data.  Any later change to those inputs makes
that verification stale and training stops until `--verify-only` is rerun.
The signal probe reads only train and validation EEG and no longer requires an
audio cache.

```bash
bash run_preprocess.sh --verify-only
bash app/run_combined_0715_v1.sh probe
bash app/run_combined_0715_v1.sh cache --rebuild
bash app/run_combined_0715_v1.sh audit-audio
bash app/run_combined_0715_v1.sh train-audio
bash app/run_combined_0715_v1.sh train-eeg
bash app/run_combined_0715_v1.sh validate
```

音频阶段现在是“监督音频初始化 → 三数据集音频微调”，不是随机初始化：
`train-audio` 会检查
`../karaone_overt_recon_bundle/artifacts/outputs_karaone_0715/karaone_0715_audio_codec_s15/checkpoints/best.pt`；
如果不存在，会先自动运行 KaraOne 0715 `prepare` 和 `audio`，再将其 EnCodec
代码生成器/条件编码器权重迁移到 30-label combined 模型并继续微调。KaraOne
的 11 个 label head 行会复制到 combined 的 KaraOne label slice，其余 FEIS 和
ds004306 行保留为新初始化并在 combined 音频监督上训练。初始化 checkpoint 的
SHA256 和迁移报告会写入 combined audio checkpoint；微调默认使用比 scratch 更
小的 `audio_model.finetune_lr=1e-4`；resume 时必须再次提供同一初始化 checkpoint。

如需显式指定已训练好的 KaraOne 音频 checkpoint：

```bash
KARAONE_AUDIO_CHECKPOINT=/absolute/path/to/karaone_0715_audio_codec_s15/checkpoints/best.pt \
bash app/run_combined_0715_v1.sh train-audio
```

`--allow-scratch-audio` 仅用于轻量诊断 smoke；它会明确标记为
`scratch_diagnostic`，不应作为最终 label-to-audio 结果。

当前 wrapper 默认将 `ALLOW_FAILED_GATE=1` 传给 combined EEG/validation
流程，因此音频 gate 未通过时会继续生成 exploratory EEG checkpoint；gate
仍然写为 `passed=false`，不会解锁 locked test。若要恢复严格阻断：

```bash
ALLOW_FAILED_GATE=0 bash run_combined_0715_full.sh
```

本轮音频优化包括按 dataset/label 加权采样和将 combined `lambda_label`
从 `0.25` 提高到 `1.0`。若明确需要重新训练 KaraOne 音频初始化模型及
combined 音频模型，必须同时打开 `RUN_AUDIO=1`：

```bash
RUN_AUDIO=1 RETRAIN_KARAONE_AUDIO=1 ALLOW_FAILED_GATE=1 \
bash run_combined_0715_full.sh
```

已有 cache v2 和 combined audio checkpoint 后，推荐的一键重跑会**复用音频
产物**，只执行 combined EEG 40 epochs、validation、FEIS/KaraOne/ds004306
全量 validation synthesis、分层 reconstruction gate 和
reference-vs-reconstruction 对比图。新训练和重建产物统一写到
`artifacts/0721v1/`，既有 cache 与已微调音频 checkpoint 仍从
`artifacts/combined_0715_v1/` 只读复用：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
RUN_NAME=0721v1 COMBINED_DEVICE=mps EEG_EPOCHS=40 bash run_combined_0715_full.sh
```

等价的显式写法是：

```bash
RUN_NAME=0721v1 RUN_AUDIO=0 REBUILD_CACHE=0 RUN_PRECHECKS=0 \
EEG_EPOCHS=40 SYNTHESIS_LIMIT=-1 PLOT_COMPARISONS=1 \
ALLOW_FAILED_GATE=1 COMBINED_DEVICE=mps bash run_combined_0715_full.sh
```

`run_combined_0715_full.sh` 的默认值即为 `RUN_NAME=0721v1`、`RUN_AUDIO=0`、
`REBUILD_CACHE=0`、`RUN_PRECHECKS=0`、combined EEG `40` epochs。启动时会先
检查已有 cache/audio checkpoint 是否存在且非空；缺失时立即报出具体路径，
不会悄悄从头微调音频。随后在全量 validation synthesis 后自动执行
reconstruction audit，并生成 reference-vs-reconstruction pair 图。
WAV 输出位于 `artifacts/0721v1/samples/<dataset>/validation/`，
对比图和 pair CSV 位于对应的 `comparison_pairs/` 子目录。可用
`SYNTHESIS_LIMIT=12` 做快速初始检查，或用 `PLOT_COMPARISONS=0` 跳过绘图。
reconstruction audit 至少需要每个数据集 12 条 validation 样本。

新的重建判据不再把 raw waveform correlation 当作唯一 gate。每条样本同时
报告严格波形相关/SI-SDR，以及 25-ms RMS envelope correlation、envelope
overlap、activity IoU、onset/offset error、20/50/100-ms RMS correlation 和
multi-scale log-spectrogram MAE。对比图中的虚线是 25-ms RMS envelope，横轴
保持真实 2 秒音频时长。

最终 validation gate 分成两层：

- `structure_reconstruction_passed`：EEG-conditioned 的 median envelope
  correlation 与 activity IoU 均至少为 `0.30`；
- `eeg_specific_reconstruction_passed`：同一 trial 下，EEG-conditioned 相对
  shuffled EEG、zero EEG 和 dataset-only 的 paired median gain 均至少为
  `0.03`，且逐 trial 胜率均至少为 `0.55`。

`label_only` 使用同一 trial EEG 预测的标签概率，因此单独报告、但不作为纯负
对照。top-level `passed` 只允许 KaraOne 的 same-trial overt 配对贡献；FEIS
只允许 subject-label canonical 声明，ds004306 只允许 category-candidate
声明。审计失败默认仍写完整报告并继续绘图，只有显式 `--strict` 才以非零状态
退出；locked test 仍要求 top-level gate 真正通过。报告位于
`artifacts/0721v1/eeg/metrics/reconstruction_validation_report.json`。

EEG 训练阶段的 `best.pt` 是按 KaraOne validation condition/envelope head、label
balanced accuracy 与 onset/duration error 得到的**代理最优**，它不是实际 EnCodec
解码音频的最终最优点。训练现在还会在 epoch 1、每 5 epoch 和最后一个 epoch 将
轻量候选保存到 `eeg/checkpoints/candidates/`。正式重建必须在 KaraOne validation
上实际解码候选，并用 EEG-conditioned 相对 shuffled/zero/dataset-only 的增益选出
`selected.pt`。该改动没有修改 combined YAML，因此现有 audio checkpoint 的
lineage 仍可直接复用。

已有 `0721v1` 无需重训，也无需重新微调音频。下面的一键脚本会实际解码并比较
当前第 4 epoch 的 `best.pt` 与第 40 epoch 的 `last.pt`，选择结果写为
`artifacts/0721v1/eeg/checkpoints/selected.pt`，随后用同一个 selected checkpoint
完成三数据集 validation WAV、六种对照、结构审计和对比图：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/feis_karaone_ds004306_bundle
COMBINED_DEVICE=mps bash app/run_0721_select_and_reconstruct.sh
```

候选选择默认使用完整 KaraOne validation（`SELECTION_LIMIT=-1`），最终也生成完整
validation（`SYNTHESIS_LIMIT=-1`），过程中保留阶段进度条和逐样本 tqdm。若只想先
验证脚本，可显式使用至少 12 条样本：

```bash
SELECTION_LIMIT=12 SYNTHESIS_LIMIT=12 PLOT_LIMIT=12 COMBINED_DEVICE=mps \
bash app/run_0721_select_and_reconstruct.sh
```

选择报告位于 `artifacts/0721v1/eeg/metrics/checkpoint_selection.json`；最终 WAV 位于
`artifacts/0721v1/selected_samples/<dataset>/validation/`。即使所有候选都未通过
EEG-specific gate，脚本仍会选择分数最高者用于 exploratory 输出，但报告中的
`passed` 保持 `false`，不会错误解锁 locked test。

### 0721v1 EEG loss recipe

0721v1 不再只靠 code CE 或逐点 waveform 指标学习。有效训练目标包括：

- dataset-specific label CE；
- EEG/audio condition alignment；
- FEIS 软多正样本与 KaraOne exact-pair contrastive；
- KaraOne 全 codebook 与 FEIS q0/q1 EnCodec code CE；
- KaraOne envelope MSE，以及 1/5/9 code-step 三尺度 envelope correlation；
- differentiable activity Dice、onset/duration Smooth-L1；
- KaraOne 同标签、不同音频 morphology ranking；
- subject-adversarial、dataset-sliced audio-label distillation 与 variance regularization。

KaraOne train loader 会显式生成同标签、不同 `audio_key` 的成对 batch，避免
morphology ranking 在 batch size 4 时长期失活。每项 loss、三个 envelope scale、
ranking active fraction 和 correct-vs-shuffled correlation 都写入逐 epoch 日志；
完整权重与 recipe version 也写入 `best.pt`/`last.pt` metadata。FEIS 没有可靠的
trial-level envelope target，ds004306 只有 category candidate audio，所以两者
不会伪装成 KaraOne 式 trial-level morphology supervision。

该 wrapper 会显示阶段级总进度条；cache、audit、训练和 synthesis 阶段还会
显示各自的 tqdm 进度。默认不会重复覆盖 KaraOne EEG 输出。如需先重建
KaraOne valid-length 预处理：

```bash
REBUILD_KARAONE=1 bash run_combined_0715_full.sh
```

常用选项包括 `RUN_SYNTHESIS=0`（只运行到 validation）、
`SYNTHESIS_LIMIT=12`（每个数据集只生成 12 条 validation 样本）和
`COMBINED_DEVICE=mps|cuda|cpu`。只有需要重新核验输入时才设置
`RUN_PRECHECKS=1`；只有需要重建 codec cache 时才设置 `REBUILD_CACHE=1`；
只有明确要重新微调音频时才设置 `RUN_AUDIO=1`。从原始预处理检查到音频
微调的完整显式运行方式为：

```bash
RUN_PRECHECKS=1 REBUILD_CACHE=1 RUN_AUDIO=1 \
COMBINED_DEVICE=mps EEG_EPOCHS=40 \
bash run_combined_0715_full.sh
```

locked test 不由该 wrapper 自动执行，必须先人工审查 validation gate。

The cache command now writes `combined-0715-cache-v2`, including source audio
paths, valid sample counts and exact EnCodec scale metadata.  A legacy cache
must be rebuilt with `--rebuild`.  Checkpoints use the v2 checkpoint/lineage
contract and bind the config, locked split, unified manifest, all referenced
preprocessed EEG payloads, QC table and cache.  Legacy smoke checkpoints are
strictly rejected and must be retrained.

`audit-audio` performs a real EnCodec decode on a deterministic stratified
validation sample (at most four unique audio keys per dataset and label).  It
does not sample locked-test waveforms.  Each dataset passes only when median
waveform correlation is at least `0.65`, median SI-SDR is at least `0 dB`, and
median RMS-normalized log-spectrogram MAE is at most `12 dB`, in addition to
all cache structure checks passing.

The locked test phase requires explicit `--allow-final-test`, a passed
validation gate, the exact validation-report SHA, and exact checkpoint/lineage
hashes.  A metadata-only preauthorization happens before the manifest, cache,
or test EEG is opened; full current-lineage validation then runs again.
`--allow-failed-gate` is forbidden for test.  The default configuration uses the first 768 EEG
samples (3 seconds) to match KaraOne 0715; the original 1280-sample output
remains available for later ablations.

## Synthesis controls

The synthesis script exports one reference directory plus six generated
controls under `<output>/<dataset>/<split>/`:

```text
reference/
codec_oracle/
eeg_conditioned/
label_only/
zero_eeg/
shuffled_eeg/
dataset_only/
synthesis_manifest.json
```

`label_only` uses the same-trial EEG label-head probabilities, not the true
label. `shuffled_eeg` is a deterministic, same-dataset/same-label derangement
without self-loops. `dataset_only` uses the empirical training-label prior.
Validation export requires a passed round-trip audit unless
`--allow-failed-gate` is supplied, in which case the manifest is explicitly
marked exploratory.  Locked-test export cannot bypass either gate.

Example validation export:

```bash
/opt/anaconda3/bin/python app/scripts/synthesize_combined_0715.py \
  --cache artifacts/combined_0715_v1/cache/combined_0715_encodec_codes.npz \
  --audio-checkpoint artifacts/combined_0715_v1/audio/checkpoints/best.pt \
  --eeg-checkpoint artifacts/0721v1/eeg/checkpoints/best.pt \
  --dataset karaone --split validation \
  --validation-gate artifacts/0721v1/eeg/metrics/validation_gate.json \
  --output artifacts/0721v1/samples
```

ds004306 audio remains category-level candidate supervision; every synthesis
manifest therefore records `ds004306_trial_level_claim_allowed=false`.
FEIS permits only canonical-audio/coarse-code claims.  Only KaraOne can support
a trial-level acoustic claim, and only after its validation controls pass.

## Remaining formal-study limitations

Dataset-sliced distillation, validation-based EEG checkpoint selection and
automatic reconstruction auditing are now implemented. The remaining limits
are scientific rather than silent code fallbacks: FEIS audio is canonical/coarse
supervision; ds004306 audio is category-candidate supervision; only KaraOne has
same-trial overt audio. Valid-audio-length handling inside the shared audio
condition encoder remains an ablation item. Runs should therefore be described
as exploratory cross-dataset EEG-to-audio reconstruction, not confirmed
reconstruction of hallucinated speech.
