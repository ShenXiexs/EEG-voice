# Joint EEG→speech-content pilot v1

## Stable interface

每个 batch 只包含一个 dataset，并返回 EEG `[B,C,T]`、电极坐标
`[B,C,3]`、channel/time mask、MFCC `[B,39,161]`、可选 HuBERT
`[B,96,768]`、pairing weight、phoneme auxiliary label，以及固定帧长的
log-mel/RMS/activity target。联合训练使用 dataset-homogeneous batch；主 batch
保持 DS004940:DS006104=1:1，每 5 个 batch 插入一个真实 label-only batch，
不把 61 通道 zero-pad 成 128 通道。

## Supervision

- `verified_exact`：content + timing/acoustic；当前只用于 DS004940。
- `candidate_filename_timing`：0.35 权重 content + phoneme；用于 DS006104
  Words/phonemes，不开放 waveform/acoustic loss。
- `label_only`：phoneme auxiliary；用于 DS006104 single-phoneme。
- dataset/subject/task/label/audio 均不进入 content inference path。

## Gates

1. Audit：DS004940 的两个 split-run boundary trial 必须保持显式排除；
   DS006104 的 S15 官方 auxiliary 必须来自固定 commit 并通过预登记 SHA256。
2. Data：HDF5 为 Volts/float32；xyz/masks/normalizer hash 完整；无 NaN/Inf。
3. M0：每 dataset 50 pair、3 seeds；pair R@1≥0.90，MFCC 相对模板改善
   ≥50%，正确 EEG 优于 zero/time/channel shuffle。
4. M1：只有三个 M0 mode × 三个 seed 全部通过才运行。独立
   `stage2_joint_ood` split 精确实现 4/1/1 subjects 和 28/6/6 contents；任何
   subject/content 交叉象限均排除。

`--smoke-model` 使用 48-dim 1-layer 网络，只证明数据、loss、梯度和
optimizer 接口可运行。它的 retrieval 或 control 数值不得用于研究结论。

## Implemented Stage-0 result (2026-08-25, v3 contract refresh)

- Audit: DS004940 17,489、DS006104 10,888 个 trial 通过 QC；DS006104 的
  S15 官方 auxiliary 已从固定 commit 获取并通过预登记 SHA256；DS004940
  的 2 个 run-boundary trial 显式排除。
- `joint_ood_fold-0` 对 subject、linguistic content 与 exact waveform 的
  train/validation/test 交集均为 0。
- 正式 M0 content shard 均为 frozen train role：每 dataset 5 subjects × 10
  common contents = 50 trials。DS004940 使用 Active；DS006104 使用 TMS0 的
  Words+phonemes（30+20）。
- DS006104 真实 single-phoneme label-only shard 为 5 subjects × 6 labels =
  30 trials；content loss 严格为 0，只进入 0.1 权重 phoneme head。
- 正式 192-dim/4-layer 模型在 `[8,128,1178]` 与 `[8,61,384]` 上完成同一
  model forward/backward，所有 loss/gradient finite。
- 20 个 included shard 共享唯一 preprocessing contract；130 条旧记录保留为
  `stale_incompatible`，不再进入 loader/normalizer。
- train-fold normalizer 使用 50+50+30 个当前 contract trial；speech target
  cache 对 24 个唯一 waveform 启用 HuBERT layer 9。缺失 target fail closed。
- raw→harmonized Welch PSD 在两个数据集各 4 个固定 trial 上全部 finite，8/8
  trial 的 45–100 Hz/passband 比率下降。
- `stage2_joint_ood_fold-0.csv` 已冻结并验证：两个数据集分别可形成
  train 4×28、validation 1×6、test 1×6 完整网格且无 subject/content 泄漏。
- 固定抽取的 20 个 DS004940 pair 已通过 audio/event hash、run-local onset
  与 epoch mapping 的机器复核；试听与括号内外 transcript 语义仍明确为
  `pending`，未被机器检查伪装成人工确认。

正式 DS004940/joint M0 会在该状态为 `pending` 时拒绝启动。人工试听并核对
`artifacts/training_data/v3/qc/ds004940_pair_review_20.csv` 后，将每行
`human_listen_transcript_status` 改为 `pass`，再运行 `validate --strict`；
验证器会保留这些人工状态。engineering smoke 与 dry-run 不绕开科学 gate，
它们始终被标成非研究结论。

## Historical 100-step capacity screen (superseded engineering evidence)

下表来自旧 artifact contract，仅保留作失败记录；它不是当前 5,000-step ×
3-seed M0，也不能与新 split/shard 直接比较。

| training | evaluation | content R@1 | MFCC L1 | dataset-mean template L1 | gate |
|---|---|---:|---:|---:|---|
| DS004940 only | DS004940 | 0.10 | 0.7777 | 0.7213 | fail |
| DS006104 only | DS006104 | 0.10 | 0.7546 | 0.6877 | fail |
| joint alternating | DS004940 | 0.06 | 0.7795 | 0.7213 | fail |
| joint alternating | DS006104 | 0.06 | 0.7575 | 0.6877 | fail |

10 个 content 下 chance R@1 为 0.10。correct EEG 未一致优于 zero、time 与
channel controls，因此不得进入 Stage 2；当前结果只说明 pipeline 可训练，
不支持正迁移或 EEG-content decoding 成功。

在这次 engineering screen 中，subject-bootstrap 的 `single error - joint
error` 对 DS004940 为 -0.00185（95% CI -0.00198, -0.00174），对
DS006104 为 -0.00298（95% CI -0.00387, -0.00251）；负值表示 joint
更差。由于只有 1 seed、100 steps 且使用缩小模型，这只能记为负迁移预警，
不能替代注册的 Stage-2 推断。

M0 的 DS004940 同 content 重复引用同一 target waveform，因此 same-content
template 是不可用的零误差分母。配置现已预注册 `dataset_mean` 作为 collapse
baseline；same-content 只作 diagnostic，不再成为一个数学上无法通过的 gate。

## Known pending provenance

- S15 pinned auxiliary CSV 已按官方固定 commit 与预登记 SHA256 确认。
- DS006104 official filename/timing join 已逐 trial 成立，但缺少最终
  presentation manifest，因此仍是 candidate pairing。
- DS004940 NPI 文件名括号内外词的 transcript 语义尚未升级为跨数据集
  lexical identity；split 当前以完整 stimulus/phoneme sequence 分组。
- audio-only MFCC→log-mel/RMS/activity renderer 已实现并通过 dry-run；只有
  validation oracle gate 通过的 renderer checkpoint 才能进入 EEG evaluator。
  未提供经过验证的 vocoder，因此 waveform 始终明确标为未生成。
