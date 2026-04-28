# EEG-Audio 数据集逐个测试 Runbook

本文档用于把当前已经筛出的 EEG-audio / EEG-speech / EEG-music 数据集逐个验证清楚。目标不是一次性下载全量数据，而是先用最小成本确认三件事：

1. EEG 文件能被读取；
2. 音频、文本、事件或节拍标注能解析；
3. EEG 时间轴和 stimulus / annotation 时间轴可以建立可复现映射。

所有命令默认在仓库根目录执行：

```bash
/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding
```

## 0. 本地目录规则

代码、报告、manifest 可以提交到 Git。原始 EEG、音频、大型特征、模型权重不要提交。

建议目录：

```bash
mkdir -p data/raw/openneuro
mkdir -p data/raw/zenodo
mkdir -p data/raw/openmiir
mkdir -p data/derived
mkdir -p data/cache
mkdir -p outputs
```

这些目录已经在 `.gitignore` 中忽略。轻量报告仍然可以提交：

- `scripts/probe_eeg_audio_datasets.py`
- `outputs/eeg_audio_dataset_probe_results.md`
- `outputs/eeg_audio_dataset_probe_detailed.md`
- `outputs/eeg_audio_dataset_probe_results.json`
- `docs/dataset_testing_guide.md`

不要提交：

- `.eeg`, `.edf`, `.bdf`, `.set`, `.fdt`, `.vhdr`, `.vmrk`, `.fif`
- `.wav`, `.mp3`, `.flac`
- `.mat`, `.h5`, `.hdf5`, `.npy`, `.npz`, `.pkl`, `.pt`, `.pth`
- `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.7z`, `.rar`
- `data/`, `outputs/probe_artifacts/`

提交前检查：

```bash
git status --short --ignored
git check-ignore -v data/raw/test.wav data/raw/test.set outputs/probe_artifacts/ds004408/01_dataset_description.json
```

`outputs/probe_artifacts/` 应显示为 ignored；`outputs/*.md` 和 `outputs/*.json` 不应被忽略。

## 1. 先跑轻量探测

先确认远端 metadata、annotation、WAV header、EEG byte-range 都可访问：

```bash
python3 scripts/probe_eeg_audio_datasets.py \
  --json-out outputs/eeg_audio_dataset_probe_results.json \
  --md-out outputs/eeg_audio_dataset_probe_results.md \
  --detail-md-out outputs/eeg_audio_dataset_probe_detailed.md \
  --artifact-dir outputs/probe_artifacts
```

验证没有目标级错误：

```bash
python3 - <<'PY'
import json
from pathlib import Path

results = json.loads(Path("outputs/eeg_audio_dataset_probe_results.json").read_text())
errors = []

for dataset in results:
    if "error" in dataset:
        errors.append((dataset["dataset_id"], dataset["error"]))
    for target in dataset.get("targets", []):
        if "error" in target:
            errors.append((dataset["dataset_id"], target["label"], target["error"]))

print("datasets:", len(results))
print("target_errors:", len(errors))
for error in errors:
    print(error)

if errors:
    raise SystemExit(1)
PY
```

当前期望结果：

- `datasets: 13`
- `target_errors: 0`
- `outputs/probe_artifacts/` 保存约 61 个轻量实测 artifact

快速查看每个数据集的 artifact：

```bash
find outputs/probe_artifacts -maxdepth 2 -type f | sort
```

## 2. 推荐测试顺序

| 顺序 | 数据集 | 用途 | 先测内容 | 进入训练前门槛 |
| --- | --- | --- | --- | --- |
| 1 | `ds004718` LPPHK | 词级/韵律对齐主数据集 | word timing、trigger、acoustic | 词 onset 能落到 EEG token grid |
| 2 | `ds005345` LPP Multi-talker | 单说话人/混合说话人自然语音 | word info、f0/intensity、FIF | single female 可完成 EEG-audio 对齐 |
| 3 | `ds004408` naturalistic speech | 英语 TextGrid/phoneme 监督 | TextGrid、BrainVision | phoneme onset 与 EEG sample 对齐 |
| 4 | `ds006104` speech decoding | 受控 phoneme/articulation probe | events、EDF header | phoneme/manner/place 标签可分类 |
| 5 | `ds006434` ABR/attention | 64 秒长 epoch 和 attention timing | events、WAV header | event sample、duration、stimulus 一致 |
| 6 | `ds007591` speech decoding | overt/covert/minimally overt 辅助验证 | events、EDF | session/task condition 可分层 |
| 7 | `zenodo-4004271` KUL | 经典 AAD baseline | README、preprocess、S1.mat | 严格 leave-trial/story/subject split |
| 8 | `zenodo-1199011` DTU | competing speech robustness | preprocessing script、预处理包 | 不先下载 16GB EEG.zip |
| 9 | `zenodo-7078451` ESAA | Mandarin AAD | readme、baseline、S1.zip | trial marker 与 target stream 可恢复 |
| 10 | `zenodo-4518754` 255ch | 高密度空间 tokenizer | misc/scripts/stimuli | channel layout 可读后再下 subject |
| 11 | `openmiir` | 音乐 beat/tempo probe | metadata、beats | beat time 可落到 token grid |
| 12 | `ds003774` MUSIN-G | 自然音乐 EEG 预训练 | events、audio header、EEGLAB | listening run 可切片 |
| 13 | `zenodo-4537751` MAD-EEG | polyphonic music attention | behavior、yaml、HDF5 | target instrument 序列可恢复 |

## 3. 通用依赖检查

```bash
python3 - <<'PY'
import importlib.util

for name in ["mne", "numpy", "scipy", "pandas", "openpyxl", "requests", "h5py"]:
    print(name, "OK" if importlib.util.find_spec(name) else "MISSING")
PY
```

如果 `mne` 或 `openpyxl` 缺失，先在当前 Python 环境补依赖；不要把虚拟环境提交进仓库。

## 4. Dataset 1: `ds004718` LPPHK

### 4.1 为什么先测

这是当前最适合作为主线的词级/韵律对齐数据集。已经实测到：

- word timing XLSX: 4474 行；
- acoustic CSV: `time`, `f0`, `intensity`；
- EEG sidecar: `SamplingFrequency=1000`；
- sentence WAV header 可解析。

### 4.2 先测 annotation

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("outputs/probe_artifacts/ds004718")
words = pd.read_excel(root / "03_word_timing_xlsx.xlsx")
triggers = pd.read_excel(root / "04_trigger_sentence_xlsx.xlsx")
acoustic = pd.read_csv(root / "05_acoustic_csv_sample.csv")

print("word rows:", len(words))
print("word cols:", list(words.columns))
print("trigger rows:", len(triggers))
print("trigger cols:", list(triggers.columns))
print("acoustic cols:", list(acoustic.columns))
print(words.head(5))

required = {"word", "onset_all", "offset_all", "POS"}
assert required.issubset(words.columns)
assert {"time", "f0", "intensity"}.issubset(acoustic.columns)
assert len(words) > 4000
PY
```

### 4.3 下载最小 EEG 样本

```bash
mkdir -p data/raw/openneuro/ds004718/sub-HK001/eeg
curl -L -o data/raw/openneuro/ds004718/sub-HK001/eeg/sub-HK001_task-lppHK_eeg.set \
  https://s3.amazonaws.com/openneuro.org/ds004718/sub-HK001/eeg/sub-HK001_task-lppHK_eeg.set
```

读取 EEG：

```bash
python3 - <<'PY'
import mne

path = "data/raw/openneuro/ds004718/sub-HK001/eeg/sub-HK001_task-lppHK_eeg.set"
raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)

print(raw)
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_sec:", raw.n_times / raw.info["sfreq"])
PY
```

### 4.4 第一个 token 对齐测试

```bash
python3 - <<'PY'
import pandas as pd

words = pd.read_excel("outputs/probe_artifacts/ds004718/03_word_timing_xlsx.xlsx")
words["token_50hz"] = (words["onset_all"].astype(float) * 50).round().astype(int)
print(words[["word", "onset_all", "offset_all", "POS", "token_50hz"]].head(20))
PY
```

通过标准：

- word onset/offset 单调递增；
- `token_50hz` 没有大规模负数或异常跳变；
- EEG duration 足够覆盖 annotation 时间轴。

优先实验：

- EEG token -> word boundary；
- EEG token 与 f0/intensity/word frequency 对齐；
- EEG/audio/text 三塔检索。

## 5. Dataset 2: `ds005345` LPP Multi-talker

### 5.1 先测 annotation

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("outputs/probe_artifacts/ds005345")
words = pd.read_csv(root / "02_female_word_info.csv")
acoustic = pd.read_csv(root / "03_female_acoustic_sample.csv")

print("word rows:", len(words))
print("word cols:", list(words.columns))
print("acoustic cols:", list(acoustic.columns))
print(words.head(5))

assert {"word", "onset", "offset", "duration", "logfreq", "pos"}.issubset(words.columns)
assert {"time", "f0", "intensity"}.issubset(acoustic.columns)
PY
```

注意：`single_female_word_information.csv` 中 onset/offset 的单位需要先和 README / stimulus duration 对齐，不要直接假设为秒。

### 5.2 下载最小 EEG 样本

优先下载预处理 FIF，比 BrainVision 三件套更适合快速打开：

```bash
mkdir -p data/raw/openneuro/ds005345/sub-01/eeg
curl -L -o data/raw/openneuro/ds005345/sub-01/eeg/sub-01_task-multitalker_run-1_eeg_preprocessed.fif \
  https://s3.amazonaws.com/openneuro.org/ds005345/derivatives/sub-01/eeg/sub-01_task-multitalker_run-1_eeg_preprocessed.fif
```

```bash
python3 - <<'PY'
import mne

path = "data/raw/openneuro/ds005345/sub-01/eeg/sub-01_task-multitalker_run-1_eeg_preprocessed.fif"
raw = mne.io.read_raw_fif(path, preload=False, verbose=False)

print(raw)
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_sec:", raw.n_times / raw.info["sfreq"])
PY
```

通过标准：

- FIF 可打开；
- EEG 时长能覆盖单说话人 stimulus；
- word/acoustic annotation 能转成统一秒级时间轴。

优先实验：

- `single_female` envelope lag scan；
- `single_female` word boundary prediction；
- 对齐稳定后扩展到 `single_male` 和 `mix`。

## 6. Dataset 3: `ds004408` Continuous Naturalistic Speech

### 6.1 先测 TextGrid

```bash
python3 - <<'PY'
import re
from pathlib import Path

tg = Path("outputs/probe_artifacts/ds004408/03_audio01_TextGrid.TextGrid").read_text(errors="replace")
labels = re.findall(r'text\s*=\s*"([^"]*)"', tg)
nonempty = [x for x in labels if x.strip()]

print("labels:", len(labels))
print("nonempty:", len(nonempty))
print("first labels:", nonempty[:30])

assert len(nonempty) > 2000
PY
```

通过标准：

- TextGrid 非空 label 数约 2465；
- 能看到 phoneme-like label，例如 `HH`, `IY1`, `W`；
- TextGrid duration 与 audio header 约 177.56 秒一致。

### 6.2 下载最小 EEG 样本

```bash
mkdir -p data/raw/openneuro/ds004408/sub-001/eeg
curl -L -o data/raw/openneuro/ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.vhdr \
  https://s3.amazonaws.com/openneuro.org/ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.vhdr
curl -L -o data/raw/openneuro/ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.vmrk \
  https://s3.amazonaws.com/openneuro.org/ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.vmrk
curl -L -o data/raw/openneuro/ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.eeg \
  https://s3.amazonaws.com/openneuro.org/ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.eeg
```

```bash
python3 - <<'PY'
import mne

path = "data/raw/openneuro/ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.vhdr"
raw = mne.io.read_raw_brainvision(path, preload=False, verbose=False)

print(raw)
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_sec:", raw.n_times / raw.info["sfreq"])
PY
```

通过标准：

- MNE 可打开 BrainVision triplet；
- sidecar 显示 512 Hz；
- channels 表显示 128 EEG channel；
- TextGrid phoneme onset 可以映射到 EEG sample。

优先实验：

- phoneme onset raster；
- audio envelope lag scan；
- phoneme auxiliary prediction。

## 7. Dataset 4: `ds006104` Speech Decoding Probe

### 7.1 先测 events

```bash
python3 - <<'PY'
import pandas as pd

events = pd.read_csv("outputs/probe_artifacts/ds006104/03_events.tsv", sep="\t")
print(events.columns.tolist())
print(events.head(10))

for col in ["trial_type", "phoneme1", "phoneme2", "manner", "place", "tms_target", "voicing"]:
    print("\n", col)
    print(events[col].value_counts(dropna=False).head(10))

assert {"phoneme1", "phoneme2", "manner", "place", "voicing"}.issubset(events.columns)
PY
```

### 7.2 下载最小 EDF 样本

EDF 约 815MB，只在 events 测试通过后下载：

```bash
mkdir -p data/raw/openneuro/ds006104/sub-P01/ses-01/eeg
curl -L -o data/raw/openneuro/ds006104/sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_eeg.edf \
  https://s3.amazonaws.com/openneuro.org/ds006104/sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_eeg.edf
```

```bash
python3 - <<'PY'
import mne

path = "data/raw/openneuro/ds006104/sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_eeg.edf"
raw = mne.io.read_raw_edf(path, preload=False, verbose=False)

print(raw)
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_sec:", raw.n_times / raw.info["sfreq"])
PY
```

优先实验：

- frozen EEG token -> phoneme category；
- frozen EEG token -> manner/place/voicing；
- subject/session split，禁止随机 segment split 作为最终指标。

## 8. Dataset 5: `ds006434` ABR / Selective Attention

### 8.1 先测 event timing

```bash
python3 - <<'PY'
import pandas as pd

events = pd.read_csv("outputs/probe_artifacts/ds006434/02_events.tsv", sep="\t")
print(events.columns.tolist())
print(events.head())
print(events[["onset", "duration", "sample", "chapter_ind", "att_story", "att_side", "EEG_trigger"]].head())

assert {"att_story", "att_side", "EEG_trigger", "sample"}.issubset(events.columns)
assert events["duration"].astype(float).median() >= 60
PY
```

### 8.2 下载最小 EEG 样本

```bash
mkdir -p data/raw/openneuro/ds006434/sub-dichotic02/eeg
curl -L -o data/raw/openneuro/ds006434/sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.vhdr \
  https://s3.amazonaws.com/openneuro.org/ds006434/sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.vhdr
curl -L -o data/raw/openneuro/ds006434/sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.vmrk \
  https://s3.amazonaws.com/openneuro.org/ds006434/sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.vmrk
curl -L -o data/raw/openneuro/ds006434/sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.eeg \
  https://s3.amazonaws.com/openneuro.org/ds006434/sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.eeg
```

```bash
python3 - <<'PY'
import mne

path = "data/raw/openneuro/ds006434/sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.vhdr"
raw = mne.io.read_raw_brainvision(path, preload=False, verbose=False)

print(raw)
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_sec:", raw.n_times / raw.info["sfreq"])
PY
```

通过标准：

- sidecar 显示 500 Hz；
- event duration 约 64 秒；
- stimulus WAV header 显示约 64 秒；
- `sample / sfreq` 与 `onset` 误差可解释。

优先实验：

- 64 秒 epoch 切片；
- attended side / attended story decoding；
- timing lag sanity check。

## 9. Dataset 6: `ds007591` Speech Decoding Secondary

### 9.1 先测 session/task condition

```bash
python3 - <<'PY'
import pandas as pd

events = pd.read_csv("outputs/probe_artifacts/ds007591/03_events.tsv", sep="\t")
channels = pd.read_csv("outputs/probe_artifacts/ds007591/04_channels.tsv", sep="\t")

print(events.columns.tolist())
print(events.head(10))
print(events[["session_type", "task_condition", "trial_type", "duration"]].value_counts().head(20))
print("channels:", channels.shape)

assert {"session_type", "task_condition", "trial_type"}.issubset(events.columns)
PY
```

### 9.2 下载最小 EDF 样本

```bash
mkdir -p data/raw/openneuro/ds007591/sub-1/ses-20230511/eeg
curl -L -o data/raw/openneuro/ds007591/sub-1/ses-20230511/eeg/sub-1_ses-20230511_task-minimallyovert_acq-calibration_run-01_eeg.edf \
  https://s3.amazonaws.com/openneuro.org/ds007591/sub-1/ses-20230511/eeg/sub-1_ses-20230511_task-minimallyovert_acq-calibration_run-01_eeg.edf
```

```bash
python3 - <<'PY'
import mne

path = "data/raw/openneuro/ds007591/sub-1/ses-20230511/eeg/sub-1_ses-20230511_task-minimallyovert_acq-calibration_run-01_eeg.edf"
raw = mne.io.read_raw_edf(path, preload=False, verbose=False)

print(raw)
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_sec:", raw.n_times / raw.info["sfreq"])
PY
```

定位：作为 speech decoding 的 secondary sanity check，不作为主自然语音对齐数据集。

## 10. Dataset 7: `zenodo-4004271` KUL AAD

### 10.1 先测 README 和严格 split 约束

```bash
python3 - <<'PY'
from pathlib import Path

text = Path("outputs/probe_artifacts/zenodo-4004271/01_README.txt").read_text(errors="replace")
for keyword in ["leave-one-trial", "leave-one-story", "leave-one-subject", "eye-gaze", "20 trials"]:
    print(keyword, keyword in text)
PY
```

这个数据集不能用随机短 segment split 作为最终结果。至少要记录：

- leave-one-trial-out；
- leave-one-story-out；
- leave-one-subject-out；
- 是否存在 gaze/attended side shortcut。

### 10.2 列出 Zenodo 文件

```bash
python3 - <<'PY'
import requests

record = requests.get("https://zenodo.org/api/records/4004271", timeout=30).json()
for f in record["files"]:
    print(f["key"], f["size"], f["links"]["self"])
PY
```

### 10.3 下载一个 subject 和 stimuli

```bash
mkdir -p data/raw/zenodo/4004271
curl -L -o data/raw/zenodo/4004271/S1.mat \
  https://zenodo.org/api/records/4004271/files/S1.mat/content
curl -L -o data/raw/zenodo/4004271/stimuli.zip \
  https://zenodo.org/api/records/4004271/files/stimuli.zip/content
```

读取 `.mat`：

```bash
python3 - <<'PY'
from pathlib import Path
import scipy.io
import h5py

path = Path("data/raw/zenodo/4004271/S1.mat")

try:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    print("scipy keys:", sorted(k for k in mat.keys() if not k.startswith("__"))[:30])
except NotImplementedError:
    with h5py.File(path, "r") as f:
        print("h5 keys:", list(f.keys())[:30])
PY
```

通过标准：

- 能找到 EEG trial、sample rate、attended ear、stimuli 文件名；
- stimuli.zip 中能找到对应 wav；
- 每个 trial 的 train/test 分组先定义，再切 segment。

## 11. Dataset 8: `zenodo-1199011` DTU AAD

### 11.1 先测 preprocessing script

```bash
python3 - <<'PY'
from pathlib import Path

text = Path("outputs/probe_artifacts/zenodo-1199011/01_preproc_script.txt").read_text(errors="replace")
print(text[:2000])
for keyword in ["AUDIO", "EEG", "fs", "trial", "envelope"]:
    print(keyword, keyword.lower() in text.lower())
PY
```

### 11.2 列文件大小

```bash
python3 - <<'PY'
import requests

record = requests.get("https://zenodo.org/api/records/1199011", timeout=30).json()
for f in record["files"]:
    print(f["key"], f["size"])
PY
```

下载策略：

1. 不先下载 `EEG.zip`，约 16GB；
2. 先下载 `DATA_preproc.zip`，约 1.8GB；
3. 只在预处理包结构确认后，再决定是否需要原始 EEG。

```bash
mkdir -p data/raw/zenodo/1199011
curl -L -o data/raw/zenodo/1199011/DATA_preproc.zip \
  https://zenodo.org/api/records/1199011/files/DATA_preproc.zip/content
```

查看 zip 结构：

```bash
python3 - <<'PY'
import zipfile

path = "data/raw/zenodo/1199011/DATA_preproc.zip"
with zipfile.ZipFile(path) as zf:
    names = zf.namelist()
    print("entries:", len(names))
    print("\n".join(names[:80]))
PY
```

通过标准：

- 能识别 subject/trial 组织；
- 能恢复 attended/unattended speech envelope；
- 训练 split 以 trial/story/subject 为单位。

## 12. Dataset 9: `zenodo-7078451` ESAA

### 12.1 先测 readme 和 baseline 包

```bash
python3 - <<'PY'
from pathlib import Path
import zipfile

print(Path("outputs/probe_artifacts/zenodo-7078451/01_readme.txt").read_text(errors="replace"))

for path in [
    "outputs/probe_artifacts/zenodo-7078451/02_preprocess_zip.zip",
    "outputs/probe_artifacts/zenodo-7078451/03_baseline_zip.zip",
]:
    print("\n#", path)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            print(name)
PY
```

### 12.2 下载一个 subject

```bash
mkdir -p data/raw/zenodo/7078451
curl -L -o data/raw/zenodo/7078451/S1.zip \
  https://zenodo.org/api/records/7078451/files/S1.zip/content
```

查看结构：

```bash
python3 - <<'PY'
import zipfile

path = "data/raw/zenodo/7078451/S1.zip"
with zipfile.ZipFile(path) as zf:
    names = zf.namelist()
    print("entries:", len(names))
    print("\n".join(names[:120]))
PY
```

通过标准：

- subject 内有 32 个 trial 或等价 marker；
- 可恢复 target stream position、speaker gender、trial start/end；
- Mandarin tonal-language AAD 可作为 `ds005345` 之后的泛化测试。

## 13. Dataset 10: `zenodo-4518754` 255-channel AAD

### 13.1 先测小文件

当前已保存 `misc.zip` 和 `scripts.zip`，先解析 channel layout：

```bash
python3 - <<'PY'
import zipfile
import pandas as pd
from io import BytesIO

misc = "outputs/probe_artifacts/zenodo-4518754/01_misc_zip.zip"
with zipfile.ZipFile(misc) as zf:
    print(zf.namelist())
    loc_name = [n for n in zf.namelist() if n.endswith("eeg255ch_locs.csv")][0]
    locs = pd.read_csv(BytesIO(zf.read(loc_name)))
    print(locs.head())
    print(locs.shape)

scripts = "outputs/probe_artifacts/zenodo-4518754/02_scripts_zip.zip"
with zipfile.ZipFile(scripts) as zf:
    print(zf.namelist())
PY
```

### 13.2 下载 stimulus，暂缓 subject tar

```bash
mkdir -p data/raw/zenodo/4518754
curl -L -o data/raw/zenodo/4518754/stimuli.zip \
  https://zenodo.org/api/records/4518754/files/stimuli.zip/content
```

subject tar 每个约 1.5-1.7GB。只有在空间 tokenizer 或 sensor ablation 确认需要后，再下载单个 subject：

```bash
curl -L -o data/raw/zenodo/4518754/S1.tar.gz \
  https://zenodo.org/api/records/4518754/files/S1.tar.gz/content
```

定位：高密度空间表征学习，不应作为第一周主数据集。

## 14. Dataset 11: OpenMIIR

### 14.1 先测 metadata 和 beat annotation

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("outputs/probe_artifacts/openmiir")
meta = pd.read_excel(root / "01_stimuli_metadata.xlsx")
beats = (root / "03_beat_annotations.txt").read_text().splitlines()

print(meta.head())
print("meta shape:", meta.shape)
print("beat lines:", beats[:20])

assert "cue bpm" in meta.columns
assert any(line.strip() and not line.startswith("#") for line in beats)
PY
```

### 14.2 beat-to-token 测试

```bash
python3 - <<'PY'
from pathlib import Path

lines = Path("outputs/probe_artifacts/openmiir/03_beat_annotations.txt").read_text().splitlines()
times = []
for line in lines:
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    fields = line.split()
    try:
        times.append(float(fields[0]))
    except ValueError:
        pass

tokens = [round(t * 50) for t in times]
print("beat count:", len(times))
print("first beat times:", times[:20])
print("first 50Hz tokens:", tokens[:20])
PY
```

定位：先做 beat/downbeat/tempo 的可控音乐对齐 probe，再考虑更大的自然音乐训练。

## 15. Dataset 12: `ds003774` MUSIN-G

### 15.1 先测 events 和 audio header

```bash
python3 - <<'PY'
import pandas as pd

events = pd.read_csv("outputs/probe_artifacts/ds003774/02_events.tsv", sep="\t")
channels = pd.read_csv("outputs/probe_artifacts/ds003774/03_channels.tsv", sep="\t")

print(events.columns.tolist())
print(events.head(20))
print("trial_type counts:")
print(events["trial_type"].value_counts(dropna=False).head(20))
print("channels:", channels.shape)

assert "trial_type" in events.columns
PY
```

### 15.2 下载最小 EEGLAB 样本

```bash
mkdir -p data/raw/openneuro/ds003774/sub-001/eeg
curl -L -o data/raw/openneuro/ds003774/sub-001/eeg/sub-001_task-ListeningandResponse_eeg.set \
  https://s3.amazonaws.com/openneuro.org/ds003774/sourcedata/sub-001/eeg/sub-001_task-ListeningandResponse_eeg.set
```

```bash
python3 - <<'PY'
import mne

path = "data/raw/openneuro/ds003774/sub-001/eeg/sub-001_task-ListeningandResponse_eeg.set"
raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)

print(raw)
print("sfreq:", raw.info["sfreq"])
print("nchan:", raw.info["nchan"])
print("duration_sec:", raw.n_times / raw.info["sfreq"])
PY
```

定位：

- 自然音乐 EEG masked modeling；
- genre / familiarity / pleasantness 等弱监督需要全量下载后再确认；
- 不作为 speech 主结果，但可检验 tokenizer 是否跨 speech/music 泛化。

## 16. Dataset 13: `zenodo-4537751` MAD-EEG

### 16.1 先测 behavior 和 YAML

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

behavior = pd.read_excel("outputs/probe_artifacts/zenodo-4537751/01_behavioral_data.xlsx")
raw_yaml = Path("outputs/probe_artifacts/zenodo-4537751/02_raw_yaml.txt").read_text(errors="replace")
seq_yaml = Path("outputs/probe_artifacts/zenodo-4537751/03_sequences_yaml.txt").read_text(errors="replace")

print("behavior shape:", behavior.shape)
print(behavior.head())
print("raw yaml head:")
print(raw_yaml[:1500])
print("sequence yaml head:")
print(seq_yaml[:1500])
PY
```

### 16.2 下载 raw HDF5 或 stimuli

先下载 raw HDF5，不直接下 3.7GB preprocessed：

```bash
mkdir -p data/raw/zenodo/4537751
curl -L -o data/raw/zenodo/4537751/madeeg_raw.hdf5 \
  https://zenodo.org/api/records/4537751/files/madeeg_raw.hdf5/content
```

查看 HDF5 结构：

```bash
python3 - <<'PY'
import h5py

path = "data/raw/zenodo/4537751/madeeg_raw.hdf5"
with h5py.File(path, "r") as f:
    def show(name, obj):
        print(name, getattr(obj, "shape", ""), getattr(obj, "dtype", ""))
    f.visititems(show)
PY
```

通过标准：

- 可以恢复 subject、trial、target instrument；
- stimuli 和 sequence mapping 可追踪；
- 作为 music attention secondary，不抢 speech 主线优先级。

## 17. 进入训练前的统一检查

每个数据集至少要产生一个本地 manifest，建议字段：

```text
dataset_id
subject_id
session_id
run_id
eeg_path
eeg_format
sfreq
n_channels
stimulus_path
annotation_path
event_path
duration_eeg_sec
duration_stim_sec
time_unit
split_group
notes
```

训练前必须确认：

- EEG sample rate 已统一记录；
- annotation 的时间单位已确认；
- audio duration 与 EEG epoch duration 的偏差可解释；
- train/test split 在切 segment 之前定义；
- AAD 数据集使用 trial/story/subject 级 split；
- 没有把同一 trial 的短片段同时放进 train 和 test；
- 所有下载数据都在 `.gitignore` 覆盖目录下。

## 18. 第一周最小交付物

建议只提交轻量文件：

- dataset manifest CSV/JSON；
- probe report；
- preprocessing / loading scripts；
- 小图或小表；
- discussion notes。

不要提交：

- full EEG/audio；
- OpenNeuro/Zenodo mirror；
- feature arrays；
- model checkpoints；
- 大 PDF 或 archive。

建议第一周完成顺序：

1. `ds004718` 跑通 word onset -> 50Hz EEG token；
2. `ds005345` 跑通 single_female EEG/audio/word 三方对齐；
3. `ds004408` 跑通 phoneme TextGrid -> EEG sample；
4. `ds006104` 跑通 phoneme/manner/place 小分类 probe；
5. `zenodo-4004271` 只做严格 split baseline 复现准备；
6. OpenMIIR 做 beat-to-token 音乐对齐 sanity check。
