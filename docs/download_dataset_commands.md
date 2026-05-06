# OpenNeuro 数据下载指令：只下框架 + 指定受试

目标：

```text
ds006104: 下载 DataLad 框架 + 少量受试完整 EEG 数据
ds005345: 下载 DataLad 框架 + stimuli/annotation + 少量受试 EEG 数据
```

不要直接全量下载：

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds006104 ...
aws s3 sync --no-sign-request s3://openneuro.org/ds005345 ...
```

这会拉完整数据；`ds005345` 全量约 160G 级别，不适合当前阶段。

## 0. 数据目录与 Git 忽略

数据放在：

```text
data/raw/openneuro/
```

`.gitignore` 已经忽略：

```gitignore
data/
data/raw/openneuro/
*.edf
*.eeg
*.vhdr
*.vmrk
*.set
*.fdt
*.fif
*.wav
*.nii
*.nii.gz
```

提交前检查：

```bash
git check-ignore -v data/raw/openneuro/ds006104_datalad
git check-ignore -v data/raw/openneuro/ds005345_datalad
git status --short
```

## 1. 推荐方式：DataLad

安装方式任选一种。

你刚才遇到的错误：

```text
zsh: command not found: datalad
```

说明本机还没有安装 DataLad。必须先安装，否则后面的 `datalad install/get` 都不会执行成功。

如果有 Homebrew：

```bash
brew install datalad git-annex
```

如果用 conda：

```bash
conda install -c conda-forge datalad git-annex
```

检查：

```bash
datalad --version
git annex version
```

如果这两条命令不能正常输出版本号，不要继续跑下载命令。

## 2. `ds006104`：框架 + 少量受试完整 EEG

用途：

```text
phoneme / CV / VC / happy-angry / F0 / timbre probe
```

### 2.1 安装 DataLad 框架

```bash
mkdir -p data/raw/openneuro

datalad install \
  -s https://github.com/OpenNeuroDatasets/ds006104.git \
  data/raw/openneuro/ds006104_datalad
```

这一步主要下载 Git/DataLad 框架和文件指针，不会把所有大文件内容拉下来。

### 2.2 只获取元数据

```bash
cd data/raw/openneuro/ds006104_datalad

datalad get \
  dataset_description.json \
  participants.tsv \
  participants.json
```

### 2.3 获取指定受试完整数据

建议先取一个 Study 1 subject 和一个 Study 2 subject：

```bash
datalad get sub-P01
datalad get sub-S01
```

如果需要再加两个：

```bash
datalad get sub-P02
datalad get sub-S02
```

说明：

- `sub-P01/sub-P02` 属于 Study 1。
- `sub-S01/sub-S02` 属于 Study 2。
- `datalad get sub-P01` 会获取该受试目录下的完整文件，而不是全数据集。

### 2.4 不要获取这些目录

当前不建议跑：

```bash
datalad get derivatives
datalad get derivatives/eeglab
```

`derivatives/eeglab/*.set` 是 `ds006104` 体积变大的主要来源之一。第一版 EEG-token probe 用 raw EDF + events 就够。

### 2.5 检查

```bash
find sub-P01 -maxdepth 4 -type f | head -50
find sub-S01 -maxdepth 4 -type f | head -50
find . -name "*events.tsv" | head
find . -name "*.edf" | head
du -sh .
```

回到项目根目录：

```bash
cd ../../../..
```

## 3. `ds005345`：框架 + stimuli/annotation + 少量受试 EEG

用途：

```text
single_female / single_male / mix speaker stream retrieval
```

注意：这里的“受试完整数据”建议理解为**该受试 EEG 相关完整数据**。不要先拉 `anat/func/fmap`，否则会把 MRI/fMRI 大文件带进来。

### 3.1 安装 DataLad 框架

```bash
mkdir -p data/raw/openneuro

datalad install \
  -s https://github.com/OpenNeuroDatasets/ds005345.git \
  data/raw/openneuro/ds005345_datalad
```

### 3.2 获取任务必需的 stimuli / annotation / metadata

```bash
cd data/raw/openneuro/ds005345_datalad

datalad get \
  dataset_description.json \
  participants.tsv \
  participants.json \
  stimuli \
  annotation \
  quiz
```

应该得到：

```text
stimuli/single_female.wav
stimuli/single_male.wav
stimuli/mix.wav
annotation/single_female_acoustic.csv
annotation/single_male_acoustic.csv
annotation/mix_acoustic.csv
annotation/single_female_word_information.csv
annotation/single_male_word_information.csv
```

### 3.3 获取少量受试 EEG raw + EEG derivatives

先取 `sub-01`：

```bash
datalad get sub-01/eeg
datalad get derivatives/sub-01/eeg
```

如果需要扩展到 3 个受试：

```bash
datalad get sub-02/eeg
datalad get sub-03/eeg
datalad get derivatives/sub-02/eeg
datalad get derivatives/sub-03/eeg
```

说明：

- `sub-*/eeg` 是 raw BrainVision EEG。
- `derivatives/sub-*/eeg` 是预处理 EEG FIF，适合先做 retrieval debug。
- 当前不要拉 `sub-*/anat`、`sub-*/func`、`sub-*/fmap`，也不要拉全量 `derivatives`。

### 3.4 如果确实要某个受试的全部模态

谨慎使用：

```bash
datalad get sub-01
datalad get derivatives/sub-01
```

这会包括该受试的 MRI/fMRI，体积会明显增加。当前 speaker stream retrieval 不需要。

### 3.5 检查

```bash
find stimuli -maxdepth 1 -type f -print
find annotation -maxdepth 1 -type f -print
find sub-01/eeg -maxdepth 1 -type f -print
find derivatives/sub-01/eeg -maxdepth 1 -type f -print
du -sh .
```

回到项目根目录：

```bash
cd ../../../..
```

## 4. 备选方式：AWS S3 只同步需要路径

如果不用 DataLad，可以用 AWS CLI 精准同步。先安装：

```bash
brew install awscli
```

### 4.1 `ds006104`：只同步指定受试

```bash
mkdir -p data/raw/openneuro/ds006104_s3

aws s3 sync --no-sign-request \
  s3://openneuro.org/ds006104 \
  data/raw/openneuro/ds006104_s3 \
  --exclude "*" \
  --include "dataset_description.json" \
  --include "participants.tsv" \
  --include "participants.json" \
  --include "sub-P01/*" \
  --include "sub-S01/*"
```

增加受试：

```bash
aws s3 sync --no-sign-request \
  s3://openneuro.org/ds006104 \
  data/raw/openneuro/ds006104_s3 \
  --exclude "*" \
  --include "sub-P02/*" \
  --include "sub-S02/*"
```

### 4.2 `ds005345`：只同步 stimuli/annotation/少量 EEG

```bash
mkdir -p data/raw/openneuro/ds005345_s3

aws s3 sync --no-sign-request \
  s3://openneuro.org/ds005345 \
  data/raw/openneuro/ds005345_s3 \
  --exclude "*" \
  --include "dataset_description.json" \
  --include "participants.tsv" \
  --include "participants.json" \
  --include "stimuli/*" \
  --include "annotation/*" \
  --include "quiz/*" \
  --include "sub-01/eeg/*" \
  --include "derivatives/sub-01/eeg/*"
```

扩展到 3 个受试：

```bash
aws s3 sync --no-sign-request \
  s3://openneuro.org/ds005345 \
  data/raw/openneuro/ds005345_s3 \
  --exclude "*" \
  --include "sub-02/eeg/*" \
  --include "sub-03/eeg/*" \
  --include "derivatives/sub-02/eeg/*" \
  --include "derivatives/sub-03/eeg/*"
```

## 5. 最推荐的最小执行版

先回到项目目录：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding
```

先安装 DataLad。你当前在 conda `base` 环境里，优先用：

```bash
conda install -c conda-forge datalad git-annex
```

然后检查：

```bash
datalad --version
git annex version
```

确认版本号正常后，再跑下载：

```bash
mkdir -p data/raw/openneuro

datalad install \
  -s https://github.com/OpenNeuroDatasets/ds006104.git \
  data/raw/openneuro/ds006104_datalad

cd data/raw/openneuro/ds006104_datalad
datalad get dataset_description.json participants.tsv participants.json
datalad get sub-P01
datalad get sub-S01
cd ../../../..

datalad install \
  -s https://github.com/OpenNeuroDatasets/ds005345.git \
  data/raw/openneuro/ds005345_datalad

cd data/raw/openneuro/ds005345_datalad
datalad get dataset_description.json participants.tsv participants.json stimuli annotation quiz
datalad get sub-01/eeg derivatives/sub-01/eeg
cd ../../../..

git check-ignore -v data/raw/openneuro/ds006104_datalad
git check-ignore -v data/raw/openneuro/ds005345_datalad
git status --short
```

## 6. 如果刚才已经跑到 `/`

你的终端现在如果显示：

```text
SamdeMacBook-Pro-5 / %
```

先回项目目录：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding
```

确认：

```bash
pwd
git status --short
```

只有确认当前目录是：

```text
/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding
```

再继续下载。
