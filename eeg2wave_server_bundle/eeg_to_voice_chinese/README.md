# eeg_to_voice_chinese

用于 ds005170 的 EEG-to-voice 实验工作区。目录结构参考
`feis_karaone_ds004306_bundle`，但数据、缓存、模型和报告均独立保存。

## 数据下载

数据集版本固定为 OpenNeuro `ds005170` `1.0.1`。请在本目录外或本目录内执行：

```bash
cd /Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/eeg2wave_server_bundle/eeg_to_voice_chinese

DATA_ROOT="$PWD/data/ds005170"
datalad clone https://github.com/OpenNeuroDatasets/ds005170.git "$DATA_ROOT"
git -C "$DATA_ROOT" checkout 1.0.1
datalad -C "$DATA_ROOT" get -r .

# 可选：确认版本和数据状态
git -C "$DATA_ROOT" describe --tags --exact-match
datalad -C "$DATA_ROOT" status
```

`datalad clone` 会取得 Git/ git-annex 元数据，`datalad get -r .` 才会把实际数据文件下载到本地。
原始数据保持在 `data/ds005170`；后续预处理结果放入 `data/derivatives`，不要覆盖原始文件。

## 目录

- `app/`：配置、训练/评估脚本、源代码和测试
- `data/`：原始数据及衍生数据
- `artifacts/`：缓存、模型和实验产物
- `eeg_output/`：音频、manifest、QC 和 subject 输出
- `reports/`：图表和技术报告
- `scripts/`：数据准备及批处理入口
