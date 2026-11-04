# 外部数据下载入口

外部数据下载代码统一放在仓库的 `data/download_data/`。
当前目录没有已有下载脚本，尚未指定要下载的数据集，因此本次只建立目录与命名约定。

文件名：`download_<数据集名>_<数据量>.py`。

- 数据集名使用小写 ASCII，名称中的分隔符使用下划线。
- 数据量指样本条数，使用 `1000`、`10k` 等数量；完整集合使用 `all`。
- 例：`download_gsm8k_1000.py`、`download_hh_rlhf_10k.py`、`download_alpaca_all.py`。
  这些仅展示命名，不表示脚本或数据已下载，也不表示已核实这些数据集的许可证。

执行方式（脚本创建后）：

```powershell
uv run python data/download_data/download_gsm8k_1000.py
```

下载脚本从 `rlflow.paths` 引入 `RAW_DIR`，原始数据保存到 raw，不写入下载代码目录；
处理结果写入 `PROCESSED_DIR`。已存在且 sha256 一致则跳过，不覆盖已有原始数据。
每次外部数据下载须在 `data/manifest.json` 记录来源、许可证、sha256、大小、路径、时间和实际样本数。
下载前先确认具体数据集、版本、样本范围和许可证。

`data/download_data/` 遵循现有 `data/` Git 忽略规则，本次不修改 `.gitignore`，不强制提交该目录。
本文件保留在版本管理中，供后续添加下载脚本时遵循。
