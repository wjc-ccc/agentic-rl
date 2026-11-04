# data/

本目录**整体不进 git**（见 `.gitignore`），只有本说明和 `manifest.json` 会提交。

```
data/
├── raw/          # 原始数据，下载后只读，不再修改
├── processed/    # 由 raw 派生的数据，必须能由脚本重建
└── manifest.json # 已下载数据的登记表（来源、sha256、许可证）
```

约定：

- 下载一律走 `scripts/download_data.py`，幂等、校验 sha256。
- 每份数据都要在 `manifest.json` 登记，字段见 `AGENTS.md` §4。
- 处理脚本请放在 `scripts/`，派生逻辑可复用的放 `src/rlflow/data/`。
- 不确定许可证的数据，先问，不要直接下载使用。
