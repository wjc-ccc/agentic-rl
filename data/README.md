# data/

本目录**整体不进 git**（见 `.gitignore`），只有本说明和 `manifest.json` 会提交。

```
data/
├── raw/          # 原始数据，下载后只读，不再修改
├── processed/    # 由 raw 派生的数据，必须能由脚本重建
└── manifest.json # 已下载数据的登记表（来源、sha256、许可证）
```

约定：

- 外部数据集下载走 `scripts/download_data.py`（尚未实现），幂等、校验 sha256，
  并在 `manifest.json` 登记来源和许可，字段见 `AGENTS.md` §4。
- 当前 toy 数据由 `scripts/prepare_data.py --config configs/sft/pytrio_toy.yaml --toy`
  自造并写入 processed，不下载外部数据；相同文件跳过，不同文件拒绝覆盖。
- 真实 JSONL 可用该脚本的 `--input <相对RAW_DIR的路径>` 校验后派生到 processed。
- 每次训练快照记录所用 processed 数据的 sha256；固定 case 集独立保存，不能用训练样本代替测试集。
- 处理脚本请放在 `scripts/`，派生逻辑可复用的放 `src/rlflow/data/`。
- 不确定许可证的数据，先问，不要直接下载使用。
