# models/

本目录**不进 git**，只有本说明会提交。

```
models/
└── checkpoints/     # <checkpoint_id>.zip + <checkpoint_id>.json 校验回执
```

- 当前模型计算在 PyTRIO 云端；本地不会下载或加载基模。
- `uv run python scripts/cloud.py download --checkpoint-id <ID>` 下载 PEFT LoRA adapter ZIP，
  不是完整模型；ID 通过 `scripts/cloud.py checkpoints` 查询，不是 sampler path。
- 云端权重引用和完整训练状态引用保存在对应 run 的 `checkpoints.json`，各自用于采样和续训。
- 重复下载只有本地回执 sha256 匹配才跳过；中断或校验失败时拒绝覆盖已有文件。
- 大模型缓存建议设 `HF_HOME` 到别的盘，避免 C 盘爆掉。
- 需要共享权重时用 release / 对象存储 / Hugging Face Hub，不要提交到 git。
