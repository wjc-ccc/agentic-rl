# models/

本目录**不进 git**，只有本说明会提交。

```
models/
└── checkpoints/     # 训练产生的权重，按 run 分子目录
```

- 本地权重、LoRA adapter、HF 下载的基座模型都可放这里。
- 大模型缓存建议设 `HF_HOME` 到别的盘，避免 C 盘爆掉。
- 需要共享权重时用 release / 对象存储 / Hugging Face Hub，不要提交到 git。
