# 2026-11-04：本地实验控制与 PyTRIO 云端计算

## 假设与范围

用户要求数据准备、参数控制、指标监控、case 查看和权重下载都在本地，
模型训练与推理通过 PyTRIO 云端执行。账号与模型尚未确定，用户选择先完成架构和离线验证。

本次实现 SFT 最小闭环，不实现 DPO/GRPO，不启动实际云端任务，也不引入外部数据集。
项目级 skill 固定为上游 0.2.2 / `cea3edbe1d6a8fbdb61c2e2ab3975b7806e1fa23`，
入口文件 Git blob SHA 与上游一致：`22a80485d4a075a6dfe8d05173a1de5c6e5b0d27`。

## 配置与运行环境

- 配置：`configs/sft/pytrio_toy.yaml` 继承 `configs/base.yaml`。
- `seed=42`，`epochs=1`，`batch_size=1`，`lr=1e-4`，`lora_rank=8`，`max_length=512`。
- `model.base_model=null`：必须在账号就绪后查询支持列表再选定模型。
- Python 3.12.13；PyTRIO 0.2.9；Transformers 5.22.1；SwanLab 0.10.3；依赖由 uv.lock 锁定。
- toy 数据由本地脚本自造：3 条训练样本、1 条独立 case，不下载第三方数据。
- SwanLab 保持 offline，实验产物未提交 Git。

## 验证与真实输出

```text
uv run pytest
14 passed in 12.87s

uv run ruff check src/rlflow scripts tests
All checks passed!

uv run python scripts/train.py --config configs/sft/pytrio_toy.yaml --dry-run
本地校验通过：3 条 SFT 样本；未连接云端，未计算训练 loss。
```

数据准备重复执行得到相同内容与 hash；不同内容拒绝覆盖。
测试覆盖：assistant-only mask 与右移、结束 token 保留、超长和模板不匹配拒绝、
非法对话、不同 CWD、目录环境变量为空、配置覆盖拼写、无模型时拒绝选择、
SwanLab 真实离线落盘、adapter 下载和 sha256 幂等性、禁止覆盖已有产物。

SDK 流程测试使用实际 `Datum` / `ModelInput` / `AdamParams` / `SamplingParams` 类型，
用实际 Client 的方法签名约束替身；远程 future、logprob、checkpoint 和采样结果由替身提供。
这些只验证调用契约，不代表真实服务的训练、采样和下载已经验证。

核对 SDK 源码时发现官网所列 `ServiceClient.run_id` 在 0.2.9 中不存在，
实现使用 `service_session_cuid`，另保存 `training_run_cuid`。

安装时 PyPI 元数据直连超时，大文件走系统代理又反复重试；
最终只在安装进程中为 PyPI 元数据使用系统代理，并让 `files.pythonhosted.org` 直连，
`uv sync --extra cloud --locked` 成功。未更改系统代理或仓库索引设置。

## 结论与后续验证

工程验证通过：本地准备、配置快照、日志、case 与下载入口已就位，模型计算只经过 PyTRIO。
当前没有训练效果结论。真实模型支持、云端训练 loss、base/adapter 生成对比和下载后的 adapter
需要用户配置账号与模型后验证；届时另存实验记录，不将本次替身结果作为模型效果。
