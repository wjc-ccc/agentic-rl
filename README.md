# sft-rl-flow

SFT 的 cloud/local 使用同一份 messages JSONL、编码规则、评测和实验命名。
YAML 的 `execution.backend` 选择 PyTRIO API 或本地 TRL SFTTrainer，训练、评测、保存和续训参数都在 `train`。
当前实现 LoRA SFT；RL 独立备用，PPO/GRPO 与 QLoRA 不提供训练入口。

- [项目约束](AGENTS.md)、[数据约束](data/AGENTS.md)
- [真实实验模板](docs/experiments/TEMPLATE.md)
- [本次实际流程验证](docs/experiments/2026-11-05-sft-gsm1k-smoke.md)

本机 6GB 显卡用 Qwen2.5-0.5B-Instruct。云端用 Qwen3.5-4B，仅上传 token 数据请求，不创建服务器、不下载基模权重。
SwanLab 与 JSONL 记录在本地控制；云端 adapter 自动下载到本地，密钥仅放 `.env` 或登录状态。

```powershell
uv sync --extra cloud --extra llm
uv run python data/download_data/download_gsm1k_40/run.py --config configs/data/gsm1k_40.yaml
uv run python data/process_data/process_gsm1k_pipeline/run.py --config configs/data/gsm1k_40.yaml
uv run python scripts/check_data.py
```

GSM1K 固定 revision：取前 40 条，按问题长度分层选 24 条，分为 16 train / 4 eval / 4 cases。
本地默认取训练集 4 条，云端取 12 条。上游只有 test split、license unknown，已获用户允许用于少量流程测试；不作为正式基准。
处理数量、seed、划分在数据 YAML；变更处理规则时指定新的 `selection.output_version`，不覆盖已有数据。

```powershell
# 本地：下载约 1GB 基模，再运行训练
uv run python scripts/download_model.py --config configs/sft/qwen2.5-0.5b-instruct-sft-lora-local-gsm1k-lr1e-4.yaml
uv run python scripts/train.py --config configs/sft/qwen2.5-0.5b-instruct-sft-lora-local-gsm1k-lr1e-4.yaml

# 云端：只下载 tokenizer，登录后训练
uv run python scripts/download_model.py --config configs/sft/qwen3.5-4b-lora-cloud-gsm1k-r8-lr1e-4-bs3.yaml --tokenizer-only
uv run trio login
uv run python scripts/train.py --config configs/sft/qwen3.5-4b-lora-cloud-gsm1k-r8-lr1e-4-bs3.yaml
```

下载源由 `model.download_source` 选择 modelscope/huggingface；HF 镜像可用进程环境变量 `HF_ENDPOINT`。
数据下载会依次尝试直连、系统代理和配置中的镜像，并保留真实来源与下载回执。
训练前可加 `--dry-run`；只校验配置、数据与快照，不初始化 SwanLab 或连接模型。

`--set train.learning_rate=1.25e-4` 等覆盖会进入快照及自动命名。
旧 `lr/epochs/batch_size/checkpoint_every` 不再使用；`max_steps > 0` 时覆盖 epoch 的训练预算。
`warmup_steps` 遵循当前锁定 Trainer：小于 1 的小数表示总步数比例，整数表示更新次数。
评测与保存支持 no/steps/epoch；logging_steps 控制日志频率，示例为每次更新记录。
云端不使用的硬件参数在启动时列出；alpha/dropout 仅本地配置。配置只检查必需字段，允许附加元数据。

```powershell
# stop_after_steps 在更新边界保存后停止，便于验证完整续训
uv run python scripts/train.py --config <配置.yaml> --set train.stop_after_steps=1
uv run python scripts/train.py --config <同一配置.yaml> --set train.resume_from_checkpoint=models/checkpoints/<来源run_id>/checkpoint-1

uv run python scripts/inspect_run.py --run outputs/<run_id> --kind metrics
uv run python scripts/inspect_run.py --run outputs/<run_id> --kind cases
uv run python scripts/sample.py --run outputs/<run_id>
uv run python scripts/cloud.py download --run outputs/<云端run_id> --step 2
```

outputs 保存完整配置、数据 hash、指标、case、SwanLab 链接和 checkpoint 索引。
checkpoint 保存于 `models/checkpoints/<run_id>/checkpoint-<step>/`；目录名含 SFT、backend、模型、数据集、lr、epoch、LoRA rank、batch、累积、长度、seed、hash 和月日时间。

本地保存原生 Trainer 的 adapter、优化器、scheduler、RNG 与进度；云端保存完整 state URI、sampler URI、scheduler、数据游标和 RNG，并通过 SDK 下载 TAR、核验内容 hash。
恢复总是新建运行目录，校验 backend、参数、数据和 tokenizer，保留来源产物；sampler URI 只用于推理。
下载失败会停止并保留可恢复 checkpoint，不重放已完成的更新。实际计费以服务方为准，代码只记请求 token 用量。

```powershell
uv run pytest
uv run ruff check src scripts tests data/download_data data/process_data
```
