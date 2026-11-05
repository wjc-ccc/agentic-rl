# 2026-11-05 SFT 本地与云端流程验证

验证数据准备、LoRA 训练、当前权重评测、完整保存、adapter 下载与中断恢复。
本次为单 seed、小样本流程测试；不据此给出模型效果结论。

## 数据与配置

- GSM1K revision：`bc09569d09a614b9b530edc7f076fb214ac10493`，license unknown，仅有 test split。
  用户已允许用于少量非正式训练测试，因此本次结果不能作为该基准的正式成绩。
- 原始 Arrow SHA256：`bf7b848f2badbdcb138d6fbc1f246a1902337ce4c00cc079e4f35baf8ed10764`。
- `gsm1k/n24-s42-v3/`：前 40 条中按问题长度分层选 24 条，16 train / 4 eval / 4 cases；集合 ID 无交叉。
- SFT train SHA256：`01d545a919bcebd9448a26a01f032b88757450711f9b1dcc82bbcbec901cd835`。
- 本地配置：`configs/sft/qwen2.5-0.5b-instruct-sft-lora-local-gsm1k-lr1e-4.yaml`，4 条训练数据。
- 云端配置：`configs/sft/qwen3.5-4b-lora-cloud-gsm1k-r8-lr1e-4-bs3.yaml`，12 条训练数据。
- seed 42，LoRA rank 8，lr 1e-4；两端共用 SFT 加载器与编码器。每步评测 2 条、固定 case 2 条。
- 实现代码随本次提交保存；每个实际 run 的完整配置、数据 hash、指标和恢复关联保留于本地 outputs。

## 实际运行

命令均为 `uv run python scripts/train.py --config <上述配置>`。
中断运行加 `--set train.stop_after_steps=1`；恢复运行加
`--set train.resume_from_checkpoint=models/checkpoints/<来源run_id>/checkpoint-1`。

| 运行 | run_id 后缀 | 状态 | SwanLab |
|---|---|---|---|
| 云端 Qwen3.5-4B，保存 step 1 | `cfg84e3dc21-1105-175744` | 按配置停止 | [记录](https://swanlab.cn/@wjcccccc/sft-rl-flow/runs/c03yxemz) |
| 云端恢复至 step 2 | `cfg84e3dc21-1105-180032` | 完成 | [记录](https://swanlab.cn/@wjcccccc/sft-rl-flow/runs/l6dljxda) |
| 本地 Qwen2.5-0.5B，保存 step 1 | `cfgf2ce69f5-1105-175754` | 按配置停止 | [记录](https://swanlab.cn/@wjcccccc/sft-rl-flow/runs/eg3lcbsv) |
| 本地恢复至 step 3 | `cfgf2ce69f5-1105-180042` | 完成 | [记录](https://swanlab.cn/@wjcccccc/sft-rl-flow/runs/od8vh1jz) |
| 本地连续训练 3 步对照 | `cfgf2ce69f5-1105-180951` | 完成 | [记录](https://swanlab.cn/@wjcccccc/sft-rl-flow/runs/tzem3m3h) |

完整 run_id 及产物位置见对应 SwanLab 配置；本地原始记录为 `outputs/<run_id>/metrics.jsonl`、
`cases.jsonl`、`checkpoints.json`、`resume.json`，权重位于 `models/checkpoints/<run_id>/checkpoint-<step>/`。

## 结果与限制

| 后端 | 更新 step | 训练 token 平均 loss | 监督 token 加权 eval loss |
|---|---:|---:|---:|
| cloud | 1 | 1.182721 | 1.196274 |
| cloud，恢复后 | 2 | 1.039199 | 0.795230 |
| local | 1 | 2.294810 | 1.541794 |
| local，恢复后 | 2 | 1.484245 | 1.274775 |
| local，恢复后 | 3 | 1.217960 | 1.218660 |

本地连续运行与中断恢复的逐步训练/评测 loss 相同，最终 adapter 张量最大绝对差为 0。
本次 CUDA 峰值分配约 1.15 GiB，未启用 vLLM、flash-attn、DDP 或 DeepSpeed。
云端通过 `create_training_client_from_state_with_optimizer` 完整恢复，step 1/2 均保留 sampler、完整 state、
本地数据游标、scheduler 和 RNG；adapter 已通过 SDK 下载为 TAR，并核验内部文件及归档 SHA256。

初次云端运行因误将 TAR 当作 ZIP 校验而停止，已完成的 step 1 状态未丢失；修复后成功下载与续训。
早期本地尝试遇到 SwanLab 旧环境变量、模板返回类型和额外 full-logits forward 的显存问题，
已迁移环境变量、显式取 token 列表并使用原生 TRL token loss，以上表中的新版运行复验通过。

云端 step 2 固定 case 的实际输出为 `3280`、`1440`，完整 prompt/reference 见 cases.jsonl。
这些少量 case 只核验采样和记录链路；不同模型与数据条数之间的 loss 不用于横向效果比较。
完整续训保证保存的状态与进度，不承诺不同硬件或远端服务的逐位一致结果。
