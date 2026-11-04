# sft-rl-flow

从经典强化学习到 LLM 后训练的**渐进式学习仓库**：每个阶段都有一个能跑通的最小实验，
讲清楚"为什么这么做"，而不只是"调用了哪个库"。

> 详细的工作规范、目录约定与硬件限制见 [`AGENTS.md`](AGENTS.md)。
> 学习路径见 [`docs/learning-roadmap.md`](docs/learning-roadmap.md)。

## 当前状态

- [x] uv + venv 环境、git 版本管理、GitHub 远端
- [x] 项目级 PyTRIO skill；本地数据/实验/日志/case，云端模型计算的 SFT 路径
- [ ] 配置 PyTRIO 账号与支持模型，验证一次真实云端训练和推理
- [ ] 阶段 1：表格型 RL（Q-Learning / SARSA）跑通 CartPole 之外的离散环境
- [ ] 阶段 2：策略梯度 → PPO（连续控制）
- [ ] 阶段 3：SFT（小模型 + LoRA）
- [ ] 阶段 4：偏好对齐（DPO / GRPO）

## 快速开始

```powershell
uv sync                      # 创建 .venv 并安装依赖
uv run pytest                # 环境自检
uv run python scripts/train.py --config configs/rl/ppo_cartpole.yaml --dry-run
```

需要 LLM 相关依赖（transformers / trl / peft）时：

```powershell
uv sync --extra llm
```

经典 RL 仍为骨架，只有 `--dry-run`，不执行本地训练。当前新增的可执行训练路径是云端 SFT。

## 本地编排 + 云端计算

```mermaid
flowchart LR
  A[本地原始数据] --> B[校验 / tokenize / mask]
  B --> C[本地训练循环与参数控制]
  C -->|token batch| D[PyTRIO 云端模型前向 / 反向 / 优化器]
  D -->|logprob / checkpoint 引用| E[本地指标 / SwanLab offline]
  F[本地固定 case] -->|prompt token| G[PyTRIO 云端推理]
  D --> G
  G -->|生成结果| H[本地 case 查看]
  D -->|主动下载 LoRA adapter| I[本地 models/checkpoints]
```

无需将整个仓库或原始数据目录上传到 PyTRIO，SDK 会发送构造后的 token batch。
权重和训练状态保存在云端，本地保存引用；下载的是 PEFT LoRA adapter ZIP。
指标和 case 默认只存本地，SwanLab 上传仍需显式开启。

先运行不需要账号、模型或网络的本地准备：

```powershell
uv run python scripts/prepare_data.py --config configs/sft/pytrio_toy.yaml --toy
uv run python scripts/train.py --config configs/sft/pytrio_toy.yaml --dry-run
```

`--dry-run` 只校验数据格式和配置并写快照，不下载 tokenizer、不校验模型特有 mask、
不执行云端调用，也不生成训练指标。数据文件相同则跳过，不同则拒绝覆盖。

账号准备好后，在本地登录并选择服务当前支持的模型：

```powershell
uv sync --extra cloud
uv run trio login
uv run python scripts/cloud.py models
uv run python scripts/train.py --config configs/sft/pytrio_toy.yaml --set "model.base_model=<查询返回的模型名>"
```

将 `<run目录>` 替换为训练入口实际打印的绝对目录；监控命令可在第二个终端运行：

```powershell
uv run python scripts/inspect_run.py --run <run目录> --kind metrics --watch
uv run python scripts/sample.py --run <run目录> --base
uv run python scripts/sample.py --run <run目录>
uv run python scripts/inspect_run.py --run <run目录> --kind cases
uv run python scripts/cloud.py checkpoints
uv run python scripts/cloud.py download --checkpoint-id <权重ID>
```

`Ctrl+C` 停止本地训练循环；已完成的云端更新和已保存的 checkpoint 保留，当前请求可能仍在云端完成。
本地程序结束后不会自动继续后续 batch。完整职责、数据格式和产物说明见
[`docs/local-cloud-architecture.md`](docs/local-cloud-architecture.md)。

项目 skill 位于 [`.agents/skills/pytrio-skill/SKILL.md`](.agents/skills/pytrio-skill/SKILL.md)，
固定上游 0.2.2，安装来源与项目规则见 [SOURCE.md](.agents/skills/pytrio-skill/SOURCE.md)。
Codex 下一轮可发现该项目级 skill，不涉及 `-g` 全局安装。

## 目录

| 路径 | 用途 | 进 git |
|---|---|---|
| `src/rlflow/` | 可复用算法与环境代码 | ✅ |
| `.agents/skills/` | 项目级 PyTRIO skill 与上游许可证 | ✅ |
| `configs/` | 实验配置（YAML） | ✅ |
| `scripts/` | 下载数据、跑实验等入口 | ✅ |
| `notebooks/` | 探索性分析 | ✅ |
| `docs/` | 笔记、路线图、实验记录 | ✅ |
| `data/` | 原始/处理后数据 | ❌ |
| `models/` | 权重与 checkpoint | ❌ |
| `outputs/` | 每次 run 的日志与产物 | ❌ |

## 环境

- Python 3.12（由 uv 管理）
- 现有 PyTorch + CUDA 12.8 学习依赖保留；云端 SFT/推理不在本机加载模型
- PyTRIO SDK 通过 `cloud` extra 安装并由 `uv.lock` 锁定
- 实验跟踪：SwanLab（默认离线）
