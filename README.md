# sft-rl-flow

从经典强化学习到 LLM 后训练的**渐进式学习仓库**：每个阶段都有一个能跑通的最小实验，
讲清楚"为什么这么做"，而不只是"调用了哪个库"。

> 详细的工作规范、目录约定与硬件限制见 [`AGENTS.md`](AGENTS.md)。
> 学习路径见 [`docs/learning-roadmap.md`](docs/learning-roadmap.md)。

## 当前状态

- [x] uv + venv 环境、git 版本管理、GitHub 远端
- [ ] 阶段 1：表格型 RL（Q-Learning / SARSA）跑通 CartPole 之外的离散环境
- [ ] 阶段 2：策略梯度 → PPO（连续控制）
- [ ] 阶段 3：SFT（小模型 + LoRA）
- [ ] 阶段 4：偏好对齐（DPO / GRPO）

## 快速开始

```powershell
uv sync                      # 创建 .venv 并安装依赖
uv run pytest                # 环境自检
uv run python scripts/train.py --config configs/rl/ppo_cartpole.yaml
```

需要 LLM 相关依赖（transformers / trl / peft）时：

```powershell
uv sync --extra llm
```

## 目录

| 路径 | 用途 | 进 git |
|---|---|---|
| `src/rlflow/` | 可复用算法与环境代码 | ✅ |
| `configs/` | 实验配置（YAML） | ✅ |
| `scripts/` | 下载数据、跑实验等入口 | ✅ |
| `notebooks/` | 探索性分析 | ✅ |
| `docs/` | 笔记、路线图、实验记录 | ✅ |
| `data/` | 原始/处理后数据 | ❌ |
| `models/` | 权重与 checkpoint | ❌ |
| `outputs/` | 每次 run 的日志与产物 | ❌ |

## 环境

- Python 3.12（由 uv 管理）
- PyTorch + CUDA 12.8（RTX 3060 Laptop 6 GB）
- 实验跟踪：SwanLab（默认离线）
