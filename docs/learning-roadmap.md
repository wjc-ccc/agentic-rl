# 强化学习 → LLM 后训练 学习路线

每个阶段都是：**先手写最小实现 → 在 toy 环境验证 → 再换成真实任务**。
不要跳过阶段 1–3 直接调 TRL，否则后面调不动也解释不了。

执行架构已切换为[本地控制 + 云端模型计算](local-cloud-architecture.md)。
当前实现的是 PyTRIO SFT 最小路径，经典 RL 算法仍待实现；下面保留概念学习顺序。
LLM 模型从 `scripts/cloud.py models` 的实际列表选择，下面的本地显存估计仅供未来本地实验参考，
不再用于限制云端模型规格。云端额度和模型可用性需要账号配置后核实。

## 阶段 0：基础（不写代码也行，但要有概念）

- MDP 五元组、回报与折扣、策略、价值函数、Bellman 方程
- on-policy vs off-policy、探索与利用、bias–variance
- 阅读：Sutton & Barto *Reinforcement Learning: An Introduction* 第 3–4 章

## 阶段 1：表格型方法（无 GPU，几秒跑完）

| 项目 | 内容 |
|---|---|
| 实验 | FrozenLake / CliffWalking 上实现 DP、Monte Carlo、Q-Learning、SARSA |
| 验收 | 画出学习曲线；能解释 Q-Learning 与 SARSA 的差别与原因 |
| 产出 | `src/rlflow/algos/tabular.py` + notebook 分析 |

## 阶段 2：值函数逼近

| 项目 | 内容 |
|---|---|
| 实验 | DQN on CartPole；加 replay buffer、target network，再分别去掉做消融 |
| 验收 | 3 个 seed 均值达到 475+；消融实验能说明每个组件的作用 |
| 注意 | 6 GB 显存完全够，MLP 即可 |

## 阶段 3：策略梯度 → PPO（主战场）

| 项目 | 内容 |
|---|---|
| 实验 | REINFORCE（带 baseline）→ A2C → PPO on CartPole / LunarLander / Pendulum |
| 重点 | advantage 估计（GAE）、ratio clipping、entropy bonus、value loss |
| 验收 | 自己实现的 PPO 与 SB3 在 CartPole 上曲线量级一致 |
| 产出 | `configs/rl/ppo_cartpole.yaml` 起的系列配置 + `docs/experiments/` 记录 |

## 阶段 4：离线 RL（可选，但对理解 LLM 对齐很有帮助）

- 概念：distribution shift、behavior cloning、CQL / IQL
- 实验：D4RL 的 Hopper/Walker 小型数据（注意：数据放 `data/raw/`，不入库）

## 阶段 5：LLM SFT

| 项目 | 内容 |
|---|---|
| 实验 | Qwen2.5-0.5B-Instruct + LoRA，在一个小指令数据集上 SFT |
| 关键 | chat template、loss mask（只对 completion 计 loss）、packing、grad accumulation |
| 显存 | 0.5B + LoRA + bf16 ≈ 2–4 GB，6 GB 可行；不要上 7B |
| 验收 | 训练 loss 下降、抽样输出风格符合数据；checkpoint 可复现 |

## 阶段 6：偏好对齐

按难度递增，每一步都能解释再往下：

1. **奖励模型**：同一 prompt 的 chosen/rejected 对，Bradley–Terry loss
2. **DPO**：离线、最省显存，0.5B 模型在 6 GB 上可行
3. **PPO / GRPO**：在线 RLHF。4 个模型同时驻留，6 GB 基本不可行；
   可行方案是 0.5B + LoRA + 4-bit + 极小 batch，或改用 GRPO（无需 critic）
   并接受很慢的速度。先跑通 100 步，再谈效果。

## 贯穿全程的习惯

- 每个实验记录：**假设 → 配置 → 曲线 → 结论**（`docs/experiments/`）
- 单一变量原则：一次只改一个超参
- 所有结论至少 2 个 seed
- 失败实验也记录，并写清楚失败原因
