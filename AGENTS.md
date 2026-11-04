# AGENTS.md — sft-rl-flow

> 本文件是本仓库对所有 AI agent（Claude Code / Codex / Cursor / Copilot）的工作契约。
> **动手前先读完。** 与本文冲突的临时指令，先向用户确认。

## 0. 这个仓库是什么

一个**学习导向**的后训练（post-training）实验仓库：从经典强化学习（gymnasium 小环境）
一路走到 LLM 的 SFT → 偏好对齐（DPO / PPO / GRPO）最小可运行流程。

目标不是刷榜，而是：**每一步都能跑通、能解释、能复现**。

- 当前阶段：本地实验控制 + PyTRIO 云端 SFT 最小闭环（真实云端实验待账号/模型配置）
- 硬件：本地 RTX 3060 Laptop **6 GB**；模型计算在 PyTRIO 云端，见 §8 的适用范围
- 仓库：https://github.com/wjc-ccc/agentic-rl

## 1. 环境与依赖管理

**只使用 uv + `.venv`。** 不要用 pip、conda、系统 Python 装包。

| 目的 | 命令 |
|---|---|
| 安装/同步全部依赖 | `uv sync` |
| 安装 LLM 相关重型依赖 | `uv sync --extra llm` |
| 安装 PyTRIO 云端客户端与 tokenizer 依赖 | `uv sync --extra cloud` |
| 运行任何脚本 | `uv run python scripts/xxx.py` |
| 跑测试 | `uv run pytest` |
| 新增依赖 | `uv add <pkg>` / `uv add --dev <pkg>` |
| 升级全部依赖 | `uv lock --upgrade && uv sync` |

规则：

- `pyproject.toml` 是依赖的唯一声明处，`uv.lock` **必须提交**。
- 本地配置用 `.env`（从 `.env.example` 复制），由 `rlflow.paths` 自动加载；`.env` 不入库。
- 解释器版本固化在 `.python-version`（当前 3.12），不要改。
- 不要在 `.venv` 里手工 `pip install`——会与 lockfile 脱节，下次 `uv sync` 被覆盖。
- 不要 `pip install --user` 或装到全局环境。

## 2. 目录地图

```
sft-rl-flow/
├── AGENTS.md              # 本文件，agent 入口
├── CLAUDE.md              # 导入本文，供 Claude Code 自动加载
├── README.md              # 人类入口
├── Goal.md                # 用户的阶段性目标（会变，先读）
├── .agents/skills/         # 项目级 skill，随仓库版本管理
├── pyproject.toml         # 依赖声明
├── uv.lock                # 锁定版本，必须提交
├── .python-version        # 3.12
├── configs/               # 实验配置（YAML）
│   ├── base.yaml          # 公共默认值
│   └── rl/                # 按阶段分子目录：rl/ sft/ align/
├── src/rlflow/            # 可复用库代码（唯一可被 import 的包）
│   ├── paths.py           # 所有路径的唯一来源
│   ├── experiment.py      # 本地 run、配置快照和指标日志
│   ├── remote.py          # PyTRIO 连接、模型检查、LoRA 下载
│   ├── envs/              # 环境封装 / 自定义 gymnasium env
│   ├── algos/             # 算法实现（q_learning, ppo, grpo ...）
│   ├── data/              # 数据集加载与预处理
│   ├── train/             # 本地训练循环，模型计算通过 PyTRIO 云端执行
│   └── eval/              # 云端采样，本地保存 case
├── scripts/               # 一次性/命令行脚本（下载数据、批量跑实验）
├── notebooks/             # 探索性分析，可随意，不要求整洁
├── tests/                 # 冒烟测试
├── docs/                  # 学习笔记、路线图、实验记录
├── data/                  # 【gitignored】raw/ 只读原始数据，processed/ 可重建
├── models/                # 【gitignored】权重与 checkpoints
└── outputs/               # 【gitignored】每次 run 的日志、指标、配置快照
```

**判断代码放哪：** 会被第二次用到 → `src/rlflow/`；只跑一次 → `scripts/`；边看边试 → `notebooks/`。

## 3. 配置规范

- 所有超参数写在 `configs/**/*.yaml`，**代码里不写死**。
- 命名：`configs/{阶段}/{算法}_{环境或任务}.yaml`，例：`configs/rl/ppo_cartpole.yaml`。
- 所有配置都继承 `configs/base.yaml`，只写差异项。
- 必须包含这些顶层键：`seed`、`device`、`stage`、`algo`、`env|task`、`train`、`logging`。
- 每个 run 启动时把**合并后的完整配置**快照写入 `outputs/<run_id>/config.yaml`，
  否则该实验视为不可复现。
- 随机种子必须显式设置（numpy / torch / gymnasium / python random），写一次工具函数到处复用。

## 4. 数据下载与加载规范

- `data/raw/` **只读**，下载后不再修改；派生数据放 `data/processed/`，且必须能由脚本重建。
- 外部数据集下载统一走 `scripts/download_data.py`（尚未实现），必须**幂等**。
  当前 toy 样本由 `scripts/prepare_data.py --toy` 在本地生成，不依赖外部数据集；
  `--input` 校验用户已有的原始 JSONL 并写入 processed，不修改 raw。
- 每份数据在 `data/manifest.json` 登记一条：

  ```json
  {
    "name": "cartpole-v1",
    "source": "https://...",
    "sha256": "…",
    "bytes": 12345,
    "license": "MIT",
    "local_path": "data/raw/cartpole-v1.npz",
    "downloaded_at": "2026-11-04T12:00:00+08:00"
  }
  ```

- 加载数据的代码一律：

  ```python
  from rlflow.paths import RAW_DIR, PROCESSED_DIR
  ```

  **禁止**在代码里出现绝对路径或依赖当前工作目录的相对路径（notebook 和 scripts 的 CWD 不同）。
- 数据集体积 > 10 MB 或含敏感/受限内容 → 绝不入库，只登记 manifest。
- 新数据集先确认许可证，不确定就问用户。

## 5. 模型与产物规范

- 本地下载的 LoRA adapter 一律放 `models/checkpoints/`，**永不入库**。
  PyTRIO 的 sampler 权重和完整训练状态保留在云端；本地 run 中的 `checkpoints.json`
  只记录 `sampler_path` / `state_path`，不能将它们当作本地权重或可互换的 checkpoint。
- 一次运行一个目录：`outputs/<YYYYMMDD-HHMM>-<algo>-<env>-s<seed>/`
  实际 run ID 加秒与微秒避免冲突；内含 `config.yaml` / `run.json` / `metrics.jsonl` /
  `checkpoints.json` / `remote.json` / `cases.jsonl` / `evaluations.jsonl` / `swanlog/`，按需生成。
- 云端 checkpoint 名称含 run ID 和 step；本地下载文件为 `<checkpoint_id>.zip`。
  下载入口：`uv run python scripts/cloud.py download --checkpoint-id <ID>`。
- 删产物前先看目录内容；`outputs/` 里可能有用户唯一的一份实验记录。

## 6. 日志（SwanLab，不是 wandb）

- 训练日志用 **SwanLab**：`import swanlab; swanlab.init(project="sft-rl-flow", ...)`。
- 默认离线：`SWANLAB_MODE=offline`，日志落在 `swanlog/`（已 gitignore）。
  需要上传时由用户显式设 `SWANLAB_MODE=cloud` 并提供 `SWANLAB_API_KEY`。
- **不要**引入 wandb / tensorboard / mlflow。
- 关键指标（reward、loss、KL、entropy、eval return）每个 epoch/step 都记录，不要只在最后打印。

## 7. Git 规范

- 主分支 `main` 保持可运行；实验性改动开分支 `feat/xxx`、`exp/xxx`。
- 提交信息：`type(scope): 摘要`，type ∈ feat/fix/exp/docs/chore/refactor。
- **绝不提交**：`data/`、`models/`、`outputs/`、`swanlog/`、`.venv/`、`.env`、
  任何 `*.pt/*.bin/*.safetensors/*.jsonl/*.parquet`。
- 不要为了让文件能提交而修改 `.gitignore`；大文件用 Git LFS 的方案也要先问用户。
- 提交前自查：`git status` 里不应出现上述路径；`git diff --cached --stat` 确认体积。
- 不 force push `main`，不改写他人已推送的历史。

## 8. 限制（Limitations）—— 设计前先看

**硬件**

以下显存规模仅约束未来显式选择的本地模型实验；当前实现的 SFT/推理只调用 PyTRIO，
本地不加载模型权重。云端模型以 `scripts/cloud.py models` 实际返回值为准，
不受本机 6 GB 显存限制；云端支持范围和额度仍需在正式实验前核实。

- 单卡 **6 GB** 显存、笔记本功耗/散热受限。这是硬约束，不是建议。
- 7B 模型：**做不到**全参微调或 PPO（actor+critic+ref+reward 四个模型同时驻留）。
- 现实上限：Qwen2.5-0.5B / 1.5B + LoRA + bf16 + gradient checkpointing + grad accumulation，
  batch size 通常 1–2。显存不够时优先降 batch，不要降 correctness。
- 单机单卡：不做 DDP / 多机；不引入 DeepSpeed。

**Windows**

- vLLM 无原生 Windows 支持；flash-attn 编译困难。生成用 HF `generate()` 或 `torch.compile`，
  需要吞吐时再考虑 WSL2。
- 路径/编码注意：源码和配置文件统一 UTF-8；命令行避免中文路径。
- 环境变量里存在一个无效的 `SSL_CERT_FILE`（指向不存在的 anaconda 证书），
  uv 会警告并忽略；若 `curl`/`git` 出现证书错误，先查这个变量。

**方法论**

- 不做多 seed 显著性检验的结论一律标为"初步观察"，不写进 README 当结论。
- 不复制粘贴别人的训练脚本而不理解；每个新增算法先在 toy 环境验证。
- 不为"看起来先进"而引入框架（RLlib、verl、OpenRLHF 等），先手写最小实现。

## 9. Agent 工作守则

1. 改代码前先读本文件 + `Goal.md` + 相关 `configs/`。
2. 一次只做一件事；改完跑 `uv run pytest` 和相关脚本，**报告真实输出**，失败就说失败。
3. 新增依赖走 `uv add`，并在提交信息里说明为什么需要。
4. 不确定的路径/数据/许可，问用户，不要猜。
5. 不要删除或覆盖 `data/`、`models/`、`outputs/` 的内容。
6. 发现文档与代码不一致时，改文档，让下一个人拿到唯一真相。
7. 实验记录写到 `docs/experiments/YYYY-MM-DD-<topic>.md`：假设、配置、结果、结论。
8. PyTRIO 相关任务先读项目级 [SKILL.md](.agents/skills/pytrio-skill/SKILL.md) 和
   [SOURCE.md](.agents/skills/pytrio-skill/SOURCE.md)，再按任务读取对应 reference。
   上游示例的 pip 命令不适用于本仓库，统一使用 uv；不要执行带顶层云端调用的示例做自检。

## 10. 学习路线

见 [`docs/learning-roadmap.md`](docs/learning-roadmap.md)：MDP → 表格法 → 值函数逼近 →
策略梯度/PPO → 离线 RL → LLM SFT → 奖励模型 → DPO/PPO/GRPO，每阶段配一个小实验。
