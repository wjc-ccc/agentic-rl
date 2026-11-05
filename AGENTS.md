# AGENTS.md — sft-rl-flow

本文件是所有编码 agent 的项目约束。只修改用户要求的内容，先检查现状，不猜测数据、路径或实验结果。

## 职责与目录

当前只做 SFT：本地负责数据准备、实验循环、指标监控、case 和下载。
`execution.backend: cloud` 使用 PyTRIO API 执行训练与推理；`local` 使用本机或 GPU 服务器上的 TRL SFTTrainer。
RL 阶段仅备用。两阶段的参数、配置加载、数据类型、模型 client 和实验记录实现完全分开，不建立共享层。

| 内容 | 唯一位置 |
|---|---|
| 完整、独立的 RL / SFT 实验配置 | `configs/rl/`、`configs/sft/` |
| SFT 数据、方法加载、LoRA 训练、固定 case 评测 | `src/sft/`；方法目录 `lora/`、`qlora/` |
| RL 独立的备用配置解析、数据类型和方法说明 | `src/rl/`；`ppo/`、`grpo/` 尚无实现 |
| SFT 配置、登记、实验记录、PyTRIO client 和归档 | `src/sft/` 内自己的模块 |
| SFT 路径和环境变量加载 | `src/sft/paths.py`，只读取 `SFT_` 目录参数 |
| 当前 SFT 的训练、采样、监控、核验、云端管理 CLI | `scripts/`，不路由 RL |
| 数据获取与处理代码、实际数据和登记 | `data/`，必须先读 [data/AGENTS.md](data/AGENTS.md) |
| 本地基模 / tokenizer、完整 checkpoint 或云端下载的 adapter | `models/base/`、`models/tokenizers/`、`models/checkpoints/<run_id>/checkpoint-<step>/` |
| 单次运行的配置、日志、指标、case、云端 checkpoint 引用 | `outputs/<run_id>/` |
| 探索代码 / 研究资料 | `notebooks/`、`search/research/` |
| 实际实验的结果分析 | `docs/experiments/`，按下述规则创建 |

不要在根目录散放脚本，不把数据 pipeline 放进 `scripts/`，不跨目录导入 CLI。
`sft`、`rl` 是两个完全独立的顶层包；禁止互相导入、禁止建立 rlflow/common 等共享业务层。
当前需要的实现都放在 sft 内；RL 不复制未使用的模型 client 或实验框架来充当已完成能力。
不要把业务代码放进 `src/sft_rl_flow.egg-info/`，它由 uv/setuptools 自动生成并忽略。
SFT 路径从 `sft.paths` 获取；RL 的备用位置只由 `rl.paths` 定义。禁止跨阶段读取路径和参数。
禁止写死本机绝对路径或依赖 CWD。
未经要求不搬移、重命名、覆盖或删除已有数据、权重、outputs。

## 环境与配置

- 只用 uv 和项目 `.venv`；Python 版本遵循 `.python-version`，不改版本，不使用 pip / conda / 系统安装。
- 同步依赖：`uv sync`；云端客户端：`uv sync --extra cloud`；新增依赖：`uv add`。
  `pyproject.toml` 是唯一依赖声明，`uv.lock` 随依赖变更提交。
- 脚本用 `uv run python <入口>`，测试用 `uv run pytest`。源码与配置统一 UTF-8。
- 每个 YAML 写全参数，RL 与 SFT 不继承、不合并、不使用 `defaults` 或公共 base。
  两阶段各自读取配置并检查阶段标记。SFT 在应用 --set 后只检查一次必需参数是否存在，
  缺项报告参数路径；不维护字段白名单，不做类型/范围检查，不拦截额外字段，不补默认值。
  参数使用 Trainer 风格名称；不保留 epochs/lr/batch_size/checkpoint_every 别名。
  LoRA alpha/dropout 仅 local 接入；cloud YAML 不配置 SDK 不支持的 4bit、alpha/dropout。
  后端不使用的硬件、下载等字段在启动时列明并写入快照。
- RL 配置命名 `<算法>_<环境>.yaml`；SFT 示例 `qwen3.5-4b-lora-cloud-gsm1k-r8-lr1e-4-bs3.yaml`。
  实际运行与产物名称使用覆盖后的真实参数，不凭手写标签推断参数。
- 配置包含 `seed/device/stage/algo/env或task/train/logging`；随机种子显式设置。
  每个 run 保存有效完整配置和所用数据 hash，不能只保存配置文件名。
- 密钥只放 `.env` 或登录状态，不写入 YAML、日志、快照或 Git。
- 日志只用 SwanLab，模式由 YAML 显式指定；当前 SFT 的覆盖只读 `SFT_SWANLAB_MODE/PROJECT`，
  YAML 保留用户选择的 cloud，纯本地可设 offline。RL 不读取这些 SFT 参数。
  指标每步写入本地 JSONL；不引入 wandb / tensorboard / mlflow。
- 实验启动顺序固定为读取配置 → --set → 一次参数检查 → 准备数据 → 运行目录与快照 →
  SwanLab init(config=实际配置) → 模型连接与训练。每步记录 loss，结束/失败时 finish。
  dry-run 不初始化 SwanLab、不连接模型。训练方法不再自行初始化日志。
- 模型计算默认在云端。本地显卡 6 GB 不作为云端模型限制；本地模型实验需用户明确要求。
  本次已按用户要求提供 Qwen2.5-0.5B-Instruct 的本地 LoRA 配置；local 不初始化 PyTRIO。
  不引入 DDP、多机、DeepSpeed 或大型训练框架。
- PyTRIO 任务先读 [.agents/skills/pytrio-skill/SKILL.md](.agents/skills/pytrio-skill/SKILL.md) 和对应 reference。
  Skill 保持项目级，版本 0.2.2，上游 commit `cea3edbe1d6a8fbdb61c2e2ab3975b7806e1fa23`；
  使用锁定 SDK 的实际签名，示例中的 pip 命令改用 uv，不执行带顶层云端调用的示例做自检。
- sampler 权重与完整训练状态用途不同，本地分别保存引用；下载的是 LoRA adapter，不是完整基模。

## 方法与模型 Interface

- SFT 配置显式指定 `stage: sft`、`algo: sft`、`method: lora`；
  `sft.load_method(method, backend)` 返回 `train(cfg, rows, experiment, client)`。
  `sft.prepare(cfg)` 只读取、校验数据并记录训练文件 hash；参数检查由入口负责。
- RL 的 `stage: rl`、`algo: ppo/grpo` 只用于备用目录，数据接口在 `rl.data`。
  LoRA SFT 已实现；QLoRA 当前无 PyTRIO 量化入口；PPO/GRPO 训练尚未实现。
  未实现方法应在创建 client/run 前明确拒绝，不静默退化到其他算法。
  当前没有 RL 执行入口；即使 --dry-run，SFT train.py 也不得读取 RL 配置。
- cloud SFT 模型入口：`sft.model_client.PyTrioClient(cfg, service=...)`。
  训练方法使用 tokenizer、start_training、forward、optim_step、sample、save_sampler/save_state；
  PyTRIO 类型、numpy 转换与 future 等待封装在 client。测试可注入 SDK 替身。
- client 不重复校验配置、batch 或调用顺序，不预查模型列表，不包装 ImportError；错误直接抛出。
  数据格式只在读取处检查，token 右移、mask、前缀和长度约束只在编码处检查。
  仍保留真实文件不覆盖、hash 校验与 checkpoint 归属匹配这些业务约束。
- SFT 共享完整 token 与监督 mask，只监督最后 assistant 和结束标记；cloud 右移一次，local labels 不预先右移。
  两端训练 loss 按完整梯度累积 batch 的有效监督 token 归一，评测按监督 token 加权。
  SFT `TrainingDatum` 只包含等长 input_tokens/target_tokens/weights，forward 固定 cross_entropy，
  返回逐 target 的 logprob。不为备用 RL 增加 importance_sampling、advantage 等参数。
  `Generation` 包含 text/tokens/logprobs；sampler/state 保存方法分别返回云端 URI。
  具体数据行、token 右移、client 输入输出与保存路径示例放对应模块 docstring，保持与实现一致。

## 产物命名

- SFT 参数名自动生成：`sft-<method>-<backend>-<模型>-<数据集>-lr<lr>-ep<epochs>-r<rank>-bs<batch>-ga<累积>-len<长度>-s<seed>-cfg<8位hash>`。
  使用 `--set` 后的实际值，短 hash 区分其余参数与训练数据 hash；排除日志选项和本机绝对路径。
  不手写 `run_name`，由运行入口自动写入完整配置快照。
- outputs：`outputs/<参数名>-<MMDD-HHMMSS>/`，同秒重名时加微秒，不覆盖旧目录。
- 云端 checkpoint：`<run_id>-step-000003-sampler` / `...-state`；step 是已完成的优化器更新次数。
  本地 `checkpoints.json` 分别记录 step、名称、sampler/state 引用，不把它们当成本地文件。
- checkpoint 放 `models/checkpoints/<run_id>/checkpoint-<step>/`；cloud 的 `adapter.tar` 和 `download.json` 记录实际模型、
  run/step、云端 checkpoint ID、sampler URI、sha256 与大小。
  命令：`uv run python scripts/cloud.py download --run <run目录> --step 3`。
  按该云端训练 session 的 sampler URI 匹配真实 ID；不猜 ID，不把完整训练状态当成 adapter。
- `train` 统一配置 eval/save/logging、resume 与自动下载；step 指完成的优化器更新。
  正常结束及 stop_after_steps 在更新边界保存最后状态；中断后从已成功保存的 checkpoint 恢复。
  `train.resume_from_checkpoint` 指向本地 checkpoint 目录，恢复新建 run，不覆盖来源。
  local 使用 Trainer 的 adapter/optimizer/scheduler/RNG/数据跳过；cloud 使用完整 state URI 与优化器恢复，
  同时恢复本地 scheduler、epoch、数据游标和 RNG。恢复核对 backend、模型/训练参数、数据 hash 和 tokenizer。
  sampler URI 不能完整续训；下载失败保留已完成状态并停止，不重放更新、不自动重试不确定的 optim_step。
  每个 cloud checkpoint 记录原所属 training session，下载按该 session 匹配真实 ID。
  实际运行产物及旧名称必须保留；训练/采样 token 数只记录请求用量，不推算收费。

## 修改与验证

1. 修改前检查 Git 状态、相关代码和配置；数据任务还要读数据约束、manifest 并核查实际目录。
2. 先说明假设、改动范围和验证标准。只实现本次需要的内容，不顺手重构或添加抽象。
3. 不覆盖其他未提交改动。只删除本次改动产生的孤儿代码，原有无关死代码不动。
4. 代码改完跑 `uv run pytest` 和相关入口；数据改动运行 `uv run python scripts/check_data.py`。
   报告真实输出，失败与未验证项如实说明。
5. 主分支保持可运行；在 `feat/` 或 `exp/` 分支工作，不 force push main，不改写已推送历史。
   提交信息使用 `type(scope): 摘要`。
6. 不提交 raw/processed 实际数据、权重、outputs、swanlog、.venv、.env 或数据缓存。
   数据代码与 manifest 可按现有白名单提交；提交前检查暂存路径与体积，不用强制添加绕过规则。

## 文档与实验记录

- 项目约束只维护本文件；数据约束只维护 `data/AGENTS.md`，不再创建重复的目录说明或规则文件。
- 不为安装、重构、单元测试、dry-run、数据核验生成实验报告；未运行实验不能填写结果或结论。
- 只有实际开始训练或评测后，才可复制 [实验模板](docs/experiments/TEMPLATE.md)，
  命名 `YYYY-MM-DD-<实验名>.md`，关联真实 run、配置、数据和指标，失败与中断也如实记录。
- 没有多 seed 支持的效果结论标为“初步观察”。研究笔记和 skill 自带文档属于资料，不复制成项目约束。
