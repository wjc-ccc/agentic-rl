# 本地控制与 PyTRIO 云端计算

## 职责和需要发送的内容

| 工作 | 执行位置 | 本地保留 / 发往云端 |
|---|---|---|
| 数据整理、校验、固定 train/case 划分 | 本地 `data/`、`prepare_data.py` | 原始文件只读，不上传整个目录 |
| chat template、tokenize、右移和 loss mask | 本地 `data/sft.py` | 构造 `ModelInput`、target、weights 后发送 batch；token 仍承载数据内容 |
| epoch、shuffle、batch、学习率、保存频率 | 本地 `train/sft.py` + YAML | SDK 请求传入实际参数 |
| 模型 forward、backward、optimizer update | PyTRIO 云端 | 本地接收 logprob，计算监督 token 平均 loss |
| 推理采样 | PyTRIO 云端 | 发送 prompt token 与 SamplingParams，接收文本和 token |
| reward / 自定义目标（后续 DPO/GRPO） | 本地算法层 | custom loss 可能在本地计算 dL/dlogprob；模型反向仍在云端，当前未实现 |
| 指标、曲线、实验状态 | 本地 JSONL + SwanLab | `inspect_run.py --kind metrics --watch`；当前配置显式选择 cloud，可改 offline |
| case 对照与人工检查 | 本地 `cases.jsonl` | 同一 case 集分别采样 base / adapter，无自动准确率结论 |
| sampler 权重、完整 optimizer 状态 | 云端 | `checkpoints.json` 分别记录 sampler / train path |
| adapter 下载与归档 | 本地 `models/checkpoints/` | 根据真实 checkpoint ID 取得短期 URL，保存 ZIP、sha256 和大小 |
| 源码、配置、skill、说明文档 | GitHub | 不包含数据、权重、密钥和实验产物 |

本地仍需在线调用 PyTRIO；本地循环是控制进程，不能提交后关机期待剩余 batch 自动训练。
此次不增加代码上传、远端容器、SSH 调度或另一套实验服务。

## 文件与配置

复用库代码在 `src/rlflow/`；脚本仅处理 CLI 参数和调用：

- `experiment.py`：创建唯一 run、保存完整配置和依赖版本、逐步写指标。
- `remote.py`：连接 SDK、核对支持模型、下载 adapter；不加载模型。
- `data/sft.py`：读取 JSONL、数据 hash、最终 assistant-only mask 和自回归右移。
- `train/sft.py`：本地 SFT 循环，等待云端 future 完成再进行下一步。
- `eval/cases.py`：云端采样和本地 case 保存。

RL 和 SFT 配置独立写全参数；已移除公共 base 和 `defaults` 继承逻辑。
`configs/sft/pytrio_toy.yaml` 保留空模型，用于离线验证。
`configs/sft/qwen3.5-4b-lora-r8-lr1e-4-bs1.yaml` 提供完整的模型调参示例；
实际训练创建 session 前仍验证服务支持情况。
LoRA rank 和训练层开关传给 TrainingClient，`train.lr` 和 `optimizer` 传给 AdamParams，
`sampling` 传给 SamplingParams；当前不支持指定 4bit 或 LoRA alpha/dropout。
`--set` 只允许覆盖已声明的键，拼错时立即失败。API Key 仅来自登录状态或 `.env`。

当前数据格式是一行一个对象：

```json
{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"参考回答"}]}
```

可有 system 和多轮历史，但只训练最后一条 assistant 回复；历史作为 prompt 不参与 loss。
采用 tokenizer 完整 chat template 的结束标记，校验生成 prompt 是完整对话 token 的前缀，
然后执行 `tokens[:-1] / tokens[1:] / weights[1:]`。前缀不一致或样本过长直接报错，
不静默截断、丢失答案或套用其他模型的特殊 token。

真实数据已有标准 JSONL 时，将其放到 `RAW_DIR`，在自己的配置中指定 processed 文件名后运行：

```powershell
uv run python scripts/prepare_data.py --config <自己的配置> --input <相对RAW_DIR的文件名>
```

该命令只准备 train；固定 case 集单独放在配置的 `data.cases` 下，避免自动复用训练样本。
当前不引入外部数据集，因此没有新增外部数据许可或下载 manifest。

## 本地产物与可复现范围

每次 run 目录独立创建，不复用已有目录：

```text
outputs/<时间>-sft-pytrio-toy-s42/
  config.yaml          # 独立配置 + CLI 覆盖、有效日志模式、训练数据 sha256
  run.json             # 状态、是否 dry-run、本地依赖版本
  remote.json          # 云端 service session ID 和 base model
  metrics.jsonl        # 每步 loss、epoch、监督 token 数、时间
  checkpoints.json     # 每次保存的 step、sampler_path、state_path
  evaluations.jsonl    # case 数据 hash、采样参数和 seed
  cases.jsonl          # prompt、reference、response、base/model_path 和 token
  swanlog/             # 本地 SwanLab 日志
models/checkpoints/<checkpoint_id>.zip
models/checkpoints/<checkpoint_id>.json  # 本地 hash 校验回执
```

所有路径通过 `rlflow.paths` 定位，脚本不依赖 CWD；`.env` 空目录值使用仓库默认值。
新增下载不解压、不合并基模、不执行本地推理。已有 archive 只有回执 sha256 匹配才跳过，
否则拒绝覆盖；下载中断留下的文件需先人工检查并移走后再重试。
`sampler_path` 与真实 checkpoint ID 用途不同，下载 ID 通过 `cloud.py checkpoints` 查询。

目前保存了可供后续恢复的云端完整状态，但尚未提供本地数据游标与 shuffle 状态恢复命令，
不能仅靠 `state_path` 声称完整实验可精确续跑。远端模型版本与计算确定性由服务提供，
本地 seed 和配置快照也不等于承诺逐位一致。

## 验证范围

离线测试验证数据格式、mask/右移、SDK 请求类型与 future 顺序、本地日志和下载幂等性。
`--dry-run` 无网络地校验数据和配置，不校验模型特有 tokenizer，也不生成训练结果。
实际模型可用性、云端训练、推理质量、checkpoint 下载需要账号和选定模型后实测。

官方依据：
[PyTRIO skill](https://github.com/SwanHubX/pytrio-skill)、
[ServiceClient](https://docs.pytrio.cn/docs/api/ServiceClient)、
[保存权重](https://docs.pytrio.cn/docs/guide/save)、
[下载权重](https://docs.pytrio.cn/docs/guide/download)。
