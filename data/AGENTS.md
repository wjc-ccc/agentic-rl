/*# 数据目录约束

本文件补充根目录 AGENTS.md，适用于 data/ 及所有子目录。涉及数据前先读本文件、相关配置和 manifest，再检查实际文件。

## 放置位置

| 内容 | 位置与命名 |
|---|---|
| 下载、解压、原始分片组织、按明确规则获取子集的代码 | `download_data/download_<数据集名>_<数据量>/run.py` |
| 清洗、转换、过滤、去重、抽样、划分和混合的代码 | `process_data/process_<数据集名>_pipeline/run.py` |
| 原始数据 | `RAW_DIR/<数据集名>/<版本或子集>/` |
| 派生数据 | `PROCESSED_DIR/<数据集名>/<处理版本>/` |
| 每个实际文件的来源、数量、hash 和处理关系 | `DATA_DIR/manifest.json` |
| 当前 SFT 的文件读写、登记与消息校验 | `src/sft/files.py`、`src/sft/registry.py` |
| SFT 对话校验与 token 右移 | `src/sft/data.py` |
| 备用 RL 自己的 prompt/rollout 类型 | `src/rl/data.py`，不复用 SFT 类型或读取函数 |

- 一个数据集一个 pipeline 目录，多步骤可拆成同目录多个文件；组合数据使用自己的目录和名称。
- 下载目录数量是请求样本数，如 `1000`、`10k`、`all`；实际条数另行登记。
  `all` 必须对应明确 revision/config/split，不能笼统声称整个上游已下载。
- 代码目录只放 `.py/.yaml` 等实现文件，不存实际数据、下载缓存、模型或运行日志。
  不再为每个目录创建 README；运行参数和步骤写在对应代码的 docstring、CLI 帮助与 YAML 中。
- 下载与处理入口留在仓库的 data/ 代码目录；当前 SFT 数据位置从 `sft.paths` 获取。
  `SFT_DATA_DIR` 只改变 SFT 实际数据根目录；RL 的备用位置由 `rl.paths` 独立定义。

## 数据保护与重建

- raw 落盘后只读；所有转换写入 processed。不得覆盖、搬移、合并或删除已有数据，除非用户明确要求。
- 下载与生成必须幂等：已存在且 sha256 一致则跳过，不一致时停止；新处理版本写新路径。
- 新数据集先核实来源和许可证；未知信息明确标为 unknown，登记不代表获得训练、分发或发送云端的许可。
- 下载入口负责获取来源；清洗、重采样和跨来源训练混合归处理 pipeline，不能改写原始数据。
- 自造测试样本可直接生成 processed，不伪造 raw 或下载记录；不得把 toy 当成真实实验数据。
- 测试输入仅在 pytest 临时目录生成；工作区不保留或自动生成 demo 数据、toy 配置和演示 pipeline。
- 固定 case 与 train 分开准备，不复制训练样本冒充测试集。

## 数据接口

尚未指定真实数据集时，只保留本接口约定，不创建假数据集目录或空壳入口。
具体获取/处理入口使用 `run.py --config <数据配置.yaml>`；数据配置写全来源 URL、
固定 revision/config/split、许可、请求量、抽样方法与随机 seed、处理/组合规则及输入输出。
每步实际数量、参数配置与父数据关系写入登记，不能只记录最终文件名。

SFT 的 cloud/local 共用 UTF-8 messages JSONL、加载器和编码规则，一行一条完整对话：

```json
{"messages":[{"role":"user","content":"<问题>"},{"role":"assistant","content":"<参考回答>"}]}
```

允许可选 system 和多轮历史；role 为 system/user/assistant，content 必须是非空文本。
每条至少两条消息，以 user、assistant 结束；当前只监督最后一条 assistant，不静默截断超长样本。
train、eval、固定 cases 分开准备。`sft.data.load_messages(path)` 返回完整对话列表，
`encode_example` 输出完整 tokens/mask；local_record 生成未右移 labels（非监督位置 -100），
cloud_record 生成右移一次的 `TrainingDatum(input_tokens, target_tokens, weights)`。

LLM RL 的 prompt 接口由 `rl.data.load_prompts(path)` 读取，消息以 user 结束：

```json
{"messages":[{"role":"user","content":"<问题>"}],"answer":"<供reward使用的参考>"}
```

answer 可选，不能将其拼进采样 prompt；rollout 记录 completion tokens、采样时的 old_logprobs 和 reward。
这里只实现数据读取 Interface，PPO/GRPO 训练尚未实现；经典环境 PPO 的 reset/step 数据不使用此格式。
DPO 等其他数据形态待对应算法实现，不冒充已支持。

模型配置通过 `data.train`、`data.eval`、`data.cases` 接入相对 PROCESSED_DIR 的路径，
例如 `<真实数据集>/<处理版本>/train.jsonl` 与独立 `cases.jsonl`；`data.max_length` 控制长度上限。
数据集未确定时路径设为 null；训练入口应清楚报错，不生成数据或实验目录。
GSM1K 流程测试使用 configs/data/gsm1k_40.yaml；固定 revision 下载前 40 条，按问题长度分层抽 24 条，
划分为 16 train / 4 eval / 4 cases，同来源 ID 不跨集合。selection.output_version 控制新版本目录。
上游只有 test split、许可 unknown；用户已允许少量流程测试，不作为正式基准成绩。

当前 SFT 登记接口 `sft.registry.register_files(paths, name=..., source=..., license=...,
pipeline=..., parents=..., selection=...)` 只登记已存在的文件；现有 JSONL 计数为非空行数，
其他格式须随具体数据集补充计数和核验。

## 登记与验证

- 每个已落盘的实际文件一条登记；`local_path` 相对 DATA_DIR，以 `raw/` 或 `processed/` 开头。
- 必填 `name/stage/local_path/source/license/sha256/bytes/records/pipeline/parents/selection/recorded_at`。
  pipeline 记录代码路径与 hash；parents 记录输入路径与 hash；未知条数为 null，不能编造。
- 外部数据补充上游 revision/config/split、真实下载时间；`recorded_at` 只是登记时间。
- 抽样记录父数据、请求/实际条数、方法和随机 seed；组合记录每个来源的比例、条数、顺序与去重规则。
  处理参数和分步过滤数量写入登记元数据，保留从原始文件到最终 train/case 的关系。
- manifest 与只读核验是数据事实来源，不另建 Markdown 数据状态清单。计划下载、只有子集、
  部分下载、登记存在但本机缺文件必须如实区分，不能凭目录名或模板声称数据可用。
- SFT 数据改动后执行 `uv run python scripts/check_data.py`，RL 尚未接入此入口；
  失败先检查，不覆盖登记或删除数据来掩盖错误。
- 数据代码与登记按 Git 白名单管理；raw/processed 实际数据、缓存和敏感元数据不入库，不使用强制添加。
