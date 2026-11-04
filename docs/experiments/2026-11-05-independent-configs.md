# 2026-11-05：独立配置与常用超参数

## 假设与要求

RL 与 SFT 的 YAML 各自写全，不再从公共 base 继承参数。
SFT 按模型、LoRA rank、学习率、batch 命名，常用调参项必须实际传到 SDK。
保留用户在原 base 中显式选择的 SwanLab cloud 模式，将其迁移到每份独立配置。
外部数据下载入口按用户指定放到 `data/download_data/`，命名为 `download_<数据集名>_<数据量>.py`。

## 配置与实现

- 删除 `configs/base.yaml`，移除递归加载与深度合并函数；遇到 `defaults` 明确报错。
- RL 独立包含环境、gamma、GAE、clip、网络和评测参数。
- toy SFT 独立包含数据、模型、训练、Adam、生成和日志参数，不再含 RL 的 gamma 或 total_steps。
- 新增 `qwen3.5-4b-lora-r8-lr1e-4-bs1.yaml`，作为模型调参示例，服务支持情况仍需登录后核实。
- LoRA 训练层开关、Adam betas/eps、权重衰减、梯度裁剪与采样参数都传入真实 SDK 类型。
- 锁定 SDK 没有 4bit、LoRA alpha/dropout 入口，未添加不生效的配置。
- 修正 CLI 科学计数法覆盖：`2e-4` / `1e-8` 按原配置声明的浮点类型写入快照，避免变成字符串。
- 当前无外部下载脚本或指定数据集可迁移；建立本地目录说明和版本化命名文档，没有下载数据。
  未修改 `.gitignore` 或强制提交 `data/download_data/`。

## 验证与真实结果

修改前：全量基线测试为 `1 failed, 13 passed in 9.82s`。
失败发生在 SwanLab 读取当前本地 `.env` 的 `SWANLAB_PROJECT` 时，字符串被按 JSON 对象解析。
该离线测试改为独立临时目录并移除其进程内的 `SWANLAB_PROJECT` 环境值，隔离用户本地配置；
没有修改用户 `.env`，也没有修复或宣称验证了该配置下的真实 SwanLab 初始化。

新增配置测试在实现前复现了继承、RL 参数进入 SFT 和缺少模型调参配置的问题。
实现后运行：

```text
uv run pytest
20 passed in 8.52s

uv run ruff check src/rlflow scripts tests
All checks passed!
```

RL、toy SFT、Qwen 配置均通过 `train.py --dry-run`。
Qwen 验证使用 lr=2e-4、batch_size=2、weight_decay=0.01、grad_clip_norm=1.0，
run_name 同步为 `qwen3.5-4b-lora-r8-lr2e-4-bs2`；完整覆盖值写入独立 run 快照。
toy 数据准备保持幂等，原有文件 hash 未变。

## 结论

配置不再跨阶段继承，参数与 SDK 调用接线通过离线检查。
未启动云端训练、推理、SwanLab 上传或外部数据下载，没有模型效果结论。
此前 2026-11-04 的记录描述当时的继承配置，本次记录取代其当前配置约定。
