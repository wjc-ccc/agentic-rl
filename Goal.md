# 当前目标：本地控制实验，云端执行模型计算

- 项目级 skill：`.agents/skills/pytrio-skill/`，固定上游版本，不安装到全局。
- 本地负责：数据整理与校验、tokenize、SFT 训练循环、实验参数控制、SwanLab 离线日志、
  指标监控、case 保存与查看、checkpoint 引用管理、LoRA adapter 下载。
- PyTRIO 云端负责：模型前向、反向、优化器更新、推理采样、权重和训练状态保存。
- 不上传完整仓库、`.env` 或原始数据目录到 PyTRIO。训练发送 token batch，推理发送 prompt token；
  这些仍然包含数据内容，正式实验前应确认数据可发往服务端。
- 当前实现最小 SFT 闭环；DPO / GRPO 等按学习路线逐步实现，不用空壳冒充已完成算法。
- 账号和模型暂未确定：先完成无需凭据的离线验证，再查询支持模型并执行真实云端实验。
- 代码、配置、文档和 skill 上传 GitHub；原始数据、权重、日志和密钥保留在本地或云端存储。
- RL / SFT 各用完整独立 YAML，不使用 base 或 defaults 继承；模型调参配置名包含模型、LoRA rank、lr、batch。
- 外部数据下载脚本位于 `data/download_data/`，命名 `download_<数据集名>_<数据量>.py`。

验收：`uv run pytest` 通过；toy 数据准备可重复执行；`train.py --dry-run` 不建立云端连接；
SDK 调用契约通过无网络测试；实际云端训练效果需有账号和模型后另行验证。
