"""QLoRA 能力状态与 Interface：当前不提供 train 实现。

预期数据与 LoRA 同为 sft.data 的 messages 对话，模型仍从 cfg.model 配置。
当前锁定的 PyTRIO create_lora_training_client 没有量化/4bit 入口，无法承诺 QLoRA。
load_method("qlora") 在创建 client 和 run 之前报错；不自动退化为普通 LoRA。
"""

UNAVAILABLE_REASON = "QLoRA 当前不支持：锁定的 PyTRIO SDK 没有量化/4bit 训练配置入口"
