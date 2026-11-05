"""独立 SFT 配置：覆盖后检查一次必需字段，不合并 base、不补默认值。"""

from pathlib import Path

import yaml

from sft.paths import project_path

TRAIN_FIELDS = {
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_steps",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "weight_decay",
    "max_grad_norm",
    "eval_strategy",
    "eval_steps",
    "save_strategy",
    "save_steps",
    "logging_steps",
    "resume_from_checkpoint",
    "download_checkpoints",
    "stop_after_steps",
    "bf16",
    "fp16",
    "gradient_checkpointing",
    "dataloader_num_workers",
    "disable_tqdm",
}


def load_config(path: str | Path) -> dict:
    path = project_path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["stage"] != "sft":
        raise ValueError("SFT 入口不能加载 RL 配置")
    cfg["_config_path"] = str(path)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for item in overrides:
        key, raw = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node[part]
        previous = node[leaf]
        value = yaml.safe_load(raw)
        node[leaf] = float(value) if isinstance(previous, float) else value


def validate_config(cfg: dict) -> None:
    required = ["seed", "device", "stage", "algo", "method", "execution.backend"]
    for section, names in {
        "model": (
            "base_model",
            "tokenizer_path",
            "cache_dir",
            "local_path",
            "download_source",
            "lora_rank",
            "train_attn",
            "train_mlp",
            "train_unembed",
            "chat_template_kwargs",
        ),
        "data": (
            "name",
            "train",
            "eval",
            "cases",
            "train_limit",
            "eval_limit",
            "case_limit",
            "max_length",
        ),
        "train": sorted(TRAIN_FIELDS),
        "sampling": ("max_tokens", "temperature", "top_p", "top_k", "stop"),
        "logging": ("backend", "project", "mode"),
    }.items():
        required.extend(f"{section}.{name}" for name in names)
    if cfg.get("execution", {}).get("backend") == "local":
        required.extend(("model.lora_alpha", "model.lora_dropout"))
    for path in required:
        node = cfg
        for key in path.split("."):
            if key not in node:
                raise KeyError(path)
            node = node[key]
    if cfg["algo"] != "sft":
        raise NotImplementedError("SFT 入口仅支持 algo: sft")
    if cfg["method"] != "lora":
        raise NotImplementedError("QLoRA 当前不支持；本次使用 LoRA")
    unused = (
        [
            "train.bf16",
            "train.fp16",
            "train.gradient_checkpointing",
            "train.dataloader_num_workers",
            "model.local_path",
        ]
        if cfg["execution"]["backend"] == "cloud"
        else ["train.download_checkpoints"]
    )
    cfg["_unused_parameters"] = unused
