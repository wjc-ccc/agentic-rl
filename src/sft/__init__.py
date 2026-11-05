"""SFT 数据准备与显式后端路由。"""

from sft.data import load_messages
from sft.files import dataset_path, fingerprint


def load_method(name, backend="cloud"):
    if name == "qlora":
        raise NotImplementedError("QLoRA 当前不支持；请选择 lora")
    if name != "lora":
        raise ValueError(f"未知 SFT 方法: {name}")
    if backend == "cloud":
        from sft.lora.trainer import train
    elif backend == "local":
        from sft.lora.local import train
    else:
        raise ValueError("backend 应为 cloud/local")
    return train


def prepare(cfg):
    rows = load_messages(dataset_path(cfg["data"]["train"]))
    cfg["data"]["train_manifest"] = fingerprint(dataset_path(cfg["data"]["train"]))
    for key in ("eval", "cases"):
        if cfg["data"][key]:
            load_messages(dataset_path(cfg["data"][key]))
            cfg["data"][key + "_manifest"] = fingerprint(dataset_path(cfg["data"][key]))
    return rows[: cfg["data"]["train_limit"]]
