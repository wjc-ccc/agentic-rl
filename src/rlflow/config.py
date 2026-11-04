"""读取单份完整 YAML 配置，不继承、不合并其他配置。

用法：
    from rlflow.config import load_config
    cfg = load_config("configs/rl/ppo_cartpole.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rlflow.paths import project_path


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = project_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置不存在: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置必须是完整的 YAML 映射: {cfg_path}")
    if "defaults" in cfg:
        raise ValueError("不支持 defaults 继承，请在当前 YAML 中写全实验参数")
    cfg["_config_path"] = str(cfg_path)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """应用命令行 YAML 覆盖项，并让无效格式及拼错的配置键立即失败。"""
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator or not key or not raw:
            raise ValueError(f"覆盖项格式应为 key=value: {item}")
        node = cfg
        *parents, leaf = key.split(".")
        for parent in parents:
            node = node[parent]
        if leaf not in node:
            raise KeyError(f"配置键不存在: {key}")
        value = yaml.safe_load(raw)
        # YAML 会把 2e-4 解析为字符串；数值调参沿用配置中声明的浮点类型。
        node[leaf] = float(value) if isinstance(node[leaf], float) else value
