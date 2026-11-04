"""YAML 配置加载：支持 `defaults: <path>` 继承，子配置覆盖父配置。

用法：
    from rlflow.config import load_config
    cfg = load_config("configs/rl/ppo_cartpole.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rlflow.paths import PROJECT_ROOT


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = _resolve(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置不存在: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("defaults", None)
    parents = [parent] if isinstance(parent, str) else list(parent or [])
    merged: dict[str, Any] = {}
    for p in parents:
        merged = _deep_merge(merged, load_config(p))
    merged = _deep_merge(merged, cfg)
    merged["_config_path"] = str(cfg_path.resolve())
    return merged


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
        node[leaf] = yaml.safe_load(raw)
