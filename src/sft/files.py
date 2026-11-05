"""SFT 自己的文件接口；实际样本位于 SFT DATA_DIR，不供 RL 复用。

例：dataset_path("math/v1/train.jsonl") -> PROCESSED_DIR / "math/v1/train.jsonl"。
fingerprint 返回 {"path": "<实际路径>", "sha256": "<64位hash>"}，写入 run 配置快照。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

from sft.paths import PROCESSED_DIR


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def dataset_path(name: str) -> Path:
    if name is None:
        raise ValueError("尚未配置数据路径，请先读 data/AGENTS.md 并填写 data.train / data.cases")
    path = (PROCESSED_DIR / name).resolve()
    if not path.is_relative_to(PROCESSED_DIR.resolve()):
        raise ValueError("data 路径必须位于 PROCESSED_DIR 中")
    return path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"数据集为空: {path}")
    return rows


def fingerprint(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
