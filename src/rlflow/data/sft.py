"""本地 JSONL 对话校验和 SFT 右移；只监督最后一条 assistant 回复。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rlflow.paths import PROCESSED_DIR


def dataset_path(name: str) -> Path:
    path = (PROCESSED_DIR / name).resolve()
    if not path.is_relative_to(PROCESSED_DIR.resolve()):
        raise ValueError("data 路径必须位于 PROCESSED_DIR 中")
    return path


def load_messages(path: Path) -> list[list[dict]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError("样本至少包含 prompt 和最终 assistant 回复")
            for msg in messages:
                if (
                    msg.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(msg.get("content"), str)
                    or not msg["content"].strip()
                ):
                    raise ValueError("messages 必须包含有效 role 和非空 content")
            if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
                raise ValueError("每个样本必须以 user、assistant 结束")
            rows.append(messages)
    if not rows:
        raise ValueError(f"数据集为空: {path}")
    return rows


def fingerprint(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def encode_sft(messages: list[dict], tokenizer, max_length: int) -> dict:
    """从完整 chat template 获取结束标记，并校验 prompt 是完整对话的前缀。"""
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if not prompt or tokens[: len(prompt)] != prompt:
        raise ValueError("chat template 的 prompt 前缀不一致；需为此模型明确实现 mask")
    if len(tokens) > max_length:
        raise ValueError(f"样本长度 {len(tokens)} 超过 max_length={max_length}；请先清理数据")
    if len(tokens) <= len(prompt):
        raise ValueError("没有可监督的 completion token")
    return {
        "input_tokens": tokens[:-1],
        "target_tokens": tokens[1:],
        "weights": ([0.0] * len(prompt) + [1.0] * (len(tokens) - len(prompt)))[1:],
    }
