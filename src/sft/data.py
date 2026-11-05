"""同一编码器供 cloud/local 使用；只在后端适配处改变 labels/Datum 的表示。"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sft.files import read_jsonl
from sft.model_client import TrainingDatum


def tokenizer_fingerprint(tokenizer):
    payload = {
        "backend": tokenizer.backend_tokenizer.to_str(),
        "chat_template": tokenizer.chat_template,
        "special_tokens": tokenizer.special_tokens_map,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class EncodedSample:
    input_ids: list[int]
    loss_mask: list[float]

    def local_record(self):
        return {
            "input_ids": self.input_ids,
            "labels": [
                t if w else -100 for t, w in zip(self.input_ids, self.loss_mask, strict=True)
            ],
        }

    def cloud_record(self):
        return TrainingDatum(self.input_ids[:-1], self.input_ids[1:], self.loss_mask[1:])


def load_messages(path: Path):
    rows = []
    for row in read_jsonl(path):
        messages = row["messages"]
        if len(messages) < 2 or [m["role"] for m in messages[-2:]] != ["user", "assistant"]:
            raise ValueError("每个样本必须以 user、assistant 结束")
        if any(
            m["role"] not in {"system", "user", "assistant"} or not m["content"].strip()
            for m in messages
        ):
            raise ValueError("messages 必须包含有效 role 和非空 content")
        rows.append(messages)
    return rows


def encode_example(messages, tokenizer, max_length, template_kwargs=None):
    options = template_kwargs or {}
    prompt = list(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True, **options
        )
    )
    tokens = list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=False, **options
        )
    )
    if not prompt or tokens[: len(prompt)] != prompt:
        raise ValueError("chat template 的 prompt 前缀不一致")
    if len(tokens) > max_length:
        raise ValueError(f"样本长度 {len(tokens)} 超过 max_length={max_length}")
    if len(tokens) <= len(prompt):
        raise ValueError("没有可监督的 completion token")
    return EncodedSample(tokens, [0.0] * len(prompt) + [1.0] * (len(tokens) - len(prompt)))


def encode_sft(messages, tokenizer, max_length, template_kwargs=None):
    return encode_example(messages, tokenizer, max_length, template_kwargs).cloud_record()
