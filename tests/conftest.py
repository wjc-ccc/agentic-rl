"""训练流程测试统一使用临时文件；不创建工作区演示数据。"""

import json
from types import SimpleNamespace

import pytest

from sft import experiment, files
from sft.config import load_config

CLOUD_CONFIG = "configs/sft/qwen3.5-4b-lora-cloud-gsm1k-r8-lr1e-4-bs3.yaml"
LOCAL_CONFIG = "configs/sft/qwen2.5-0.5b-instruct-sft-lora-local-gsm1k-lr1e-4.yaml"
MESSAGES = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]


class FixedTokenizer:
    backend_tokenizer = SimpleNamespace(to_str=lambda: "fixed-vocabulary-and-merges")
    chat_template = "fixed-last-assistant"
    special_tokens_map = {"eos_token": "<end>"}

    def apply_chat_template(
        self, messages, *, tokenize, return_dict, add_generation_prompt, **kwargs
    ):
        assert tokenize and not return_dict
        return [11, 12, 13] if add_generation_prompt else [11, 12, 13, 21, 22, 99]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(experiment, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(experiment, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setenv("SFT_SWANLAB_MODE", "offline")
    monkeypatch.setenv("SFT_SWANLAB_PROJECT", "pytest")
    result = load_config(CLOUD_CONFIG)
    for split in ("train", "eval", "cases"):
        path = files.PROCESSED_DIR / (split + ".jsonl")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"messages": MESSAGES}) + "\n", encoding="utf-8")
        result["data"][split] = path.name
    result["train"].update(
        max_steps=2,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        download_checkpoints=False,
        disable_tqdm=True,
    )
    return result
