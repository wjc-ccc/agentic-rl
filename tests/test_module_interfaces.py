"""从公开 Interface 验证独立包、数据与 client；样本和产物仅写 pytest 临时目录。"""

import json
import os
import subprocess
import sys

import pytest

import rl
import sft
from rl.data import load_prompts
from sft.paths import PROJECT_ROOT

CONFIG = "configs/sft/qwen3.5-4b-lora-cloud-gsm1k-r8-lr1e-4-bs3.yaml"


def test_installed_packages_import_outside_repo_without_loading_cloud_sdk(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys, rl, sft; from importlib.metadata import distribution; "
                "from importlib.util import find_spec; assert find_spec('rlflow') is None; "
                "assert sft.load_method('lora').__module__ == 'sft.lora.trainer'; "
                "assert 'pytrio' not in sys.modules; "
                "assert set(distribution('sft-rl-flow').read_text('top_level.txt').split()) "
                "== {'sft', 'rl'}"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["ppo", "grpo"])
def test_unimplemented_rl_methods_fail_explicitly(name):
    with pytest.raises(NotImplementedError, match="尚未实现"):
        rl.load_method(name)


def test_unsupported_qlora_fails_before_output_or_cloud_calls(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train.py"),
            "--config",
            CONFIG,
            "--set",
            "method=qlora",
            "--dry-run",
        ],
        env={
            **os.environ,
            "SFT_OUTPUTS_DIR": str(tmp_path / "outputs"),
            "PYTHONIOENCODING": "utf-8",
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode != 0 and "QLoRA 当前不支持" in result.stderr
    assert not (tmp_path / "outputs").exists()
    with pytest.raises(ValueError, match="未知 SFT"):
        sft.load_method("unknown")
    with pytest.raises(ValueError, match="未知 RL"):
        rl.load_method("unknown")


def test_rl_data_keeps_reference_out_of_prompt(tmp_path):
    path = tmp_path / "prompts.jsonl"
    row = {"messages": [{"role": "user", "content": "1+1?"}], "answer": "2"}
    path.write_text(json.dumps(row), encoding="utf-8")
    assert load_prompts(path) == [row]
    row["messages"].append({"role": "assistant", "content": "2"})
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="以 user 结束"):
        load_prompts(path)
