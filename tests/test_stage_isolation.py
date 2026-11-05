"""SFT 当前运行，RL 仅备用；代码与参数不得跨阶段复用。"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_CONFIG = PROJECT_ROOT / "configs/sft/qwen3.5-4b-lora-cloud-gsm1k-r8-lr1e-4-bs3.yaml"
RL_CONFIG = PROJECT_ROOT / "configs/rl/ppo_cartpole.yaml"


def test_stage_packages_have_no_cross_stage_or_shared_project_imports():
    for package in ("sft", "rl"):
        forbidden = {"rl" if package == "sft" else "sft", "rlflow"}
        for path in (PROJECT_ROOT / "src" / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imported = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    imported = {(node.module or "").split(".")[0]}
                else:
                    continue
                assert not imported & forbidden, f"跨阶段依赖: {path}: {imported & forbidden}"


@pytest.mark.parametrize("package", ["sft", "rl"])
def test_one_stage_import_does_not_load_the_other(package, tmp_path):
    other = "rl" if package == "sft" else "sft"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                f"import sys, {package}; import {package}.config; "
                f"assert '{other}' not in sys.modules; assert 'rlflow' not in sys.modules"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_each_stage_rejects_the_other_stage_configuration():
    from rl.config import load_config as load_rl
    from sft.config import load_config as load_sft

    assert load_sft(SFT_CONFIG)["stage"] == "sft"
    assert load_rl(RL_CONFIG)["stage"] == "rl"
    with pytest.raises(ValueError, match="SFT"):
        load_sft(RL_CONFIG)
    with pytest.raises(ValueError, match="RL"):
        load_rl(SFT_CONFIG)


def test_sft_training_entry_does_not_accept_reserved_rl_config(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train.py"),
            "--config",
            str(RL_CONFIG),
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            **os.environ,
            "SFT_OUTPUTS_DIR": str(tmp_path / "outputs"),
            "RLFLOW_OUTPUTS_DIR": str(tmp_path / "outputs"),
            "PYTHONIOENCODING": "utf-8",
        },
    )
    assert result.returncode != 0
    assert "SFT" in result.stderr and "RL" in result.stderr
    assert not (tmp_path / "outputs").exists()
