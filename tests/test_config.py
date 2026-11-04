"""独立实验配置与显式超参数的回归检查。"""

import pytest
import yaml

from rlflow.config import load_config
from rlflow.paths import CONFIGS_DIR


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.rglob("*.yaml")))
def test_each_config_is_complete_without_inheritance(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "defaults" not in raw
    assert {"seed", "device", "stage", "algo", "train", "logging"} <= raw.keys()
    assert "env" in raw or "task" in raw
    assert load_config(path) == {**raw, "_config_path": str(path.resolve())}


def test_rl_and_sft_have_separate_parameters():
    rl = load_config("configs/rl/ppo_cartpole.yaml")
    sft = load_config("configs/sft/pytrio_toy.yaml")
    assert rl["train"]["gamma"] == 0.99
    assert "env" not in sft
    assert {"gamma", "total_steps", "clip_ratio", "gae_lambda"}.isdisjoint(sft["train"])
    assert "sampling" not in rl
    assert sft["model"]["lora_rank"] == 8
    assert {"beta1", "beta2", "eps", "weight_decay", "grad_clip_norm"} <= sft["optimizer"].keys()


def test_defaults_are_rejected_instead_of_followed(tmp_path):
    config = tmp_path / "inherited.yaml"
    config.write_text("defaults: missing-parent.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="defaults"):
        load_config(config)


def test_model_config_name_describes_the_parameters():
    path = CONFIGS_DIR / "sft" / "qwen3.5-4b-lora-r8-lr1e-4-bs1.yaml"
    cfg = load_config(path)
    assert cfg["model"]["base_model"] == "Qwen/Qwen3.5-4B"
    assert cfg["model"]["lora_rank"] == 8
    assert cfg["train"]["lr"] == 1e-4
    assert cfg["train"]["batch_size"] == 1
    assert cfg["run_name"] == path.stem
