"""独立 YAML 与一次性缺项检查；不用别名或隐式 defaults。"""

import pytest
import yaml

from rl.config import load_config as load_rl_config
from rl.paths import CONFIGS_DIR as RL_CONFIGS_DIR
from sft.config import apply_overrides, load_config, validate_config
from sft.paths import CONFIGS_DIR


@pytest.mark.parametrize(
    "path, loader",
    [
        *[(path, load_config) for path in sorted(CONFIGS_DIR.rglob("*.yaml"))],
        *[(path, load_rl_config) for path in sorted(RL_CONFIGS_DIR.rglob("*.yaml"))],
    ],
)
def test_each_config_is_complete_without_inheritance(path, loader):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "defaults" not in raw
    assert {"seed", "device", "stage", "algo", "train", "logging"} <= raw.keys()
    assert "env" in raw or "task" in raw
    assert loader(path) == {**raw, "_config_path": str(path.resolve())}


def test_rl_and_sft_have_separate_parameters(cfg):
    rl = load_rl_config("configs/rl/ppo_cartpole.yaml")
    assert rl["train"]["gamma"] == 0.99
    assert {"gamma", "total_steps", "gae_lambda", "lr", "epochs", "batch_size"}.isdisjoint(
        cfg["train"]
    )
    assert "sampling" not in rl and "optimizer" not in cfg
    apply_overrides(cfg, ["train.learning_rate=1.25e-4"])
    assert cfg["train"]["learning_rate"] == 1.25e-4
    assert rl["train"]["lr"] == 3e-4


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.yaml")))
def test_native_parameter_configs_are_complete(path):
    cfg = load_config(path)
    validate_config(cfg)
    assert cfg["execution"]["backend"] in {"cloud", "local"}
    assert cfg["model"]["lora_rank"] == 8
    assert cfg["train"]["learning_rate"] == 1e-4
    assert "run_name" not in cfg
    if cfg["execution"]["backend"] == "cloud":
        assert "lora_alpha" not in cfg["model"] and "lora_dropout" not in cfg["model"]


def test_required_check_allows_metadata_without_type_or_range_policy(cfg):
    cfg["train"]["note"] = "allowed"
    cfg["train"]["learning_rate"] = 0
    validate_config(cfg)
    assert cfg["train"]["note"] == "allowed"
