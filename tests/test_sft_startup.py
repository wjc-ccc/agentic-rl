"""真实入口顺序：配置、数据、快照、SwanLab、模型；测试无网络。"""

import copy
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest
import yaml
from conftest import FixedTokenizer

from sft import config
from sft.paths import PROJECT_ROOT


@pytest.fixture
def startup(cfg, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "sft_train_cli", PROJECT_ROOT / "scripts/train.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    events, logged, losses, checks = [], [], [], []
    monkeypatch.setenv("SFT_SWANLAB_PROJECT", "startup-test")

    def init(**kwargs):
        events.append("swanlab.init")
        logged.append(copy.deepcopy(kwargs["config"]))

    def log(metrics, step):
        if "loss" in metrics:
            losses.append((step, metrics["loss"]))

    monkeypatch.setitem(
        sys.modules,
        "swanlab",
        SimpleNamespace(init=init, log=log, finish=lambda: events.append("swanlab.finish")),
    )

    class Client:
        tokenizer = FixedTokenizer()
        usage = {}

        def __init__(self, cfg):
            events.append("client.create")

        def start_training(self, state_path=None):
            events.append("training.start")
            return {"training_run_id": "test"}

        def forward(self, batch, **kwargs):
            return [[-100, -100, -1, -2, -3] for _ in batch]

        def optim_step(self, lr):
            pass

        def sample(self, prompt, **kwargs):
            return [SimpleNamespace(text="result", tokens=[21, 99])]

        def save_sampler(self, name):
            return "sampler://" + name

        def save_state(self, name):
            return "state://" + name

    original = config.validate_config

    def validate(actual):
        checks.append(True)
        original(actual)

    monkeypatch.setattr(cli, "validate_config", validate)
    monkeypatch.setattr(cli, "PyTrioClient", Client)
    monkeypatch.setattr(cli, "seed_everything", lambda seed: None)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(path)])
    return SimpleNamespace(
        cli=cli,
        cfg=cfg,
        cfg_path=path,
        events=events,
        logged=logged,
        losses=losses,
        checks=checks,
        outputs=tmp_path / "outputs",
    )


def test_start_logs_effective_parameters_once_before_connection(startup, monkeypatch):
    startup.cfg["note"] = "extra metadata"
    startup.cfg["train"]["note"] = "no whitelist"
    startup.cfg_path.write_text(yaml.safe_dump(startup.cfg), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(startup.cfg_path),
            "--set",
            "train.learning_rate=2e-4",
            "--set",
            "train.per_device_train_batch_size=2",
        ],
    )
    startup.cli.main()
    assert startup.events == ["swanlab.init", "client.create", "training.start", "swanlab.finish"]
    assert len(startup.checks) == 1
    recorded = startup.logged[0]
    assert recorded["train"]["learning_rate"] == 2e-4
    assert recorded["train"]["per_device_train_batch_size"] == 2
    assert recorded["logging"]["project"] == "startup-test"
    assert recorded["note"] == "extra metadata"
    run = next(startup.outputs.iterdir())
    assert yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) == recorded
    assert startup.losses == [(1, 2), (2, 2)]


@pytest.mark.parametrize(
    "missing",
    ["seed", "train.learning_rate", "model.lora_rank", "logging.project", "execution.backend"],
)
def test_missing_field_fails_before_outputs_logging_and_connection(startup, missing):
    node = startup.cfg
    *parents, leaf = missing.split(".")
    for part in parents:
        node = node[part]
    del node[leaf]
    startup.cfg_path.write_text(yaml.safe_dump(startup.cfg), encoding="utf-8")
    with pytest.raises(KeyError) as error:
        startup.cli.main()
    assert error.value.args == (missing,)
    assert startup.events == [] and not startup.outputs.exists()


def test_dry_run_does_not_initialize_logging_or_model(startup, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(startup.cfg_path), "--dry-run"])
    startup.cli.main()
    assert len(startup.checks) == 1 and startup.events == []


def test_connection_failure_finishes_experiment(startup, monkeypatch):
    def fail(cfg):
        startup.events.append("client.create")
        raise RuntimeError("connection failed")

    monkeypatch.setattr(startup.cli, "PyTrioClient", fail)
    with pytest.raises(RuntimeError, match="connection failed"):
        startup.cli.main()
    assert startup.events == ["swanlab.init", "client.create", "swanlab.finish"]
    run = next(startup.outputs.iterdir())
    assert json.loads((run / "run.json").read_text())["status"] == "failed"
    assert json.loads((run / "error.json").read_text())["type"] == "RuntimeError"
