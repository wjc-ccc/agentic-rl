"""不访问云端：验证数据契约、本地产物和 SDK 请求流程。"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest
import yaml

from rlflow.config import apply_overrides, load_config
from rlflow.data.sft import encode_sft, load_messages
from rlflow.experiment import Experiment
from rlflow.paths import PROJECT_ROOT, _dir
from rlflow.remote import check_model, download_weights

MESSAGES = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"},
]


class ToyTokenizer:
    """固定 token 序列，让预期的 mask 与右移位置可独立核对。"""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        if add_generation_prompt:
            assert messages == MESSAGES[:-1]
            return [11, 12, 13]
        assert messages == MESSAGES
        return [11, 12, 13, 21, 22, 99]  # 包含模板的结束 token


def test_sft_masks_and_shifts_exactly():
    encoded = encode_sft(MESSAGES, ToyTokenizer(), max_length=6)
    assert encoded == {
        "input_tokens": [11, 12, 13, 21, 22],
        "target_tokens": [12, 13, 21, 22, 99],
        "weights": [0.0, 0.0, 1.0, 1.0, 1.0],
    }


def test_sft_rejects_truncation_and_template_mismatch():
    with pytest.raises(ValueError, match="max_length"):
        encode_sft(MESSAGES, ToyTokenizer(), max_length=5)

    class MismatchTokenizer(ToyTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [7] if kwargs["add_generation_prompt"] else [8, 9]

    with pytest.raises(ValueError, match="前缀"):
        encode_sft(MESSAGES, MismatchTokenizer(), max_length=6)


@pytest.mark.parametrize("messages", [[], MESSAGES[:-1], list(reversed(MESSAGES))])
def test_invalid_data_fails_before_remote_calls(tmp_path, messages):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_messages(path)


def test_paths_and_config_work_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLFLOW_TEST_DIR", "")
    assert _dir("RLFLOW_TEST_DIR", "outputs") == PROJECT_ROOT / "outputs"
    monkeypatch.setenv("RLFLOW_TEST_DIR", "models")
    assert _dir("RLFLOW_TEST_DIR", "outputs") == PROJECT_ROOT / "models"
    cfg = load_config("configs/sft/pytrio_toy.yaml")
    apply_overrides(cfg, ["train.lr=2e-4"])
    assert cfg["train"]["lr"] == 2e-4
    apply_overrides(cfg, ["optimizer.eps=1e-8"])
    assert cfg["optimizer"]["eps"] == 1e-8
    assert cfg["seed"] == 42
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.lrr=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["train.lr"])


def test_prepare_and_dry_run_cli_are_idempotent_and_local(tmp_path):
    env = {
        **os.environ,
        "RLFLOW_DATA_DIR": str(tmp_path / "data"),
        "RLFLOW_OUTPUTS_DIR": str(tmp_path / "outputs"),
        "PYTHONIOENCODING": "utf-8",
    }

    def run_script(script, *args):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout

    config = ["--config", "configs/sft/pytrio_toy.yaml"]
    first = run_script("prepare_data.py", *config, "--toy")
    assert run_script("prepare_data.py", *config, "--toy") == first
    output = run_script("train.py", *config, "--dry-run")
    assert "3 条 SFT 样本" in output
    run = next((tmp_path / "outputs").iterdir())
    assert json.loads((run / "run.json").read_text())["status"] == "validated"
    snapshot = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["model"]["base_model"] is None
    assert len(snapshot["data"]["train_manifest"]["sha256"]) == 64
    assert not (run / "metrics.jsonl").exists()
    assert not (run / "remote.json").exists()
    assert "validated" in run_script("inspect_run.py", "--run", str(run))


def test_model_validation_does_not_choose_a_model():
    service = SimpleNamespace(get_supported_models=lambda: ["supported"])
    for model in (None, "unavailable"):
        with pytest.raises(ValueError):
            check_model(service, model)
    check_model(service, "supported")


def test_swanlab_logging_stays_local(tmp_path, monkeypatch):
    from rlflow import experiment as experiment_module

    monkeypatch.setattr(experiment_module, "OUTPUTS_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SWANLAB_PROJECT", raising=False)
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    cfg = load_config("configs/sft/pytrio_toy.yaml")
    run = Experiment(cfg)
    run.start_logging(cfg)
    try:
        run.log(1, {"validation/ping": 1})
    finally:
        run.finish("validated")
    assert (run.path / "swanlog").is_dir()
    assert json.loads((run.path / "metrics.jsonl").read_text())["validation/ping"] == 1


def test_adapter_download_is_idempotent_and_never_overwrites(tmp_path, monkeypatch):
    from rlflow import remote

    calls = []
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("adapter_config.json", "{}")
    payload = archive.getvalue()
    monkeypatch.setattr(remote, "CHECKPOINTS_DIR", tmp_path)
    monkeypatch.setattr(remote, "urlopen", lambda url, timeout: io.BytesIO(payload))

    def archive_url(checkpoint_id):
        calls.append(checkpoint_id)
        return SimpleNamespace(result=lambda: SimpleNamespace(url="https://example.test/temporary"))

    service = SimpleNamespace(
        create_rest_client=lambda: SimpleNamespace(
            get_checkpoint_archive_url=archive_url,
        )
    )
    path = download_weights(service, "ckpt-123")
    assert path.read_bytes() == payload
    assert download_weights(service, "ckpt-123") == path
    assert calls == ["ckpt-123"]
    assert "temporary" not in path.with_suffix(".json").read_text()
    path.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        download_weights(service, "ckpt-123")
    assert path.read_bytes() == b"changed"
    with pytest.raises(ValueError):
        download_weights(service, "../escape")


def test_sft_and_sampling_use_real_sdk_types_without_network(tmp_path, monkeypatch):
    trio = pytest.importorskip("pytrio")
    transformers = pytest.importorskip("transformers")
    from rlflow import experiment as experiment_module
    from rlflow.eval.cases import sample_cases
    from rlflow.train.sft import train_sft

    monkeypatch.setattr(experiment_module, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: ToyTokenizer()
    )
    cfg = load_config("configs/sft/pytrio_toy.yaml")
    cfg["model"]["base_model"] = "test-model"
    cfg["train"]["epochs"] = 2
    cfg["train"]["checkpoint_every"] = 1
    cfg["train"]["lr"] = 2e-4
    cfg["model"]["lora_rank"] = 16
    cfg["model"]["train_mlp"] = False
    cfg["optimizer"].update(
        beta1=0.8,
        beta2=0.98,
        eps=1e-8,
        weight_decay=0.1,
        grad_clip_norm=1.0,
    )
    cfg["sampling"].update(top_p=0.8, top_k=20, stop=["END"])
    events = []
    run = Experiment(cfg)
    # 此测试聚焦 SDK 契约和本地文件；SwanLab 离线另作真实脚本验证。
    monkeypatch.setattr(run, "start_logging", lambda cfg: None)

    def future(event, value):
        def result():
            events.append(event)
            return value

        return SimpleNamespace(result=result)

    class Training:
        training_run_cuid = "training-test"

        def forward_backward(self, data, loss_fn):
            assert loss_fn == "cross_entropy"
            assert all(isinstance(d, trio.Datum) for d in data)
            assert data[0].loss_fn_inputs["weights"].tolist() == [0, 0, 1, 1, 1]
            # Prompt 的 logprob 应被 mask；其值不能影响报告的平均 loss。
            outputs = [{"logprobs": np.asarray([-100, -100, -1, -2, -3])} for _ in data]
            return future("forward", SimpleNamespace(loss_fn_outputs=outputs))

        def optim_step(self, params):
            assert isinstance(params, trio.AdamParams)
            assert params.model_dump() == {
                "learning_rate": 2e-4,
                "beta1": 0.8,
                "beta2": 0.98,
                "eps": 1e-8,
                "weight_decay": 0.1,
                "grad_clip_norm": 1.0,
            }
            assert events[-1] == "forward"
            return future("optim", None)

        def save_weights_for_sampler(self, name):
            return future("sampler-save", SimpleNamespace(path=f"sampler://{name}"))

        def save_state(self, name):
            return future("state-save", SimpleNamespace(path=f"train://{name}"))

    class Sampler:
        def get_tokenizer(self):
            return ToyTokenizer()

        def sample(self, prompt, sampling_params, num_samples):
            assert isinstance(prompt, trio.ModelInput)
            assert isinstance(sampling_params, trio.SamplingParams)
            assert sampling_params.top_p == 0.8
            assert sampling_params.top_k == 20
            assert sampling_params.stop == ["END"]
            assert num_samples == 1
            return future(
                "sample",
                SimpleNamespace(
                    sequences=[
                        SimpleNamespace(
                            text="cloud response",
                            tokens=[21, 22],
                        )
                    ]
                ),
            )

    # 除了请求对象类型，还用 SDK 自身的方法签名约束替身调用。
    service = create_autospec(trio.ServiceClient, instance=True)
    service.service_session_cuid = "service-test"
    service.get_supported_models.return_value = ["test-model"]
    training = create_autospec(trio.TrainingClient, instance=True)
    training.training_run_cuid = "training-test"
    for method in ("forward_backward", "optim_step", "save_weights_for_sampler", "save_state"):
        getattr(training, method).side_effect = getattr(Training(), method)
    service.create_lora_training_client.return_value = training
    sampler = create_autospec(trio.SamplingClient, instance=True)
    sampler.get_tokenizer.side_effect = Sampler().get_tokenizer
    sampler.sample.side_effect = Sampler().sample
    service.create_sampling_client.return_value = sampler

    train_sft(cfg, [MESSAGES], run, service)
    service.create_lora_training_client.assert_called_once_with(
        base_model="test-model",
        rank=16,
        seed=42,
        train_mlp=False,
        train_attn=True,
        train_unembed=True,
    )
    run.finish("completed")
    metrics = [json.loads(line) for line in (run.path / "metrics.jsonl").read_text().splitlines()]
    assert [m["loss"] for m in metrics] == [2.0, 2.0]
    assert [m["step"] for m in metrics] == [1, 2]
    assert events == ["forward", "optim", "sampler-save", "state-save"] * 2
    checkpoints = json.loads((run.path / "checkpoints.json").read_text())["checkpoints"]
    assert len(checkpoints) == 2
    assert checkpoints[-1]["state_path"].startswith("train://")
    sample_cases(cfg, [MESSAGES], run.path, service, checkpoints[-1]["sampler_path"])
    case = json.loads((run.path / "cases.jsonl").read_text())
    assert case["response"] == "cloud response"
    assert case["reference"] == "answer"
    assert case["messages"] == MESSAGES[:-1]
    assert case["model_path"].startswith("sampler://")
