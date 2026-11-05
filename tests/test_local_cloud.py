"""临时数据与 SDK 替身验证：编码、token 归一、保存、完整恢复和下载失败。"""

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest
from conftest import CLOUD_CONFIG, MESSAGES, FixedTokenizer

import sft
from sft.checkpoint import due, load_resume
from sft.config import apply_overrides, load_config
from sft.data import encode_example, encode_sft, load_messages
from sft.experiment import Experiment
from sft.model_client import PyTrioClient, TrainingDatum
from sft.paths import PROJECT_ROOT, _dir
from sft.remote import download_entry, download_weights


def adapter_tar():
    files = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"test-adapter"}
    manifest = {
        "files": [
            {"path": k, "size": len(v), "sha256": hashlib.sha256(v).hexdigest()}
            for k, v in files.items()
        ]
    }
    files["trio_checkpoint_complete.json"] = json.dumps(manifest).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def future(value):
    return SimpleNamespace(result=lambda: value)


def make_service():
    trio = pytest.importorskip("pytrio")
    service = create_autospec(trio.ServiceClient, instance=True)
    training = create_autospec(trio.TrainingClient, instance=True)
    service.service_session_cuid = "service"
    training.training_run_cuid = "training"
    service.create_lora_training_client.return_value = training
    service.create_training_client_from_state_with_optimizer.return_value = training
    records, updates = [], []

    def forward(data, loss_fn):
        assert loss_fn == "cross_entropy"
        records.append(data)
        return future(
            SimpleNamespace(
                loss_fn_outputs=[
                    {
                        "logprobs": np.array(
                            [-100, -100]
                            + [
                                -i
                                for i in range(
                                    1, len(row.loss_fn_inputs["target_tokens"].tolist()) - 1
                                )
                            ]
                        )
                    }
                    for row in data
                ]
            )
        )

    training.forward_backward.side_effect = forward
    training.forward.side_effect = forward
    training.optim_step.side_effect = lambda params: (updates.append(params), future(None))[1]
    training.save_weights_for_sampler.side_effect = lambda name: future(
        SimpleNamespace(path="sampler://" + name)
    )
    training.save_state.side_effect = lambda name: future(SimpleNamespace(path="state://" + name))
    sampler = create_autospec(trio.SamplingClient, instance=True)
    sampler.sample.return_value = future(
        SimpleNamespace(
            sequences=[SimpleNamespace(text="result", tokens=[21, 99], logprobs=[-0.1, -0.2])]
        )
    )
    training.save_weights_and_get_sampling_client.return_value = sampler
    service.create_sampling_client.return_value = sampler
    return service, training, records, updates


def make_client(cfg, service):
    result = PyTrioClient(cfg, service=service)
    result._tokenizer = FixedTokenizer()
    return result


def test_cloud_and_local_share_unshifted_tokens_masks_and_end_token():
    shared = encode_example(MESSAGES, FixedTokenizer(), 6)
    assert shared.local_record() == {
        "input_ids": [11, 12, 13, 21, 22, 99],
        "labels": [-100, -100, -100, 21, 22, 99],
    }
    assert shared.cloud_record() == TrainingDatum(
        [11, 12, 13, 21, 22], [12, 13, 21, 22, 99], [0, 0, 1, 1, 1]
    )
    with pytest.raises(ValueError, match="max_length"):
        encode_sft(MESSAGES, FixedTokenizer(), 5)

    class Bad(FixedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [7] if kwargs["add_generation_prompt"] else [8, 9]

    with pytest.raises(ValueError, match="前缀"):
        encode_sft(MESSAGES, Bad(), 6)


@pytest.mark.parametrize("messages", [[], MESSAGES[:-1], list(reversed(MESSAGES))])
def test_invalid_messages_fail_at_reader(tmp_path, messages):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_messages(path)


def test_paths_and_native_overrides_work_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SFT_TEST_DIR", "")
    assert _dir("SFT_TEST_DIR", "outputs") == PROJECT_ROOT / "outputs"
    cfg = load_config(CLOUD_CONFIG)
    apply_overrides(cfg, ["train.learning_rate=2e-4", "train.adam_epsilon=1e-7"])
    assert cfg["train"]["learning_rate"] == 2e-4 and cfg["train"]["adam_epsilon"] == 1e-7
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.lr=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["train.learning_rate"])


def test_dry_run_and_unconfigured_data_do_not_contact_models(tmp_path):
    env = {
        **os.environ,
        "SFT_DATA_DIR": str(tmp_path / "data"),
        "SFT_OUTPUTS_DIR": str(tmp_path / "outputs"),
        "PYTHONIOENCODING": "utf-8",
    }
    path = tmp_path / "data/processed/fixture/train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"messages": MESSAGES}), encoding="utf-8")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config",
        CLOUD_CONFIG,
        "--set",
        "data.train=fixture/train.jsonl",
        "--set",
        "data.eval=null",
        "--set",
        "data.cases=null",
        "--dry-run",
    ]
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
    assert "1 条 SFT 样本" in result.stdout
    run = next((tmp_path / "outputs").iterdir())
    assert json.loads((run / "run.json").read_text())["status"] == "validated"
    assert not (run / "remote.json").exists() and not (run / "swanlog").exists()
    result = subprocess.run(
        command[:4] + ["--set", "data.train=null", "--dry-run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode != 0 and "尚未配置数据路径" in result.stderr


def test_sdk_types_logprob_loss_and_logical_batch_normalization(cfg):
    service, training, records, updates = make_service()
    cfg["train"].update(max_steps=1, gradient_accumulation_steps=2, eval_strategy="no")
    run = Experiment(cfg)
    client = make_client(cfg, service)
    sft.load_method("lora")(cfg, [MESSAGES, MESSAGES], run, client)
    assert len(updates) == 1 and len(records) == 2
    assert updates[0].learning_rate == cfg["train"]["learning_rate"]
    assert updates[0].beta2 == cfg["train"]["adam_beta2"]
    assert updates[0].grad_clip_norm == cfg["train"]["max_grad_norm"]
    assert sum(
        sum(d.loss_fn_inputs["weights"].tolist()) for batch in records for d in batch
    ) == pytest.approx(1)
    metrics = json.loads((run.path / "metrics.jsonl").read_text())
    assert metrics["loss"] == 2.0  # 只对 1,2,3 三个监督位置求平均
    service.get_supported_models.assert_not_called()
    assert training.optim_step.call_count == 1
    sampler = training.save_weights_and_get_sampling_client
    sampler.assert_not_called()


def test_scheduled_eval_uses_current_weights_and_records_cases(cfg):
    service, training, records, updates = make_service()
    cfg["train"]["max_steps"] = 1
    run = Experiment(cfg)
    sft.load_method("lora")(cfg, [MESSAGES], run, make_client(cfg, service))
    assert len(updates) == 1 and len(records) == 2  # train + forward-only eval
    training.forward.assert_called_once()
    training.save_weights_and_get_sampling_client.assert_called_once()
    case = json.loads((run.path / "cases.jsonl").read_text())
    assert case["step"] == 1 and case["model_path"] == "current"
    assert case["messages"] == MESSAGES[:-1]
    assert json.loads((run.path / "usage.json").read_text())["sample_output_tokens"] == 2


def test_unequal_lengths_use_full_accumulation_token_denominator(cfg):
    class VariableTokenizer(FixedTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            tokens = super().apply_chat_template(messages, **kwargs)
            if not kwargs["add_generation_prompt"] and messages[-1]["content"] == "long":
                return tokens[:-1] + [23, 24, 25, 99]
            return tokens

    rows = [MESSAGES, [MESSAGES[0], {"role": "assistant", "content": "long"}]]
    service, _, records, _ = make_service()
    cfg["train"].update(max_steps=1, eval_strategy="no", gradient_accumulation_steps=2)
    client = make_client(cfg, service)
    client._tokenizer = VariableTokenizer()
    run = Experiment(cfg)
    sft.load_method("lora")(cfg, rows, run, client)
    weights = [sum(d.loss_fn_inputs["weights"].tolist()) for batch in records for d in batch]
    assert sorted(weights) == pytest.approx([1 / 3, 2 / 3])
    assert json.loads((run.path / "metrics.jsonl").read_text())["loss"] == pytest.approx(3)


def test_resume_restores_optimizer_and_data_cursor_without_replaying_step(cfg):
    service, training, records, updates = make_service()
    cfg["train"].update(stop_after_steps=1, eval_strategy="no")
    run = Experiment(cfg)
    sft.load_method("lora")(cfg, [MESSAGES] * 4, run, make_client(cfg, service))
    source = run.checkpoint_root / "checkpoint-1"
    state = json.loads((source / "progress.json").read_text())
    assert state["start"] == 1 and state["epoch"] == 0
    cfg["train"].update(stop_after_steps=None, resume_from_checkpoint=str(source))
    resumed = Experiment(cfg)
    next_service, next_training, next_records, next_updates = make_service()
    sft.load_method("lora")(cfg, [MESSAGES] * 4, resumed, make_client(cfg, next_service))
    next_service.create_training_client_from_state_with_optimizer.assert_called_once_with(
        state["state_path"]
    )
    next_service.create_lora_training_client.assert_not_called()
    assert len(next_updates) == 1
    metrics = json.loads((resumed.path / "metrics.jsonl").read_text())
    assert metrics["step"] == 2
    assert source.parent != resumed.checkpoint_root
    cfg["train"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="参数或数据 hash"):
        load_resume(cfg)


def test_backend_or_tokenizer_change_is_not_complete_resume(cfg):
    service, _, _, _ = make_service()
    cfg["train"].update(max_steps=1, eval_strategy="no")
    run = Experiment(cfg)
    sft.load_method("lora")(cfg, [MESSAGES], run, make_client(cfg, service))
    cfg["train"]["resume_from_checkpoint"] = str(run.checkpoint_root / "checkpoint-1")
    changed = copy.deepcopy(cfg)
    changed["execution"]["backend"] = "local"
    with pytest.raises(ValueError, match="backend"):
        load_resume(changed)
    client = make_client(cfg, service)
    client._tokenizer = FixedTokenizer()
    client._tokenizer.chat_template = "changed-template"
    with pytest.raises(ValueError, match="tokenizer"):
        sft.load_method("lora")(cfg, [MESSAGES], Experiment(cfg), client)


def test_state_save_failure_preserves_sampler_and_no_resume_marker(cfg):
    service, training, _, _ = make_service()
    cfg["train"].update(max_steps=1, eval_strategy="no")

    def fail(name):
        raise RuntimeError("state save failed")

    training.save_state.side_effect = fail
    run = Experiment(cfg)
    with pytest.raises(RuntimeError, match="state save failed"):
        sft.load_method("lora")(cfg, [MESSAGES], run, make_client(cfg, service))
    entry = json.loads((run.path / "checkpoints.json").read_text())["checkpoints"][0]
    assert entry["sampler_path"].startswith("sampler://") and not entry["resumable"]
    assert not (run.checkpoint_root / "checkpoint-1/progress.json").exists()


def test_download_matches_owning_session_is_idempotent_and_recovers_complete_tar(cfg):
    run = Experiment(cfg)
    folder = run.checkpoint_root / "checkpoint-3"
    entry = {
        "step": 3,
        "path": str(folder),
        "training_run_id": "original-session",
        "sampler_path": "sampler://original",
        "state_path": "state://original",
    }
    run.record_checkpoint(entry)
    payload, downloads, sessions = adapter_tar(), [], []

    def listing(session):
        sessions.append(session)
        return future(
            SimpleNamespace(
                checkpoints=[
                    SimpleNamespace(path="state://original", checkpoint_id="wrong"),
                    SimpleNamespace(path="sampler://original", checkpoint_id="correct"),
                ]
            )
        )

    def download(checkpoint, path, *, resume):
        downloads.append(checkpoint)
        path.write_bytes(payload)
        return future(None)

    service = SimpleNamespace(
        create_rest_client=lambda: SimpleNamespace(
            list_checkpoints=listing, download_checkpoint=download
        )
    )
    target = download_weights(service, run.path, 3)
    assert target.name == "adapter.tar" and target.read_bytes() == payload
    assert download_weights(service, run.path, 3) == target
    assert downloads == ["correct"] and sessions == ["original-session"]
    receipt = json.loads((folder / "download.json").read_text())
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    target.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        download_weights(service, run.path, 3)
    second = run.checkpoint_root / "checkpoint-4"
    second.mkdir()
    (second / "adapter.download").write_bytes(payload)
    download_entry(service, second, entry)
    assert downloads == ["correct"]  # 完整临时归档不再发起 HTTP Range 请求


def test_download_failure_preserves_completed_update_and_resumable_state(cfg, monkeypatch):
    from sft.lora import trainer

    service, training, _, updates = make_service()
    cfg["train"].update(download_checkpoints=True, eval_strategy="no")

    def fail(*args):
        raise RuntimeError("download failed")

    monkeypatch.setattr(trainer, "download_entry", fail)
    run = Experiment(cfg)
    with pytest.raises(RuntimeError, match="download failed"):
        trainer.train(cfg, [MESSAGES] * 3, run, make_client(cfg, service))
    assert len(updates) == 1
    entry = json.loads((run.path / "checkpoints.json").read_text())["checkpoints"][0]
    assert entry["resumable"] and entry["download"] == "failed"
    assert (run.checkpoint_root / "checkpoint-1/progress.json").exists()


@pytest.mark.parametrize(
    "strategy,step,epoch,expected",
    [
        ("no", 2, True, False),
        ("steps", 1, False, False),
        ("steps", 2, False, True),
        ("epoch", 1, True, True),
        ("epoch", 2, False, False),
    ],
)
def test_eval_and_save_count_optimizer_updates(cfg, strategy, step, epoch, expected):
    for kind in ("eval", "save"):
        cfg["train"][kind + "_strategy"] = strategy
        cfg["train"][kind + "_steps"] = 2
        assert due(cfg, kind, step, epoch) == expected
