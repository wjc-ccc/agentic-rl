"""临时目录中的真实 TRL LoRA：训练、评测、完整保存和确定性续训。"""

import copy
import json

import pytest
import torch
from conftest import LOCAL_CONFIG

import sft
from sft.config import load_config
from sft.experiment import Experiment, seed_everything


@pytest.fixture
def tiny_local(cfg, tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    model_path = tmp_path / "base"
    vocab = {
        token: i
        for i, token in enumerate(
            ["<pad>", "<unk>", "<eos>", "user", "assistant", "x", "y", "a", "b", "c"]
        )
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{m['role'] + ' ' + m['content'] + ' <eos> '}}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    seed_everything(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    local = load_config(LOCAL_CONFIG)
    local["data"] = cfg["data"]
    local["model"].update(local_path=str(model_path), base_model="fixture-qwen2", lora_rank=2)
    local["device"] = "cpu"
    local["train"].update(
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        disable_tqdm=True,
        max_steps=3,
        lr_scheduler_type="linear",
    )
    local["sampling"]["max_tokens"] = 2
    rows = [
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": answer}]
        for answer in ("a", "a b c", "b", "c b")
    ]
    from sft.files import dataset_path

    for split in ("train", "eval", "cases"):
        dataset_path(local["data"][split]).write_text(
            "".join(json.dumps({"messages": row}) + "\n" for row in rows), encoding="utf-8"
        )
    local["data"].update(eval_limit=4, case_limit=1)
    return local


def execute(cfg):
    rows = sft.prepare(cfg)
    run = Experiment(cfg)
    seed_everything(cfg["seed"])
    stopped = sft.load_method("lora", "local")(cfg, rows, run)
    return run, stopped


def test_real_trl_checkpoint_resume_matches_continuous_training(tiny_local):
    from safetensors.torch import load_file

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        continuous, stopped = execute(copy.deepcopy(tiny_local))
        assert not stopped
        interrupted_cfg = copy.deepcopy(tiny_local)
        interrupted_cfg["train"]["stop_after_steps"] = 1
        interrupted, stopped = execute(interrupted_cfg)
        assert stopped
        source = interrupted.checkpoint_root / "checkpoint-1"
        before = (source / "progress.json").read_bytes()
        assert all(
            (source / name).exists()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
        )
        resumed_cfg = copy.deepcopy(tiny_local)
        resumed_cfg["train"]["resume_from_checkpoint"] = str(source)
        resumed, stopped = execute(resumed_cfg)
        assert not stopped and (source / "progress.json").read_bytes() == before
        for run in (continuous, resumed):
            state = json.loads(
                (run.checkpoint_root / "checkpoint-3/trainer_state.json").read_text()
            )
            assert state["global_step"] == 3 and state["epoch"] == 1.5
        expected = load_file(continuous.checkpoint_root / "checkpoint-3/adapter_model.safetensors")
        actual = load_file(resumed.checkpoint_root / "checkpoint-3/adapter_model.safetensors")
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-7, rtol=1e-5)
        metrics = [
            json.loads(line) for line in (resumed.path / "metrics.jsonl").read_text().splitlines()
        ]
        assert [m["step"] for m in metrics if "loss" in m] == [2, 3]
        assert all(m["eval_loss"] > 0 for m in metrics if "eval_loss" in m)
    finally:
        torch.set_num_threads(previous_threads)


def test_native_token_loss_is_invariant_to_accumulation_split(tiny_local):
    from safetensors.torch import load_file

    runs = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for batch, accumulation in ((1, 2), (2, 1)):
            cfg = copy.deepcopy(tiny_local)
            cfg["train"].update(
                max_steps=1,
                eval_strategy="no",
                save_strategy="no",
                per_device_train_batch_size=batch,
                gradient_accumulation_steps=accumulation,
            )
            run, _ = execute(cfg)
            runs.append(run)
        states = [
            load_file(r.checkpoint_root / "checkpoint-1/adapter_model.safetensors") for r in runs
        ]
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], atol=1e-7, rtol=1e-5)
        losses = [
            [
                json.loads(line)["loss"]
                for line in (r.path / "metrics.jsonl").read_text().splitlines()
                if "loss" in json.loads(line)
            ]
            for r in runs
        ]
        assert losses[0] == pytest.approx(losses[1], abs=1e-5)
    finally:
        torch.set_num_threads(previous_threads)
