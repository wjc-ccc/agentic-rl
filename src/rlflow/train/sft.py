"""本地编排 SFT；全部模型 forward/backward/optimizer 由 PyTRIO 执行。"""

from __future__ import annotations

import random

import numpy as np

from rlflow.data.sft import encode_sft
from rlflow.remote import check_model


def train_sft(cfg: dict, rows: list[list[dict]], experiment, service) -> None:
    import pytrio as trio

    check_model(service, cfg["model"]["base_model"])
    # 先在本地下载 tokenizer 并验证所有样本，再创建训练 session。
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    encoded = [encode_sft(row, tokenizer, cfg["data"]["max_length"]) for row in rows]
    datums = [
        trio.Datum(
            model_input=trio.ModelInput.from_ints(row["input_tokens"]),
            loss_fn_inputs={
                "target_tokens": np.asarray(row["target_tokens"], dtype=np.int64),
                "weights": np.asarray(row["weights"], dtype=np.float32),
            },
        )
        for row in encoded
    ]
    training = service.create_lora_training_client(
        base_model=cfg["model"]["base_model"],
        rank=cfg["model"]["lora_rank"],
        seed=cfg["seed"],
        train_mlp=cfg["model"]["train_mlp"],
        train_attn=cfg["model"]["train_attn"],
        train_unembed=cfg["model"]["train_unembed"],
    )
    experiment.write(
        "remote.json",
        {
            "service_session_id": service.service_session_cuid,
            "training_run_id": training.training_run_cuid,
            "base_model": cfg["model"]["base_model"],
        },
    )
    checkpoints = []

    def save(step):
        # 立即记录 sampler 路径，即使随后的完整状态保存失败也不会丢失引用。
        sampler = training.save_weights_for_sampler(
            name=f"{experiment.path.name}-step-{step}"
        ).result()
        entry = {"step": step, "sampler_path": sampler.path}
        checkpoints.append(entry)
        experiment.write("checkpoints.json", {"checkpoints": checkpoints})
        state = training.save_state(name=f"{experiment.path.name}-step-{step}-train").result()
        entry["state_path"] = state.path
        experiment.write("checkpoints.json", {"checkpoints": checkpoints})

    experiment.start_logging(cfg)
    rng = random.Random(cfg["seed"])
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        order = list(range(len(datums)))
        rng.shuffle(order)
        size = cfg["train"]["batch_size"]
        for start in range(0, len(order), size):
            batch = [datums[i] for i in order[start : start + size]]
            result = training.forward_backward(batch, loss_fn="cross_entropy").result()
            training.optim_step(
                trio.AdamParams(learning_rate=float(cfg["train"]["lr"]), **cfg["optimizer"])
            ).result()
            logprobs = np.concatenate(
                [np.asarray(out["logprobs"].tolist()) for out in result.loss_fn_outputs]
            )
            weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
            loss = float(-np.dot(logprobs, weights) / weights.sum())
            step += 1
            metrics = {"loss": loss, "epoch": epoch + 1, "supervised_tokens": float(weights.sum())}
            experiment.log(step, metrics)
            print(f"step={step} loss={loss:.6f}", flush=True)
            if step % cfg["train"]["checkpoint_every"] == 0:
                save(step)
    if not checkpoints or checkpoints[-1]["step"] != step:
        save(step)
