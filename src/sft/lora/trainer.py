"""Cloud SFT：API 累积梯度；评测、完整状态与自动下载由配置调度。"""

import math
import random

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import get_scheduler

from sft.checkpoint import due, load_resume, restore_rng, save_rng, write_progress
from sft.data import encode_sft, load_messages, tokenizer_fingerprint
from sft.evaluation import sample_cases
from sft.files import dataset_path
from sft.remote import download_entry


def evaluate(cfg, client, experiment, step):
    rows = load_messages(dataset_path(cfg["data"]["eval"]))[: cfg["data"]["eval_limit"]]
    encoded = [
        encode_sft(
            r, client.tokenizer, cfg["data"]["max_length"], cfg["model"]["chat_template_kwargs"]
        )
        for r in rows
    ]
    size = cfg["train"]["per_device_eval_batch_size"]
    logprobs = []
    for start in range(0, len(encoded), size):
        logprobs.extend(client.forward(encoded[start : start + size], backward=False))
    weights = np.concatenate([r.weights for r in encoded])
    loss = float(-np.dot(np.concatenate(logprobs), weights) / weights.sum())
    experiment.log(step, {"eval_loss": loss})
    cases = load_messages(dataset_path(cfg["data"]["cases"]))[: cfg["data"]["case_limit"]]
    sample_cases(cfg, cases, experiment.path, client, "current", step)
    experiment.write("usage.json", client.usage)


def train(cfg, rows, experiment, client):
    t = cfg["train"]
    encoded = [
        encode_sft(
            r, client.tokenizer, cfg["data"]["max_length"], cfg["model"]["chat_template_kwargs"]
        )
        for r in rows
    ]
    source, state = load_resume(cfg)
    signature = tokenizer_fingerprint(client.tokenizer)
    if state and signature != state["tokenizer_hash"]:
        raise ValueError("checkpoint tokenizer 与当前 tokenizer 不一致")
    remote = client.start_training(state["state_path"] if state else None)
    experiment.write("remote.json", remote)
    if source:
        experiment.write("resume.json", {"source": str(source), "step": state["step"]})
        if t["download_checkpoints"]:
            download_entry(client.service, source, state)
        restore_rng(source)
    micro = t["per_device_train_batch_size"]
    batch_size = micro * t["gradient_accumulation_steps"]
    epoch_steps = math.ceil(len(encoded) / batch_size)
    total = t["max_steps"] if t["max_steps"] > 0 else math.ceil(t["num_train_epochs"] * epoch_steps)
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=t["learning_rate"])
    scheduler = get_scheduler(
        t["lr_scheduler_type"],
        dummy,
        num_warmup_steps=(
            int(t["warmup_steps"])
            if t["warmup_steps"] >= 1
            else math.ceil(total * t["warmup_steps"])
        ),
        num_training_steps=total,
    )
    step = state["step"] if state else 0
    client.global_step = step
    if state:
        scheduler.load_state_dict(state["scheduler"])
        dummy.param_groups[0]["lr"] = scheduler.get_last_lr()[0]

    def save(next_epoch, next_start):
        folder = experiment.checkpoint_root / f"checkpoint-{step}"
        entry = {"step": step, "path": str(folder), **remote}
        name = f"{experiment.path.name}-step-{step:06d}"
        entry["sampler_path"] = client.save_sampler(name + "-sampler")
        experiment.record_checkpoint({**entry, "resumable": False})
        entry["state_path"] = client.save_state(name + "-state")
        folder.mkdir(parents=True)
        save_rng(folder)
        state_saved = write_progress(
            folder,
            cfg,
            step,
            **{k: v for k, v in entry.items() if k not in {"step", "path"}},
            epoch=next_epoch,
            start=next_start,
            scheduler=scheduler.state_dict(),
            tokenizer_hash=signature,
            usage=client.usage,
        )
        experiment.write("usage.json", client.usage)
        experiment.record_checkpoint({**entry, "resumable": True})
        if t["download_checkpoints"]:
            try:
                download_entry(client.service, folder, state_saved)
            except Exception:
                experiment.record_checkpoint({**entry, "resumable": True, "download": "failed"})
                raise

    epoch = state["epoch"] if state else 0
    initial_start = state["start"] if state else 0
    progress = tqdm(total=total, initial=step, desc="cloud SFT", disable=t["disable_tqdm"])
    while step < total:
        order = list(range(len(encoded)))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        for start in range(initial_start, len(order), batch_size):
            batch = [encoded[i] for i in order[start : start + batch_size]]
            weights = np.concatenate([r.weights for r in batch])
            values = []
            for i in range(0, len(batch), micro):
                values.extend(
                    client.forward(batch[i : i + micro], denominator=float(weights.sum()))
                )
            lr = scheduler.get_last_lr()[0]
            client.optim_step(lr)
            dummy.step()
            scheduler.step()
            step += 1
            client.global_step = step
            epoch_end = start + batch_size >= len(order)
            next_epoch, next_start = (epoch + 1, 0) if epoch_end else (epoch, start + batch_size)
            loss = float(-np.dot(np.concatenate(values), weights) / weights.sum())
            if step % t["logging_steps"] == 0:
                experiment.log(step, {"loss": loss, "learning_rate": lr, **client.usage})
            progress.update()
            progress.set_postfix(loss=f"{loss:.4f}")
            if due(cfg, "eval", step, epoch_end):
                evaluate(cfg, client, experiment, step)
            stop = t["stop_after_steps"] is not None and step >= t["stop_after_steps"]
            if due(cfg, "save", step, epoch_end) or stop or step >= total:
                save(next_epoch, next_start)
            if stop or step >= total:
                progress.close()
                return step < total
        initial_start = 0
        epoch += 1
    progress.close()
    return False
