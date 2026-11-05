"""TRL SFTTrainer：共享已编码样本，原生优化器/调度/保存/续训，日志与 case 用 callback。"""

from dataclasses import fields

import torch

from sft.checkpoint import load_resume, write_progress
from sft.data import encode_example, load_messages, tokenizer_fingerprint
from sft.evaluation import sample_cases
from sft.files import dataset_path
from sft.paths import project_path


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        width = max(len(r["input_ids"]) for r in rows)
        ids, labels, masks = [], [], []
        for row in rows:
            length = len(row["input_ids"])
            ids.append(list(row["input_ids"]) + [self.tokenizer.pad_token_id] * (width - length))
            labels.append(list(row["labels"]) + [-100] * (width - length))
            masks.append([1] * length + [0] * (width - length))
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
        }


def lora_config(cfg):
    from peft import LoraConfig

    m = cfg["model"]
    modules = []
    if m["train_attn"]:
        modules += ["q_proj", "k_proj", "v_proj", "o_proj"]
    if m["train_mlp"]:
        modules += ["gate_proj", "up_proj", "down_proj"]
    if m["train_unembed"]:
        modules += ["lm_head"]
    return LoraConfig(
        r=m["lora_rank"],
        lora_alpha=m["lora_alpha"],
        lora_dropout=m["lora_dropout"],
        target_modules=modules,
        task_type="CAUSAL_LM",
    )


def train(cfg, rows, experiment, client=None):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from sft.lora.sampling import LocalSamplingClient

    t, m = cfg["train"], cfg["model"]
    cache = str(project_path(m["cache_dir"]))
    model_path = str(project_path(m["local_path"])) if m["local_path"] else m["base_model"]
    tokenizer_path = str(project_path(m["tokenizer_path"])) if m["tokenizer_path"] else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    signature = tokenizer_fingerprint(tokenizer)
    source, state = load_resume(cfg)
    if state and signature != state["tokenizer_hash"]:
        raise ValueError("checkpoint tokenizer 与当前 tokenizer 不一致")
    if source:
        experiment.write("resume.json", {"source": str(source), "step": state["step"]})

    def dataset(messages):
        return Dataset.from_list(
            [
                encode_example(
                    row, tokenizer, cfg["data"]["max_length"], m["chat_template_kwargs"]
                ).local_record()
                for row in messages
            ]
        )

    train_dataset = dataset(rows)
    eval_dataset, cases = None, []
    if t["eval_strategy"] != "no":
        evaluation = load_messages(dataset_path(cfg["data"]["eval"]))[: cfg["data"]["eval_limit"]]
        eval_dataset = dataset(evaluation)
        cases = load_messages(dataset_path(cfg["data"]["cases"]))[: cfg["data"]["case_limit"]]
    dtype = torch.bfloat16 if t["bf16"] else torch.float16 if t["fp16"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache, dtype=dtype, attn_implementation="eager"
    )
    model.config.use_cache = False
    native_fields = {field.name for field in fields(SFTConfig)}
    arguments = {k: v for k, v in t.items() if k in native_fields}
    arguments.update(
        output_dir=str(experiment.checkpoint_root),
        report_to="none",
        seed=cfg["seed"],
        use_cpu=cfg["device"] == "cpu",
        remove_unused_columns=False,
        packing=False,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        save_only_model=False,
        ignore_data_skip=False,
    )

    def persist(step):
        folder = experiment.checkpoint_root / f"checkpoint-{step}"
        write_progress(folder, cfg, step, tokenizer_hash=signature)
        experiment.record_checkpoint(
            {"step": step, "path": str(folder), "resumable": True, "backend": "local"}
        )

    class Logging(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                experiment.log(state.global_step, logs)

        def on_step_end(self, args, state, control, **kwargs):
            if t["stop_after_steps"] and state.global_step >= t["stop_after_steps"]:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            persist(state.global_step)

        def on_evaluate(self, args, state, control, **kwargs):
            sampling = LocalSamplingClient(cfg, kwargs["model"], tokenizer, state.global_step)
            sample_cases(cfg, cases, experiment.path, sampling, "current", state.global_step)

    class TokenMeanTrainer(SFTTrainer):
        # 原生 SFTTrainer 按完整累积 batch 的有效 token 归一训练 loss。
        # Trainer 的评测聚合默认按样本均值加权，这里改为监督 token 加权。
        def prediction_step(self, model, inputs, *args, **kwargs):
            count = (inputs["labels"][:, 1:] != -100).sum().item()
            loss, logits, labels = super().prediction_step(model, inputs, *args, **kwargs)
            self.eval_loss_sum += loss.item() * count
            self.eval_token_count += count
            return loss, logits, labels

        def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
            self.eval_loss_sum, self.eval_token_count = 0.0, 0
            result = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
            result.metrics[f"{metric_key_prefix}_loss"] = self.eval_loss_sum / self.eval_token_count
            return result

    trainer = TokenMeanTrainer(
        model=model,
        args=SFTConfig(**arguments),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(tokenizer),
        peft_config=lora_config(cfg),
        callbacks=[Logging()],
    )
    trainer.train(resume_from_checkpoint=str(source) if source else None)
    step = trainer.state.global_step
    folder = experiment.checkpoint_root / f"checkpoint-{step}"
    if not (folder / "progress.json").exists():
        trainer._save_checkpoint(trainer.model, trial=None)
        persist(step)
    if torch.cuda.is_available():
        experiment.log(step, {"gpu_peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30})
    return bool(t["stop_after_steps"] and step < trainer.state.max_steps)
