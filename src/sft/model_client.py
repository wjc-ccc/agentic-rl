"""PyTRIO SFT Adapter；所有模型计算在 API 中，不加载本地基模。"""

from dataclasses import dataclass

import numpy as np

from sft.paths import project_path
from sft.remote import connect


@dataclass
class TrainingDatum:
    input_tokens: list[int]
    target_tokens: list[int]
    weights: list[float]


@dataclass
class Generation:
    text: str | None
    tokens: list[int]
    logprobs: list[float]


class PyTrioClient:
    def __init__(self, cfg, *, service=None):
        self.cfg = cfg
        self.service = service if service is not None else connect()
        self._tokenizer = None
        self._sampler = None
        self._sampler_path = None
        self._current = None
        self.usage = {
            "train_input_tokens": 0,
            "eval_input_tokens": 0,
            "sample_input_tokens": 0,
            "sample_output_tokens": 0,
        }

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(project_path(self.cfg["model"]["tokenizer_path"]))
                if self.cfg["model"]["tokenizer_path"]
                else self.cfg["model"]["base_model"],
                cache_dir=str(project_path(self.cfg["model"]["cache_dir"])),
            )
        return self._tokenizer

    def start_training(self, state_path=None):
        model = self.cfg["model"]
        if state_path:
            self._training = self.service.create_training_client_from_state_with_optimizer(
                state_path
            )
        else:
            self._training = self.service.create_lora_training_client(
                base_model=model["base_model"],
                rank=model["lora_rank"],
                seed=self.cfg["seed"],
                train_mlp=model["train_mlp"],
                train_attn=model["train_attn"],
                train_unembed=model["train_unembed"],
            )
        return {
            "service_session_id": self.service.service_session_cuid,
            "training_run_id": self._training.training_run_cuid,
            "base_model": model["base_model"],
        }

    def forward(self, batch, *, denominator=1.0, backward=True):
        import pytrio as trio

        data = [
            trio.Datum(
                model_input=trio.ModelInput.from_ints(row.input_tokens),
                loss_fn_inputs={
                    "target_tokens": np.asarray(row.target_tokens, dtype=np.int64),
                    "weights": np.asarray(row.weights, dtype=np.float32) / denominator,
                },
            )
            for row in batch
        ]
        operation = self._training.forward_backward if backward else self._training.forward
        result = operation(data, loss_fn="cross_entropy").result()
        key = "train_input_tokens" if backward else "eval_input_tokens"
        self.usage[key] += sum(len(row.input_tokens) for row in batch)
        return [out["logprobs"].tolist() for out in result.loss_fn_outputs]

    def optim_step(self, learning_rate):
        import pytrio as trio

        t = self.cfg["train"]
        self._training.optim_step(
            trio.AdamParams(
                learning_rate=learning_rate,
                beta1=t["adam_beta1"],
                beta2=t["adam_beta2"],
                eps=t["adam_epsilon"],
                weight_decay=t["weight_decay"],
                grad_clip_norm=t["max_grad_norm"],
            )
        ).result()
        self._current = None

    def step(self, batch):
        values = self.forward(batch, denominator=sum(sum(r.weights) for r in batch))
        self.optim_step(self.cfg["train"]["learning_rate"])
        return values

    def save_sampler(self, name):
        return self._training.save_weights_for_sampler(name=name).result().path

    def save_state(self, name):
        return self._training.save_state(name=name).result().path

    def sample(self, prompt_tokens, *, model_path=None, num_samples=1, params=None):
        import pytrio as trio

        if model_path == "current":
            if self._current is None:
                self._current = self._training.save_weights_and_get_sampling_client()
            sampler = self._current
        else:
            if self._sampler is None or self._sampler_path != model_path:
                self._sampler = self.service.create_sampling_client(
                    base_model=self.cfg["model"]["base_model"], model_path=model_path
                )
                self._sampler_path = model_path
            sampler = self._sampler
        values = params if params is not None else self.cfg["sampling"]
        response = sampler.sample(
            prompt=trio.ModelInput.from_ints(prompt_tokens),
            num_samples=num_samples,
            sampling_params=trio.SamplingParams(
                seed=self.cfg["seed"] + getattr(self, "global_step", 0), **values
            ),
        ).result()
        self.usage["sample_input_tokens"] += len(prompt_tokens)
        self.usage["sample_output_tokens"] += sum(len(s.tokens) for s in response.sequences)
        return [Generation(s.text, list(s.tokens), list(s.logprobs)) for s in response.sequences]
