"""本地当前权重推理；SFT cloud/local 共用 case 记录格式与 sampling 配置。"""

import torch

from sft.model_client import Generation
from sft.paths import project_path


def load_sampling_client(cfg, checkpoint=None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m, t = cfg["model"], cfg["train"]
    base = str(project_path(m["local_path"])) if m["local_path"] else m["base_model"]
    tokenizer_path = checkpoint or (
        str(project_path(m["tokenizer_path"])) if m["tokenizer_path"] else base
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dtype = torch.bfloat16 if t["bf16"] else torch.float16 if t["fp16"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, attn_implementation="eager")
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint)
    model.to(cfg["device"])
    return LocalSamplingClient(cfg, model, tokenizer)


class LocalSamplingClient:
    def __init__(self, cfg, model, tokenizer, step=0):
        self.cfg, self.model, self.tokenizer, self.step = cfg, model, tokenizer, step

    def sample(self, prompt_tokens, *, model_path=None):
        s = self.cfg["sampling"]
        inputs = torch.tensor([prompt_tokens], device=self.model.device)
        options = {
            "max_new_tokens": s["max_tokens"],
            "do_sample": s["temperature"] > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if s["temperature"] > 0:
            options.update(temperature=s["temperature"], top_p=s["top_p"], top_k=max(s["top_k"], 0))
        if s["stop"]:
            options.update(stop_strings=s["stop"], tokenizer=self.tokenizer)
        was_training = self.model.training
        self.model.eval()
        devices = [inputs.device.index] if inputs.is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(self.cfg["seed"] + self.step)
            output = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), **options
            )
        self.model.train(was_training)
        tokens = output[0, len(prompt_tokens) :].tolist()
        return [Generation(self.tokenizer.decode(tokens, skip_special_tokens=True), tokens, [])]
