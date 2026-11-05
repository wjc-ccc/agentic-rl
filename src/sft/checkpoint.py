"""同后端的完整状态、随机状态与数据进度；不覆盖来源 checkpoint。"""

import json
import random

import numpy as np
import torch

from sft.experiment import training_fingerprint
from sft.paths import project_path


def write_progress(folder, cfg, step, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        "backend": cfg["execution"]["backend"],
        "step": step,
        "fingerprint": training_fingerprint(cfg),
        **extra,
    }
    (folder / "progress.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


### 加载是否 checkpoint使用
def load_resume(cfg):
    source = cfg["train"]["resume_from_checkpoint"]
    if not source:
        return None, None
    folder = project_path(source)
    state = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
    if state["backend"] != cfg["execution"]["backend"]:
        raise ValueError("checkpoint backend 不一致，不支持跨后端完整续训")
    if state["fingerprint"] != training_fingerprint(cfg):
        raise ValueError("checkpoint 的模型、训练参数或数据 hash 与本次配置不一致")
    return folder, state


def save_rng(folder):
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        folder / "client_rng.pt",
    )


def restore_rng(folder):
    state = torch.load(folder / "client_rng.pt", map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def due(cfg, kind, step, epoch_end):
    strategy = cfg["train"][f"{kind}_strategy"]
    return (strategy == "steps" and step % cfg["train"][f"{kind}_steps"] == 0) or (
        strategy == "epoch" and epoch_end
    )
