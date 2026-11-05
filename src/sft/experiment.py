"""真实参数命名与 SwanLab/JSONL 记录；不加载训练模型。"""

import hashlib
import json
import os
import random
import re
from datetime import UTC, datetime

import numpy as np
import yaml

from sft.paths import CHECKPOINTS_DIR, OUTPUTS_DIR


def training_fingerprint(cfg):
    payload = {
        k: v for k, v in cfg.items() if not k.startswith("_") and k not in {"logging", "run_name"}
    }
    payload = json.loads(json.dumps(payload))
    for name in ("resume_from_checkpoint", "stop_after_steps", "download_checkpoints"):
        payload["train"].pop(name, None)
    for key in ("cache_dir", "local_path", "tokenizer_path", "download_source"):
        payload["model"].pop(key, None)
    for key in ("train_manifest", "eval_manifest", "cases_manifest"):
        if key in payload["data"]:
            payload["data"][key] = {"sha256": payload["data"][key]["sha256"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parameter_name(cfg):
    t, m = cfg["train"], cfg["model"]
    model = m["base_model"].rstrip("/").rsplit("/", 1)[-1]
    mantissa, exponent = f"{t['learning_rate']:.8e}".split("e")
    lr = f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    label = (
        f"{cfg['stage']}-{cfg['method']}-{cfg['execution']['backend']}-"
        f"{model}-{cfg['data']['name']}-lr{lr}-ep{t['num_train_epochs']:g}-"
        f"r{m['lora_rank']}-bs{t['per_device_train_batch_size']}-"
        f"ga{t['gradient_accumulation_steps']}-len{cfg['data']['max_length']}-"
        f"s{cfg['seed']}-cfg{training_fingerprint(cfg)[:8]}"
    )
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", label).lower()


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def append_record(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class Experiment:
    def __init__(self, cfg, *, dry_run=False):
        stamp = datetime.now().strftime("%m%d-%H%M%S")
        name = parameter_name(cfg)
        run_id = f"{name}-{stamp}"
        if (OUTPUTS_DIR / run_id).exists():
            run_id += "-" + datetime.now().strftime("%f")
        self.path = OUTPUTS_DIR / run_id
        self.path.mkdir(parents=True)
        self.checkpoint_root = CHECKPOINTS_DIR / run_id
        self.cfg = cfg
        self.tracker = None
        self.checkpoints = []
        cfg["run_name"] = name
        prefix = cfg["stage"].upper()
        cfg["logging"]["mode"] = os.environ.get(f"{prefix}_SWANLAB_MODE") or cfg["logging"]["mode"]
        cfg["logging"]["project"] = (
            os.environ.get(f"{prefix}_SWANLAB_PROJECT") or cfg["logging"]["project"]
        )
        (self.path / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        self.write(
            "run.json",
            {
                "run_id": run_id,
                "status": "prepared",
                "dry_run": dry_run,
                "backend": cfg["execution"]["backend"],
                "unused_parameters": cfg.get("_unused_parameters", []),
            },
        )

    def write(self, name, record):
        (self.path / name).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def start_logging(self, cfg):
        import swanlab

        run = swanlab.init(
            project=cfg["logging"]["project"],
            name=self.path.name,
            config=cfg,
            mode={"cloud": "online"}.get(cfg["logging"]["mode"], cfg["logging"]["mode"]),
            log_dir=str(self.path / "swanlog"),
        )
        self.tracker = swanlab
        if run is not None:
            self.write("swanlab.json", {"id": run.id, "url": run.url})

    def log(self, step, metrics):
        metrics = {
            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.number))
        }
        append_record(
            self.path / "metrics.jsonl",
            {"step": step, "time": datetime.now(UTC).isoformat(), **metrics},
        )
        if self.tracker:
            self.tracker.log(metrics, step=step)

    def record_checkpoint(self, entry):
        self.checkpoints = [e for e in self.checkpoints if e["step"] != entry["step"]] + [entry]
        self.write("checkpoints.json", {"checkpoints": self.checkpoints})

    def finish(self, status):
        record = json.loads((self.path / "run.json").read_text(encoding="utf-8"))
        record["status"] = status
        self.write("run.json", record)
        if self.tracker:
            self.tracker.finish()
