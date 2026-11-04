"""本地实验目录、配置快照和逐步日志；不执行模型计算。"""

from __future__ import annotations

import json
import os
import random
import re
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import yaml

from rlflow.paths import OUTPUTS_DIR


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    # Gymnasium 环境在创建后还需 env.reset(seed=seed)。


def append_record(path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class Experiment:
    def __init__(self, cfg: dict, *, dry_run: bool = False):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        name = re.sub(r"[^a-zA-Z0-9_.-]", "-", cfg.get("run_name", cfg["algo"]))
        self.path = OUTPUTS_DIR / f"{stamp}-{name}-s{cfg['seed']}"
        self.path.mkdir(parents=True, exist_ok=False)
        logging = cfg["logging"]
        logging["mode"] = os.environ.get("SWANLAB_MODE") or logging["mode"]
        logging["project"] = os.environ.get("SWANLAB_PROJECT") or logging["project"]
        (self.path / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        packages = {}
        for package in ("pytrio", "numpy", "transformers", "swanlab"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                pass
        self.tracker = None
        self.write("run.json", {"status": "prepared", "dry_run": dry_run, "packages": packages})

    def write(self, filename: str, record: dict) -> None:
        (self.path / filename).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    def start_logging(self, cfg: dict) -> None:
        import swanlab

        self.tracker = swanlab
        swanlab.init(
            project=cfg["logging"]["project"],
            experiment_name=self.path.name,
            config=cfg,
            mode=cfg["logging"]["mode"],
            logdir=str(self.path / "swanlog"),
        )

    def log(self, step: int, metrics: dict) -> None:
        append_record(
            self.path / "metrics.jsonl",
            {
                "step": step,
                "time": datetime.now(UTC).isoformat(),
                **metrics,
            },
        )
        if self.tracker:
            self.tracker.log(metrics, step=step)

    def finish(self, status: str) -> None:
        metadata = json.loads((self.path / "run.json").read_text(encoding="utf-8"))
        metadata["status"] = status
        self.write("run.json", metadata)
        if self.tracker:
            self.tracker.finish()
