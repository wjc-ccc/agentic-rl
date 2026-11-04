"""本地实验入口：--dry-run 校验准备；SFT 的模型计算仅通过 PyTRIO 执行。"""

from __future__ import annotations

import argparse

from rlflow.config import apply_overrides, load_config
from rlflow.data.sft import dataset_path, fingerprint, load_messages
from rlflow.experiment import Experiment, seed_everything
from rlflow.remote import connect
from rlflow.train.sft import train_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], help="如 model.base_model=模型名")
    parser.add_argument("--dry-run", action="store_true", help="只校验本地数据/配置，不连接云端")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    is_sft = cfg["stage"] == "sft" and cfg["algo"] == "sft"
    rows = []
    if is_sft:
        if cfg["execution"]["backend"] != "pytrio":
            raise ValueError("SFT 仅支持 execution.backend=pytrio")
        for key in ("epochs", "batch_size", "checkpoint_every"):
            if not isinstance(cfg["train"][key], int) or cfg["train"][key] <= 0:
                raise ValueError(f"train.{key} 必须是正整数")
        if float(cfg["train"]["lr"]) <= 0 or cfg["data"]["max_length"] < 2:
            raise ValueError("学习率必须为正，max_length 必须至少为 2")
        if not args.dry_run and not cfg["model"]["base_model"]:
            raise ValueError("先运行 scripts/cloud.py models，再用 --set model.base_model=模型名")
        path = dataset_path(cfg["data"]["train"])
        rows = load_messages(path)
        cfg["data"]["train_manifest"] = fingerprint(path)
    elif not args.dry_run:
        raise NotImplementedError("经典 RL 尚未实现；可用 --dry-run 检查配置并创建快照")
    experiment = Experiment(cfg, dry_run=args.dry_run)
    print(f"run_dir={experiment.path}", flush=True)
    if args.dry_run:
        experiment.finish("validated")
        print(f"本地校验通过：{len(rows)} 条 SFT 样本；未连接云端，未计算训练 loss。")
        return
    try:
        seed_everything(cfg["seed"])
        train_sft(cfg, rows, experiment, connect())
    except KeyboardInterrupt:
        experiment.finish("interrupted")
        raise
    except Exception:
        experiment.finish("failed")
        raise
    experiment.finish("completed")


if __name__ == "__main__":
    main()
