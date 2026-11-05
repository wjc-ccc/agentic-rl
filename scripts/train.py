"""SFT 统一入口：execution.backend 选择 PyTRIO API 或本地 TRL。"""

import argparse

import sft
from sft.config import apply_overrides, load_config, validate_config
from sft.experiment import Experiment, seed_everything
from sft.model_client import PyTrioClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    validate_config(cfg)
    train = sft.load_method(cfg["method"], cfg["execution"]["backend"])
    rows = sft.prepare(cfg)
    experiment = Experiment(cfg, dry_run=args.dry_run)
    print(f"run_dir={experiment.path}", flush=True)
    print("不适用参数:", cfg.get("_unused_parameters", []), flush=True)
    if args.dry_run:
        experiment.finish("validated")
        print(f"本地校验通过：{len(rows)} 条 SFT 样本；未连接模型。")
        return
    try:
        experiment.start_logging(cfg)
        seed_everything(cfg["seed"])
        if cfg["execution"]["backend"] == "cloud":
            client = PyTrioClient(cfg)
        else:
            client = None
        stopped = train(cfg, rows, experiment, client)
    except KeyboardInterrupt:
        experiment.finish("interrupted")
        raise
    except Exception as exc:
        experiment.write("error.json", {"type": type(exc).__name__, "message": str(exc)})
        experiment.finish("failed")
        raise
    experiment.finish("interrupted" if stopped else "completed")


if __name__ == "__main__":
    main()
