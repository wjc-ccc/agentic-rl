"""训练入口骨架。

当前只做：加载配置 → 建 run 目录 → 打印。具体算法在后续阶段填进 src/rlflow/algos/。

    uv run python scripts/train.py --config configs/rl/ppo_cartpole.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import yaml

from rlflow.config import load_config
from rlflow.paths import OUTPUTS_DIR, ensure_dirs


def make_run_dir(cfg: dict) -> "object":
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    name = cfg.get("run_name") or f"{cfg.get('algo', 'run')}-{cfg.get('env', {}).get('id', 'task')}"
    run_dir = OUTPUTS_DIR / f"{stamp}-{name}-s{cfg.get('seed', 0)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/ 下的 YAML 路径")
    parser.add_argument("--set", action="append", default=[], help="覆盖项，如 train.lr=1e-4")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)

    ensure_dirs()
    run_dir = make_run_dir(cfg)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), "config": cfg}, ensure_ascii=False, indent=2))
    print("\n[骨架] 训练循环尚未实现，下一步见 docs/learning-roadmap.md 阶段 3。")


if __name__ == "__main__":
    main()
