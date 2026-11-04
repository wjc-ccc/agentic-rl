"""使用某次实验的配置和固定 case 集，在云端推理并将结果保存到本地。"""

from __future__ import annotations

import argparse
import json

from rlflow.config import load_config
from rlflow.data.sft import dataset_path, fingerprint, load_messages
from rlflow.eval.cases import sample_cases
from rlflow.experiment import append_record
from rlflow.paths import project_path
from rlflow.remote import connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="本地 run 目录")
    parser.add_argument("--base", action="store_true", help="使用相同 case 比较未微调的基模")
    args = parser.parse_args()
    run_dir = project_path(args.run)
    cfg = load_config(run_dir / "config.yaml")
    model_path = None
    if not args.base:
        checkpoints = json.loads((run_dir / "checkpoints.json").read_text(encoding="utf-8"))
        model_path = checkpoints["checkpoints"][-1]["sampler_path"]
    path = dataset_path(cfg["data"]["cases"])
    rows = load_messages(path)
    append_record(
        run_dir / "evaluations.jsonl",
        {
            "data": fingerprint(path),
            "model_path": model_path,
            "sampling": cfg["sampling"],
            "seed": cfg["seed"],
        },
    )
    sample_cases(cfg, rows, run_dir, connect(), model_path)


if __name__ == "__main__":
    main()
