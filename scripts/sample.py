"""按 run 的 backend 从已保存权重采样固定 case，结果追加到本地。"""

from __future__ import annotations

import argparse
import json

from sft.config import load_config
from sft.data import load_messages
from sft.evaluation import sample_cases
from sft.experiment import append_record
from sft.files import dataset_path, fingerprint
from sft.model_client import PyTrioClient
from sft.paths import project_path


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
        entry = checkpoints["checkpoints"][-1]
        model_path = (
            entry["sampler_path"] if cfg["execution"]["backend"] == "cloud" else entry["path"]
        )
    path = dataset_path(cfg["data"]["cases"])
    rows = load_messages(path)[: cfg["data"]["case_limit"]]
    append_record(
        run_dir / "evaluations.jsonl",
        {
            "data": fingerprint(path),
            "model_path": model_path,
            "sampling": cfg["sampling"],
            "seed": cfg["seed"],
        },
    )
    if cfg["execution"]["backend"] == "cloud":
        client = PyTrioClient(cfg)
    else:
        from sft.lora.sampling import load_sampling_client

        client = load_sampling_client(cfg, model_path)
    sample_cases(cfg, rows, run_dir, client, model_path)


if __name__ == "__main__":
    main()
