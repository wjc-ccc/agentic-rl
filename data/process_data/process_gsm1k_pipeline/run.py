"""确定性抽条、长度分层和不重叠划分；SFT/RL 使用同一组来源 ID。"""

import argparse
import json
import random
from pathlib import Path

import yaml

from sft.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, project_path
from sft.registry import register_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    source = RAW_DIR / cfg["name"] / cfg["revision"][:8] / f"first-{cfg['download']['count']}.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    rows = sorted(rows, key=lambda row: len(row["question"]))
    bins = [rows[i * len(rows) // 3 : (i + 1) * len(rows) // 3] for i in range(3)]
    rng = random.Random(cfg["selection"]["seed"])
    for bucket in bins:
        rng.shuffle(bucket)
    selected = [bucket.pop() for _ in range(max(map(len, bins))) for bucket in bins if bucket][
        : cfg["selection"]["count"]
    ]
    assert sum(cfg["splits"].values()) == len(selected)
    # eval/cases/train 轮转分配，避免按长度顺序切分造成明显偏差。
    assigned = {name: [] for name in cfg["splits"]}
    names = ["eval", "cases", "train"]
    for row in selected:
        for _ in range(len(names)):
            name = names.pop(0)
            names.append(name)
            if len(assigned[name]) < cfg["splits"][name]:
                assigned[name].append(row)
                break
    root = PROCESSED_DIR / cfg["name"] / cfg["selection"]["output_version"]
    root.mkdir(parents=True, exist_ok=True)
    pipeline = Path(__file__).relative_to(PROJECT_ROOT).as_posix()
    for split, split_rows in assigned.items():

        def prompt(row):
            return [
                {
                    "role": "user",
                    "content": row["question"] + "\nReturn only the final numeric answer.",
                }
            ]

        for stage in ("sft", "rl"):
            records = []
            for row in split_rows:
                record = {"id": row["id"], "answer": str(row["answer"])}
                record["messages"] = prompt(row)
                if stage == "sft":
                    record["messages"] += [{"role": "assistant", "content": str(row["answer"])}]
                records.append(record)
            target = root / f"{stage}-{split}.jsonl"
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
            if target.exists():
                assert target.read_text(encoding="utf-8") == text
            else:
                target.write_text(text, encoding="utf-8")
            register_files(
                [target],
                name=cfg["name"],
                source=f"{cfg['source']}@{cfg['revision']}",
                license=cfg["license"],
                pipeline=pipeline,
                parents=[source],
                selection={
                    "method": "seeded_length_strata",
                    "seed": cfg["selection"]["seed"],
                    "source_ids": [r["id"] for r in split_rows],
                    "split": split,
                    "counts": cfg["splits"],
                    "benchmark_use": "pipeline_smoke_only",
                },
            )
            print(f"{stage}/{split}: {len(records)} records -> {target}")


if __name__ == "__main__":
    main()
