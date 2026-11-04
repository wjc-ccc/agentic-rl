"""本地查看实验状态、指标和 case；--watch 持续读取新增行。"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

from rlflow.paths import project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--kind", choices=["summary", "metrics", "cases"], default="summary")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 必须是正整数")
    run_dir = project_path(args.run)
    if not run_dir.is_dir():
        parser.error(f"run 不存在: {run_dir}")
    if args.kind == "summary":
        if args.watch:
            parser.error("--watch 仅支持 metrics 或 cases")
        print((run_dir / "run.json").read_text(encoding="utf-8"))
        return
    path = run_dir / f"{args.kind}.jsonl"
    position = 0
    try:
        while True:
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    f.seek(position)
                    lines = []
                    while line := f.readline():
                        if not line.endswith("\n"):
                            break
                        lines.append(line)
                        position = f.tell()
                    for line in deque(lines, maxlen=args.limit):
                        print(
                            json.dumps(json.loads(line), ensure_ascii=False, indent=2), flush=True
                        )
            if not args.watch:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
