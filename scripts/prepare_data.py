"""准备本地 SFT JSONL：自造 toy 数据，或校验用户已有的原始 JSONL。"""

from __future__ import annotations

import argparse
import json

from rlflow.config import load_config
from rlflow.data.sft import dataset_path, fingerprint, load_messages
from rlflow.paths import RAW_DIR


def write_once(path, rows):
    payload = "".join(json.dumps({"messages": row}, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"已有内容不同，拒绝覆盖: {path}")
    else:
        with path.open("x", encoding="utf-8") as f:
            f.write(payload)
    print(json.dumps(fingerprint(path), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--toy", action="store_true", help="生成无外部许可依赖的自造样本")
    source.add_argument("--input", help="相对 RAW_DIR 的、已按 messages 格式整理的 JSONL")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.toy:

        def conversation(question, answer):
            return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

        rows = [
            conversation("Where does rlflow prepare data?", "rlflow prepares data locally."),
            conversation(
                "Where does rlflow run model training?", "Model training runs in the cloud."
            ),
            conversation(
                "Where are experiment logs stored?", "Experiment logs are stored locally."
            ),
        ]
        cases = [
            conversation("Where does model inference run?", "Model inference runs in the cloud.")
        ]
        write_once(dataset_path(cfg["data"]["train"]), rows)
        write_once(dataset_path(cfg["data"]["cases"]), cases)
    else:
        path = (RAW_DIR / args.input).resolve()
        if not path.is_relative_to(RAW_DIR.resolve()):
            raise ValueError("--input 必须位于 RAW_DIR 中")
        write_once(dataset_path(cfg["data"]["train"]), load_messages(path))


if __name__ == "__main__":
    main()
