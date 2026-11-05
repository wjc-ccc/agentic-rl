"""从本地查询云端模型、checkpoint，或下载 LoRA adapter。"""

from __future__ import annotations

import argparse
import json

from sft.remote import connect, download_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models")
    checkpoints = sub.add_parser("checkpoints")
    checkpoints.add_argument("--offset", type=int, default=0, help="每页 100 条，下一页加 100")
    download = sub.add_parser("download")
    download.add_argument("--run", required=True, help="本地 run 目录，包含配置和 checkpoint 引用")
    download.add_argument("--step", type=int, required=True, help="已保存的训练 step")
    args = parser.parse_args()
    service = connect()
    if args.command == "models":
        print(json.dumps(service.get_supported_models(), ensure_ascii=False, indent=2))
    elif args.command == "checkpoints":
        result = (
            service.create_rest_client()
            .list_user_checkpoints(limit=100, offset=args.offset)
            .result()
        )
        print(result.model_dump_json(indent=2))
    else:
        print(download_weights(service, args.run, args.step))


if __name__ == "__main__":
    main()
