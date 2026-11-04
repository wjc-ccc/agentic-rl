"""从本地查询云端模型、checkpoint，或下载 LoRA adapter。"""

from __future__ import annotations

import argparse
import json

from rlflow.remote import connect, download_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models")
    checkpoints = sub.add_parser("checkpoints")
    checkpoints.add_argument("--offset", type=int, default=0, help="每页 100 条，下一页加 100")
    download = sub.add_parser("download")
    download.add_argument("--checkpoint-id", required=True, help="checkpoint ID，非 sampler path")
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
        print(download_weights(service, args.checkpoint_id))


if __name__ == "__main__":
    main()
