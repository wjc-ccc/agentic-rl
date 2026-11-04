"""PyTRIO 连接和 LoRA 下载；不加载本地模型权重。"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from urllib.request import urlopen

from rlflow.paths import CHECKPOINTS_DIR


def connect():
    try:
        import pytrio as trio
    except ImportError as exc:
        raise RuntimeError("请先执行 uv sync --extra cloud，再执行 uv run trio login") from exc
    return trio.ServiceClient()


def check_model(service, model: str | None) -> None:
    if not model:
        raise ValueError("先用 scripts/cloud.py models 查询，再设置 model.base_model")
    supported = service.get_supported_models()
    if model not in supported:
        raise ValueError(f"服务当前不支持 {model}，可用模型: {supported}")


def download_weights(service, checkpoint_id: str):
    """下载官方提供的 adapter ZIP；已存在且本地 sha256 匹配则跳过。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", checkpoint_id):
        raise ValueError("checkpoint_id 只能包含字母、数字、下划线和连字符")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    target = CHECKPOINTS_DIR / f"{checkpoint_id}.zip"
    receipt = target.with_suffix(".json")
    if target.exists():
        with target.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8"))["sha256"] == digest:
            return target
        raise FileExistsError(f"已有文件未通过 sha256 校验，拒绝覆盖: {target}")
    url = service.create_rest_client().get_checkpoint_archive_url(checkpoint_id).result().url
    # 不保存短期下载 URL；独占创建，防止覆盖用户已有产物。
    with urlopen(url, timeout=60) as response, target.open("xb") as f:
        shutil.copyfileobj(response, f)
    if not zipfile.is_zipfile(target):
        raise ValueError(f"下载结果不是有效 ZIP，保留文件供检查: {target}")
    with target.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    receipt.write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "sha256": digest,
                "bytes": target.stat().st_size,
                "format": "peft-lora-zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target
