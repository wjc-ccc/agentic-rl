"""云端连接与 SDK adapter 下载；checkpoint 独立保存所属 session。"""

import hashlib
import json
import tarfile

from sft.paths import project_path


def connect():
    import pytrio as trio

    return trio.ServiceClient()


def _complete_adapter(path):
    if not path.exists():
        return False
    try:
        with tarfile.open(path) as archive:
            manifest = json.load(archive.extractfile("trio_checkpoint_complete.json"))
            for item in manifest["files"]:
                payload = archive.extractfile(item["path"]).read()
                if (
                    len(payload) != item["size"]
                    or hashlib.sha256(payload).hexdigest() != item["sha256"]
                ):
                    return False
    except (tarfile.ReadError, KeyError):
        return False
    return True


def download_entry(service, folder, entry):
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "download.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        target = folder / saved["archive"]
        if saved["sha256"] != hashlib.sha256(target.read_bytes()).hexdigest():
            raise FileExistsError(f"已下载权重 hash 不一致，拒绝覆盖: {target}")
        return target
    rest = service.create_rest_client()
    candidates = rest.list_checkpoints(entry["training_run_id"]).result().checkpoints
    checkpoint = next(c for c in candidates if c.path == entry["sampler_path"])
    partial = folder / "adapter.download"
    if not _complete_adapter(partial):
        rest.download_checkpoint(checkpoint.checkpoint_id, partial, resume=True).result()
    if not _complete_adapter(partial):
        raise ValueError(f"下载结果不是有效 adapter 归档: {partial}")
    target = folder / "adapter.tar"
    partial.rename(target)
    receipt.write_text(
        json.dumps(
            {
                **entry,
                "checkpoint_id": checkpoint.checkpoint_id,
                "archive": target.name,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "bytes": target.stat().st_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def download_weights(service, run_dir, step):
    run_dir = project_path(run_dir)
    entries = json.loads((run_dir / "checkpoints.json").read_text())["checkpoints"]
    entry = next(e for e in entries if e["step"] == step)
    return download_entry(service, project_path(entry["path"]), entry)
