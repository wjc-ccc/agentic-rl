"""实际数据文件的登记与只读核验；路径均相对 DATA_DIR，可在不同机器复核。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from sft.files import fingerprint
from sft.paths import DATA_DIR, PROJECT_ROOT


def _local_path(path: Path) -> str:
    relative = path.resolve().relative_to(DATA_DIR.resolve())
    if relative.parts[0] not in {"raw", "processed"}:
        raise ValueError("实际数据必须位于 RAW_DIR 或 PROCESSED_DIR 中")
    return relative.as_posix()


def _facts(path: Path) -> dict:
    local_path = _local_path(path)
    records = None
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            records = sum(1 for line in f if line.strip())
    return {
        "stage": local_path.split("/")[0],
        "local_path": local_path,
        "sha256": fingerprint(path)["sha256"],
        "bytes": path.stat().st_size,
        "records": records,
    }


def _read_manifest() -> dict:
    path = DATA_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"datasets": []}


def register_files(
    paths: list[Path],
    *,
    name: str,
    source: str,
    license: str,
    pipeline: str,
    parents: list[Path],
    selection: dict,
) -> None:
    """只登记已存在的文件；已有登记不一致时拒绝修改，换新产物路径后再运行。"""
    manifest = _read_manifest()
    entries = manifest["datasets"]
    indexed = {entry["local_path"]: entry for entry in entries}
    if len(indexed) != len(entries):
        raise ValueError("manifest 中存在重复路径，先核查登记")
    lineage = [{"local_path": _local_path(p), "sha256": fingerprint(p)["sha256"]} for p in parents]
    changed = False
    for path in paths:
        facts = _facts(path)
        entry = {
            "name": name,
            **facts,
            "source": source,
            "license": license,
            "pipeline": {
                "path": pipeline,
                "sha256": fingerprint(PROJECT_ROOT / pipeline)["sha256"],
            },
            "parents": lineage,
            "selection": selection,
        }
        old = indexed.get(entry["local_path"])
        if old is not None:
            previous = {k: v for k, v in old.items() if k != "recorded_at"}
            # 文件和处理参数一致时沿用原始生成代码的 hash，不用当前代码重写来源。
            candidate = {**entry, "pipeline": old["pipeline"]}
            if previous != candidate or old["pipeline"]["path"] != pipeline:
                raise ValueError(f"已有登记不一致，拒绝覆盖: {entry['local_path']}")
        else:
            entry["recorded_at"] = datetime.now(UTC).isoformat()
            entries.append(entry)
            indexed[entry["local_path"]] = entry
            changed = True
    if changed:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=DATA_DIR, delete=False) as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        Path(f.name).replace(DATA_DIR / "manifest.json")


def check_manifest() -> list[str]:
    """核查实际文件、数量、hash 和父数据引用，不创建或修改任何文件。"""
    entries = _read_manifest()["datasets"]
    indexed = {entry["local_path"]: entry for entry in entries}
    errors = []
    if len(indexed) != len(entries):
        errors.append("manifest 中存在重复路径")
    for entry in entries:
        for field in (
            "name",
            "source",
            "license",
            "pipeline",
            "selection",
            "recorded_at",
        ):
            if field not in entry:
                errors.append(f"登记缺少 {field}: {entry['local_path']}")
        path = DATA_DIR / entry["local_path"]
        if not path.is_file():
            errors.append(f"文件缺失: {entry['local_path']}")
            continue
        for key, value in _facts(path).items():
            if entry[key] != value:
                errors.append(f"{key} 不一致: {entry['local_path']}")
        for parent in entry["parents"]:
            registered = indexed.get(parent["local_path"])
            if registered is None or registered["sha256"] != parent["sha256"]:
                errors.append(f"父数据登记不一致: {parent['local_path']}")
    for stage in ("raw", "processed"):
        for path in sorted((DATA_DIR / stage).rglob("*")):
            if path.is_file() and _local_path(path) not in indexed:
                errors.append(f"文件未登记: {_local_path(path)}")
    return errors
