"""固定 revision 下载 GSM1K；原始全量 Arrow 和前 N 条子集均登记，不覆盖文件。"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from urllib.request import ProxyHandler, build_opener

import yaml

from sft.paths import PROJECT_ROOT, RAW_DIR, project_path
from sft.registry import register_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    root = RAW_DIR / cfg["name"] / cfg["revision"][:8]
    root.mkdir(parents=True, exist_ok=True)
    target = root / "source.arrow"
    receipt = root / "source.download.json"
    if target.exists():
        saved = json.loads(receipt.read_text())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == saved["sha256"]
    else:
        errors = []
        for endpoint in cfg["download"]["endpoints"]:
            url = (
                f"{endpoint}/datasets/{cfg['source']}/resolve/{cfg['revision']}/"
                f"{cfg['download']['file']}"
            )
            for direct in [True, False] if cfg["download"]["proxy"] == "auto" else [True]:
                started = time.monotonic()
                try:
                    opener = build_opener(ProxyHandler({})) if direct else build_opener()
                    with opener.open(url, timeout=30) as response:
                        payload = response.read()
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                target.write_bytes(payload)
                receipt.write_text(
                    json.dumps(
                        {
                            "source": url,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "bytes": len(payload),
                            "seconds": time.monotonic() - started,
                            "network": "direct" if direct else "system-proxy",
                            "downloaded_at": datetime.now(UTC).isoformat(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(receipt.read_text(), flush=True)
                break
            if target.exists():
                break
        else:
            raise RuntimeError("下载失败: " + "; ".join(errors))
    from datasets import Dataset

    rows = list(Dataset.from_file(str(target)))[: cfg["download"]["count"]]
    subset = root / f"first-{len(rows)}.jsonl"
    text = "".join(
        json.dumps({"id": i, **row}, ensure_ascii=False) + "\n" for i, row in enumerate(rows)
    )
    if subset.exists():
        assert subset.read_text(encoding="utf-8") == text
    else:
        subset.write_text(text, encoding="utf-8")
    pipeline = str(__import__("pathlib").Path(__file__).relative_to(PROJECT_ROOT)).replace(
        "\\", "/"
    )
    common = dict(
        name=cfg["name"],
        source=f"{cfg['source']}@{cfg['revision']}:{cfg['split']}",
        license=cfg["license"],
        pipeline=pipeline,
    )
    register_files(
        [target], parents=[], selection={"method": "fixed_revision_full_split"}, **common
    )
    register_files([receipt], parents=[], selection={"method": "download_receipt"}, **common)
    register_files(
        [subset], parents=[target], selection={"method": "first_n", "count": len(rows)}, **common
    )
    print(f"已登记原始数据与 {len(rows)} 条子集: {subset}")


if __name__ == "__main__":
    main()
