"""仅在 pytest 临时目录验证登记接口，不在工作区生成演示数据。"""

import json

import pytest

from sft import registry


@pytest.fixture
def registered_files(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(registry, "DATA_DIR", data)
    monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pipeline.py").write_text("# temporary test fixture\n", encoding="utf-8")
    raw = data / "raw" / "fixture.jsonl"
    train = data / "processed" / "fixture" / "train.jsonl"
    cases = train.with_name("cases.jsonl")
    for path in (raw, train, cases):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"fixture": 1}\n', encoding="utf-8")
    options = dict(
        name="fixture",
        source="pytest",
        license="test_fixture",
        pipeline="pipeline.py",
        selection={"method": "all"},
    )
    registry.register_files([raw], parents=[], **options)
    registry.register_files([train, cases], parents=[raw], **options)
    return data, raw, train, cases, options


def test_registration_records_real_files_and_lineage_idempotently(registered_files):
    data, raw, train, cases, options = registered_files
    manifest = data / "manifest.json"
    before = manifest.read_bytes()
    entries = json.loads(before)["datasets"]
    assert all(row["records"] == 1 and len(row["sha256"]) == 64 for row in entries)
    assert entries[1]["parents"] == [
        {
            "local_path": "raw/fixture.jsonl",
            "sha256": entries[0]["sha256"],
        }
    ]
    registry.register_files([train, cases], parents=[raw], **options)
    assert manifest.read_bytes() == before
    assert registry.check_manifest() == []


def test_audit_reports_unregistered_missing_and_changed_files_without_writing(registered_files):
    data, raw, train, cases, _ = registered_files
    before = (data / "manifest.json").read_bytes()
    train.write_bytes(train.read_bytes() + b"\n")
    cases.unlink()
    raw.with_name("unregistered.txt").write_text("unknown", encoding="utf-8")
    errors = "\n".join(registry.check_manifest())
    assert "未登记" in errors and "缺失" in errors and "不一致" in errors
    assert (data / "manifest.json").read_bytes() == before


def test_incomplete_provenance_is_never_silently_rewritten(registered_files):
    data, raw, train, cases, options = registered_files
    path = data / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["datasets"][1]["source"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes()
    assert any("登记缺少 source" in error for error in registry.check_manifest())
    with pytest.raises(ValueError, match="已有登记不一致"):
        registry.register_files([train, cases], parents=[raw], **options)
    assert path.read_bytes() == before


def test_lineage_cannot_point_to_an_unregistered_parent(registered_files):
    data, _, _, _, _ = registered_files
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"].pop(0)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any("父数据登记不一致" in error for error in registry.check_manifest())
