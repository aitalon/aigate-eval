"""Ensure a clone actually ships the documented data and exact reference bytes."""
import hashlib
import json
from pathlib import Path


def test_builtin_manifest_matches_shipped_files():
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    manifest = json.loads((root / "builtin_manifest.json").read_text(encoding="utf-8"))
    actual = set(root.glob("pii/curated_*.jsonl")) | {root / "injection/core.jsonl"} | set(root.glob("*/example.jsonl"))
    declared = {root / item["file"] for item in manifest["files"]}
    assert declared == actual
    count = 0
    ids = set()
    texts = set()
    for entry in manifest["files"]:
        raw = (root / entry["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        assert len(rows) == entry["rows"]
        for row in rows:
            assert row["id"] not in ids
            ids.add(row["id"])
            # The legacy noise example also demonstrates the unlabelled form of a clean row.
            if row.get("entities", "injection") is not None:
                assert row["text"] not in texts
                texts.add(row["text"])
        count += len(rows)
    assert count == manifest["total_rows"] == 639
