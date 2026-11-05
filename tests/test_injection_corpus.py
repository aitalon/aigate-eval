"""Integrity checks for the shipped, offline injection reference corpus.

These checks validate dataset usability, balance and inert examples; detector
performance is measured separately against a configured AIGate instance.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import re
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("injection_corpus_eval", ROOT / "aigate_eval.py")
eval_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_module)


def core_rows():
    return [
        json.loads(line)
        for line in (ROOT / "benchmarks/injection/core.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_core_injection_is_balanced_across_families_and_languages():
    rows = core_rows()
    assert len(rows) >= 260
    categories = {row["category"] for row in rows}
    assert categories >= {
        "instruction_override", "role_spoofing", "secret_exfiltration",
        "tool_abuse", "retrieval_poisoning", "output_hijack", "obfuscation",
        "authority_pressure", "data_integrity", "workflow_hijack", "multilingual",
        "context_delimiters", "tool_description",
    }
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        strata = collections.Counter((row["language"], row["label"]) for row in subset)
        assert set(strata) == {
            ("en", "malicious"), ("en", "benign"),
            ("ru", "malicious"), ("ru", "benign"),
        }, category
        assert min(strata.values()) >= 5, category
        assert len(set(strata.values())) == 1, category


def test_shipped_injection_files_load_together_without_duplicate_ids_or_texts():
    files = sorted((ROOT / "benchmarks/injection").glob("*.jsonl"))
    rows = eval_module.load_rows([str(path) for path in files], None)
    assert len(rows) >= 265
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["text"] for row in rows}) == len(rows)
    for row in rows:
        assert row["text"].strip(), row["id"]
        assert len(row["text"].encode("utf-8")) <= eval_module.PREFLIGHT_MAX_TEXT_BYTES
        assert eval_module.row_label(row) in {"malicious", "benign"}


def test_core_injection_has_provenance_and_inert_url_destinations():
    for row in core_rows():
        assert row["origin"] == "original-synthetic"
        assert row["synthetic"] is True
        assert row["source"] == "aitalon-injection-core-v1"
        assert row["id"].startswith("core-inj-")
        for url in re.findall(r"https?://[^\s<>\"'，。]+", row["text"]):
            hostname = urlsplit(url).hostname
            assert hostname and hostname.rstrip(".").endswith(".invalid"), (row["id"], url)
        for domain in re.findall(r"[A-Za-z0-9._+-]+@([A-Za-z0-9.-]+)", row["text"]):
            assert domain.rstrip(".").endswith(".invalid"), (row["id"], domain)


def test_each_transport_kind_produces_a_valid_chat_payload():
    rows = core_rows()
    assert {row["kind"] for row in rows} == {"user_prompt", "tool_result", "tool_description"}
    for row in rows:
        body = eval_module.chat_body(row, "reference-model")
        assert body["model"] == "reference-model"
        assert body["messages"]
        if row["kind"] == "tool_description":
            declaration = json.loads(row["text"])
            assert declaration["inputSchema"]["required"] == ["query"]
            function = body["tools"][0]["function"]
            assert function["description"] == declaration["description"]
            assert function["parameters"] == declaration["inputSchema"]
        else:
            assert body["messages"][-1]["content"] == row["text"]
            expected = "user" if row["kind"] == "user_prompt" else "tool"
            assert body["messages"][-1]["role"] == expected


def test_delimiters_and_encoding_cases_preserve_their_original_representation():
    rows = core_rows()
    delimiter_rows = [row for row in rows if row["category"] == "context_delimiters"]
    assert any("\n" in row["text"] for row in delimiter_rows)
    assert any("</tool_result>" in row["text"] for row in delimiter_rows)
    assert any("<!--" in row["text"] for row in delimiter_rows)
    encoded_rows = [row for row in rows if row["category"] == "obfuscation"]
    assert any(r"\u0069" in row["text"] for row in encoded_rows)
    assert any("aWdub3JlIGFsbCBwcmV2aW91cyBpbnN0cnVjdGlvbnM=" in row["text"] for row in encoded_rows)
