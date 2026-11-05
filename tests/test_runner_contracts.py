"""Selected transport and immutable-input evidence contracts, without network I/O."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_contracts", ROOT / "aigate_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def write_config(tmp_path, *, via="chat", model=""):
    config = tmp_path / "config.ini"
    config.write_text(
        "[aigate]\nurl=https://gate.invalid\njobs=1\nretries=0\n"
        "[pii]\nfolder=pii\n"
        f"[injection]\nfolder=injection\nvia={via}\nmodel={model}\n",
        encoding="utf-8",
    )
    return config


def test_check_chat_requires_model_even_when_both_preflight_samples_pass(
    tmp_path, monkeypatch, capsys
):
    config = write_config(tmp_path)
    calls = []

    def preflight(self, text, row_id):
        calls.append(row_id)
        if row_id == "check-pii":
            return {"decision": "mask", "text": "[PII]"}
        return {"decision": "block"}

    monkeypatch.setattr(ev.Gate, "preflight", preflight)
    monkeypatch.setattr(
        ev.Gate, "post", lambda *args: pytest.fail("A missing model must not dispatch chat")
    )
    assert ev.main(["check", "--config", str(config)]) == 1
    assert calls == ["check-pii", "check-injection"]
    assert "[injection] model" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("preflight_decision", "chat_status", "chat_body", "expected_exit"),
    [
        ("allow", 403, {"error": {"code": "REQUEST_BLOCKED"}}, 0),
        ("block", 200, {"choices": [{"message": {"content": "OK"}}]}, 1),
    ],
)
def test_check_chat_result_is_authoritative_for_the_selected_transport(
    tmp_path, monkeypatch, preflight_decision, chat_status, chat_body, expected_exit
):
    config = write_config(tmp_path, model="test-chat-model")
    sent = []

    def preflight(self, text, row_id):
        if row_id == "check-pii":
            return {"decision": "mask", "text": "[PII]"}
        return {"decision": preflight_decision}

    def post(self, path, body):
        sent.append((path, body))
        return chat_status, chat_body, {}

    monkeypatch.setattr(ev.Gate, "preflight", preflight)
    monkeypatch.setattr(ev.Gate, "post", post)
    assert ev.main(["check", "--config", str(config)]) == expected_exit
    assert len(sent) == 1
    path, body = sent[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "test-chat-model"
    assert any(
        message["role"] == "tool" and message["content"] == ev.INJECTION_SAMPLE
        for message in body["messages"]
    )


@pytest.mark.parametrize("command", ["pii", "injection"])
def test_report_hashes_loaded_input_before_dispatch_even_if_file_changes_during_work(
    command, tmp_path, monkeypatch
):
    folder = tmp_path / command
    folder.mkdir()
    dataset = folder / "mutable.jsonl"
    original_row = {
        "id": "original-input",
        "text": "This loaded reference text contains no personal details.",
        "entities": [],
        "label": "benign",
    }
    original_bytes = (json.dumps(original_row) + "\n").encode("utf-8")
    dataset.write_bytes(original_bytes)
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    config = write_config(tmp_path, via="preflight")
    sent = []

    def preflight(self, text, row_id):
        sent.append((text, row_id))
        replacement = {
            "id": "different-input",
            "text": "A concurrently saved replacement must not become the run provenance.",
            "entities": [],
            "label": "benign",
        }
        dataset.write_text(json.dumps(replacement) + "\n", encoding="utf-8")
        return {"decision": "allow", "reason": "server_policy_passed"}

    monkeypatch.setattr(ev.Gate, "preflight", preflight)
    assert ev.main([command, "--config", str(config)]) == 0
    assert sent == [(original_row["text"], original_row["id"])]
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() != original_hash
    summary = json.loads(
        (tmp_path / "results" / f"{command}.json").read_text(encoding="utf-8")
    )
    assert summary["meta"]["dataset_sha256"] == {"mutable.jsonl": original_hash}
    assert summary["meta"]["rows"] == 1
    detail = json.loads(
        (tmp_path / "results" / f"{command}.raw.jsonl").read_text(encoding="utf-8")
    )
    assert detail["id"] == "original-input"
    assert detail["decision"] == "allow"
