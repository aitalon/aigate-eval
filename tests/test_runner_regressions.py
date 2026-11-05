"""Offline regressions: bad responses must never improve quality metrics."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_regressions", ROOT / "aigate_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def gate_for(status, body):
    gate = ev.Gate(argparse.Namespace(
        url="https://gate.invalid", timeout=1, retries=0, token="", integration="",
        header=[], insecure=False, ca="",
    ))
    gate.post = lambda *a: (status, body, {})
    return gate


@pytest.mark.parametrize("body", [
    {"detail": [{"loc": ["body", "text"], "msg": "required"}]},
    {"error": {"code": "PRIVACY_INPUT_UNSUPPORTED"}},
    {"error": {"type": "local_preflight_error", "scan_truncated": True}},
    {"error": {"type": "local_preflight_error", "message": "could not scan"}},
])
def test_validation_or_incomplete_422_is_an_error(body):
    gate = gate_for(422, body)
    assert gate.preflight("text", "one")["decision"] == "error"
    assert ev.classify_chat(gate, {"text": "text"}, "test")["decision"] == "error"


def test_bridge_finding_is_a_real_block():
    body = {"error": {"type": "local_preflight_error", "blocked_by": {
        "category": "prompt_injection", "part": "tool_result", "detector": "test",
    }, "scan_truncated": False}}
    assert gate_for(422, body).preflight("text", "one")["decision"] == "block"


@pytest.mark.parametrize("reason", ["server_scan_incomplete", "server_mask_invalid", "server_rate_limit_exceeded"])
def test_fail_closed_server_response_is_not_detection(reason):
    assert gate_for(200, {"decision": "block", "reason": reason}).preflight("text", "one")["decision"] == "error"


@pytest.mark.parametrize("text", [None, "", 42])
def test_invalid_mask_response_is_an_error(text):
    assert gate_for(200, {"decision": "mask", "text": text}).preflight("text", "one")["decision"] == "error"


@pytest.mark.parametrize("entity", [
    {"start": 0.5, "end": 2, "type": "PII.PERSON"},
    {"start": True, "end": 2, "type": "PII.PERSON"},
    {"start": 0, "end": 2, "type": ""},
    {"start": 0, "end": 2, "type": "PII.PERSON", "text": "wrong"},
    None,
])
def test_invalid_annotations_rejected(entity):
    with pytest.raises(SystemExit):
        ev.check_entities({"id": "bad", "text": "abc", "entities": [entity]})


def test_datasets_works_without_config_or_credentials(tmp_path, capsys):
    for suite, row in [
        ("pii", {"text": "No personal data.", "entities": []}),
        ("injection", {"text": "The build passed.", "label": "benign"}),
    ]:
        folder = tmp_path / "benchmarks" / suite
        folder.mkdir(parents=True)
        (folder / "custom.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert ev.main(["datasets", "--config", str(tmp_path / "missing.ini")]) == 0
    inventory = json.loads(capsys.readouterr().out)["datasets"]
    assert [item["rows"] for item in inventory] == [1, 1]
    assert len(inventory[0]["sha256"]["custom.jsonl"]) == 64


@pytest.mark.parametrize("command", ["pii", "injection"])
def test_api_errors_return_nonzero_and_keep_reports(command, tmp_path, monkeypatch):
    for suite in ("pii", "injection"):
        folder = tmp_path / suite
        folder.mkdir()
        row = {"id": "test", "text": "Clean text", "entities": [], "label": "benign"}
        (folder / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = tmp_path / "config.ini"
    config.write_text("[aigate]\nurl=https://gate.invalid\n[pii]\nfolder=pii\n[injection]\nfolder=injection\n", encoding="utf-8")
    monkeypatch.setattr(ev.Gate, "preflight", lambda *a: {"decision": "error", "reason": "http_401"})
    assert ev.main([command, "--config", str(config)]) == 1
    report = json.loads((tmp_path / "results" / f"{command}.json").read_text(encoding="utf-8"))
    assert report["decisions"] == {"error": 1}
    assert report["meta"]["dataset_sha256"]


def test_check_fails_when_injection_is_allowed(tmp_path, monkeypatch):
    config = tmp_path / "config.ini"
    config.write_text("[aigate]\nurl=https://gate.invalid\n", encoding="utf-8")
    monkeypatch.setattr(ev.Gate, "preflight", lambda self, text, row_id: (
        {"decision": "mask", "text": "[PII]"} if row_id == "check-pii" else {"decision": "allow"}
    ))
    assert ev.main(["check", "--config", str(config)]) == 1


def test_category_slices_and_raw_metadata():
    row = {"id": "x", "source": "s", "text": "x", "category": "role_spoofing", "_label": "malicious"}
    result = ev.score_injection([row], [{"decision": "block"}])
    assert result["groups"]["category:role_spoofing"] == {"tp": 1}
    assert result["details"][0]["category"] == "role_spoofing"


def test_chat_incomplete_scan_is_error_even_with_block_code():
    gate = gate_for(403, {"error": {"code": "REQUEST_BLOCKED", "message": "Проверка неполна: sensitivecore_unavailable"}})
    assert ev.classify_chat(gate, {"text": "text"}, "test")["decision"] == "error"


def test_unchanged_mask_is_invalid():
    assert gate_for(200, {"decision": "mask", "text": "original"}).preflight("original", "x")["decision"] == "error"


def test_local_block_fallback_does_not_override_truncation():
    assert not ev._local_block({"decision": "block", "error": {"type": "local_preflight_error", "scan_truncated": True}})


def test_preflight_check_never_calls_chat_even_when_model_is_set(tmp_path, monkeypatch):
    config = tmp_path / "config.ini"
    config.write_text("[aigate]\nurl=https://gate.invalid\n[injection]\nmodel=test\nvia=preflight\n", encoding="utf-8")
    monkeypatch.setattr(ev.Gate, "preflight", lambda self, text, row_id: (
        {"decision": "mask", "text": "[PII]"} if row_id == "check-pii" else {"decision": "block"}
    ))
    monkeypatch.setattr(ev, "classify_chat", lambda *a: pytest.fail("preflight must not call the LLM"))
    assert ev.main(["check", "--config", str(config)]) == 0


def test_existing_injection_marker_with_only_pii_mask_is_not_detection(tmp_path, monkeypatch):
    folder = tmp_path / "injection"
    folder.mkdir()
    row = {"id": "literal", "text": "Marker [INJECTION_REDACTED], email test@example.com", "label": "benign"}
    (folder / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = tmp_path / "config.ini"
    config.write_text("[aigate]\nurl=https://gate.invalid\n[injection]\nfolder=injection\n", encoding="utf-8")
    monkeypatch.setattr(ev.Gate, "preflight", lambda *a: {"decision": "mask", "text": "Marker [INJECTION_REDACTED], email [PII]"})
    assert ev.main(["injection", "--config", str(config)]) == 0
    result = json.loads((tmp_path / "results/injection.json").read_text(encoding="utf-8"))
    assert result["groups"]["all"] == {"tn": 1}


@pytest.mark.parametrize("original,masked,start,end", [
    ("phone +1 650-555-0126", "phone [REDACTED_PHONE_16505550126]", 6, 21),
    ("CVC 271", "CVC [REDACTED_CVC_271]", 4, 7),
    ("passport 40 19 638205", "passport [REDACTED_PASSPORT_4019638205]", 9, 21),
    ("password ERASED", "password [SECRET_REDACTED]", 9, 15),
    ("name PERSON", "name [PERSON_REDACTED]", 5, 11),
    ("CVC CVC", "CVC [CVC_REDACTED]", 4, 7),
])
def test_opaque_redaction_ids_do_not_leak_matching_original_digits(original, masked, start, end):
    kept = ev.kept_mask(original, masked)
    assert all(kept[:start])
    assert not any(kept[start:end])


@pytest.mark.parametrize("text", ["Literal [REDACTED_PHONE_123] in a manual", "Marker [CUSTOM_123]"])
def test_original_or_unknown_markers_are_not_assumed_to_hide_data(text):
    assert all(ev.kept_mask(text, text))


def test_unknown_marker_does_not_hide_matching_value():
    kept = ev.kept_mask("CVC 271", "CVC [CUSTOM_271]")
    assert all(kept[-3:])
