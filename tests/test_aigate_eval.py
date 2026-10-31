"""aigate_eval.py against a fake AIGate.

The fake follows the public contracts the script relies on: /v1/preflight
(allow / mask with text / block, 429 with Retry-After) and
/v1/chat/completions (403 REQUEST_BLOCKED before the model).
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("aigate_eval", ROOT / "aigate_eval.py")
aigate_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aigate_eval)

TOKEN = "test-token"
PII = re.compile(r"\+7[\d \-]{10,16}\d|[\w.]+@[\w.]+\w|\d{4} \d{6}|[А-Я][а-я]+ой Анне Сергеевне")
ATTACK = re.compile(r"<INFORMATION>|игнорируй предыдущие инструкции", re.IGNORECASE)


class FakeGate(BaseHTTPRequestHandler):
    throttle_once = True
    seen: list = []

    def log_message(self, *args):
        pass

    def _send(self, status, body, headers=None):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self._send(401, {"error": {"code": "HTTP_ERROR", "message": "Authentication required"}})
        if type(self).throttle_once:
            type(self).throttle_once = False
            return self._send(429, {"ok": False}, {"Retry-After": "0.5"})
        type(self).seen.append((self.path, dict(self.headers), body))
        if self.path == "/v1/preflight":
            assert set(body) == {"text", "site", "domain", "submission_id"}
            assert re.fullmatch(r"[A-Za-z0-9._-]{1,128}", body["submission_id"])
            if ATTACK.search(body["text"]):
                return self._send(200, {"ok": True, "decision": "block", "reason": "server_policy_blocked", "finding_count": 1})
            masked = PII.sub("[PII]", body["text"])
            if masked != body["text"]:
                return self._send(200, {"ok": True, "decision": "mask", "reason": "server_policy_masked", "finding_count": 1, "text": masked})
            return self._send(200, {"ok": True, "decision": "allow", "reason": "server_policy_passed", "finding_count": 0})
        if self.path == "/v1/chat/completions":
            if ATTACK.search(json.dumps(body, ensure_ascii=False)):
                return self._send(403, {"error": {"code": "REQUEST_BLOCKED", "message": "Request blocked by policy"}})
            if any(m["role"] == "tool" for m in body["messages"]):
                # A thinking-mode provider rejects tool history the gate already let through.
                return self._send(400, {"error": {"code": "HTTP_ERROR", "message": "Upstream LLM API error (400): pass reasoning_content"}})
            return self._send(200, {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        return self._send(404, {"error": {"code": "HTTP_ERROR", "message": "not found"}})


@pytest.fixture()
def gate_url():
    FakeGate.throttle_once = True
    FakeGate.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGate)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@pytest.fixture()
def workdir(tmp_path):
    for kind in ("pii", "injection"):
        target = tmp_path / "benchmarks" / kind
        target.mkdir(parents=True)
        shutil.copy(ROOT / "benchmarks" / kind / "example.jsonl", target / "example.jsonl")
    return tmp_path


def write_config(workdir, url, *, token=TOKEN, via="preflight", extra=""):
    path = workdir / "config.ini"
    path.write_text(
        f"[aigate]\nurl = {url}\ntoken = {token}\njobs = 2\nretries = 3\n{extra}\n"
        f"[injection]\nvia = {via}\nmodel = eval-model\n",
        encoding="utf-8",
    )
    return path


def test_all_runs_both_checks_from_the_config_folders(gate_url, workdir):
    config = write_config(workdir, gate_url, extra="[headers]\nX-Request-Source = aigate-eval\n")
    assert aigate_eval.main(["--config", str(config)]) == 0
    pii = json.loads((workdir / "results" / "pii.json").read_text())
    # Phone, passport, email and one of the two names (the fake masks only one name form).
    assert pii["total"]["entities"] == 5 and pii["total"]["protected"] == 4
    assert pii["by_type"]["PII.PERSON"] == {"protected": 1, "entities": 2}
    assert pii["total"]["clean_docs"] == 1 and pii["total"]["false_alarm"] == 0
    assert pii["total"]["noise_rows"] == 1 and pii["total"]["noise_changed"] == 0
    assert pii["meta"]["throttled"] == 1 and pii["meta"]["files"] == ["example.jsonl"]
    assert "Защищено сущностей | 80.0%" in (workdir / "results" / "pii.md").read_text()
    injection = json.loads((workdir / "results" / "injection.json").read_text())
    assert injection["groups"]["all"] == {"tp": 2, "tn": 3}
    assert all(headers.get("X-Request-Source") == "aigate-eval" for _, headers, _ in FakeGate.seen)


def test_injection_via_chat_sends_tool_results_and_declarations(gate_url, workdir):
    config = write_config(workdir, gate_url, via="chat")
    assert aigate_eval.main(["injection", "--config", str(config)]) == 0
    groups = json.loads((workdir / "results" / "injection.json").read_text())["groups"]
    assert groups["all"] == {"tp": 2, "tn": 3} and groups["language:ru"] == {"tp": 1, "tn": 1}
    sent = [json.dumps(body, ensure_ascii=False) for _, _, body in FakeGate.seen]
    assert any('"role": "tool"' in body for body in sent)
    assert any('"name": "create_note"' in body and '"tools"' in body for body in sent)
    assert all('"max_tokens": 1' in body for body in sent)


def test_token_from_environment_and_bad_token_reported(gate_url, workdir, monkeypatch, capsys):
    config = write_config(workdir, gate_url, token="")
    monkeypatch.setenv("AIGATE_TOKEN", "wrong")
    assert aigate_eval.main(["check", "--config", str(config)]) == 1
    assert "Authentication required" in capsys.readouterr().out


def test_partially_labeled_sets_do_not_count_false_alarms(gate_url, workdir):
    folder = workdir / "benchmarks" / "pii"
    clean_with_phone = {"id": "p-1", "text": "Звоните +7 916 555-12-34", "entities": []}
    (folder / "names_only.jsonl").write_text(json.dumps(clean_with_phone, ensure_ascii=False) + "\n", encoding="utf-8")
    (workdir / "benchmarks" / "open_datasets_manifest.json").write_text(
        json.dumps({"datasets": [{"file": "pii/names_only.jsonl", "labeled": "partial"}]}), encoding="utf-8"
    )
    config = write_config(workdir, gate_url)
    assert aigate_eval.main(["pii", "--config", str(config), "--limit", "1"]) == 0
    summary = json.loads((workdir / "results" / "pii.json").read_text())
    # --limit applies per file: the first example row plus the partial-label row.
    assert summary["meta"]["rows"] == 2
    assert summary["by_source"]["names_only"] == {"clean_unscored": 1}
    assert summary["total"].get("false_alarm", 0) == 0


def test_missing_config_points_to_the_example(tmp_path):
    with pytest.raises(SystemExit, match="config.example.ini"):
        aigate_eval.main(["--config", str(tmp_path / "config.ini")])


def test_kept_mask_marks_only_surviving_characters():
    kept = aigate_eval.kept_mask("тел. +7 903 123-45-67, ок", "тел. [PII], ок")
    assert bytes(kept[:5]) == b"\1" * 5
    assert not any(kept[5:21])
    assert all(kept[21:])


def test_rejects_entities_outside_the_text():
    with pytest.raises(SystemExit, match="outside the text"):
        aigate_eval.check_entities({"id": "x", "text": "abc", "entities": [{"start": 1, "end": 9, "type": "PII.PHONE"}]})
