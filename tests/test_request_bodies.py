"""Byte-for-byte invariance of what Jev and Luna send for the prompt_injection task.

tests/fixtures/request_bodies.json was captured from the code BEFORE tasks existed
(monkeypatching the HTTP layer and recording the JSON for one fixed prompt). The
published results depend on these bodies, so they must never change silently.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardrails import _http, jev, llm_judge, registry  # noqa: E402
from guardrails.task import load_task  # noqa: E402

FROZEN = json.loads((Path(__file__).parent / "fixtures" / "request_bodies.json").read_text(encoding="utf-8"))


class FakeResp:
    status_code = 200
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


@pytest.fixture
def sent(monkeypatch):
    """Record every request; answer like Jev for the decisions URL and like Luna otherwise."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(_http.time, "sleep", lambda s: None)
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        key = next(iter(json["questions"])) if "questions" in json else None
        return FakeResp({"answers": {key: {"noul": 0.9}}, "choices": [
            {"message": {"content": '{"attack": true, "confidence": 90}'}}]})

    monkeypatch.setattr(_http.requests, "post", post)
    return calls


def test_jev_body_is_unchanged(sent):
    jev.make().check(FROZEN["prompt"])
    assert sent[0]["url"] == FROZEN["jev"]["url"]
    assert json.dumps(sent[0]["json"]) == json.dumps(FROZEN["jev"]["json"])  # same keys, same order


def test_luna_body_is_unchanged(sent):
    llm_judge.make().check(FROZEN["prompt"])
    assert sent[0]["url"] == FROZEN["luna"]["url"]
    assert json.dumps(sent[0]["json"]) == json.dumps(FROZEN["luna"]["json"])


def test_explicit_task_gives_the_same_bodies(sent):
    task = load_task("prompt_injection")
    registry.load("jev", task).check(FROZEN["prompt"])
    registry.load("luna", task).check(FROZEN["prompt"])
    assert json.dumps(sent[0]["json"]) == json.dumps(FROZEN["jev"]["json"])
    assert json.dumps(sent[1]["json"]) == json.dumps(FROZEN["luna"]["json"])


def test_other_tasks_flow_into_both_bodies(sent, tmp_path):
    (tmp_path / "toxicity.toml").write_text(
        (Path(__file__).resolve().parents[1] / "tasks" / "_template.toml").read_text())
    task = load_task("toxicity", tmp_path)
    jev.make(task).check("you are awful")
    llm_judge.make(task).check("you are awful")
    j, l = sent[0]["json"], sent[1]["json"]
    assert j["state"] == {"message": "you are awful"}
    q = j["questions"]["is_toxic"]
    assert q["instructions"] == task.question
    assert q["criteria"] == {"true": task.criteria_true, "false": task.criteria_false}
    system = l["messages"][0]["content"]
    assert system.startswith("You are a content moderation classifier for a chat product.\n")
    assert 'Answer toxic=true when: ' + task.criteria_true in system
    assert '{"message": "..."}' in system and '{"toxic": true|false,' in system
    assert json.loads(l["messages"][1]["content"]) == {"message": "you are awful"}


def test_judge_parses_the_task_answer_key(monkeypatch, tmp_path):
    (tmp_path / "toxicity.toml").write_text(
        (Path(__file__).resolve().parents[1] / "tasks" / "_template.toml").read_text())
    task = load_task("toxicity", tmp_path)
    assert llm_judge._parse('{"toxic": true, "confidence": 80}', task.answer_key) == (True, 80.0)
    with pytest.raises(ValueError, match="missing boolean 'toxic'"):
        llm_judge._parse('{"attack": true, "confidence": 80}', task.answer_key)
    assert llm_judge._parse('{"attack": false, "confidence": 5}') == (False, 5.0)  # default key unchanged
