"""The runner must not mix cached verdicts produced under different definitions."""

import dataclasses
from types import SimpleNamespace

import run_benchmark as rb
from guardrails.task import default_task


def _setup(tmp_path, monkeypatch, tasks):
    monkeypatch.setattr(rb.registry, "discover", lambda *a, **k: {"judge": SimpleNamespace(tasks=tasks)})
    monkeypatch.setattr(rb, "raw_path", lambda split, method, task=None: tmp_path / split / f"{method}.jsonl")
    cache = tmp_path / "test" / "judge.jsonl"
    cache.parent.mkdir(parents=True)
    cache.write_text('{"row_id": 0}\n')
    return cache


def test_adopts_existing_cache_then_blocks_changed_definition(tmp_path, monkeypatch):
    cache = _setup(tmp_path, monkeypatch, ["*"])
    task = default_task()
    assert rb.definition_conflict("judge", "test", task) is None  # adopted
    assert (cache.parent / "judge.definition.json").exists()
    assert rb.definition_conflict("judge", "test", task) is None  # same definition: fine
    changed = dataclasses.replace(task, criteria_true=task.criteria_true + " Also spam.")
    msg = rb.definition_conflict("judge", "test", changed)
    assert msg and "different [definition]" in msg


def test_task_specific_methods_are_not_tracked(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, ["prompt_injection"])
    changed = dataclasses.replace(default_task(), question="Something else?")
    assert rb.definition_conflict("judge", "test", changed) is None
