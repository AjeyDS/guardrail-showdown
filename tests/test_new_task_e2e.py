"""A brand-new task end to end: template-based task file, a CSV with only text,label,
a fake task-agnostic guardrail, run_benchmark -> analyze -> make_visuals.

Everything lives in tmp dirs; nothing touches data/ or results/ of the repo.
"""

import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze as an  # noqa: E402
import make_visuals as mv  # noqa: E402
import run_benchmark as rb  # noqa: E402
from guardrails import registry, task as task_mod  # noqa: E402
from guardrails.base import Verdict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ROWS = [("you are awful", 1), ("I hate you all", 1), ("go away idiot", 1),
        ("thanks, that helps", 0), ("see you tomorrow", 0), ("I disagree with your plan", 0)]


class Fake:
    """Task-agnostic fake: flags text containing an insult word."""

    def __init__(self, name):
        self.name = name

    def check(self, prompt):
        flagged = any(w in prompt for w in ("awful", "hate", "idiot"))
        return Verdict(flagged, 0.9 if flagged else 0.1, 5.0, cost_usd=0.00001, raw={})


@pytest.fixture
def world(tmp_path, monkeypatch):
    # repo layout in tmp: tasks/, data/toxicity/, guardrails/ (fakes), results/ comes from the task
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "toxicity.toml").write_text((ROOT / "tasks" / "_template.toml").read_text())
    data = tmp_path / "data" / "toxicity"
    data.mkdir(parents=True)
    for split in ("val", "test"):
        pd.DataFrame(ROWS, columns=["text", "label"]).to_csv(data / f"{split}.csv", index=False)
    monkeypatch.setattr(task_mod, "ROOT", tmp_path)
    monkeypatch.setattr(task_mod, "TASKS_DIR", tasks)

    gdir = tmp_path / "guardrails"
    gdir.mkdir()
    for method, tasks_line in (("fakeany", '["*"]'), ("fakepi", '["prompt_injection"]')):
        (gdir / f"{method}.py").write_text(textwrap.dedent(f'''
            METHOD = "{method}"
            DISPLAY_NAME = "Fake {method}"
            LOCAL = True
            REQUIRES_KEYS: list[str] = []
            TASKS: list[str] = {tasks_line}
            def make(task=None):
                raise RuntimeError("replaced in test")
        '''))
    monkeypatch.setattr(registry, "GUARDRAILS_DIR", gdir)
    monkeypatch.setattr(rb, "load_guardrail", lambda m, task=None: Fake(m))
    # the runner builds these at import from the real guardrails/: point them at the fakes
    monkeypatch.setattr(rb, "REGISTRY", {m: i.module for m, i in registry.discover().items()})
    monkeypatch.setattr(rb, "API_METHODS", set())
    return tmp_path


def test_new_task_end_to_end(world, capsys):
    # 1. the run: default methods are the ones that support the task; task-specific ones are refused
    assert rb.main(["--task", "toxicity", "--split", "val"]) == 0
    assert rb.main(["--task", "toxicity", "--split", "test"]) == 0
    with pytest.raises(SystemExit):
        rb.main(["--task", "toxicity", "--split", "test", "--methods", "fakepi"])
    assert "do not support task 'toxicity'" in capsys.readouterr().err

    results = world / "results" / "toxicity"
    assert (results / "raw" / "test" / "fakeany.jsonl").exists()
    assert not (results / "raw" / "test" / "fakepi.jsonl").exists()
    df = pd.read_csv(results / "results.csv")
    assert set(df["method"]) == {"fakeany"} and len(df) == 12
    assert set(df["category"].fillna("")) == {""} and not df["is_hard_negative"].any()

    # 2. the scorecard
    assert an.main(["--task", "toxicity"]) == 0
    card = (results / "scorecard.md").read_text()
    assert card.startswith("# Guardrail Showdown: toxicity scorecard (test split)")
    assert "3 toxic messages and 3 civil ones" in card  # labels from the task
    assert "Catches the most toxic messages" in card
    assert "Fake fakeany" not in card and "| fakeany |" in card
    # optional sections are skipped when the task defines no families / hard negatives
    assert "Category breakdown" not in card and "tricky-but" not in card and "hard toxic" not in card
    assert "Jev's verdict" not in card
    assert "**Not applicable to this task**: " in card and "fakepi" in card
    assert (results / "figures" / "catch_vs_false_blocks.png").exists()
    assert not (results / "figures" / "category_heatmap.png").exists()

    # 3. the visuals that need only results + data
    assert mv.main(["--task", "toxicity"]) == 0
    visuals = {p.name for p in (results / "visuals").glob("*.png")}
    assert {"confusion_test.png", "per_1000_messages.png", "speed_vs_cost.png"} <= visuals
    assert "one_sentence.png" not in visuals and "example_prompts.png" not in visuals


def test_prompt_injection_paths_are_unchanged_by_the_task_machinery():
    t = task_mod.load_task("prompt_injection")
    assert t.data_dir == ROOT / "data" and t.results_dir == ROOT / "results"
