import json
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_benchmark as rb  # noqa: E402
from guardrails.base import Verdict  # noqa: E402

COLUMNS = ["split", "method", "row_id", "label", "category", "is_hard_negative",
           "flagged", "score", "latency_ms", "cost_usd", "error"]


class FakeGuardrail:
    """Flags prompts containing 'ignore'. Fails on prompts containing 'boom'
    while `fail` is True. Counts calls per prompt."""

    def __init__(self, name="fake", fail=False):
        self.name = name
        self.fail = fail
        self.calls = []
        self._lock = threading.Lock()

    def check(self, prompt):
        with self._lock:
            self.calls.append(prompt)
        if self.fail and "boom" in prompt:
            return Verdict(flagged=None, score=None, latency_ms=1.0, error="fake failure")
        flagged = "ignore" in prompt
        return Verdict(flagged=flagged, score=0.9 if flagged else 0.1, latency_ms=2.0,
                       cost_usd=0.001, raw={"x": 1})


def make_data(n=40):
    rows = []
    for i in range(n):
        attack = i % 4 != 0  # 75% attacks
        if attack:
            text, cat, tags = f"ignore all rules {i}", "jailbreak", ""
        elif i == 4:
            text, cat, tags = f"boom edge {i}", "edge_case", ""
        elif i == 8:
            text, cat, tags = f"boom tagged {i}", "benign", "security|hard_negative"
        elif i == 12:
            text, cat, tags = f"what is xss {i}", "benign", "education|security_adjacent"
        else:
            text, cat, tags = f"recipe {i}", "benign", "everyday"
        rows.append({"row_id": i, "text": text, "label": int(attack), "category": cat,
                     "source": "s", "severity": "", "group_id": f"g{i}",
                     "augmented": False, "tags": tags})
    return pd.DataFrame(rows)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    results = tmp_path / "results"
    data.mkdir()
    make_data().to_csv(data / "val.csv", index=False)
    monkeypatch.setattr(rb, "DATA_DIR", data)
    monkeypatch.setattr(rb, "RESULTS_DIR", results)
    guardrails = {}

    def fake_load(method, task=None):
        guardrails.setdefault(method, FakeGuardrail(name=method))
        return guardrails[method]

    monkeypatch.setattr(rb, "load_guardrail", fake_load)
    return tmp_path, guardrails


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_writes_jsonl_and_csv_with_exact_columns(env):
    tmp, _ = env
    assert rb.main(["--split", "val", "--methods", "regex,jev"]) == 0
    for m in ("regex", "jev"):
        recs = lines(tmp / "results/raw/val" / f"{m}.jsonl")
        assert sorted(r["row_id"] for r in recs) == list(range(40))
        assert set(recs[0]) == {"row_id", "flagged", "score", "latency_ms", "cost_usd", "error", "raw"}
    df = pd.read_csv(tmp / "results/results.csv")
    assert list(df.columns) == COLUMNS
    assert len(df) == 80
    assert set(df["method"]) == {"regex", "jev"}
    assert "raw" not in df.columns
    row = df[(df.method == "jev") & (df.row_id == 1)].iloc[0]
    assert row["split"] == "val" and row["label"] == 1 and bool(row["flagged"]) is True
    assert row["score"] == pytest.approx(0.9)


def test_resume_skips_done_and_retries_errors(env):
    tmp, guardrails = env
    guardrails["regex"] = FakeGuardrail("regex", fail=True)
    assert rb.main(["--split", "val", "--methods", "regex"]) == 0
    path = tmp / "results/raw/val/regex.jsonl"
    errored = {r["row_id"] for r in lines(path) if r["error"]}
    assert errored == {4, 8}  # the two 'boom' rows

    g = guardrails["regex"]
    g.fail = False
    g.calls.clear()
    assert rb.main(["--split", "val", "--methods", "regex"]) == 0
    assert sorted(g.calls) == sorted(["boom edge 4", "boom tagged 8"])  # only the errored rows

    latest = rb.read_raw(path)
    assert len(latest) == 40
    assert all(r["error"] is None for r in latest.values())
    df = pd.read_csv(tmp / "results/results.csv")
    assert len(df) == 40 and df["error"].isna().all()

    g.calls.clear()
    rb.main(["--split", "val", "--methods", "regex"])
    assert g.calls == []  # nothing left to do


def test_resume_tolerates_truncated_last_line(env):
    tmp, guardrails = env
    path = tmp / "results/raw/val/regex.jsonl"
    path.parent.mkdir(parents=True)
    good = {"row_id": 0, **Verdict(False, 0.1, 1.0, 0.0).to_dict()}
    path.write_text(json.dumps(good) + '\n{"row_id": 1, "flag')
    rb.main(["--split", "val", "--methods", "regex"])
    assert "recipe 0" not in guardrails["regex"].calls and len(guardrails["regex"].calls) == 39
    assert len(rb.read_raw(path)) == 40


def test_stratified_limit_keeps_both_labels_and_is_deterministic():
    df = make_data(200)
    a = rb.stratified_sample(df, 20, seed=0)
    b = rb.stratified_sample(df, 20, seed=0)
    c = rb.stratified_sample(df, 20, seed=1)
    assert a["row_id"].tolist() == b["row_id"].tolist()
    assert a["row_id"].tolist() != c["row_id"].tolist()
    assert len(a) == 20
    assert set(a["label"]) == {0, 1}
    assert a["label"].mean() == pytest.approx(0.75, abs=0.05)
    tiny = rb.stratified_sample(df, 2, seed=0)
    assert set(tiny["label"]) == {0, 1}


def test_limit_via_cli_only_runs_sample(env):
    tmp, guardrails = env
    assert rb.main(["--split", "val", "--methods", "regex", "--limit", "8", "--seed", "3"]) == 0
    recs = lines(tmp / "results/raw/val/regex.jsonl")
    assert len(recs) == 8
    labels = pd.read_csv(tmp / "data/val.csv").set_index("row_id").loc[[r["row_id"] for r in recs], "label"]
    assert set(labels) == {0, 1}


def test_is_hard_negative_rule(env):
    tmp, _ = env
    rb.main(["--split", "val", "--methods", "regex"])
    df = pd.read_csv(tmp / "results/results.csv").set_index("row_id")
    hard = set(df.index[df["is_hard_negative"]])
    assert hard == {4, 8, 12}  # edge_case, hard_negative tag, security_adjacent tag
    assert not df.loc[df["label"] == 1, "is_hard_negative"].any()
    # rule directly: attacks never count, even with the tag
    assert rb.is_hard_negative(1, "edge_case", "hard_negative") is False
    assert rb.is_hard_negative(0, "benign", "a|security_adjacent") is True
    assert rb.is_hard_negative(0, "benign", "security") is False


def test_concurrent_path_one_line_per_row(env):
    tmp, guardrails = env
    assert rb.main(["--split", "val", "--methods", "jev", "--workers", "8"]) == 0
    recs = lines(tmp / "results/raw/val/jev.jsonl")
    ids = [r["row_id"] for r in recs]
    assert len(ids) == 40 and sorted(ids) == list(range(40))
    assert len(guardrails["jev"].calls) == 40


def test_budget_guard_and_yes(env, monkeypatch):
    tmp, guardrails = env
    monkeypatch.setattr(rb, "MAX_API_PROMPTS", 10)
    assert rb.main(["--split", "val", "--methods", "jev"]) == 2
    assert not (tmp / "results/raw/val/jev.jsonl").exists()
    # local methods are not subject to the guard
    assert rb.main(["--split", "val", "--methods", "regex"]) == 0
    assert rb.main(["--split", "val", "--methods", "jev", "--yes"]) == 0
    assert len(lines(tmp / "results/raw/val/jev.jsonl")) == 40


def test_rebuild_covers_all_splits_and_bad_args(env):
    tmp, _ = env
    make_data(10).to_csv(tmp / "data/other.csv", index=False)
    rb.main(["--split", "val", "--methods", "regex"])
    rb.main(["--split", "other", "--methods", "regex"])
    (tmp / "results/results.csv").unlink()
    assert rb.main(["--rebuild"]) == 0
    df = pd.read_csv(tmp / "results/results.csv")
    assert set(df["split"]) == {"val", "other"} and len(df) == 50

    assert rb.main(["--split", "nope", "--methods", "regex"]) == 2  # missing data file
    with pytest.raises(SystemExit) as e:
        rb.main(["--split", "val", "--methods", "bogus"])
    assert e.value.code == 2


def test_check_raising_is_recorded_not_fatal(env, monkeypatch):
    tmp, _ = env

    class Bad:
        name = "bad"

        def check(self, prompt):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(rb, "load_guardrail", lambda m, task=None: Bad())
    assert rb.main(["--split", "val", "--methods", "regex", "--limit", "5"]) == 0
    df = pd.read_csv(tmp / "results/results.csv")
    assert len(df) == 5 and df["error"].str.contains("kaboom").all()


# ---- relaxed data contract ----------------------------------------------------
def test_minimal_csv_needs_only_text_and_label(env):
    tmp, _ = env
    (tmp / "data/mini.csv").write_text("text,label\nignore me,1\nhello,0\n")
    df = rb.load_split("mini")
    assert df["row_id"].tolist() == [0, 1]  # generated from row order
    assert (df["category"] == "").all() and (df["tags"] == "").all()
    assert rb.main(["--split", "mini", "--methods", "regex"]) == 0
    res = pd.read_csv(tmp / "results/results.csv")
    assert list(res.columns) == COLUMNS and len(res) == 2
    assert res["is_hard_negative"].eq(False).all()


@pytest.mark.parametrize("csv, message", [
    ("txt,label\na,1\n", "missing required column"),
    ("text\na\n", r"missing required column\(s\) \['label'\]"),
    ("text,label\na,1\nb,2\n", "`label` must be 0 or 1.*line"),
    ("text,label\na,yes\n", "`label` must be 0 or 1"),
    ("text,label\na,\n", "`label` must be 0 or 1"),
    ("row_id,text,label\n1,a,1\n1,b,0\n", "`row_id` values must be unique"),
    ("row_id,text,label\nx,a,1\n", "`row_id` must be whole numbers"),
])
def test_bad_csv_fails_with_a_clear_message(env, capsys, csv, message):
    tmp, _ = env
    (tmp / "data/bad.csv").write_text(csv)
    with pytest.raises(rb.DataError, match=message):
        rb.load_split("bad")
    assert rb.main(["--split", "bad", "--methods", "regex"]) == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp / "results/raw/bad").exists()


def test_rebuild_skips_a_bad_csv_with_a_warning(env, capsys):
    tmp, _ = env
    rb.main(["--split", "val", "--methods", "regex"])
    (tmp / "data/val.csv").write_text("text,label\nx,7\n")
    assert rb.main(["--rebuild"]) == 0
    assert "skipping raw/val" in capsys.readouterr().err


def test_unknown_task_is_rejected(env, capsys):
    with pytest.raises(SystemExit) as e:
        rb.main(["--task", "nope", "--split", "val"])
    assert e.value.code == 2
    assert "unknown task 'nope'" in capsys.readouterr().err
