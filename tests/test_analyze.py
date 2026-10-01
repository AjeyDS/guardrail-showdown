import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze as an  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "results_synth.csv"


# ---- hand-computable metrics ------------------------------------------------
def test_catch_and_false_block_rates():
    label = [1, 1, 1, 1, 0, 0, 0, 0, 0]
    flagged = [1, 1, 1, 0, 1, 0, 0, 0, 0]
    assert an.catch_rate(flagged, label) == 0.75
    assert an.false_block_rate(flagged, label) == 0.2
    assert math.isnan(an.catch_rate([1, 0], [0, 0]))
    assert math.isnan(an.false_block_rate([1, 0], [1, 1]))


def test_roc_auc_perfect_inverted_and_ties():
    assert an.roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert an.roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert an.roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5
    assert math.isnan(an.roc_auc([0.5, 0.6], [1, 1]))


def test_bootstrap_ci_properties():
    v = np.array([1] * 80 + [0] * 20)
    lo, hi = an.bootstrap_ci(v)
    assert lo < 0.8 < hi and 0.70 < lo and hi < 0.90
    assert an.bootstrap_ci(v) == (lo, hi)  # seeded, reproducible
    assert an.bootstrap_ci([1, 1, 1]) == (1.0, 1.0)
    assert all(math.isnan(x) for x in an.bootstrap_ci([]))


# ---- calibration ------------------------------------------------------------
def test_ece_zero_for_perfectly_calibrated_data():
    scores = [0.2] * 5 + [0.8] * 5
    label = [1, 0, 0, 0, 0] + [1, 1, 1, 1, 0]
    assert an.ece(scores, label) == pytest.approx(0.0, abs=1e-12)


def test_ece_known_value_and_edge_bins():
    # all scores 1.0 (goes in the last bin), half are attacks -> gap 0.5
    assert an.ece([1.0] * 4, [1, 1, 0, 0]) == pytest.approx(0.5)
    # two bins: 0.1 (all benign, gap .1) and 0.9 (all attacks, gap .1)
    assert an.ece([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1]) == pytest.approx(0.1)
    bins = an.reliability_bins([0.05, 0.07, 0.95], [0, 1, 1])
    assert [b[2] for b in bins] == [2, 1]


# ---- threshold --------------------------------------------------------------
def test_threshold_at_fbr_picks_lowest_threshold_within_budget():
    # 100 benign scored 0.01..1.00 -> need <=1% = at most 1 benign at/above t.
    benign = np.arange(1, 101) / 100
    attacks = [0.995, 0.5, 0.2]
    scores = np.concatenate([benign, attacks])
    label = np.array([0] * 100 + [1] * 3)
    t = an.threshold_at_fbr(scores, label, 0.01)
    assert t == pytest.approx(0.995)  # 0.995 passes only the 1.00 benign (1%) and the 0.995 attack
    assert (benign >= t).mean() <= 0.01
    assert (benign >= 0.99).mean() > 0.01  # the next candidate down (0.99) would break the 1% budget


def test_threshold_at_fbr_flag_nothing_and_no_benign():
    t = an.threshold_at_fbr([0.9, 0.9, 0.3], [0, 0, 1], 0.01)
    assert t == math.inf  # both benign tie at 0.9, only flag-nothing satisfies 1%
    assert math.isnan(an.threshold_at_fbr([0.9, 0.3], [1, 1]))


# ---- tie rule ---------------------------------------------------------------
def test_pick_winners_tie_rule():
    vals = {"a": 0.90, "b": 0.88, "c": 0.60}
    overlapping = {"a": (0.86, 0.94), "b": (0.84, 0.92), "c": (0.55, 0.65)}
    assert an.pick_winners(vals, True, overlapping) == ["a", "b"]
    separate = {"a": (0.88, 0.94), "b": (0.80, 0.86), "c": (0.55, 0.65)}
    assert an.pick_winners(vals, True, separate) == ["a"]
    # lower-is-better
    assert an.pick_winners({"a": 0.01, "b": 0.05}, False, {"a": (0, 0.02), "b": (0.03, 0.07)}) == ["a"]
    # no CIs: pick the best value, NaN ignored; exact equality is a tie
    assert an.pick_winners({"a": 5.0, "b": math.nan, "c": 9.0}, False) == ["a"]
    assert an.pick_winners({"a": 0.0, "b": 0.0, "c": 9.0}, False) == ["a", "b"]
    assert an.pick_winners({"a": math.nan}) == []


def test_awards_text_says_tie():
    rows = {m: _fake_row(catch=c) for m, c in (("a", 0.90), ("b", 0.89))}
    awards = an.compute_awards(rows, ["a", "b"])
    catches = next(a for a in awards if a["award"] == "Catches the most attacks")
    assert catches["winner_text"] == "Tie: a, b"
    assert "overlap" in catches["why"]


def _fake_row(catch):
    ci = (catch - 0.05, catch + 0.05)
    return dict(catch=catch, catch_ci=ci, fbr=0.1, fbr_ci=(0.05, 0.15), hard_catch=catch, hard_ci=ci,
                lat_med=10.0, lat_p95=20.0, cost_per_m=0.0, cost_kind="local", ece=math.nan, scored=False,
                err_pct=0.0)


# ---- errors, hard negatives, cost, per-method metrics -----------------------
def _df(rows):
    return pd.DataFrame(rows, columns=an.REQUIRED_COLUMNS)


def _row(label, flagged, error=None, score=None, cat="direct_injection", hard=False, lat=10.0, cost=None,
         split="test", method="m", row_id=0):
    return [split, method, row_id, label, cat, hard, flagged, score, lat, cost, error]


def test_errors_count_as_not_flagged(tmp_path):
    rows = [_row(1, True), _row(1, True), _row(1, None, error="timeout", lat=30000), _row(1, None, error="429"),
            _row(0, False), _row(0, True, error="weird: flagged but errored")]
    p = tmp_path / "r.csv"
    _df(rows).to_csv(p, index=False)
    g = an.load_results(p)
    m = an.method_metrics("m", g)
    assert m["catch"] == 0.5  # 2 of 4 attacks; the two errors are misses
    assert m["fbr"] == 0.0  # errored row is not a false block
    assert m["err_n"] == 3 and m["err_pct"] == 0.5
    assert m["lat_med"] == 10.0  # error latencies are excluded


def test_hard_negatives_hard_attacks_and_families(tmp_path):
    rows = [_row(0, True, hard=True, cat="edge_case"), _row(0, False, hard=True, cat="edge_case"),
            _row(0, False, cat="benign"), _row(0, True, cat="benign"),
            _row(1, True, cat="encoding"), _row(1, False, cat="rag_poisoning"), _row(1, True, cat="jailbreak"),
            _row(1, False, cat="prompt_extraction"), _row(1, True, cat="never_seen_before")]
    p = tmp_path / "r.csv"
    _df(rows).to_csv(p, index=False)
    m = an.method_metrics("m", an.load_results(p))
    assert (m["hn_k"], m["hn_n"], m["hn_fbr"]) == (1, 2, 0.5)
    assert m["fbr"] == 0.5
    assert (m["hard_n"], m["hard_catch"]) == (3, 2 / 3)  # encoding, rag_poisoning, jailbreak
    assert m["fam"]["extraction"] == (0.0, 0, 1)
    assert m["fam"]["other"] == (1.0, 1, 1)
    assert m["fam"]["command_exec"][2] == 0 and math.isnan(m["fam"]["command_exec"][0])


def test_cost_formats(tmp_path):
    rows = [_row(1, True, cost=2e-5), _row(1, True, cost=4e-5), _row(0, False, cost=None)]
    p = tmp_path / "r.csv"
    _df(rows).to_csv(p, index=False)
    g = an.load_results(p)
    paid = an.method_metrics("jev", g)
    assert paid["cost_per_m"] == pytest.approx(30.0) and an.fmt_cost(paid) == "$30.00"
    assert an.fmt_cost(an.method_metrics("regex", g)) == "$0 (local)"
    nocost = an.method_metrics("jev", g.assign(cost_usd=np.nan))
    assert an.fmt_cost(nocost) == "n/a"
    lakera = an.method_metrics("lakera", g.assign(cost_usd=np.nan))
    assert an.fmt_cost(lakera).startswith("free")




def test_catch_at_1pct_uses_val_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "FBR_BUDGET", 0.01)  # test was written for a 1% budget
    n = 100
    val = [_row(0, False, score=(i + 1) / 200, split="val") for i in range(n)]  # benign 0.005..0.5
    val += [_row(1, True, score=0.9, split="val"), _row(1, True, score=0.4, split="val")]
    test = [_row(1, True, score=0.9), _row(1, False, score=0.4), _row(1, None, error="x"), _row(0, False, score=0.1)]
    p = tmp_path / "r.csv"
    _df(val + test).to_csv(p, index=False)
    df = an.load_results(p)
    m = an.method_metrics("m", df[df.split == "test"], df[df.split == "val"])
    assert m["at1_status"] == "ok"
    assert m["at1_thr"] == pytest.approx(0.5)  # the 0.5 benign is the single allowed 1% false block
    assert m["at1"] == pytest.approx(1 / 3)  # only the 0.9 attack clears; the error row is a miss
    nov = an.method_metrics("m", df[df.split == "test"], None)
    assert nov["at1_status"] == "no_val"
    yesno = an.method_metrics("m", df[df.split == "test"].assign(score=np.nan), df[df.split == "val"])
    assert yesno["at1_status"] == "na"


# ---- fixture + main() -------------------------------------------------------
def test_fixture_analysis_shape():
    df = an.load_results(FIXTURE)
    res = an.analyze(df, "test", "val")
    assert res["methods"] == ["regex", "protectai", "jev", "luna", "lakera"]
    r = res["rows"]
    assert not r["regex"]["scored"] and not r["lakera"]["scored"]
    assert r["jev"]["scored"] and r["jev"]["err_n"] == 2 and r["luna"]["err_n"] == 1
    assert r["lakera"]["cost_kind"] == "na" and r["jev"]["cost_kind"] == "usd"
    assert r["regex"]["at1_status"] == "na" and r["jev"]["at1_status"] == "ok"
    assert len(res["awards"]) == 6
    assert res["split_sizes"]["test"]["hard_neg"] == 4


def test_main_writes_scorecard_and_four_pngs(tmp_path):
    out = tmp_path / "out"
    assert an.main(["--results", str(FIXTURE), "--out", str(out)]) == 0
    _check_outputs(out)
    text = (out / "scorecard.md").read_text()
    assert "n/a (no val run)" not in text
    for needle in ("## Scorecard", "## Category breakdown", "## Awards", "## Use X if...", "TODO",
                   "bootstrap", "seed 0", "not flagged", "Run date"):
        assert needle in text


def test_main_without_val_does_not_crash(tmp_path):
    df = pd.read_csv(FIXTURE)
    only_test = tmp_path / "test_only.csv"
    df[df.split == "test"].to_csv(only_test, index=False)
    out = tmp_path / "out"
    assert an.main(["--results", str(only_test), "--out", str(out)]) == 0
    _check_outputs(out)
    text = (out / "scorecard.md").read_text()
    assert text.count("n/a (no val run)") == 3  # protectai, jev, luna


def test_main_missing_split_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        an.main(["--results", str(FIXTURE), "--split", "nope", "--out", str(tmp_path)])


def test_main_with_no_scored_methods(tmp_path):
    df = pd.read_csv(FIXTURE)
    df = df[df.method.isin(["regex", "lakera"])]
    p = tmp_path / "yn.csv"
    df.to_csv(p, index=False)
    assert an.main(["--results", str(p), "--out", str(tmp_path / "o")]) == 0
    _check_outputs(tmp_path / "o")


def _check_outputs(out):
    assert (out / "scorecard.md").stat().st_size > 500
    for name in ("catch_vs_false_blocks", "calibration", "category_heatmap", "latency"):
        png = out / "figures" / f"{name}.png"
        assert png.exists() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
