import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import make_visuals as mv  # noqa: E402

EXPECTED = [
    "confusion_test.png", "confusion_hard.png",
    "per_1000_messages.png", "per_1000_messages_square.png",
    "per_1000_messages_hard.png", "per_1000_messages_hard_square.png",
    "one_sentence.png", "one_sentence_square.png",
    "default_vs_tuned.png", "example_prompts.png", "speed_vs_cost.png",
]


# ---- per-1000 expected values ------------------------------------------------
def test_per_1000_expected_values():
    r = mv.per_1000(catch=0.625, fbr=0.028)
    assert r["missed"] == 3.8 and r["missed_icons"] == 4          # 3.75 -> 3.8, icons round half up
    assert r["blocked"] == 27.7 and r["blocked_icons"] == 28       # 0.028 * 990 = 27.72
    assert (r["n_attacks"], r["n_safe"]) == (10, 990)


def test_per_1000_extremes_and_rounding():
    perfect = mv.per_1000(1.0, 0.0)
    assert perfect["missed"] == 0 and perfect["blocked"] == 0 and perfect["blocked_icons"] == 0
    worst = mv.per_1000(0.0, 1.0)
    assert worst["missed_icons"] == 10 and worst["blocked_icons"] == 990
    assert mv.round_half_up(2.5) == 3 and mv.round_half_up(0.49) == 0


# ---- example selection -------------------------------------------------------
ROWS = [  # (row_id, label, n_right, n_wrong)
    (10, 1, 2, 2), (11, 1, 4, 0), (12, 1, 0, 4), (13, 1, 3, 1), (14, 1, 2, 2), (15, 1, 1, 3),
    (16, 1, 2, 2), (17, 1, 2, 2), (18, 1, 3, 1),
    (20, 0, 2, 2), (21, 0, 4, 0), (22, 0, 3, 1), (23, 0, 1, 3), (24, 0, 2, 2), (25, 0, 2, 2),
    (26, 0, 0, 4), (27, 0, 2, 2), (28, 0, 2, 2),
]


def test_select_examples_most_disagreement_ties_by_row_id():
    chosen = mv.select_examples(ROWS)
    assert chosen == [10, 14, 16, 17, 20, 24, 25, 27]  # 2-2 splits first, lowest row_id wins ties
    assert len(chosen) == 8


def test_select_examples_is_deterministic_and_order_independent():
    first = mv.select_examples(ROWS)
    rng = random.Random(7)
    for _ in range(5):
        shuffled = ROWS[:]
        rng.shuffle(shuffled)
        assert mv.select_examples(shuffled) == first


def test_shorten():
    assert mv.shorten("a  b\nc") == "a b c"
    s = mv.shorten("x" * 200, 70)
    assert len(s) == 70 and s.endswith("…")


# ---- one-sentence loader -----------------------------------------------------
def test_definition_effect_and_jsonl_latest(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [
        {"row_id": 1, "flagged": True, "error": None},
        {"row_id": 1, "flagged": None, "error": "timeout"},   # an error never replaces a clean record
        {"row_id": 2, "flagged": False, "error": None},
        {"row_id": 3, "flagged": None, "error": "boom"},
    ]))
    old = mv.load_jsonl_latest(f)
    assert old[1]["flagged"] is True and old[3]["error"] == "boom"
    new = {1: {"flagged": True}, 2: {"flagged": True}, 3: {"flagged": True}, 9: {"flagged": True}}
    labels = pd.Series({1: 1, 2: 1, 3: 0, 4: 1})
    e = mv.definition_effect(old, new, labels)
    assert e["n"] == 3 and e["n_attacks"] == 2 and e["n_safe"] == 1   # row 9 / 4 are not shared
    assert e["catch_old"] == 0.5 and e["catch_new"] == 1.0
    assert e["fb_old"] == 0 and e["fb_new"] == 1                      # errored old record = not flagged


# ---- end to end on synthetic data ---------------------------------------------
def _synthetic(tmp_path):
    rng = np.random.default_rng(0)
    res_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    (res_dir / "raw" / "val").mkdir(parents=True)
    (res_dir / "archive").mkdir()
    data_dir.mkdir()

    def split_rows(n, offset=0):
        return [(offset + i, int(i % 2 == 0), "direct_injection" if i % 2 == 0 else "benign")
                for i in range(n)]

    sizes = {"val": 60, "test": 60, "hard": 56}
    rows = []
    for split, n in sizes.items():
        for method, acc in [("regex", 0.55), ("protectai", 0.8), ("jev", 0.85), ("luna", 0.85),
                            ("lakera", 0.8)]:
            if method == "lakera" and split != "val":
                continue
            for rid, label, cat in split_rows(n):
                right = rng.random() < acc
                flagged = bool(label) if right else not label
                score = None if method in ("regex", "lakera") else float(
                    np.clip((0.75 if flagged else 0.2) + rng.normal(0, 0.1), 0, 1))
                rows.append(dict(split=split, method=method, row_id=rid, label=label, category=cat,
                                 is_hard_negative=False, flagged=flagged, score=score,
                                 latency_ms=float(rng.uniform(1, 500)),
                                 cost_usd=0.0 if method in ("regex", "protectai") else (
                                     None if method == "lakera" else 2e-5),
                                 error=None))
    pd.DataFrame(rows).to_csv(res_dir / "results.csv", index=False)

    def frame(n, source=None):
        out = []
        for rid, label, cat in split_rows(n):
            out.append(dict(row_id=rid, text=f"prompt number {rid} " * 8, label=label, category=cat,
                            source=source or "synthetic", severity="", group_id="", augmented=False,
                            tags=""))
        return pd.DataFrame(out)

    frame(60).to_csv(data_dir / "val.csv", index=False)
    hard = frame(56)
    hard.loc[hard["row_id"] >= 40, "source"] = "hand_written"   # 16 hand-written: 8 attacks, 8 safe
    hard.to_csv(data_dir / "hard.csv", index=False)

    val_ids = list(range(0, 20))
    for m in ("jev", "luna"):
        for name, rate in ((f"archive/val_{m}_narrow_def.jsonl", 0.3), (f"raw/val/{m}.jsonl", 0.9)):
            with open(res_dir / name, "w") as fh:
                for rid in val_ids:
                    fh.write(json.dumps({"row_id": rid, "flagged": bool(rid % 2 == 0 and
                                         rng.random() < rate), "score": 0.5, "error": None}) + "\n")
    return res_dir, data_dir


def test_main_writes_all_pngs(tmp_path, capsys):
    res_dir, data_dir = _synthetic(tmp_path)
    out = tmp_path / "visuals"
    rc = mv.main(["--results", str(res_dir / "results.csv"), "--out", str(out), "--data", str(data_dir)])
    assert rc == 0
    for name in EXPECTED:
        f = out / name
        assert f.exists(), name
        assert f.stat().st_size > 8_000, name
    # sizes: landscape 1600x900, square 1080x1080
    from matplotlib import image as mpimg
    assert mpimg.imread(out / "confusion_test.png").shape[:2] == (900, 1600)
    assert mpimg.imread(out / "per_1000_messages_square.png").shape[:2] == (1080, 1080)
    assert mpimg.imread(out / "one_sentence_square.png").shape[:2] == (1080, 1080)


def test_lakera_is_excluded_from_test_visual_data(tmp_path):
    res_dir, _ = _synthetic(tmp_path)
    d = mv.load_data(res_dir / "results.csv")
    assert "lakera" in set(d["all"]["method"]) and "lakera" not in set(d["nl"]["method"])
    pts = mv.speed_cost_points(d["all"])
    assert np.isnan(pts["lakera"]["cost"]) and pts["lakera"]["lat"] > 0
    assert pts["regex"]["cost"] == 0
