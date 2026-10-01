"""Shareable, plain-English charts for the guardrail benchmark.

    .venv/bin/python make_visuals.py [--task prompt_injection] [--results results/results.csv]
                                     [--out results/visuals] [--data data]

Default paths come from the task's `results_dir` / `data_dir` (tasks/<task>.toml)
and the words for the two classes from its labels. Charts whose inputs are absent
(the one-sentence chart needs results/archive/, the example prompts need a hard
split with hand-written rows) are skipped.

Every number comes from `results/` and `data/` files; rates are computed with the
pure functions in analyze.py so the visuals always agree with the scorecards.
Errored calls count as NOT flagged (same rule as the scorecard).  Lakera has only
partial data, so it is left out of every test/hard visual (it appears, clearly
marked, only on the speed chart using its val-subset latency).
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

import analyze as an  # noqa: E402
from guardrails import registry  # noqa: E402
from guardrails.task import DEFAULT_TASK, Task, TaskError, default_task, load_task, show  # noqa: E402

# --------------------------------------------------------------------------
# House style
# --------------------------------------------------------------------------
BG = "#F7F3EA"
INK = "#1B2A41"
GREY = "#8A94A6"
LIGHT = "#D9DCE3"
AMBER = "#E8A33D"
TEAL = "#2E8B7A"

DISPLAY = {"regex": "Regex", "protectai": "ProtectAI", "jev": "Jev",
           "luna": "Luna (GPT-6)", "lakera": "Lakera Guard"}
SUBJECT = "jev"  # the write-up's subject: slightly bolder, never a "better" colour

DPI = 200
WIDE = (8.0, 4.5)     # 1600 x 900 px
SQUARE = (5.4, 5.4)   # 1080 x 1080 px

N_MESSAGES = 1000     # "if 1,000 messages hit your app ..."
N_ATTACKS = 10        # "... and 10 are attacks"

LAKERA_NOTE = "Lakera Guard is left out: only partial data."
SPLIT_LABEL = {"test": "test set", "hard": "hard set"}
PARTIAL = "lakera"  # left out of the test/hard visuals (partial data), see load_data


def name_of(m: str) -> str:
    if m in DISPLAY:
        return DISPLAY[m]
    try:
        return registry.discover()[m].display_name
    except KeyError:
        return m


def note(nl: pd.DataFrame) -> str:
    """The 'Lakera is left out' footnote, only when Lakera is in the results."""
    return LAKERA_NOTE if nl.attrs.get("lakera_left_out", True) else ""


def join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------
def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def per_1000(catch: float, fbr: float, n_messages: int = N_MESSAGES,
             n_attacks: int = N_ATTACKS) -> dict:
    """Expected outcomes for `n_messages` messages of which `n_attacks` are attacks.

    missed = (1 - catch) * n_attacks;  blocked = fbr * (n_messages - n_attacks).
    Values are rounded to 1 decimal; the *_icons are the integer rounding.
    """
    n_safe = n_messages - n_attacks
    missed = (1.0 - catch) * n_attacks
    blocked = fbr * n_safe
    return {"missed": round(missed, 1), "blocked": round(blocked, 1),
            "missed_icons": round_half_up(missed), "blocked_icons": round_half_up(blocked),
            "n_attacks": n_attacks, "n_safe": n_safe}


def select_examples(rows: list, per_class: int = 4) -> list:
    """Pick the prompts the methods disagree on most.

    rows: list of (row_id, label, n_right, n_wrong).  Disagreement = min(right, wrong)
    (a 2-2 split beats a 4-0 sweep); ties broken by row_id.  Returns row_ids, attacks
    first then safe, each group in row_id order.  Pure and order-independent.
    """
    picked = []
    for label in (1, 0):
        cand = [r for r in rows if r[1] == label]
        cand.sort(key=lambda r: (-min(r[2], r[3]), r[0]))
        picked.append(sorted(r[0] for r in cand[:per_class]))
    return picked[0] + picked[1]


def shorten(text: str, limit: int = 70) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def load_jsonl_latest(path) -> dict:
    """row_id -> record; later lines win but a clean record is never replaced by an errored one."""
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = int(rec["row_id"])
        if rid in out and not out[rid].get("error") and rec.get("error"):
            continue
        out[rid] = rec
    return out


def _rec_flagged(rec: dict) -> bool:
    return bool(rec.get("flagged")) and not rec.get("error")


def definition_effect(old: dict, new: dict, labels: pd.Series) -> dict:
    """Catch / false-block rate on the row_ids both files share (labels: row_id -> 0/1)."""
    ids = sorted(set(old) & set(new) & set(labels.index))
    y = np.array([int(labels.loc[i]) for i in ids])
    fo = np.array([_rec_flagged(old[i]) for i in ids])
    fn = np.array([_rec_flagged(new[i]) for i in ids])
    return {"n": len(ids), "n_attacks": int((y == 1).sum()), "n_safe": int((y == 0).sum()),
            "fb_old": int(fo[y == 0].sum()), "fb_new": int(fn[y == 0].sum()),
            "catch_old": an.catch_rate(fo, y), "catch_new": an.catch_rate(fn, y),
            "fbr_old": an.false_block_rate(fo, y), "fbr_new": an.false_block_rate(fn, y)}


def confusion_counts(g: pd.DataFrame) -> dict:
    y, f = g["label"].to_numpy(), g["flagged_b"].to_numpy()
    att, ben = y == 1, y == 0
    tp, fp = int(f[att].sum()), int(f[ben].sum())
    return {"tp": tp, "fn": int(att.sum()) - tp, "fp": fp, "tn": int(ben.sum()) - fp,
            "n_attacks": int(att.sum()), "n_safe": int(ben.sum())}


# --------------------------------------------------------------------------
# Drawing toolkit
# --------------------------------------------------------------------------
class Canvas:
    """A figure with one full-bleed axes whose units are inches (origin bottom-left)."""

    def __init__(self, size, title, subtitle, footnote):
        self.w, self.h = size
        self.k = 1.0 if self.w >= 7 else 0.8  # type scale: squares are narrower
        self.fig = plt.figure(figsize=size, dpi=DPI, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)
        self.ax.axis("off")
        self.margin = 0.4
        self.top = self._header(title, subtitle)
        self.bottom = self._footer(footnote)

    def _chars(self, fontsize, bold=False):
        usable = (self.w - 2 * self.margin) * 72
        return max(20, int(usable / (fontsize * (0.60 if bold else 0.54))))

    def _header(self, title, subtitle):
        t = 19 * self.k
        lines = textwrap.wrap(title, self._chars(t, True))
        y = self.h - 0.28
        self.ax.text(self.margin, y, "\n".join(lines), fontsize=t, fontweight="bold", color=INK,
                     va="top", ha="left", linespacing=1.15)
        y -= len(lines) * t * 1.15 / 72 + 0.07
        s = 11.5 * self.k
        sub = textwrap.wrap(subtitle, self._chars(s))
        self.ax.text(self.margin, y, "\n".join(sub), fontsize=s, color=GREY, va="top", ha="left",
                     linespacing=1.2)
        y -= len(sub) * s * 1.2 / 72 + 0.12
        return y

    def _footer(self, note):
        s = 9.0 * self.k
        lines = textwrap.wrap(note, self._chars(s))
        self.ax.text(self.margin, 0.17, "\n".join(lines), fontsize=s, color=GREY, va="bottom",
                     ha="left", linespacing=1.25)
        return 0.17 + len(lines) * s * 1.25 / 72 + 0.1

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=DPI, facecolor=BG)
        plt.close(self.fig)


def _blend(color: str, alpha: float) -> tuple:
    """`color` over the background at `alpha`."""
    c = np.array(matplotlib.colors.to_rgb(color))
    b = np.array(matplotlib.colors.to_rgb(BG))
    return tuple(b * (1 - alpha) + c * alpha)


def draw_person(ax, x, y, h, color, edge=None, lw=0):
    """Tiny person glyph (head + shoulders) with its base-centre at (x, y), height h (inches)."""
    th = np.linspace(0, math.pi, 20)
    rx, ry = 0.36 * h, 0.5 * h
    pts = np.column_stack([x + rx * np.cos(th), y + ry * np.sin(th)])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=edge or color, linewidth=lw,
                         zorder=3))
    ax.add_patch(Circle((x, y + 0.79 * h), 0.2 * h, facecolor=color, edgecolor=edge or color,
                        linewidth=lw, zorder=3))


def glyph_ok(ch: str) -> bool:
    try:
        font = fm.get_font(fm.findfont(fm.FontProperties(family=plt.rcParams["font.sans-serif"])))
        return font.get_char_index(ord(ch)) != 0
    except Exception:  # pragma: no cover - defensive
        return False


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data(results_path: Path, task: Task | None = None) -> dict:
    full = an.load_results(results_path, task)
    nl = full[full["method"] != PARTIAL].copy()
    nl.attrs["lakera_left_out"] = bool((full["method"] == PARTIAL).any())
    return {"all": full, "nl": nl}


def ordered(methods) -> list:
    """Known methods in their usual order, then any others alphabetically."""
    present = set(methods)
    return [m for m in an.METHOD_ORDER if m in present] + sorted(present - set(an.METHOD_ORDER))


def methods_in(nl: pd.DataFrame, split: str) -> list:
    return ordered(nl.loc[nl["split"] == split, "method"])


def split_sizes(nl: pd.DataFrame, split: str) -> tuple:
    ms = methods_in(nl, split)
    g = nl[(nl["split"] == split) & (nl["method"] == ms[0])] if ms else nl.iloc[0:0]
    return len(g), int((g["label"] == 1).sum()), int((g["label"] == 0).sum())


# --------------------------------------------------------------------------
# 1. Confusion matrices
# --------------------------------------------------------------------------
def plot_confusion(nl: pd.DataFrame, split: str, path: Path, task: Task | None = None):
    task = task or default_task()
    pos, safe = task.positive_label, task.neg_adj
    n, n_att, n_safe = split_sizes(nl, split)
    ms = methods_in(nl, split)
    cv = Canvas(WIDE, "Where each guardrail gets it right and wrong",
                f"{SPLIT_LABEL[split].capitalize()}: {n:,} prompts ({n_att:,} {task.pos_plural}, {n_safe:,} {safe}), "
                "default settings", join(note(nl), "Percentages are within each row. "
                "A failed call counts as let through."))
    ax, k = cv.ax, cv.k
    x0, x1 = cv.margin, cv.w - cv.margin
    y_top, y_bot = cv.top, cv.bottom
    gap_x, gap_y = 0.25, 0.14
    pw = (x1 - x0 - gap_x) / 2
    ph = (y_top - y_bot - gap_y) / 2
    lab_w = 1.0
    for i, m in enumerate(ms):
        col, row = i % 2, i // 2
        px = x0 + col * (pw + gap_x)
        py = y_top - row * (ph + gap_y)  # top of panel
        c = confusion_counts(nl[(nl["split"] == split) & (nl["method"] == m)])
        hdr = 0.24
        cw = (pw - lab_w) / 2
        chh = (ph - hdr - 0.04) / 2
        bold = "bold" if m == SUBJECT else "normal"
        ax.text(px, py - hdr / 2, name_of(m), fontsize=12 * k, fontweight="bold", color=INK,
                va="center", ha="left")
        for j, head in enumerate(("Blocked", "Let through")):
            ax.text(px + lab_w + cw * (j + 0.5), py - hdr / 2, head, fontsize=11 * k, color=GREY,
                    va="center", ha="center")
        cells = [
            # row 0: really an attack
            [(f"{cap(pos)} blocked", c["tp"], c["n_attacks"], True),
             (f"{cap(pos)} missed", c["fn"], c["n_attacks"], False)],
            # row 1: really safe
            [(f"{cap(safe)} user wrongly blocked", c["fp"], c["n_safe"], False),
             (f"{cap(safe)} user passed", c["tn"], c["n_safe"], True)],
        ]
        rows_lab = ((cap(task.pos_plural), c["n_attacks"]), (cap(safe), c["n_safe"]))
        for r in range(2):
            cy_top = py - hdr - 0.04 - r * chh
            ax.text(px, cy_top - chh / 2 + 0.07, rows_lab[r][0], fontsize=11.5 * k, color=INK,
                    va="center", ha="left", fontweight=bold)
            ax.text(px, cy_top - chh / 2 - 0.1, f"n = {rows_lab[r][1]:,}", fontsize=10 * k,
                    color=GREY, va="center", ha="left")
            for j in range(2):
                label, cnt, denom, good = cells[r][j]
                rate = cnt / denom if denom else 0.0
                alpha = 0.12 + 0.78 * rate
                fc = _blend(TEAL if good else AMBER, alpha)
                cx = px + lab_w + j * cw
                ax.add_patch(Rectangle((cx + 0.025, cy_top - chh + 0.025), cw - 0.05, chh - 0.05,
                                       facecolor=fc, edgecolor="none", zorder=2))
                tc = "white" if (good and alpha > 0.6) else INK
                wrapped = "\n".join(textwrap.wrap(label, 15))
                nl_ = wrapped.count("\n") + 1
                ax.text(cx + cw / 2, cy_top - chh / 2 + 0.05 * nl_ + 0.0, wrapped, fontsize=10 * k,
                        color=tc, va="center", ha="center", zorder=4, linespacing=1.05)
                ax.text(cx + cw / 2, cy_top - chh + 0.05 + 0.0 + 0.1, f"{cnt:,} ({100 * rate:.1f}%)",
                        fontsize=10.8 * k, fontweight="bold", color=tc, va="center", ha="center",
                        zorder=4)
    cv.save(path)


# --------------------------------------------------------------------------
# 2. Per 1,000 messages
# --------------------------------------------------------------------------
def plot_per_1000(nl: pd.DataFrame, split: str, path: Path, size=WIDE, task: Task | None = None):
    task = task or default_task()
    n, n_att, n_safe = split_sizes(nl, split)
    ms = methods_in(nl, split)
    stats = {}
    for m in ms:
        g = nl[(nl["split"] == split) & (nl["method"] == m)]
        stats[m] = per_1000(an.catch_rate(g["flagged_b"], g["label"]),
                            an.false_block_rate(g["flagged_b"], g["label"]))
    title = ("Per 1,000 messages: what each one gets wrong" if split == "test"
             else "On the hard set, per 1,000 messages")
    cv = Canvas(size, title,
                f"If 1,000 messages hit your app and 10 are {task.pos_plural} ({SPLIT_LABEL[split]} rates, "
                f"{n:,} prompts).",
                join("Expected values from measured rates; real traffic differs.", note(nl)))
    ax, k = cv.ax, cv.k
    wide = cv.w >= 7
    x0, x1 = cv.margin, cv.w - cv.margin
    y_top, y_bot = cv.top, cv.bottom
    max_users = max(s["blocked_icons"] for s in stats.values())
    name_w, num_w = 1.4, 0.6
    num_font = 16 * k
    bad = lambda v: AMBER if v >= 0.5 else GREY  # noqa: E731
    if wide:
        att_sp, att_h = 0.19, 0.28
        icons_x = x0 + name_w + num_w
        usr_num_x = icons_x + N_ATTACKS * att_sp + 0.15 + num_w
        usr_x = usr_num_x + 0.12
        usr_sp = 0.09
        per_row = max(1, int((x1 - usr_x) / usr_sp))
        rows_needed = max(1, math.ceil(max_users / per_row))
        hdr_h = 0.5
        block_h = (y_top - y_bot - hdr_h) / len(ms)
        usr_h = min(0.13, (block_h - 0.1) / rows_needed / 1.08)
        hy = y_top - 0.02
        ax.text(icons_x - num_w, hy, f"{cap(task.pos_plural)} missed", fontsize=11.5 * k, color=INK,
                fontweight="bold", va="top")
        ax.text(icons_x - num_w, hy - 0.2, "out of 10", fontsize=10.5 * k, color=GREY, va="top")
        ax.text(usr_num_x - num_w, hy, "Real users wrongly blocked", fontsize=11.5 * k, color=INK,
                fontweight="bold", va="top")
        ax.text(usr_num_x - num_w, hy - 0.2, "out of 990", fontsize=10.5 * k, color=GREY, va="top")
        y = y_top - hdr_h
        for m in ms:
            s = stats[m]
            cy = y - block_h / 2
            _row_band(ax, m, x0 - 0.12, cy, x1 - x0 + 0.24, block_h)
            ax.text(x0, cy, name_of(m), fontsize=13 * k, color=INK, va="center",
                    fontweight="bold" if m == SUBJECT else "normal")
            ax.text(icons_x - 0.1, cy, f"{s['missed']:.1f}", fontsize=num_font, fontweight="bold",
                    color=bad(s["missed"]), va="center", ha="right")
            for i in range(N_ATTACKS):
                draw_person(ax, icons_x + i * att_sp + att_sp / 2, cy - att_h / 2, att_h,
                            AMBER if i < s["missed_icons"] else LIGHT)
            ax.text(usr_num_x, cy, f"{s['blocked']:.1f}", fontsize=num_font, fontweight="bold",
                    color=bad(s["blocked"]), va="center", ha="right")
            n_rows = max(1, math.ceil(s["blocked_icons"] / per_row))
            for i in range(s["blocked_icons"]):
                r_, c_ = divmod(i, per_row)
                gy = cy + (n_rows - 1) * usr_h * 0.54 - r_ * usr_h * 1.08 - usr_h / 2
                draw_person(ax, usr_x + c_ * usr_sp + usr_sp / 2, gy, usr_h, AMBER)
            if s["blocked_icons"] == 0:
                ax.text(usr_x, cy, "0 in the measured sample", fontsize=10.5 * k, color=GREY, va="center",
                        style="italic")
            y -= block_h
    else:
        att_sp, att_h = 0.21, 0.27
        icons_x = x0 + name_w + num_w
        usr_sp = 0.088
        usr_x = icons_x
        per_row = max(1, int((x1 - usr_x) / usr_sp))
        rows_needed = max(1, math.ceil(max_users / per_row))
        key_h = 0.42
        ax.text(x0, y_top, f"Top row: {task.pos_plural} missed, out of 10.\nBottom row: real users wrongly "
                "blocked, out of 990.", fontsize=10 * k, color=GREY, va="top", linespacing=1.3)
        block_h = (y_top - y_bot - key_h) / len(ms)
        usr_h = min(0.12, (block_h - att_h - 0.16) / rows_needed / 1.08)
        y = y_top - key_h
        for m in ms:
            s = stats[m]
            _row_band(ax, m, x0 - 0.12, y - block_h / 2, x1 - x0 + 0.24, block_h)
            ay = y - 0.08 - att_h
            ax.text(x0, ay + att_h / 2, name_of(m), fontsize=13 * k, color=INK, va="center",
                    fontweight="bold" if m == SUBJECT else "normal")
            ax.text(icons_x - 0.1, ay + att_h / 2, f"{s['missed']:.1f}", fontsize=num_font,
                    fontweight="bold", color=bad(s["missed"]), va="center", ha="right")
            for i in range(N_ATTACKS):
                draw_person(ax, icons_x + i * att_sp + att_sp / 2, ay, att_h,
                            AMBER if i < s["missed_icons"] else LIGHT)
            uy = ay - 0.07 - usr_h
            ax.text(icons_x - 0.1, uy + usr_h / 2 + (0.0), f"{s['blocked']:.1f}", fontsize=num_font,
                    fontweight="bold", color=bad(s["blocked"]), va="center", ha="right")
            for i in range(s["blocked_icons"]):
                r_, c_ = divmod(i, per_row)
                draw_person(ax, usr_x + c_ * usr_sp + usr_sp / 2, uy - r_ * usr_h * 1.08, usr_h,
                            AMBER)
            if s["blocked_icons"] == 0:
                ax.text(usr_x, uy + usr_h / 2, "0 in the measured sample", fontsize=10 * k, color=GREY,
                        va="center", style="italic")
            y -= block_h
    cv.save(path)
    return stats


def _row_band(ax, m, x, cy, w, h):
    if m == SUBJECT:
        ax.add_patch(Rectangle((x, cy - h / 2 + 0.03), w, h - 0.06, facecolor=_blend(LIGHT, 0.45),
                               edgecolor="none", zorder=0))


# --------------------------------------------------------------------------
# 3. One sentence
# --------------------------------------------------------------------------
def one_sentence_data(results_dir: Path, data_dir: Path) -> dict:
    val = pd.read_csv(Path(data_dir) / "val.csv")
    if "row_id" not in val.columns:  # same rule as run_benchmark.load_split
        val.insert(0, "row_id", range(len(val)))
    labels = val.set_index("row_id")["label"]
    out = {}
    for m in ("jev", "luna"):
        old = load_jsonl_latest(Path(results_dir) / "archive" / f"val_{m}_narrow_def.jsonl")
        new = load_jsonl_latest(Path(results_dir) / "raw" / "val" / f"{m}.jsonl")
        out[m] = definition_effect(old, new, labels)
    return out


def _legend(cv, x, y, items, step):
    for col, lab in items:
        cv.ax.add_patch(Circle((x + 0.07, y), 0.07, facecolor=col, edgecolor="none"))
        cv.ax.text(x + 0.2, y, lab, fontsize=11.5 * cv.k, color=INK, va="center")
        x += step


def plot_one_sentence(eff: dict, path: Path, size=WIDE, task: Task | None = None, nl=None):
    task = task or default_task()
    luna = eff["luna"]
    n, n_att, n_safe = luna["n"], luna["n_attacks"], luna["n_safe"]
    title = (f"One sentence in the definition moved Luna from {100 * luna['catch_old']:.0f}% "
             f"to {100 * luna['catch_new']:.0f}%")
    fb = ", ".join(f"{name_of(m).split(' (')[0]} {eff[m]['fb_old']} then {eff[m]['fb_new']}"
                   for m in ("jev", "luna"))
    cv = Canvas(size, title,
                f"Same {n} validation prompts ({n_att} {task.pos_plural}, {n_safe} {task.neg_adj}): "
                f"share of {task.pos_plural} blocked.",
                join(f"Small sample: with {n_att} {task.pos_plural}, one prompt moves a bar by about "
                     f"{100 / n_att:.0f} points. {cap(task.neg_plural)} wrongly blocked (of {n_safe}), before then "
                     f"after: {fb}.", note(nl) if nl is not None else LAKERA_NOTE))
    k = cv.k
    leg_y = cv.top - 0.12
    _legend(cv, cv.margin + 0.55, leg_y, ((GREY, "Before: narrow definition"),
                                          (INK, "After: new definition")), 2.5 if cv.w > 7 else 2.15)
    left, right = cv.margin + 0.55, cv.w - cv.margin
    axp = cv.fig.add_axes([left / cv.w, (cv.bottom + 0.4) / cv.h, (right - left) / cv.w,
                           (cv.top - cv.bottom - 0.75) / cv.h])
    axp.set_facecolor(BG)
    for s in ("top", "right", "left"):
        axp.spines[s].set_visible(False)
    axp.spines["bottom"].set_color(LIGHT)
    methods = ["jev", "luna"]
    bw = 0.34
    for i, m in enumerate(methods):
        for j, (key, col) in enumerate((("catch_old", GREY), ("catch_new", INK))):
            v = eff[m][key]
            x = i + (j - 0.5) * (bw + 0.04)
            axp.bar(x, v * 100, width=bw, color=col, linewidth=0, zorder=3)
            axp.text(x, v * 100 + 2, f"{100 * v:.0f}%", ha="center", va="bottom",
                     fontsize=17 * k, fontweight="bold", color=col, zorder=4)
    axp.set_xticks(range(len(methods)))
    axp.set_xticklabels([name_of(m) for m in methods], fontsize=13 * k, color=INK)
    for lbl, m in zip(axp.get_xticklabels(), methods):
        lbl.set_fontweight("bold" if m == SUBJECT else "normal")
    axp.set_ylim(0, 110)
    axp.set_yticks([0, 50, 100])
    axp.set_yticklabels(["0%", "50%", "100%"], fontsize=11.5 * k, color=GREY)
    axp.yaxis.grid(True, color=LIGHT, linewidth=0.8, zorder=0)
    axp.tick_params(length=0)
    axp.set_xlim(-0.6, len(methods) - 0.4)
    cv.save(path)


# --------------------------------------------------------------------------
# 4. Default vs tuned cut-off
# --------------------------------------------------------------------------
def tuned_rows(nl: pd.DataFrame, task: Task | None = None) -> list:
    """[(split, method, catch@0.5, catch@val-cutoff, fbr@0.5, fbr@val-cutoff)] for scored methods."""
    out = []
    for split in ("test", "hard"):
        if not (nl["split"] == split).any():
            continue
        res = an.analyze(nl, split, "val", task)
        for m in res["methods"]:  # methods without a score have no cut-off to tune
            r = res["rows"].get(m)
            if r and r.get("at1_status") == "ok":
                out.append((split, m, r["catch"], r["at1"], r["fbr"], r["at1_fbr"]))
    return out


def plot_default_vs_tuned(nl: pd.DataFrame, path: Path, task: Task | None = None):
    task = task or default_task()
    rows = tuned_rows(nl, task)
    if not rows:  # no scored method with a val run: nothing to show
        return rows, None
    best = max(rows, key=lambda r: (round(r[3] - r[2], 9), r[1]))
    sizes = {s: split_sizes(nl, s) for s in ("test", "hard")}
    title = (f"Tuning {name_of(best[1]).split(' (')[0]}'s cut-off on practice data lifted its catch "
             f"rate from {100 * best[2]:.1f}% to {100 * best[3]:.1f}%")
    cv = Canvas(WIDE, title,
                f"Biggest jump, on the {SPLIT_LABEL[best[0]]}. Cut-off chosen on val to wrongly block "
                f"at most {int(an.FBR_BUDGET * 100)}% of {task.neg_plural}.",
                join(note(nl), "Catching more usually means wrongly blocking more real users: the "
                     f"right-hand column shows {task.neg_plural} wrongly blocked, default to tuned."))
    k = cv.k
    _legend(cv, cv.margin + 1.35, cv.top - 0.1, ((GREY, "Default cut-off (0.5)"),
                                                 (INK, "Cut-off tuned on val")), 2.5)
    left, right = cv.margin + 1.3, cv.w - cv.margin - 1.55
    axp = cv.fig.add_axes([left / cv.w, (cv.bottom + 0.3) / cv.h, (right - left) / cv.w,
                           (cv.top - cv.bottom - 0.65) / cv.h])
    axp.set_facecolor(BG)
    for s in ("top", "right", "left"):
        axp.spines[s].set_visible(False)
    axp.spines["bottom"].set_color(LIGHT)
    xmax = 112.0
    ys, items, group_pos = [], [], {}
    y = 0.0
    for split in ("hard", "test"):  # plotted bottom-up, so test ends on top
        grp = [r for r in rows if r[0] == split]
        if not grp:
            continue
        start = y
        for r in reversed(grp):
            ys.append(y)
            items.append(r)
            y += 1
        group_pos[split] = (start, y - 1)
        y += 0.95
    yax = axp.get_yaxis_transform()
    for y_, (split, m, c0, c1, f0, f1) in zip(ys, items):
        bold = m == SUBJECT
        axp.plot([c0 * 100, c1 * 100], [y_, y_], color=GREY if bold else LIGHT, lw=4, zorder=2,
                 solid_capstyle="round")
        axp.scatter([c0 * 100], [y_], s=150, color=GREY, zorder=3, edgecolor=INK if bold else GREY,
                    linewidth=2 if bold else 0)
        axp.scatter([c1 * 100], [y_], s=150, color=INK, zorder=3, linewidth=0)
        lo, hi = sorted((c0, c1))
        axp.text(lo * 100 - 2.3, y_, f"{100 * lo:.1f}%", ha="right", va="center", fontsize=11.5 * k,
                 color=GREY if lo == c0 else INK, fontweight="bold")
        axp.text(hi * 100 + 2.3, y_, f"{100 * hi:.1f}%", ha="left", va="center", fontsize=11.5 * k,
                 color=INK if hi == c1 else GREY, fontweight="bold")
        axp.text(1.03, y_, f"{100 * f0:.1f}% → {100 * f1:.1f}%", transform=yax, ha="left",
                 va="center", fontsize=11.5 * k, color=AMBER if f1 > f0 else INK)
        axp.text(-0.03, y_, name_of(m).split(" (")[0], transform=yax, ha="right", va="center",
                 fontsize=13 * k, color=INK, fontweight="bold" if bold else "normal")
    for split, (lo_y, hi_y) in group_pos.items():
        n = sizes[split][0]
        axp.text(0.0, hi_y + 0.9, f"{SPLIT_LABEL[split].capitalize()} ({n:,} prompts)",
                 transform=yax, ha="left", va="center", fontsize=12 * k, color=GREY,
                 fontweight="bold")
    axp.text(1.03, max(ys) + 0.9, f"{cap(task.neg_adj)} blocked", transform=yax, ha="left", va="center",
             fontsize=11.5 * k, color=GREY, fontweight="bold")
    axp.set_xlim(0, xmax)
    axp.set_ylim(-0.6, max(ys) + 1.3)
    axp.set_yticks([])
    ticks = [0, 25, 50, 75, 100]
    axp.set_xticks(ticks)
    axp.set_xticklabels([f"{t}%" for t in ticks], fontsize=11 * k, color=GREY)
    axp.tick_params(length=0)
    axp.xaxis.grid(True, color=LIGHT, linewidth=0.6, zorder=0)
    cv.save(path)
    return rows, best


# --------------------------------------------------------------------------
# 5. Example prompts report card
# --------------------------------------------------------------------------
def example_table(nl: pd.DataFrame, data_dir: Path) -> tuple:
    hard = pd.read_csv(Path(data_dir) / "hard.csv")
    hand = hard[hard["source"] == "hand_written"].set_index("row_id")
    g = nl[(nl["split"] == "hard") & nl["row_id"].isin(hand.index)]
    ms = methods_in(nl, "hard")
    verdict: dict = {}
    for m in ms:
        gm = g[g["method"] == m].set_index("row_id")
        for rid, r in gm.iterrows():
            verdict.setdefault(rid, {})[m] = (bool(r["flagged_b"]) == bool(r["label"] == 1))
    rows = []
    for rid, vs in verdict.items():
        if len(vs) == len(ms):
            right = sum(vs.values())
            rows.append((int(rid), int(hand.loc[rid, "label"]), right, len(ms) - right))
    chosen = select_examples(rows)
    return chosen, hand, verdict, ms


def plot_example_prompts(nl: pd.DataFrame, data_dir: Path, path: Path, task: Task | None = None):
    task = task or default_task()
    chosen, hand, verdict, ms = example_table(nl, data_dir)
    ok_g, bad_g = ("✓", "✗") if glyph_ok("✓") and glyph_ok("✗") else ("OK", "X")
    n_att = sum(int(hand.loc[r, "label"]) == 1 for r in chosen)
    cv = Canvas(WIDE, "Same tricky prompts, very different verdicts",
                f"Hand-written hard-set prompts the guardrails disagreed on most "
                f"({n_att} {task.pos_plural}, {len(chosen) - n_att} {task.neg_adj}).",
                join(f"{ok_g} = right call, {bad_g} = wrong call. Picked for disagreement, so this is not "
                     f"a typical sample; ties go to the lower row id.", note(nl)))
    ax, k = cv.ax, cv.k
    x0, x1 = cv.margin, cv.w - cv.margin
    col_w = 0.78
    ans_w = 0.7
    m_x0 = x1 - col_w * len(ms)
    ans_x = m_x0 - ans_w
    top = cv.top
    for j, m in enumerate(ms):
        ax.text(m_x0 + col_w * (j + 0.5), top - 0.1, name_of(m).replace(" (GPT-6)", "\n(GPT-6)"),
                fontsize=10.5 * k, fontweight="bold" if m == SUBJECT else "normal", color=INK,
                ha="center", va="center", linespacing=0.95)
    ax.text(ans_x + ans_w / 2, top - 0.1, "Right\nanswer", fontsize=10.5 * k, color=GREY, ha="center",
            va="center", linespacing=0.95)
    ax.text(x0, top - 0.1, "Prompt (shortened)", fontsize=10.5 * k, color=GREY, va="center")
    row_h = (top - 0.28 - cv.bottom) / len(chosen)
    for i, rid in enumerate(chosen):
        y = top - 0.28 - i * row_h
        label = int(hand.loc[rid, "label"])
        prev = int(hand.loc[chosen[i - 1], "label"]) if i else label
        if prev != label:
            ax.plot([x0, x1], [y + 0.01, y + 0.01], color=GREY, lw=0.8, zorder=1)
        elif i > 0:
            ax.plot([x0, x1], [y, y], color=LIGHT, lw=0.5, zorder=1)
        txt = "\n".join(textwrap.wrap(shorten(hand.loc[rid, "text"]), 40)[:2])
        ax.text(x0, y - row_h / 2, txt, fontsize=9.5 * k, color=INK, va="center", linespacing=1.1)
        ans = "BLOCK" if label == 1 else "PASS"
        ax.text(ans_x + ans_w / 2, y - row_h / 2, ans, fontsize=10.5 * k, fontweight="bold",
                color=INK, ha="center", va="center")
        for j, m in enumerate(ms):
            right = verdict[rid][m]
            cx = m_x0 + col_w * (j + 0.5)
            ax.add_patch(Rectangle((cx - col_w / 2 + 0.07, y - row_h + 0.04), col_w - 0.14, row_h - 0.08,
                                   facecolor=_blend(TEAL if right else AMBER, 0.28), edgecolor="none",
                                   zorder=2))
            ax.text(cx, y - row_h / 2, ok_g if right else bad_g, fontsize=15 * k, fontweight="bold",
                    color=TEAL if right else "#B9791A", ha="center", va="center", zorder=3)
    cv.save(path)
    return chosen


# --------------------------------------------------------------------------
# 6. Speed vs cost
# --------------------------------------------------------------------------
def speed_cost_points(df_all: pd.DataFrame, task: Task | None = None) -> dict:
    nl = df_all[df_all["method"] != PARTIAL]
    pts = {}
    for m in methods_in(nl, "test"):
        g = nl[(nl["split"] == "test") & (nl["method"] == m)]
        met = an.method_metrics(m, g, task=task)
        pts[m] = {"lat": met["lat_med"], "cost": met["cost_per_m"], "n": len(g)}
    lk = df_all[(df_all["method"] == PARTIAL) & (df_all["split"] == "val") & ~df_all["error_flag"]]
    if (df_all["method"] == PARTIAL).any():
        lat = lk["latency_ms"].dropna()
        pts[PARTIAL] = {"lat": float(lat.median()) if len(lat) else math.nan, "cost": math.nan,
                        "n": len(lat)}
    return pts


def plot_speed_vs_cost(df_all: pd.DataFrame, path: Path, task: Task | None = None):
    pts = speed_cost_points(df_all, task)
    shown = [m for m in pts if m != PARTIAL]
    n_test = pts.get("jev", pts[shown[0]])["n"]
    if "jev" in pts:
        title = (f"Free local checkers answer in milliseconds; Jev costs ${pts['jev']['cost']:.2f} per million checks")
    else:
        title = "Median response time against cost per million checks"
    cv = Canvas(WIDE, title,
                f"Test set, {n_test:,} prompts per method: median response time vs cost per 1M checks.",
                join("Cost is projected from real per-call cost; local methods are $0.",
                     f"Lakera Guard has partial data (val subset, {pts[PARTIAL]['n']} successful calls), "
                     "so no cost is shown." if PARTIAL in pts else ""))
    k = cv.k
    left, right = cv.margin + 0.8, cv.w - cv.margin - 0.1
    axp = cv.fig.add_axes([left / cv.w, (cv.bottom + 0.5) / cv.h, (right - left) / cv.w,
                           (cv.top - cv.bottom - 0.55) / cv.h])
    axp.set_facecolor(BG)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axp.spines[s].set_color(LIGHT)
    axp.set_xscale("log")
    top_cost = max([p["cost"] for m, p in pts.items() if m != PARTIAL] + [1.0])
    axp.set_ylim(-0.5 * top_cost, top_cost * 1.2)
    axp.set_xlim(0.008, 9000)
    axp.axhline(0, color=GREY, lw=1, zorder=1, linestyle=(0, (4, 3)))
    axp.text(0.0125, -0.03 * top_cost, "local = $0 baseline", color=GREY, fontsize=10.5 * k,
             va="top", ha="left")
    for m in shown:
        p = pts[m]
        bold = m == SUBJECT
        axp.scatter([p["lat"]], [p["cost"]], s=190 if bold else 150, color=INK, zorder=4,
                    edgecolor=INK, linewidth=0)
        lab = f"{name_of(m)}\n{an.fmt_ms(p['lat'])} ms, " + ("$0" if p["cost"] == 0 else f"${p['cost']:.2f}")
        kw = dict(fontsize=11.5 * k, color=INK, fontweight="bold" if bold else "normal",
                  linespacing=1.1, textcoords="offset points")
        if m in ("regex", "protectai"):
            axp.annotate(lab, (p["lat"], p["cost"]), xytext=(0, 13), ha="center", va="bottom", **kw)
        elif m == "jev":
            axp.annotate(lab, (p["lat"], p["cost"]), xytext=(14, 0), ha="left", va="center", **kw)
        else:
            axp.annotate(lab, (p["lat"], p["cost"]), xytext=(-14, 0), ha="right", va="center", **kw)
    lk = pts.get(PARTIAL)
    if lk and not math.isnan(lk["lat"]):
        axp.scatter([lk["lat"]], [0], s=150, facecolor=BG, edgecolor=INK, linewidth=2, zorder=4)
        axp.annotate(f"Lakera Guard (partial data, free tier)\n{an.fmt_ms(lk['lat'])} ms, cost n/a",
                     (lk["lat"], 0), xytext=(14, -18), textcoords="offset points", ha="right",
                     va="top", fontsize=11.5 * k, color=INK, linespacing=1.1)
    axp.set_xlabel("Median response time in ms (log scale: each gridline is 10x slower)",
                   fontsize=11.5 * k, color=INK, labelpad=6)
    axp.set_ylabel("Cost per 1M checks (USD)", fontsize=11.5 * k, color=INK, labelpad=10)
    axp.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    axp.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1,000"], fontsize=11 * k, color=GREY)
    ticks = [t for t in range(0, 100, 10) if t <= top_cost * 1.1]
    axp.set_yticks(ticks)
    axp.set_yticklabels([f"${t}" for t in ticks], fontsize=11 * k, color=GREY)
    axp.tick_params(length=0)
    axp.xaxis.grid(True, color=LIGHT, linewidth=0.6, zorder=0)
    axp.minorticks_off()
    cv.save(path)
    return pts


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _one_sentence_inputs(results_dir: Path) -> bool:
    return all((results_dir / "archive" / f"val_{m}_narrow_def.jsonl").exists() for m in ("jev", "luna"))


def _example_inputs(data_dir: Path) -> bool:
    path = data_dir / "hard.csv"
    return path.exists() and "source" in pd.read_csv(path, nrows=0).columns \
        and (pd.read_csv(path)["source"] == "hand_written").any()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Make plain-English shareable charts.")
    p.add_argument("--task", default=DEFAULT_TASK, help=f"task name: tasks/<task>.toml (default {DEFAULT_TASK})")
    p.add_argument("--results", default=None, help="results CSV (default: <task results_dir>/results.csv)")
    p.add_argument("--out", default=None, help="output folder (default: <task results_dir>/visuals)")
    p.add_argument("--data", default=None, help="folder with val.csv and hard.csv (default: the task's data_dir)")
    a = p.parse_args(argv)
    try:
        task = load_task(a.task)
    except TaskError as e:
        p.error(str(e))

    results_path = Path(a.results) if a.results else task.results_dir / "results.csv"
    results_dir = results_path.parent
    out = Path(a.out) if a.out else task.results_dir / "visuals"
    data_dir = Path(a.data) if a.data else task.data_dir
    out.mkdir(parents=True, exist_ok=True)
    d = load_data(results_path, task)
    nl, full = d["nl"], d["all"]
    made = []

    def done(name):
        made.append(out / name)
        return out / name

    for split in ("test", "hard"):
        if (nl["split"] == split).any():
            plot_confusion(nl, split, done(f"confusion_{split}.png"), task)
    if (nl["split"] == "test").any():
        plot_per_1000(nl, "test", done("per_1000_messages.png"), task=task)
        plot_per_1000(nl, "test", done("per_1000_messages_square.png"), SQUARE, task)
    if (nl["split"] == "hard").any():
        plot_per_1000(nl, "hard", done("per_1000_messages_hard.png"), task=task)
        plot_per_1000(nl, "hard", done("per_1000_messages_hard_square.png"), SQUARE, task)
    if _one_sentence_inputs(results_dir):
        eff = one_sentence_data(results_dir, data_dir)
        plot_one_sentence(eff, done("one_sentence.png"), task=task, nl=nl)
        plot_one_sentence(eff, done("one_sentence_square.png"), SQUARE, task, nl)
    else:
        print(f"skipped one_sentence charts: no val_<method>_narrow_def.jsonl in {results_dir / 'archive'}")
    if tuned_rows(nl, task):
        plot_default_vs_tuned(nl, done("default_vs_tuned.png"), task)
    if _example_inputs(data_dir):
        plot_example_prompts(nl, data_dir, done("example_prompts.png"), task)
    else:
        print(f"skipped example_prompts: needs {data_dir / 'hard.csv'} with `source` = hand_written rows")
    if (nl["split"] == "test").any():
        plot_speed_vs_cost(full, done("speed_vs_cost.png"), task)
    print(f"Wrote {len(made)} images to {show(out)}/")
    for m in made:
        print("  ", m.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
