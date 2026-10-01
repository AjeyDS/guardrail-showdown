"""Benchmark runner: run guardrails over a data split and cache raw verdicts.

    .venv/bin/python run_benchmark.py --split val [--methods regex,jev] [--limit N]
    .venv/bin/python run_benchmark.py --rebuild

Raw verdicts go to results/raw/<split>/<method>.jsonl (one line per finished
prompt, appended and flushed immediately, so a crash loses nothing). Re-running
resumes: rows already present without an error are skipped, errored rows are
retried. results/results.csv is rebuilt from every raw file afterwards.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from guardrails import registry
from guardrails.base import Verdict

ROOT = Path(__file__).resolve().parent
# Overridable (tests point these at a tmp dir); read at call time.
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Guardrails are auto-discovered from guardrails/*.py (see guardrails/registry.py).
# REGISTRY: method -> module name. Modules are imported lazily by load_guardrail().
_DISCOVERED = registry.discover()
REGISTRY: dict[str, str] = {m: info.module for m, info in _DISCOVERED.items()}
METHOD_ORDER = list(REGISTRY)
API_METHODS = {m for m, info in _DISCOVERED.items() if not info.local}  # run concurrently

# Rough USD per prompt, for the pre-run estimate only (real cost comes from the
# providers' responses). Deliberately pessimistic round numbers.
# ~2x the per-call cost measured on live calls (jev $0.000016, luna $0.000028);
# lakera is on the free Community tier.
EST_COST_PER_PROMPT = {"jev": 0.00003, "luna": 0.00006, "lakera": 0.0}
MAX_API_PROMPTS = 1200

CSV_COLUMNS = [
    "split", "method", "row_id", "label", "category", "is_hard_negative",
    "flagged", "score", "latency_ms", "cost_usd", "error",
]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_split(split: str) -> pd.DataFrame:
    path = Path(DATA_DIR) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    df["row_id"] = df["row_id"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Seeded sample of n rows that keeps the label ratio (>=1 per label if n allows)."""
    if n >= len(df):
        return df.sort_values("row_id").reset_index(drop=True)
    if n <= 0:
        raise ValueError("--limit must be positive")
    groups = {lab: sorted(g["row_id"].tolist()) for lab, g in df.groupby("label")}
    labels = sorted(groups)
    total = len(df)
    # Largest-remainder allocation, with a floor of 1 per label.
    quotas = {lab: n * len(groups[lab]) / total for lab in labels}
    alloc = {lab: min(len(groups[lab]), max(1, int(quotas[lab]))) for lab in labels}
    while sum(alloc.values()) < n:
        cand = [l for l in labels if alloc[l] < len(groups[l])]
        lab = max(cand, key=lambda l: quotas[l] - alloc[l])
        alloc[lab] += 1
    while sum(alloc.values()) > n:
        cand = [l for l in labels if alloc[l] > 1]
        lab = min(cand, key=lambda l: quotas[l] - alloc[l])
        alloc[lab] -= 1
    rng = random.Random(seed)
    keep: list[int] = []
    for lab in labels:
        keep += rng.sample(groups[lab], alloc[lab])
    out = df[df["row_id"].isin(keep)]
    return out.sort_values("row_id").reset_index(drop=True)


def is_hard_negative(label: int, category: str, tags: str) -> bool:
    tagset = {t for t in str(tags).split("|") if t}
    return int(label) == 0 and (
        category == "edge_case" or "hard_negative" in tagset or "security_adjacent" in tagset
    )


# --------------------------------------------------------------------------
# raw cache
# --------------------------------------------------------------------------

def raw_path(split: str, method: str) -> Path:
    return Path(RESULTS_DIR) / "raw" / split / f"{method}.jsonl"


def read_raw(path: Path) -> dict[int, dict]:
    """Latest record per row_id (last line wins). Malformed lines are skipped."""
    latest: dict[int, dict] = {}
    if not path.exists():
        return latest
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                latest[int(rec["row_id"])] = rec
            except (ValueError, KeyError, TypeError):
                continue  # e.g. a half-written final line after a crash
    return latest


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def load_guardrail(method: str):
    """Import the method's module lazily and build its guardrail."""
    return registry.load(method)


def run_method(method: str, guardrail, df: pd.DataFrame, split: str, workers: int,
               show_progress: bool = True) -> None:
    """Run `guardrail` over rows of df that lack a clean cached verdict."""
    path = raw_path(split, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {rid for rid, rec in read_raw(path).items() if rec.get("error") is None}
    todo = [(int(r), t) for r, t in zip(df["row_id"], df["text"]) if int(r) not in done]
    if not todo:
        return

    n_workers = max(1, workers) if method in API_METHODS else 1
    lock = threading.Lock()
    bar = tqdm(total=len(todo), desc=method, disable=not show_progress)

    # A crash can leave a half-written last line with no newline; terminate it
    # so the first new record doesn't get glued onto it.
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
    else:
        needs_newline = False

    with open(path, "a", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")

        def work(item: tuple[int, str]) -> None:
            rid, text = item
            try:
                verdict = guardrail.check(text)
            except Exception as e:  # check() must not raise, but never lose a row
                verdict = Verdict(flagged=None, score=None, latency_ms=0.0,
                                  error=f"check raised {type(e).__name__}: {e}"[:200])
            line = json.dumps({"row_id": rid, **verdict.to_dict()}, ensure_ascii=False)
            with lock:
                fh.write(line + "\n")
                fh.flush()
                bar.update(1)

        try:
            if n_workers == 1:
                for item in todo:
                    work(item)
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    list(ex.map(work, todo))
        finally:
            bar.close()


def summarize(method: str, df: pd.DataFrame, split: str) -> dict:
    recs = read_raw(raw_path(split, method))
    rows = [recs[int(r)] for r in df["row_id"] if int(r) in recs]
    errors = sum(1 for r in rows if r.get("error") is not None)
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    lats = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    return {
        "method": method,
        "n_done": len(rows) - errors,
        "n_errors": errors,
        "n_missing": len(df) - len(rows),
        "cost_usd": sum(costs),
        "median_latency_ms": statistics.median(lats) if lats else float("nan"),
    }


def print_summary(summaries: list[dict], split: str) -> None:
    print(f"\nSummary for split '{split}':")
    print(f"{'method':<10}{'done':>7}{'errors':>8}{'missing':>9}{'cost_usd':>12}{'median_ms':>11}")
    for s in summaries:
        print(f"{s['method']:<10}{s['n_done']:>7}{s['n_errors']:>8}{s['n_missing']:>9}"
              f"{s['cost_usd']:>12.5f}{s['median_latency_ms']:>11.1f}")


# --------------------------------------------------------------------------
# results.csv
# --------------------------------------------------------------------------

def rebuild_results() -> pd.DataFrame:
    """Rebuild results/results.csv from every raw file under results/raw/."""
    raw_root = Path(RESULTS_DIR) / "raw"
    frames: list[pd.DataFrame] = []
    split_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir()) if raw_root.exists() else []
    for split_dir in split_dirs:
        split = split_dir.name
        try:
            data = load_split(split)
        except FileNotFoundError:
            print(f"warning: no data/{split}.csv; skipping raw/{split}", file=sys.stderr)
            continue
        meta = data.set_index("row_id")
        files = {p.stem: p for p in split_dir.glob("*.jsonl")}
        ordered = [m for m in METHOD_ORDER if m in files] + sorted(m for m in files if m not in METHOD_ORDER)
        for method in ordered:
            recs = read_raw(files[method])
            out = []
            for rid in sorted(recs):
                if rid not in meta.index:
                    print(f"warning: {split}/{method} row_id {rid} not in data; skipped", file=sys.stderr)
                    continue
                m, r = meta.loc[rid], recs[rid]
                out.append({
                    "split": split, "method": method, "row_id": rid,
                    "label": int(m["label"]), "category": m["category"],
                    "is_hard_negative": is_hard_negative(m["label"], m["category"], m["tags"]),
                    "flagged": r.get("flagged"), "score": r.get("score"),
                    "latency_ms": r.get("latency_ms"), "cost_usd": r.get("cost_usd"),
                    "error": r.get("error"),
                })
            if out:
                frames.append(pd.DataFrame(out, columns=CSV_COLUMNS))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CSV_COLUMNS)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(RESULTS_DIR) / "results.csv", index=False)
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run prompt-injection guardrails over a data split.")
    p.add_argument("--split", help="split name: data/<split>.csv (e.g. val, test)")
    p.add_argument("--methods", default=",".join(METHOD_ORDER),
                   help=f"comma-separated subset of: {','.join(METHOD_ORDER)}")
    p.add_argument("--limit", type=int, help="stratified, seeded sample of N rows (smoke test)")
    p.add_argument("--workers", type=int, default=4, help="threads for API methods (default 4)")
    p.add_argument("--seed", type=int, default=0, help="seed for --limit sampling")
    p.add_argument("--rebuild", action="store_true", help="only rebuild results/results.csv")
    p.add_argument("--yes", action="store_true", help=f"allow API methods on >{MAX_API_PROMPTS} prompts")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.rebuild:
        df = rebuild_results()
        print(f"Rebuilt {Path(RESULTS_DIR) / 'results.csv'} ({len(df)} rows)")
        return 0

    if not args.split:
        parser.error("--split is required unless --rebuild is given")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in REGISTRY]
    if unknown or not methods:
        parser.error(f"unknown method(s) {unknown}; choose from {METHOD_ORDER}")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    try:
        df = load_split(args.split)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.limit is not None:
        df = stratified_sample(df, args.limit, args.seed)
    n = len(df)
    print(f"Split '{args.split}': {n} prompts "
          f"({int((df['label'] == 1).sum())} attacks, {int((df['label'] == 0).sum())} benign)")

    api = [m for m in methods if m in API_METHODS]
    if api:
        est = sum(EST_COST_PER_PROMPT.get(m, 0.0) for m in api) * n
        print(f"Estimated API cost for {', '.join(api)}: about ${est:.3f} (rough upper-ish guess)")
        if n > MAX_API_PROMPTS and not args.yes:
            print(f"error: refusing to run API methods on {n} prompts (> {MAX_API_PROMPTS}); "
                  f"pass --yes to override", file=sys.stderr)
            return 2

    failed_load = False
    summaries = []
    for method in methods:  # sequential: heavy local models never compete with API calls
        try:
            guardrail = load_guardrail(method)
        except Exception as e:
            print(f"error: could not load method '{method}': {type(e).__name__}: {e}", file=sys.stderr)
            failed_load = True
            continue
        run_method(method, guardrail, df, args.split, args.workers)
        summaries.append(summarize(method, df, args.split))
    print_summary(summaries, args.split)

    out = rebuild_results()
    print(f"\nWrote {Path(RESULTS_DIR) / 'results.csv'} ({len(out)} rows)")
    return 1 if failed_load else 0


if __name__ == "__main__":
    sys.exit(main())
