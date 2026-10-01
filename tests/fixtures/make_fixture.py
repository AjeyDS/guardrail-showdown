"""Regenerate tests/fixtures/results_synth.csv (synthetic, deterministic; no real data).

    .venv/bin/python tests/fixtures/make_fixture.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ATTACK_CATS = (["direct_injection"] * 6 + ["instruction_override"] * 2 + ["jailbreak"] * 4 + ["encoding"] * 4
               + ["rag_poisoning"] * 4 + ["prompt_extraction"] * 3 + ["code_execution"] * 2 + ["weird_new_cat"])
BENIGN_CATS = ["benign"] * 12 + ["edge_case"] * 3 + ["benign"]  # last 'benign' is tagged hard-negative below


def prompts(split):
    rows = []
    for i, cat in enumerate(ATTACK_CATS):
        rows.append((i, 1, cat, False))
    base = len(ATTACK_CATS)
    for j, cat in enumerate(BENIGN_CATS):
        rows.append((base + j, 0, cat, cat == "edge_case" or j == len(BENIGN_CATS) - 1))
    return rows


def build():
    rng = np.random.default_rng(7)
    out = []
    for split in ("val", "test"):
        for method in ("regex", "protectai", "jev", "luna", "lakera"):
            for row_id, label, cat, hard in prompts(split):
                scored = method in ("protectai", "jev", "luna")
                if scored:
                    score = float(np.clip(rng.beta(5, 2) if label else rng.beta(2, 5), 0.01, 0.99))
                    if hard and method == "luna":
                        score = 0.9  # a tricky-but-safe prompt that fools one method
                    flagged = score >= 0.5
                else:
                    score = None
                    flagged = bool(rng.random() < (0.7 if label else 0.05)) or (hard and method == "regex")
                lat = {"regex": 0.05, "protectai": 40, "jev": 300, "luna": 1500, "lakera": 150}[method]
                latency = float(lat * rng.lognormal(0, 0.3))
                cost = {"regex": 0.0, "protectai": 0.0, "jev": 1e-5, "luna": 4e-5, "lakera": None}[method]
                error = None
                if method == "jev" and row_id in (3, 30):
                    error, flagged, score, cost, latency = "timeout", None, None, None, 30000.0
                if method == "luna" and row_id == 5:
                    error, flagged, score, cost = "refusal", None, None, None
                out.append(dict(split=split, method=method, row_id=row_id, label=label, category=cat,
                                is_hard_negative=hard, flagged=flagged, score=score, latency_ms=latency,
                                cost_usd=cost, error=error))
    return pd.DataFrame(out)


if __name__ == "__main__":
    path = Path(__file__).with_name("results_synth.csv")
    build().to_csv(path, index=False)
    print("wrote", path)
