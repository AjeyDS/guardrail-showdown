"""Download the benchmark dataset once and save fixed CSV copies.

Run:  .venv/bin/python prepare_data.py
Writes data/val.csv and data/test.csv and prints a short profile.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET = "neuralchemy/Prompt-injection-dataset"
CONFIG = "core"
DEEPSET = "deepset/prompt-injections"
OUT = Path(__file__).parent / "data"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ds = load_dataset(DATASET, CONFIG)
    for split, name in [("validation", "val"), ("test", "test")]:
        df = ds[split].to_pandas()
        df.insert(0, "row_id", range(len(df)))
        df["tags"] = df["tags"].apply(lambda t: "|".join(t) if t is not None else "")
        df.to_csv(OUT / f"{name}.csv", index=False)

        print(f"\n=== {name}: {len(df)} rows ===")
        print("label counts:", df["label"].value_counts().to_dict())
        print("text length (chars): median", int(df["text"].str.len().median()),
              "p99", int(df["text"].str.len().quantile(0.99)),
              "max", int(df["text"].str.len().max()))
        print(pd.crosstab(df["category"], df["label"]).to_string())

    build_hard_set()


def build_hard_set() -> None:
    """Second, separately reported test: realistic + deliberately tricky prompts.

    - deepset/prompt-injections test split: natural user questions and real
      injections (some German). Not in ProtectAI v2's training data.
    - data/hand_written.csv: 20 attacks hidden inside content (emails, pages,
      tool results) and 20 tricky-but-safe prompts that *sound* like attacks.
    """
    deepset = load_dataset(DEEPSET)["test"].to_pandas()[["text", "label"]]
    deepset["category"] = deepset["label"].map({0: "benign", 1: "deepset_injection"})
    deepset["source"] = "deepset"
    deepset["tags"] = ""

    hand = pd.read_csv(OUT / "hand_written.csv")
    hand["source"] = "hand_written"

    df = pd.concat([deepset, hand], ignore_index=True)
    for col, val in [("severity", ""), ("group_id", ""), ("augmented", False)]:
        df[col] = val
    df.insert(0, "row_id", range(len(df)))
    df = df[["row_id", "text", "label", "category", "source", "severity", "group_id", "augmented", "tags"]]
    df.to_csv(OUT / "hard.csv", index=False)

    print(f"\n=== hard: {len(df)} rows ===")
    print(pd.crosstab([df["source"], df["category"]], df["label"]).to_string())


if __name__ == "__main__":
    main()
