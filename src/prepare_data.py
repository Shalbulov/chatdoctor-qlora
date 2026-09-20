"""Build the instruction-response dataset from lavita/ChatDoctor-HealthCareMagic-100k.

Pipeline order matters: cheap row-level rules run over all 112k rows first, then we
draw a fixed-seed candidate pool and do the O(n^2) near-duplicate pass inside it.
Running near-dup over the full corpus would be 112k x 112k similarities for a payoff
that src/explore.py measured at roughly 0.2% of rows.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from clean import (
    ARTIFACT, BAD_OPENERS, CORRUPTION_MARK, INLINE_LIST, MARKDOWN, SENTENCE_END,
    first_word, normalise, strip_pleasantries,
)

DATASET = "lavita/ChatDoctor-HealthCareMagic-100k"
SEED = 42

# Bounds come from the length quantiles in explore.py. p05/p99 of the input is
# 251/1354 chars and of the output 264/1428, so these keep the bulk of the
# distribution and cut the tails where answers are either stubs or transcripts.
MIN_IN, MAX_IN = 150, 1400
MIN_OUT, MAX_OUT = 150, 1300


class Funnel:
    """Row counts after each rule, so the README can show where the data went."""

    def __init__(self, total: int):
        self.rows = [("loaded", total, 0)]
        self.prev = total

    def step(self, name: str, remaining: int):
        self.rows.append((name, remaining, self.prev - remaining))
        self.prev = remaining

    def render(self) -> str:
        w = max(len(n) for n, _, _ in self.rows)
        head = f"{'stage'.ljust(w)}  {'kept':>7}  {'dropped':>8}"
        body = "\n".join(
            f"{n.ljust(w)}  {k:>7}  {d:>8}" if d else f"{n.ljust(w)}  {k:>7}  {'':>8}"
            for n, k, d in self.rows
        )
        return f"{head}\n{'-' * len(head)}\n{body}"


def drop_near_duplicates(texts: pd.Series, threshold: float, block: int = 512) -> np.ndarray:
    """Greedy: keep the first row of every near-identical group.

    Cosine over char 3-5 grams rather than word tokens, because the duplicates we saw
    were re-posts of the same complaint with small edits, not paraphrases.
    """
    X = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000
    ).fit_transform(texts)

    keep = np.ones(X.shape[0], dtype=bool)
    for start in range(0, X.shape[0], block):
        stop = min(start + block, X.shape[0])
        sims = cosine_similarity(X[start:stop], X[:stop], dense_output=False).toarray()
        for local, row_idx in enumerate(range(start, stop)):
            if not keep[row_idx]:
                continue
            earlier = sims[local, :row_idx]
            if earlier.size and earlier.max() >= threshold:
                keep[row_idx] = False
    return keep


def write_jsonl(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in frame.to_dict("records"):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--pool", type=int, default=6000,
                    help="candidate pool size for the near-duplicate pass")
    ap.add_argument("--near-dup-threshold", type=float, default=0.85)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    df = load_dataset(DATASET, split="train").to_pandas()
    funnel = Funnel(len(df))

    df["instruction"] = df["input"].map(normalise)
    df["response"] = df["output"].map(normalise)

    df = df[~df["instruction"].str.lower().duplicated()]
    funnel.step("exact duplicate question", len(df))

    df["response"] = df["response"].map(strip_pleasantries)
    df = df[~df["response"].str.contains(ARTIFACT)]
    funnel.step("'Chat Doctor' left mid-answer", len(df))

    df = df[~df["response"].map(first_word).isin(BAD_OPENERS)]
    funnel.step("corrupted opening word", len(df))

    df = df[~df["response"].str.contains(CORRUPTION_MARK)
            & ~df["instruction"].str.contains(CORRUPTION_MARK)]
    funnel.step("'.*' corruption marker", len(df))

    # Answers that survive cleaning but begin mid-word are upstream glue damage,
    # e.g. "come have evaluated your query" from a mangled "Welcome, I have".
    df = df[~df["response"].str.contains(MARKDOWN)
            & ~df["response"].str.contains(INLINE_LIST)]
    funnel.step("markdown or enumerated list", len(df))

    df = df[df["response"].str[:1].str.isupper()]
    funnel.step("answer starts mid-word", len(df))

    df = df[df["response"].str.contains(SENTENCE_END)]
    funnel.step("answer cut off mid-sentence", len(df))

    df = df[df["instruction"].str.len().between(MIN_IN, MAX_IN)
            & df["response"].str.len().between(MIN_OUT, MAX_OUT)]
    funnel.step("length outside bounds", len(df))

    pool = df.sample(min(args.pool, len(df)), random_state=SEED).reset_index(drop=True)
    funnel.step(f"sampled candidate pool (seed {SEED})", len(pool))

    keep = drop_near_duplicates(pool["instruction"], args.near_dup_threshold)
    pool = pool[keep].reset_index(drop=True)
    funnel.step(f"near-duplicate question (cos>={args.near_dup_threshold})", len(pool))

    need = args.n_train + args.n_val + args.n_eval
    if len(pool) < need:
        raise SystemExit(f"pool has {len(pool)} rows, need {need}")

    pool = pool.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    cols = ["instruction", "response"]
    splits = {
        "train": pool.iloc[:args.n_train][cols],
        "val": pool.iloc[args.n_train:args.n_train + args.n_val][cols],
        "eval_set": pool.iloc[args.n_train + args.n_val:need][cols],
    }
    for name, frame in splits.items():
        write_jsonl(args.out / f"{name}.jsonl", frame)

    print(funnel.render())
    print("\nsplits")
    for name, frame in splits.items():
        print(f"  {name:<9} {len(frame):>5} rows  "
              f"median answer {int(frame.response.str.len().median())} chars")

    report = args.out / "funnel.txt"
    report.write_text(funnel.render() + "\n", encoding="utf-8")
    print(f"\nfunnel written to {report}")


if __name__ == "__main__":
    main()
