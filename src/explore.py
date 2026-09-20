"""One-off look at the raw dataset before deciding on filter thresholds.
Not part of the pipeline, kept in the repo so the numbers in the README are reproducible."""

import re
import numpy as np
import pandas as pd
from datasets import load_dataset

ARTIFACT = re.compile(r"chat\s*doctor", re.I)
SENTENCE_END = re.compile(r"[.!?\"')\]]\s*$")

ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
df = ds.to_pandas()
print(f"rows: {len(df)}")
print(f"unique instructions: {df.instruction.nunique()}")
print(df.instruction.value_counts().head(3).to_string(), "\n")

df["in_len"] = df.input.str.len()
df["out_len"] = df.output.str.len()
print("char length quantiles")
print(df[["in_len", "out_len"]].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(0).to_string(), "\n")

has_artifact = df.output.apply(lambda s: bool(ARTIFACT.search(s)))
truncated = ~df.output.str.strip().apply(lambda s: bool(SENTENCE_END.search(s)))
dup_input = df.input.str.lower().str.strip().duplicated()
dup_pair = (df.input + "||" + df.output).str.lower().duplicated()

for name, mask in [
    ("output contains 'Chat Doctor'", has_artifact),
    ("output has no terminal punctuation", truncated),
    ("duplicate input (exact, normalized)", dup_input),
    ("duplicate input+output pair", dup_pair),
]:
    print(f"{name:42s} {mask.sum():6d}  ({mask.mean():5.1%})")

# where does the artifact sit: a greeting we could strip, or mid-sentence damage
sample = df.loc[has_artifact, "output"].head(400)
pos = sample.apply(lambda s: ARTIFACT.search(s).start() / max(len(s), 1))
print(f"\nartifact relative position in output: median {np.median(pos):.2f}, "
      f"share in first 15% of text {np.mean(pos < 0.15):.1%}")

non_latin = df.output.apply(lambda s: sum(ord(c) > 127 for c in s) / max(len(s), 1))
print(f"outputs with >5% non-ascii chars: {(non_latin > 0.05).mean():.2%}")
