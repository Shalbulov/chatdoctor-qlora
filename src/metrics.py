"""Scoring the two answer columns against the held-out doctor replies.

Three groups of numbers, because no single one of them settles the question:

  ROUGE-L        lexical overlap with the reference. On free-form medical advice it is
                 a weak proxy: two correct answers can share almost no wording. It is
                 here because the task asks for a standard metric and because a large
                 move in it is still informative.
  BERTScore      embedding overlap, less sensitive to paraphrase. Rescaled against the
                 package baseline, otherwise every score sits near 0.85 and differences
                 are invisible.
  Style markers  length, bullet lists, disclaimers. This is what the fine-tune was
                 actually aimed at, so it is measured directly rather than inferred
                 from a metric that only sees it sideways.

Runs on CPU in about a minute; only generation needs a GPU.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

COLUMNS = ("base", "tuned")

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
BOLD = re.compile(r"\*\*.+?\*\*")
HEADING = re.compile(r"^\s*#{1,6}\s+", re.M)
DISCLAIMER = re.compile(
    r"consult (?:a|your|with) (?:doctor|physician|healthcare|medical)|"
    r"see (?:a|your) (?:doctor|physician|specialist)|"
    r"seek (?:immediate |urgent )?medical (?:attention|advice|help)|"
    r"i am (?:an ai|not a doctor)|this is not (?:a substitute|medical advice)",
    re.I,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=Path("reports/predictions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports/metrics.json"))
    ap.add_argument("--markdown", type=Path, default=Path("reports/metrics.md"))
    ap.add_argument("--no-bertscore", action="store_true",
                    help="skip the roberta-large download when only ROUGE is wanted")
    return ap.parse_args()


def rouge_l(preds: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def bert_score_f1(preds: list[str], refs: list[str]) -> list[float]:
    from bert_score import score

    _, _, f1 = score(preds, refs, lang="en", rescale_with_baseline=True, verbose=False)
    return f1.tolist()


def style(texts: list[str]) -> dict:
    return {
        "chars": statistics.mean(len(t) for t in texts),
        "words": statistics.mean(len(t.split()) for t in texts),
        "bullet_share": sum(bool(BULLET.search(t)) for t in texts) / len(texts),
        "bold_share": sum(bool(BOLD.search(t)) for t in texts) / len(texts),
        "heading_share": sum(bool(HEADING.search(t)) for t in texts) / len(texts),
        "disclaimer_share": sum(bool(DISCLAIMER.search(t)) for t in texts) / len(texts),
    }


def render(report: dict) -> str:
    def row(label: str, path: list, fmt: str = "{:.3f}") -> str:
        vals = []
        for col in COLUMNS:
            node = report[col]
            for key in path:
                node = node[key]
            vals.append(fmt.format(node))
        return f"| {label} | {vals[0]} | {vals[1]} |"

    ref = report["reference"]
    lines = [
        f"# Evaluation, {report['n']} held-out examples",
        "",
        "| | base | base + adapter |",
        "|---|---|---|",
        row("ROUGE-L F1", ["rougeL"]),
    ]
    if "bertscore" in report["base"]:
        lines.append(row("BERTScore F1 (rescaled)", ["bertscore"]))
    lines += [
        row("wins vs the other column", ["rouge_wins"], "{:d}"),
        row("mean words", ["style", "words"], "{:.0f}"),
        row("mean characters", ["style", "chars"], "{:.0f}"),
        row("answers with a bullet list", ["style", "bullet_share"], "{:.0%}"),
        row("answers with bold text", ["style", "bold_share"], "{:.0%}"),
        row("answers with a heading", ["style", "heading_share"], "{:.0%}"),
        row("answers with a disclaimer", ["style", "disclaimer_share"], "{:.0%}"),
        "",
        f"Reference answers for comparison: {ref['words']:.0f} words, "
        f"{ref['bullet_share']:.0%} with a bullet list, "
        f"{ref['disclaimer_share']:.0%} with a disclaimer.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    rows = [json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    refs = [r["reference"] for r in rows]

    report = {"n": len(rows), "reference": style(refs)}
    per_example = {}
    for col in COLUMNS:
        preds = [r[col] for r in rows]
        scores = rouge_l(preds, refs)
        per_example[col] = {"rougeL": scores}
        report[col] = {"rougeL": statistics.mean(scores), "style": style(preds)}

    if not args.no_bertscore:
        for col in COLUMNS:
            f1 = bert_score_f1([r[col] for r in rows], refs)
            per_example[col]["bertscore"] = f1
            report[col]["bertscore"] = statistics.mean(f1)

    # a paired count, because a mean over 20 examples can be carried by one outlier
    base_wins = sum(b > t for b, t in zip(per_example["base"]["rougeL"],
                                          per_example["tuned"]["rougeL"]))
    report["base"]["rouge_wins"] = base_wins
    report["tuned"]["rouge_wins"] = len(rows) - base_wins

    report["per_example"] = per_example
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = render(report)
    args.markdown.write_text(table, encoding="utf-8")
    print(table)
    print(f"written to {args.out} and {args.markdown}")


if __name__ == "__main__":
    main()
