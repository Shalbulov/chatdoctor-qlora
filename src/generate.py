"""Answers for the held-out set, from the base model and from base plus adapter.

Both sets of answers come out of one loaded model. The adapter is attached once and
switched off for the base pass through `disable_adapter()`, which means the comparison
is against exactly the weights the adapter sits on, with the same 4-bit quantisation on
both sides. Loading an unquantised base separately would have made quantisation a second
difference between the two columns and muddied the metrics.

Decoding is greedy. Sampling would put run-to-run noise on top of a 20 example
comparison, and the thing being measured is style, which greedy decoding shows plainly.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, default=Path("data/eval_set.jsonl"))
    ap.add_argument("--adapter", type=Path, default=Path("adapter"))
    ap.add_argument("--out", type=Path, default=Path("reports/predictions.jsonl"))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="generous on purpose: truncating the wordy base model would "
                         "flatter it on length and cut its answers mid-sentence")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def load_model(name: str, adapter: Path, cuda: bool):
    if cuda:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=quant,
            device_map={"": 0},
            attn_implementation="sdpa",
            dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


def generate(model, tok, prompts: list[str], max_new_tokens: int) -> list[str]:
    batch = tok(prompts, return_tensors="pt", padding=True,
                add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    # left padding, so every row in the batch starts generating at the same offset
    new_tokens = out[:, batch["input_ids"].shape[1]:]
    return [t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True)]


def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for batched generation

    rows = [json.loads(line) for line in args.eval_set.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    cuda = torch.cuda.is_available()
    if not cuda:
        print("no GPU visible, generation on CPU will be very slow")
    model = load_model(args.base_model, args.adapter, cuda)

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": r["instruction"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for r in rows
    ]

    results = []
    for start in range(0, len(rows), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        with model.disable_adapter():
            base = generate(model, tok, chunk, args.max_new_tokens)
        tuned = generate(model, tok, chunk, args.max_new_tokens)
        for row, b, t in zip(rows[start:start + args.batch_size], base, tuned):
            results.append({
                "instruction": row["instruction"],
                "reference": row["response"],
                "base": b,
                "tuned": t,
            })
        print(f"{len(results)}/{len(rows)} done")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
