"""QLoRA fine-tuning of Qwen2.5-7B-Instruct on the cleaned HealthCareMagic split.

Written for a free Colab T4, which drives most of the configuration:
  - Turing has no bf16, so the run is fp16 and the 4-bit compute dtype is fp16 too
  - flash-attention needs sm80+, so attention falls back to sdpa
  - 16 GB VRAM means gradient checkpointing and a micro-batch of 1

Only the adapter is saved. The base model stays untouched on the Hub.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sft_data import PadCollator, SFTDataset, read_jsonl

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("adapter"))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=640,
                    help="longest sequence in the cleaned corpus is 524 tokens")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--limit-train", type=int, default=None,
                    help="cut the training set down, for a smoke run")
    ap.add_argument("--limit-val", type=int, default=None,
                    help="evaluate on the first N validation rows only")
    ap.add_argument("--wandb-project", default="chatdoctor-qlora")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args()


def load_model(name: str, cuda: bool):
    """4-bit on GPU. Without a GPU the model loads unquantised on CPU, which is only
    useful for checking the wiring on a small model before paying for a GPU session:
    bitsandbytes has no CPU kernels for nf4."""
    if not cuda:
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa")
        model.config.use_cache = False
        return model

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
    model.config.use_cache = False          # incompatible with gradient checkpointing
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def main():
    args = parse_args()
    set_seed(SEED)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_rows = read_jsonl(args.data / "train.jsonl")
    val_rows = read_jsonl(args.data / "val.jsonl")
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_val:
        val_rows = val_rows[: args.limit_val]

    train_ds = SFTDataset(train_rows, tok, args.max_len)
    val_ds = SFTDataset(val_rows, tok, args.max_len)
    print(f"train {len(train_ds)} rows, val {len(val_ds)} rows")

    # warmup in steps, not as a ratio: transformers 5 dropped warmup_ratio, and the
    # number is easier to sanity-check against the step count printed by the Trainer
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, round(0.03 * total_steps))

    cuda = torch.cuda.is_available()
    if not cuda:
        print("no GPU visible, running unquantised on CPU (smoke test only)")
    model = load_model(args.base_model, cuda)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # attention plus MLP. Attention-only trains faster but adapts style less,
        # and the MLP blocks are where the wordiness of the base model lives.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    report_to = [] if args.no_wandb else ["wandb"]
    if report_to:
        import os
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    targs = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        max_grad_norm=0.3,
        # paged optimiser so a long sequence cannot push the run over the 16 GB edge
        optim="paged_adamw_8bit" if cuda else "adamw_torch",
        fp16=cuda,
        bf16=False,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=report_to,
        run_name=args.run_name,
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tok.pad_token_id),
    )
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    history = args.out / "log_history.json"
    history.write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"adapter saved to {args.out}, log history to {history}")
    if train_ds.n_truncated or val_ds.n_truncated:
        print(f"truncated at max_len: train {train_ds.n_truncated}, val {val_ds.n_truncated}")


if __name__ == "__main__":
    main()
