"""Turning JSONL rows into tensors for supervised fine-tuning.

The one thing worth getting right here is the label mask. The prompt tokens are set
to -100 so the loss only covers the doctor's answer; without that the model spends
half its capacity learning to reproduce patient complaints.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from prompt import SYSTEM_PROMPT

IGNORE = -100


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int = 1024):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.n_truncated = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        # Prompt and answer are tokenised separately rather than tokenising the whole
        # conversation and slicing it: the chat template can merge tokens across the
        # boundary, and an off-by-one there silently shifts the whole mask.
        # tokenize=False and a separate tokeniser call: apply_chat_template returns a
        # BatchEncoding on transformers 5.x and a plain list on 4.x, and the string
        # path behaves the same on both. Qwen's template emits no BOS, hence
        # add_special_tokens=False.
        prompt_text = self.tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["instruction"]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(
            row["response"].strip() + self.tok.eos_token, add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [IGNORE] * len(prompt_ids) + answer_ids

        if len(input_ids) > self.max_len:
            self.n_truncated += 1
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]

        return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    """Right-padding to the longest sequence in the batch. Padding positions are
    masked out of the loss as well, otherwise they leak into it as valid targets."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            gap = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * gap)
            labels.append(b["labels"] + [IGNORE] * gap)
            attention.append([1] * len(b["input_ids"]) + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
