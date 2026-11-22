"""Training pairs: read a JSONL file of questions and answers and turn them into tokens.

Each line is a JSON object with at least "question" and "answer". The model is
only graded on the answer; the question tokens are masked out of the loss.
"""
from __future__ import annotations

import json
import random
from typing import List, Tuple

import torch

Example = Tuple[torch.Tensor, torch.Tensor]  # (input ids [1, T], labels [1, T])


def read_pairs(path: str) -> List[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path} line {n} is not valid JSON: {e}") from None
            if not isinstance(row, dict) or not row.get("question") or not row.get("answer"):
                raise ValueError(f"{path} line {n} needs non-empty 'question' and 'answer'")
            pairs.append(row)
    if not pairs:
        raise ValueError(f"{path} has no pairs")
    return pairs


def tokenize_pairs(pairs: List[dict], tokenizer, max_tokens: int) -> Tuple[List[Example], int]:
    """Returns (examples, number of pairs cut short to fit max_tokens)."""
    examples: List[Example] = []
    truncated = 0
    for row in pairs:
        user = [{"role": "user", "content": row["question"]}]
        full = user + [{"role": "assistant", "content": row["answer"]}]
        prompt_ids = tokenizer.apply_chat_template(user, tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
        prompt_ids = list(getattr(prompt_ids, "input_ids", prompt_ids))
        full_ids = list(getattr(full_ids, "input_ids", full_ids))
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("this tokenizer's chat template does not put the question first; cannot mask it")
        if len(full_ids) > max_tokens:
            full_ids = full_ids[:max_tokens]
            truncated += 1
        if len(full_ids) - len(prompt_ids) < 1:
            continue  # nothing of the answer survived; skip
        ids = torch.tensor([full_ids], dtype=torch.long)
        labels = ids.clone()
        labels[0, :len(prompt_ids)] = -100
        examples.append((ids, labels))
    if not examples:
        raise ValueError("no usable pairs after tokenizing; is --max-tokens too small?")
    return examples, truncated


class Schedule:
    """Fixed, repeatable order of examples, so a resumed run replays the same data."""

    def __init__(self, num_examples: int, seed: int):
        self.n = num_examples
        self.seed = seed
        self._epoch = -1
        self._order: List[int] = []

    def index(self, position: int) -> int:
        epoch, offset = divmod(position, self.n)
        if epoch != self._epoch:
            order = list(range(self.n))
            random.Random(f"{self.seed}-{epoch}").shuffle(order)
            self._epoch, self._order = epoch, order
        return self._order[offset]
