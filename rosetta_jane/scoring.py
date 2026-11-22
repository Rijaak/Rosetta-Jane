"""Scoring a trained adapter by comparison: is the model better with it than without it?

The same questions are put to the unchanged model and to the model with the adapter,
through the swarm, and the two are compared on how well they predict the right answers.
A score only means something on questions the adapter never saw while training, so a
share of every training file is set aside for this and never trained on.

A score card belongs to whoever made it. It is a file on their device, built from their
own questions. Nothing sends it anywhere. People compare cards by choosing to share them,
because a score reported by a stranger, on questions you cannot see, cannot be checked.

A score informs the decision to use an adapter. It does not make it: nothing in
Rosetta-Jane applies an adapter, whatever it scores.

Plain Python, shared by the desktop program and the phone app.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

HOLD_OUT_EVERY = 10           # one pair in ten is kept back for scoring
MIN_PAIRS_TO_HOLD_OUT = 5     # with fewer pairs than this, all are trained on and none scored


def split_holdout(examples: Sequence, every: int = HOLD_OUT_EVERY) -> Tuple[list, list]:
    """(pairs to train on, pairs kept back for scoring). The same file always splits the same way."""
    examples = list(examples)
    if len(examples) < MIN_PAIRS_TO_HOLD_OUT:
        return examples, []
    every = max(2, min(every, len(examples)))
    held = [e for i, e in enumerate(examples) if i % every == every - 1]
    kept = [e for i, e in enumerate(examples) if i % every != every - 1]
    return kept, held


def tally(loss_sum: float, right: int, tokens: int) -> dict:
    """One side of a comparison, from totals over every graded answer token."""
    if tokens <= 0:
        return {"loss": None, "right_next_word": None}
    return {"loss": loss_sum / tokens, "right_next_word": right / tokens}


def card(name: str, fingerprint: str, pairs: int, tokens: int, before: dict, after: dict, *,
         held_out: bool, checked: int = 0, unchecked: int = 0) -> dict:
    return {"adapter": name, "base_model_fingerprint": fingerprint,
            "scored": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": pairs, "answer_tokens": tokens,
            "never_trained_on": held_out, "before": before, "after": after,
            "work_checked": checked, "work_unchecked": unchecked, "verdict": verdict(before, after, pairs)}


def verdict(before: dict, after: dict, pairs: int) -> str:
    """One cautious sentence. Small differences and small samples are called what they are."""
    if before.get("loss") is None or after.get("loss") is None:
        return "not scored"
    if not (math.isfinite(before["loss"]) and math.isfinite(after["loss"])):
        return "broken: the scores are not numbers"
    change = (before["loss"] - after["loss"]) / max(before["loss"], 1e-9)
    if change > 0.02:
        text = "better with the adapter"
    elif change < -0.02:
        text = "worse with the adapter"
    else:
        text = "about the same with and without the adapter"
    if pairs < 20:
        text += f" (on only {pairs} pair{'s' if pairs != 1 else ''}; too few to rely on)"
    return text


def share_text(c: Optional[dict]) -> str:
    """A score card as a short message its owner can choose to send or post. It holds no questions or answers."""
    if not c or c.get("before", {}).get("loss") is None:
        return ""
    b, a = c["before"], c["after"]
    lines = [f"Rosetta-Jane score card: {c['adapter'] or 'adapter'}",
             f"Base model: {c['base_model_fingerprint']}",
             f"Scored {c['scored']} on {c['pairs']} pairs "
             + ("kept back from training (never trained on)." if c["never_trained_on"] else "of the scorer's own choosing."),
             f"Right next word: {b['right_next_word']:.1%} without the adapter, {a['right_next_word']:.1%} with it.",
             f"Loss (lower is better): {b['loss']:.3f} without, {a['loss']:.3f} with.",
             f"Result: {c['verdict']}.",
             f"Scoring calculations double-checked: {c.get('work_checked', 0)}. Resting on one peer's word: "
             f"{c.get('work_unchecked', 0)}.",
             "One person's score on one set of questions. Score it yourself before you rely on it."]
    return "\n".join(lines)


def describe(c: Optional[dict]) -> List[str]:
    """A score card as lines a person can read."""
    if not c or c.get("before", {}).get("loss") is None:
        return ["Not scored: there were too few pairs to keep any back for scoring."]
    b, a = c["before"], c["after"]
    lines = [f"Scored on {c['pairs']} pairs "
             + ("the adapter never saw in training." if c["never_trained_on"] else "chosen by the person scoring."),
             f"  Right next word: {b['right_next_word']:.1%} without the adapter, {a['right_next_word']:.1%} with it.",
             f"  Loss (lower is better): {b['loss']:.3f} without, {a['loss']:.3f} with.",
             f"  Result: {c['verdict']}."]
    if c.get("work_unchecked"):
        lines.append(f"  {c['work_unchecked']} of the scoring calculations rested on one peer's word.")
    return lines
