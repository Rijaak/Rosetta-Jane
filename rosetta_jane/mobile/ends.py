"""The asking side's own part of the model on a device without PyTorch.

The token table, the first layers, the final norm and the output layer stay on the
device that asks: text goes in here and comes out here, and only numbers from
partway through the model ever leave. Mirrors the desktop's LocalEnds.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import st
from .engine import Stack, _rms, _rms_back, f32, scan


class Ends:
    def __init__(self, model_dir: str, own_layers: int):
        self.stack = Stack(model_dir, 0, own_layers, own=True)
        self.fingerprint, self.num_layers = self.stack.fingerprint, self.stack.total
        self.own_layers, self.hidden = own_layers, self.stack.hidden
        self.eos_token_id = self.stack.cfg.get("eos_token_id")
        found = scan(model_dir)
        base = self.stack.prefix[:-len("layers.")]
        self._embed_key = base + "embed_tokens.weight"
        if self._embed_key not in found or base + "norm.weight" not in found:
            raise ValueError("this model's token table or final norm could not be found")
        self._embed_file = found[self._embed_key][0]
        self.vocab = int(found[self._embed_key][1][0])
        self._norm = found[base + "norm.weight"][0].get(base + "norm.weight")
        head = "lm_head.weight" if "lm_head.weight" in found else self._embed_key      # shared when absent
        self._head = found[head][0].get(head)
        for f in {id(v[0]): v[0] for v in found.values()}.values():
            if f is not self._embed_file:
                f.close()

    def embed(self, ids: Sequence[int]) -> np.ndarray:
        return self._embed_file.rows(self._embed_key, [int(i) for i in ids])

    # -- answering ---------------------------------------------------------------------------
    def head(self, new_ids: List[int], cache):
        out, cache = self.stack.step(0, self.own_layers, self.embed(new_ids), cache)
        return out[None], cache

    def logits(self, h_last: np.ndarray) -> np.ndarray:
        """Scores for the next token, given the last position's final hidden state."""
        h = np.asarray(h_last, dtype=f32).reshape(-1, self.hidden)[-1]
        normed, _ = _rms(h, self._norm, self.stack.eps)
        return self._head @ normed

    # -- training ----------------------------------------------------------------------------
    def train_forward(self, ids: Sequence[int], adapters: Dict[str, np.ndarray], scale: float):
        """This device's first layers for one example, keeping what the backward pass needs."""
        return self.stack.forward(0, self.own_layers, self.embed(ids), adapters, scale)

    def train_backward(self, graph, g: np.ndarray) -> Dict[str, np.ndarray]:
        return self.stack.backward(graph, g)[1]

    def grade(self, h_last: np.ndarray, labels: Sequence[int]) -> Tuple[float, int, int]:
        """For scoring: (summed loss, answer words predicted right, answer words) for one example."""
        h = np.asarray(h_last, dtype=f32).reshape(-1, self.hidden)
        normed, _ = _rms(h, self._norm, self.stack.eps)
        logits = (normed @ self._head.T)[:-1].astype(np.float64)
        targets = np.asarray(labels[1:], dtype=np.int64)
        rows = np.nonzero(targets != -100)[0]
        if len(rows) == 0:
            return 0.0, 0, 0
        z = logits[rows] - logits[rows].max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        want = targets[rows]
        return float(-logp[np.arange(len(rows)), want].sum()), int((logits[rows].argmax(-1) == want).sum()), len(rows)

    def loss(self, h_last: np.ndarray, labels: Sequence[int]) -> Tuple[float, np.ndarray]:
        """How wrong the model was on the answer part of one example, and the gradient to send back.

        labels use -100 for positions that are not graded (the question).
        """
        h = np.asarray(h_last, dtype=f32).reshape(-1, self.hidden)
        normed, r = _rms(h, self._norm, self.stack.eps)
        logits = (normed @ self._head.T)[:-1].astype(np.float64)       # position t predicts token t+1
        targets = np.asarray(labels[1:], dtype=np.int64)
        graded = targets != -100
        count = int(graded.sum())
        if count == 0:
            raise ValueError("nothing to learn from: this example has no answer tokens")
        z = logits - logits.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        rows = np.nonzero(graded)[0]
        loss = float(-logp[rows, targets[rows]].mean())
        d = np.exp(logp)
        d[rows, targets[rows]] -= 1.0
        d[~graded] = 0.0
        d = (d / count).astype(f32)
        g_normed = np.zeros_like(h)
        g_normed[:-1] = d @ self._head
        return loss, _rms_back(g_normed, h, self._norm, r)
