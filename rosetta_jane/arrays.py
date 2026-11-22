"""The few operations on numbers that the answering and checking logic needs, for either kind of array.

A desktop holds PyTorch tensors, a phone holds NumPy arrays. Everything that decides
whom to believe is written once against these helpers, so both run the same rules.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:                               # a phone has no PyTorch
    torch = None


def is_numpy(x) -> bool:
    return isinstance(x, np.ndarray)


def ndim(x) -> int:
    return x.ndim if is_numpy(x) else x.dim()


def cat(blocks: list):
    """Join blocks of hidden states [1, tokens, hidden] along the token axis."""
    return np.concatenate(blocks, axis=1) if is_numpy(blocks[0]) else torch.cat(blocks, dim=1)


def agree(a, b, tolerance: float) -> bool:
    """Same result, allowing for the small differences between different hardware."""
    if tuple(a.shape) != tuple(b.shape):
        return False
    if is_numpy(a) or is_numpy(b):
        a = np.asarray(a, dtype=np.float32) if is_numpy(a) else a.float().numpy()
        b = np.asarray(b, dtype=np.float32) if is_numpy(b) else b.float().numpy()
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return False
        return float(np.linalg.norm((a - b).ravel())) <= tolerance * (float(np.linalg.norm(a.ravel())) + 1e-6)
    a, b = a.float(), b.float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return False
    return float((a - b).norm()) <= tolerance * (float(a.norm()) + 1e-6)


def groups(outs: List[Tuple[object, object]], tolerance: float) -> List[List[Tuple[object, object]]]:
    """Sort results into sets that agree with each other, largest set first."""
    found: List[List[Tuple[object, object]]] = []
    for item in outs:
        for g in found:
            if agree(g[0][1], item[1], tolerance):
                g.append(item)
                break
        else:
            found.append([item])
    return sorted(found, key=len, reverse=True)


def make_generator(seed: Optional[int]):
    if seed is None:
        return None
    return torch.Generator().manual_seed(seed) if torch is not None else np.random.default_rng(seed)


def pick_token(logits, temperature: float, top_p: float, gen) -> int:
    if is_numpy(logits):
        logits = logits.reshape(-1).astype(np.float64)
        if temperature <= 0:
            return int(np.argmax(logits))
        z = logits / temperature
        probs = np.exp(z - z.max())
        probs /= probs.sum()
        rng = gen if gen is not None else np.random.default_rng()
        if 0 < top_p < 1:
            order = np.argsort(-probs, kind="stable")
            ordered = probs[order]
            keep = np.cumsum(ordered) - ordered < top_p     # always keeps the most likely token
            ordered = ordered * keep
            return int(order[rng.choice(len(ordered), p=ordered / ordered.sum())])
        return int(rng.choice(len(probs), p=probs))
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    if 0 < top_p < 1:
        ordered, idx = probs.sort(descending=True)
        keep = ordered.cumsum(0) - ordered < top_p      # always keeps the most likely token
        ordered = ordered * keep
        return int(idx[torch.multinomial(ordered / ordered.sum(), 1, generator=gen)])
    return int(torch.multinomial(probs, 1, generator=gen))
