"""The layer arithmetic of a language model, written with NumPy only.

A desktop peer lets PyTorch do this. A phone cannot, so the same calculation is
spelled out here: normalise, attend, feed forward, for each layer in turn, and
the reverse of each step for training. It reads the ordinary model files, so a
phone uses the same model copy as everyone else and returns the same numbers.

Covers the model families that share the Llama layer design. Others are refused
rather than computed wrongly. Calculations are done in 32-bit floats, which is
what a desktop peer started with `--dtype float32` does; the two can check each
other.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import st

# Families checked against PyTorch on small random models (see tests/test_mobile.py).
SUPPORTED = frozenset({"llama", "mistral", "qwen2"})
VARIANT = "float32"
_LAYER_RE = re.compile(r"^(?P<prefix>(?:.*\.)?layers\.)(?P<index>\d+)\.(?P<rest>.+)$")
_LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
f32 = np.float32


class Unsupported(ValueError):
    pass


def read_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    kind = str(cfg.get("model_type", "unknown"))
    if kind not in SUPPORTED:
        raise Unsupported(f"model family {kind!r} cannot run on this device yet. "
                          f"Supported here: {', '.join(sorted(SUPPORTED))}.")
    if str(cfg.get("hidden_act", "silu")) != "silu":
        raise Unsupported("only models using the SiLU activation are supported on this device")
    for key in ("hidden_size", "num_attention_heads", "num_hidden_layers"):
        if not isinstance(cfg.get(key), int) or cfg[key] <= 0:
            raise Unsupported(f"config.json has no usable {key}")
    return cfg


def scan(model_dir: str) -> Dict[str, Tuple[st.File, List[int], str]]:
    found = {}
    for path in st.weight_files(model_dir):
        f = st.File(path)
        for key in f.keys():
            found[key] = (f, f.shape(key), f.dtype(key))
    return found


def layout(keys) -> Tuple[str, int]:
    """The name prefix of the decoder layers and how many there are."""
    seen: Dict[str, set] = {}
    for key in keys:
        m = _LAYER_RE.match(key)
        if m:
            seen.setdefault(m.group("prefix"), set()).add(int(m.group("index")))
    if not seen:
        raise Unsupported("no decoder layers found in the weight files")
    prefix = max(seen, key=lambda p: len(seen[p]))
    count = len(seen[prefix])
    if seen[prefix] != set(range(count)):
        raise Unsupported("decoder layer numbering has gaps; unsupported model layout")
    return prefix, count


def fingerprint(model_dir: str, found=None) -> str:
    """The same model ID the desktop program computes, so phones and PCs recognise the same model."""
    import hashlib
    found = found if found is not None else scan(model_dir)
    h = hashlib.sha256()
    for key, (_, shape, dtype) in sorted(found.items()):
        h.update(f"{key}|{shape}|{dtype}\n".encode())
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        cfg = json.load(f)
    for volatile in ("_name_or_path", "transformers_version"):
        cfg.pop(volatile, None)
    h.update(json.dumps(cfg, sort_keys=True).encode())
    return h.hexdigest()[:16]


# -- the pieces of a layer, each with its reverse ---------------------------------------------
def _rms(x: np.ndarray, w: np.ndarray, eps: float):
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=f32) + f32(eps))
    return (x * r) * w, r


def _rms_back(g: np.ndarray, x: np.ndarray, w: np.ndarray, r: np.ndarray) -> np.ndarray:
    gz = g * w
    return gz * r - x * (r ** 3) * np.mean(gz * x, axis=-1, keepdims=True, dtype=f32)


def _silu(z: np.ndarray):
    sig = 1.0 / (1.0 + np.exp(-z))
    return z * sig, sig


def _rot(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _rot_back(z: np.ndarray) -> np.ndarray:
    half = z.shape[-1] // 2
    return np.concatenate([z[..., half:], -z[..., :half]], axis=-1)


class Layer:
    """One decoder layer's weights."""

    def __init__(self, index: int, found, prefix: str, cfg: dict):
        self.index = index
        base = f"{prefix}{index}."

        def get(name: str, required: bool = True) -> Optional[np.ndarray]:
            key = base + name
            if key not in found:
                if required:
                    raise Unsupported(f"weight {key} is missing; this model's layers are laid out differently")
                return None
            return found[key][0].get(key)
        self.w = {name: get(name + ".weight") for name in _LINEARS}
        self.b = {name: get(name + ".bias", required=False) for name in _LINEARS}
        self.norm1 = get("input_layernorm.weight")
        self.norm2 = get("post_attention_layernorm.weight")
        expected = {base + n + s for n in _LINEARS for s in (".weight", ".bias")} | \
            {base + "input_layernorm.weight", base + "post_attention_layernorm.weight"}
        extra = [k for k in found if k.startswith(base) and k not in expected]
        if extra:                                  # a part this code does not know how to apply
            raise Unsupported(f"layer {index} has parts this device cannot run: {extra[0][len(base):]}")

    def shapes(self, targets: Sequence[str]) -> Dict[str, Tuple[int, int]]:
        return {f"{self.index}.{name}": (int(self.w[name].shape[1]), int(self.w[name].shape[0]))
                for name in _LINEARS if name.split(".")[-1] in targets}


class Stack:
    """A run of consecutive layers, loaded into memory, ready to compute."""

    def __init__(self, model_dir: str, start: int, end: int, own: bool = False):
        self.cfg = read_config(model_dir)
        found = scan(model_dir)
        self.prefix, self.total = layout(found.keys())
        if own and not (start == 0 < end <= self.total):
            raise ValueError(f"keep between 1 and {self.total} layers on your own device")
        if not own and not 1 <= start < end <= self.total:
            raise ValueError(f"layers must be within 1:{self.total}; layer 0 stays with the person asking")
        self.fingerprint = fingerprint(model_dir, found)
        self.start, self.end = start, end
        self.layers = [Layer(i, found, self.prefix, self.cfg) for i in range(start, end)]
        for f in {id(v[0]): v[0] for v in found.values()}.values():
            f.close()
        c = self.cfg
        self.hidden = int(c["hidden_size"])
        self.heads = int(c["num_attention_heads"])
        self.kv_heads = int(c.get("num_key_value_heads") or self.heads)
        self.head_dim = int(c.get("head_dim") or self.hidden // self.heads)
        if self.heads % self.kv_heads or self.head_dim % 2 or self.head_dim * self.heads != self.hidden:
            raise Unsupported("this model's attention shape is not supported on this device")
        self.eps = float(c.get("rms_norm_eps", 1e-6))
        theta = float(c.get("rope_theta", 10000.0))
        self._inv_freq = (1.0 / (theta ** (np.arange(0, self.head_dim, 2, dtype=f32) / f32(self.head_dim)))).astype(f32)
        self._scale = f32(1.0 / math.sqrt(self.head_dim))

    # -- shared pieces ----------------------------------------------------------------------
    def _rope(self, first: int, count: int):
        freqs = np.arange(first, first + count, dtype=f32)[:, None] * self._inv_freq[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(f32), np.sin(emb).astype(f32)

    def _split(self, x: np.ndarray, heads: int) -> np.ndarray:           # [T, heads*d] -> [heads, T, d]
        return x.reshape(x.shape[0], heads, self.head_dim).transpose(1, 0, 2)

    def _merge(self, x: np.ndarray) -> np.ndarray:                       # [heads, T, d] -> [T, heads*d]
        return x.transpose(1, 0, 2).reshape(x.shape[1], -1)

    def _linear(self, layer: Layer, name: str, x: np.ndarray, lora=None, tape=None) -> np.ndarray:
        y = x @ layer.w[name].T
        if layer.b[name] is not None:
            y = y + layer.b[name]
        if lora is not None:
            pair = lora.get(f"{layer.index}.{name}")
            if pair is not None:
                a, b, scale = pair
                xa = x @ a.T
                y = y + (xa @ b.T) * f32(scale)
                if tape is not None:
                    tape[name] = (x, xa)
                return y
        if tape is not None:
            tape[name] = (x, None)
        return y

    def _linear_back(self, layer: Layer, name: str, g: np.ndarray, lora, tape, grads) -> np.ndarray:
        x, xa = tape[name]
        gx = g @ layer.w[name]
        if xa is not None:
            a, b, scale = lora[f"{layer.index}.{name}"]
            gb = g @ b                                  # [T, r]
            gx = gx + (gb @ a) * f32(scale)
            grads[f"{layer.index}.{name}.A"] = (gb.T @ x) * f32(scale)
            grads[f"{layer.index}.{name}.B"] = (g.T @ xa) * f32(scale)
        return gx

    def _attend(self, layer: Layer, x: np.ndarray, past, lora=None, tape=None):
        t = x.shape[0]
        before = 0 if past is None else past[0].shape[1]
        h, r = _rms(x, layer.norm1, self.eps)
        lt = {} if tape is not None else None
        q = self._split(self._linear(layer, "self_attn.q_proj", h, lora, lt), self.heads)
        k = self._split(self._linear(layer, "self_attn.k_proj", h, lora, lt), self.kv_heads)
        v = self._split(self._linear(layer, "self_attn.v_proj", h, lora, lt), self.kv_heads)
        cos, sin = self._rope(before, t)
        q = q * cos + _rot(q) * sin
        k = k * cos + _rot(k) * sin
        if past is not None:
            k = np.concatenate([past[0], k], axis=1)
            v = np.concatenate([past[1], v], axis=1)
        group = self.heads // self.kv_heads
        kr = np.repeat(k, group, axis=0) if group > 1 else k
        vr = np.repeat(v, group, axis=0) if group > 1 else v
        scores = (q @ kr.transpose(0, 2, 1)) * self._scale            # [heads, T, before+T]
        if t > 1:                                                     # a token may not look ahead
            blocked = np.triu(np.ones((t, t), dtype=bool), k=1)
            scores[:, :, before:][:, blocked] = -np.inf
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        merged = self._merge(p @ vr)
        out = self._linear(layer, "self_attn.o_proj", merged, lora, lt)
        if tape is not None:
            tape.update(attn=(x, h, r, q, kr, vr, p, cos, sin, lt))
        return x + out, (k, v)

    def _attend_back(self, layer: Layer, g: np.ndarray, lora, tape, grads) -> np.ndarray:
        x, h, r, q, kr, vr, p, cos, sin, lt = tape["attn"]
        t = x.shape[0]
        group = self.heads // self.kv_heads
        ga = self._split(self._linear_back(layer, "self_attn.o_proj", g, lora, lt, grads), self.heads)
        gp = ga @ vr.transpose(0, 2, 1)
        gvr = p.transpose(0, 2, 1) @ ga
        gs = p * (gp - np.sum(gp * p, axis=-1, keepdims=True))
        gq = (gs @ kr) * self._scale
        gkr = (gs.transpose(0, 2, 1) @ q) * self._scale
        if group > 1:
            gk = gkr.reshape(self.kv_heads, group, t, self.head_dim).sum(axis=1)
            gv = gvr.reshape(self.kv_heads, group, t, self.head_dim).sum(axis=1)
        else:
            gk, gv = gkr, gvr
        gq = gq * cos + _rot_back(gq * sin)
        gk = gk * cos + _rot_back(gk * sin)
        gh = self._linear_back(layer, "self_attn.q_proj", self._merge(gq), lora, lt, grads)
        gh = gh + self._linear_back(layer, "self_attn.k_proj", self._merge(gk), lora, lt, grads)
        gh = gh + self._linear_back(layer, "self_attn.v_proj", self._merge(gv), lora, lt, grads)
        return g + _rms_back(gh, x, layer.norm1, r)

    def _feed(self, layer: Layer, x: np.ndarray, lora=None, tape=None) -> np.ndarray:
        h, r = _rms(x, layer.norm2, self.eps)
        lt = {} if tape is not None else None
        a = self._linear(layer, "mlp.gate_proj", h, lora, lt)
        u = self._linear(layer, "mlp.up_proj", h, lora, lt)
        act, sig = _silu(a)
        out = self._linear(layer, "mlp.down_proj", act * u, lora, lt)
        if tape is not None:
            tape.update(feed=(x, r, a, u, act, sig, lt))
        return x + out

    def _feed_back(self, layer: Layer, g: np.ndarray, lora, tape, grads) -> np.ndarray:
        x, r, a, u, act, sig, lt = tape["feed"]
        gt = self._linear_back(layer, "mlp.down_proj", g, lora, lt, grads)
        ga = gt * u * (sig * (1.0 + a * (1.0 - sig)))
        gh = self._linear_back(layer, "mlp.gate_proj", ga, lora, lt, grads)
        gh = gh + self._linear_back(layer, "mlp.up_proj", gt * act, lora, lt, grads)
        return g + _rms_back(gh, x, layer.norm2, r)

    def _pick(self, first: int, last: int) -> List[Layer]:
        if not self.start <= first < last <= self.end:
            raise ValueError(f"this device holds layers {self.start}:{self.end}")
        return self.layers[first - self.start:last - self.start]

    # -- answering: new tokens in, continuing from what was seen before ------------------------
    def step(self, first: int, last: int, x: np.ndarray, cache: Optional[list],
             lora: Optional[dict] = None) -> Tuple[np.ndarray, list]:
        """Apply layers [first, last) to new hidden states [tokens, hidden]. Returns (output, new cache).

        lora, when given, is a team's adapter as {"<layer>.<module>": (A, B, scale)}: the answer then
        comes from the model with that adapter. Without it this is the unchanged model.
        """
        layers = self._pick(first, last)
        x = np.ascontiguousarray(x, dtype=f32)
        new_cache = []
        for i, layer in enumerate(layers):
            x, kv = self._attend(layer, x, None if cache is None else cache[i], lora)
            x = self._feed(layer, x, lora)
            new_cache.append(kv)
        return x, new_cache

    # -- training: one example forward, then backward ------------------------------------------
    def shapes(self, first: int, last: int, targets: Sequence[str]) -> Dict[str, Tuple[int, int]]:
        out = {}
        for layer in self._pick(first, last):
            out.update(layer.shapes(targets))
        return out

    def forward(self, first: int, last: int, x: np.ndarray, adapters: Dict[str, np.ndarray], scale: float):
        """One example forward with LoRA adapters applied. Returns (output, what backward needs)."""
        lora = {key[:-2]: (adapters[key], adapters[key[:-2] + ".B"], scale) for key in adapters if key.endswith(".A")}
        layers = self._pick(first, last)
        x = np.ascontiguousarray(x, dtype=f32)
        tapes = []
        for layer in layers:
            tape: dict = {}
            x, _ = self._attend(layer, x, None, lora, tape)
            x = self._feed(layer, x, lora, tape)
            tapes.append(tape)
        return x, (layers, lora, tapes)

    def backward(self, graph, g: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Given the gradient at the output, return the gradient at the input and for every adapter."""
        layers, lora, tapes = graph
        g = np.ascontiguousarray(g, dtype=f32)
        grads: Dict[str, np.ndarray] = {}
        for layer, tape in zip(reversed(layers), reversed(tapes)):
            g = self._feed_back(layer, g, lora, tape, grads)
            g = self._attend_back(layer, g, lora, tape, grads)
        return g, {k: np.ascontiguousarray(v, dtype=f32) for k, v in grads.items()}
