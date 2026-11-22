"""The p2p service on a device without PyTorch.

Answers exactly the requests a desktop peer answers, with the same checks and
the same replies, using the NumPy engine. To everyone else on the network it is
simply a peer running precision 'float32'.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from ..lan.protocol import Channel, ProtocolError
from . import st
from .engine import VARIANT, Stack

MAX_RANK = 256
_KEY_RE = re.compile(r"^(\d+)\.([A-Za-z0-9_.]+)\.(A|B)$")


def check_spec(rank, alpha, targets):
    if not (isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= MAX_RANK):
        raise ValueError(f"LoRA rank must be a whole number from 1 to {MAX_RANK}")
    if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and 0 < alpha <= 4096):
        raise ValueError("bad LoRA alpha")
    if not (isinstance(targets, (list, tuple)) and 1 <= len(targets) <= 16
            and all(isinstance(t, str) and t.replace("_", "").isalnum() and len(t) <= 40 for t in targets)):
        raise ValueError("bad LoRA target names")
    return rank, float(alpha), list(targets)


def check_adapters(adapters: Dict[str, np.ndarray], shapes: Dict[str, tuple], rank: int) -> None:
    """Refuse adapters that do not exactly fit the layers they are meant for."""
    if set(adapters) != {f"{name}.{part}" for name in shapes for part in ("A", "B")}:
        raise ValueError("adapters do not match these layers")
    for key, value in adapters.items():
        fan_in, fan_out = shapes[key[:-2]]
        want = (rank, fan_in) if _KEY_RE.match(key).group(3) == "A" else (fan_out, rank)
        if tuple(value.shape) != want or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"adapter {key} has the wrong shape or contents")


def _hidden_states(tensor, hidden: int, limit: int) -> bool:
    return isinstance(tensor, np.ndarray) and tensor.ndim == 3 and tensor.shape[0] == 1 \
        and tensor.shape[2] == hidden and 1 <= tensor.shape[1] <= limit and tensor.dtype.kind == "f" \
        and bool(np.isfinite(tensor).all())


class NumpyLayerServer:
    def __init__(self, model_dir: str, start: int, end: int, *, max_sessions: int = 2, max_tokens: int = 4096,
                 session_ttl: float = 600.0, training: bool = False, max_train_tokens: int = 1024):
        self.stack = Stack(model_dir, start, end)
        self.start, self.end = start, end
        self.fingerprint, self.hidden, self.variant = self.stack.fingerprint, self.stack.hidden, VARIANT
        self.training, self.paused = training, False
        self.max_sessions, self.max_tokens, self.session_ttl = max_sessions, max_tokens, session_ttl
        self.max_train_tokens = max_train_tokens
        self._lock = threading.Lock()             # one piece of work at a time on this hardware
        self._sessions: List[dict] = []
        self.steps_served = self.tokens_served = self.examples_trained = 0

    def service(self) -> dict:
        with self._lock:
            self._expire()
            free = max(0, self.max_sessions - len(self._sessions))
        return {"model": self.fingerprint, "variant": self.variant, "layers": [self.start, self.end],
                "hidden": self.hidden, "free": free, "train": self.training}

    def attach(self, node) -> None:
        for op in ("open", "step", "close", "t_open", "t_adapters", "t_fwd", "t_bwd", "t_grads", "t_reset"):
            node.handlers[op] = getattr(self, "_" + op)
        node.disconnect_hooks.append(self._drop)
        node.services_fn = lambda: [] if self.paused else [self.service()]

    # -- shared bookkeeping -------------------------------------------------------------------
    def _expire(self) -> None:
        cutoff = time.time() - self.session_ttl
        self._sessions[:] = [s for s in self._sessions if s["last"] >= cutoff]

    def _drop_locked(self, state: dict) -> None:
        for name in ("session", "training"):
            session = state.pop(name, None)
            if session is not None:
                self._sessions[:] = [s for s in self._sessions if s is not session]

    def _drop(self, state: dict) -> None:
        with self._lock:
            self._drop_locked(state)

    def _range(self, header: dict):
        if self.paused:
            raise RuntimeError("this peer is not serving right now")
        x, y = header.get("start"), header.get("end")
        if header.get("model") != self.fingerprint:
            raise ValueError("this peer serves a different model")
        if not (isinstance(x, int) and isinstance(y, int) and not isinstance(x, bool) and not isinstance(y, bool)
                and self.start <= x < y <= self.end):
            raise ValueError(f"this peer serves layers {self.start}:{self.end}")
        return x, y

    def _begin(self, state: dict, name: str, session: dict) -> None:
        with self._lock:
            self._drop_locked(state)
            self._expire()
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("busy: no free session slots")
            session["last"] = time.time()
            self._sessions.append(session)
            state[name] = session

    def _current(self, state: dict, name: str, missing: str) -> dict:
        session = state.get(name)
        if session is None or not any(session is s for s in self._sessions):
            raise RuntimeError(missing)
        session["last"] = time.time()
        return session

    # -- answering ----------------------------------------------------------------------------
    def _open(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        x, y = self._range(header)
        self._begin(state, "session", {"range": (x, y), "cache": None, "tokens": 0})
        ch.send({"op": "open_ok", "variant": self.variant})

    def _step(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = self._current(state, "session", "no open session (it may have timed out)")
        if not _hidden_states(tensor, self.hidden, 1 << 30):
            raise ProtocolError("hidden states must have shape [1, tokens, hidden size]")
        if session["tokens"] + tensor.shape[1] > self.max_tokens:
            raise RuntimeError(f"session is limited to {self.max_tokens} tokens")
        with self._lock:
            out, session["cache"] = self.stack.step(*session["range"], tensor[0], session["cache"])
            session["tokens"] += tensor.shape[1]
            self.steps_served += 1
            self.tokens_served += tensor.shape[1]
        ch.send({"op": "step_ok"}, tensor=out[None])

    def _close(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        self._drop(state)
        ch.send({"op": "close_ok"})

    # -- training: one example at a time, nothing kept -------------------------------------------
    def _training(self, state: dict) -> dict:
        return self._current(state, "training", "no open training session (it may have timed out)")

    def _t_open(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        if self.paused:
            raise RuntimeError("this peer is not serving right now")
        if not self.training:
            raise RuntimeError("this peer has not agreed to help with training")
        x, y = self._range(header)
        rank, alpha, targets = check_spec(header.get("rank"), header.get("alpha"), header.get("targets"))
        shapes = self.stack.shapes(x, y, targets)
        if not shapes:
            raise ValueError(f"these layers have no modules named {targets}")
        self._begin(state, "training", {"range": (x, y), "rank": rank, "scale": alpha / rank, "shapes": shapes,
                                        "adapters": None, "graph": None, "grads": None})
        ch.send({"op": "t_open_ok", "variant": self.variant})

    def _t_adapters(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = self._training(state)
        blob = state.get("blob")
        if not blob or len(blob) > 1 << 30:
            raise ProtocolError("adapters missing or too large")
        try:
            adapters = st.loads(blob)
        except st.FormatError as e:
            raise ProtocolError(f"adapters could not be read: {e}") from None
        check_adapters(adapters, session["shapes"], session["rank"])
        with self._lock:
            session.update(adapters=adapters, graph=None, grads=None)
        ch.send({"op": "t_adapters_ok"})

    def _t_fwd(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = self._training(state)
        if session["adapters"] is None:
            raise RuntimeError("send adapters before asking for a calculation")
        if not _hidden_states(tensor, self.hidden, self.max_train_tokens):
            raise ProtocolError(f"hidden states must have shape [1, up to {self.max_train_tokens} tokens, hidden size]")
        with self._lock:
            out, graph = self.stack.forward(*session["range"], tensor[0], session["adapters"], session["scale"])
            session.update(graph=(graph, out.shape), grads=None)
        ch.send({"op": "t_fwd_ok"}, tensor=out[None])

    def _t_bwd(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = self._training(state)
        if session["graph"] is None:
            raise RuntimeError("backward asked with no forward calculation waiting")
        graph, shape = session["graph"]
        if not isinstance(tensor, np.ndarray) or tensor.shape != (1,) + tuple(shape) or tensor.dtype.kind != "f" \
                or not np.isfinite(tensor).all():
            raise ProtocolError("gradient does not match the result it belongs to")
        with self._lock:
            g_in, grads = self.stack.backward(graph, tensor[0])
            session.update(graph=None, grads=grads)
            self.examples_trained += 1
        ch.send({"op": "t_bwd_ok"}, tensor=g_in[None])

    def _t_grads(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = self._training(state)
        if session["grads"] is None or session["graph"] is not None:
            raise RuntimeError("no finished calculation to report")
        grads, session["grads"] = session["grads"], None
        ch.send({"op": "t_grads_ok"}, blob=st.dumps(grads))

    def _t_reset(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = self._training(state)
        with self._lock:
            session.update(graph=None, grads=None)
        ch.send({"op": "t_reset_ok"})
