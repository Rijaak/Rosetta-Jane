"""Running pieces of a model.

LayerServer: what a volunteer runs. It holds a run of middle layers and applies
them to whatever hidden states a client sends, keeping a small per-conversation
memory so each new word costs one small message.

LocalEnds: what the person asking runs. It holds the first few layers and the
final output layer, so the words going in and the words coming out exist only
on that person's own device.
"""
from __future__ import annotations

import os
import threading
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .lan.protocol import Channel, ProtocolError
from .lan.slicing import _scan, check_model_type, detect_layout, make_slice, model_fingerprint
from .lan.stage import _find_decoder


def model_shape(model_dir: str) -> Tuple[str, int]:
    """(layer key prefix, number of layers) of a local model copy."""
    return detect_layout(_scan(model_dir).keys())


def _slice(model_dir: str, work_dir: str, a: int, b: int, ends: bool, allow_untested: bool) -> str:
    """Cut layers [a, b) out of the local copy; with ends=True also the embeddings and output layer."""
    model_type = check_model_type(model_dir, allow_untested)
    prefix, n = model_shape(model_dir)
    if not 0 <= a < b <= n:
        raise ValueError(f"layer range {a}:{b} is outside this model's 0:{n}")
    if ends:
        if a != 0:
            raise ValueError("the asking side always starts at layer 0")
        stages = [{"name": "host", "layers": [0, b]}]
        if b < n:
            stages.append({"name": "rest", "layers": [b, n]})
        name = "host"
    else:
        if a == 0:
            raise ValueError("layer 0 always stays with the person asking; serve from layer 1 upward")
        stages = [{"name": "host", "layers": [0, a]}, {"name": "me", "layers": [a, b]}]
        if b < n:
            stages.append({"name": "rest", "layers": [b, n]})
        name = "me"
    plan = {"plan_version": 1, "fingerprint": model_fingerprint(model_dir), "model_type": model_type,
            "allow_untested": allow_untested, "num_layers": n, "layer_prefix": prefix, "stages": stages}
    return make_slice(model_dir, os.path.join(work_dir, f"slice-{a}-{b}{'-ends' if ends else ''}"), plan, name)


def _load(slice_dir: str, dtype, load_4bit, device_map, max_memory):
    from transformers import AutoModelForCausalLM
    kwargs: dict = {}
    if dtype:
        kwargs["dtype"] = getattr(torch, dtype)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if max_memory:
        kwargs["max_memory"] = max_memory
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype) if dtype else torch.float16)
        kwargs.setdefault("device_map", "auto")
    model = AutoModelForCausalLM.from_pretrained(slice_dir, **kwargs).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _variant(dtype: Optional[str], load_4bit: bool, model) -> str:
    if load_4bit:
        return "4bit"
    return dtype or str(next(model.parameters()).dtype).replace("torch.", "")


def _numbered(layers) -> List[list]:
    """For each layer, the inner parts that remember their own position in the stack."""
    return [[m for m in layer.modules() if isinstance(getattr(m, "layer_idx", None), int)] for layer in layers]


def _run(decoder, layers, h, cache, keep_cache: bool = True):
    """Apply a run of layers to new hidden states, continuing from an earlier cache.

    Each layer files its per-conversation memory under its own position number. When
    only some of a peer's layers are used, they must be renumbered from zero for the
    duration of the call, or one layer's memory lands in another layer's slot.
    """
    whole = decoder.layers
    marked = _numbered(layers)
    before = [[m.layer_idx for m in parts] for parts in marked]
    # Some families keep a per-layer list in their settings (which kind of attention each
    # layer uses). Narrow those lists to the layers in use, for the same reason.
    where = {id(layer): i for i, layer in enumerate(whole)}
    picked = [where[id(layer)] for layer in layers]
    config = decoder.config
    lists = {k: v for k, v in vars(config).items() if isinstance(v, list) and len(v) == len(whole)}
    decoder.layers = layers
    try:
        if len(picked) != len(whole):
            for key, value in lists.items():
                setattr(config, key, [value[i] for i in picked])
        for position, parts in enumerate(marked):
            for m in parts:
                m.layer_idx = position
        if not keep_cache:                        # training: no per-conversation memory wanted
            out = decoder(inputs_embeds=h, use_cache=False)
        else:
            try:
                out = decoder(inputs_embeds=h, cache=cache, use_cache=True)
            except TypeError:                     # older library versions use the previous name
                out = decoder(inputs_embeds=h, past_key_values=cache, use_cache=True)
    finally:
        decoder.layers = whole
        for key, value in lists.items():
            setattr(config, key, value)
        for parts, old in zip(marked, before):
            for m, index in zip(parts, old):
                m.layer_idx = index
    if not keep_cache:
        return out.last_hidden_state, None
    new_cache = getattr(out, "cache", None)
    if new_cache is None:
        new_cache = out.past_key_values
    return out.last_hidden_state, new_cache


class LayerServer:
    def __init__(self, model_dir: str, start: int, end: int, work_dir: str, *, dtype: Optional[str] = None,
                 load_4bit: bool = False, device_map=None, max_memory: Optional[dict] = None,
                 max_sessions: int = 4, max_tokens: int = 8192, session_ttl: float = 600.0,
                 allow_untested: bool = False, training: bool = False, max_train_tokens: int = 4096):
        self.training = training                  # the owner has agreed to help with training too
        self.paused = False                       # the owner can stop serving without unloading
        self.max_train_tokens = max_train_tokens
        self.examples_trained = 0
        self.start, self.end = start, end
        self.fingerprint = model_fingerprint(model_dir)
        model = _load(_slice(model_dir, work_dir, start, end, False, allow_untested),
                      dtype, load_4bit, device_map, max_memory)
        self._decoder = _find_decoder(model)
        self._decoder.norm = nn.Identity()        # the final norm belongs only at the very end of the model
        self._layers = list(self._decoder.layers)
        self.hidden = int(model.config.hidden_size)
        self.variant = _variant(dtype, load_4bit, model)
        self._dtype = (getattr(torch, dtype) if dtype else torch.float16) if load_4bit else \
            next(p for p in self._layers[0].parameters() if p.is_floating_point()).dtype
        self._device = next(self._layers[0].parameters()).device
        self.max_sessions, self.max_tokens, self.session_ttl = max_sessions, max_tokens, session_ttl
        self._lock = threading.Lock()             # one piece of work at a time on this hardware
        self._sessions: List[dict] = []
        self.steps_served = 0
        self.tokens_served = 0

    def service(self) -> dict:
        with self._lock:
            self._expire()
            free = max(0, self.max_sessions - len(self._sessions))
        return {"model": self.fingerprint, "variant": self.variant, "layers": [self.start, self.end],
                "hidden": self.hidden, "free": free, "train": self.training}

    def _expire(self) -> None:
        cutoff = time.time() - self.session_ttl
        self._sessions[:] = [s for s in self._sessions if s["last"] >= cutoff]

    # -- requests --------------------------------------------------------------
    def attach(self, node) -> None:
        node.handlers["open"] = self._open
        node.handlers["step"] = self._step
        node.handlers["close"] = self._close
        for op in ("t_open", "t_adapters", "t_fwd", "t_bwd", "t_grads", "t_reset"):
            node.handlers[op] = getattr(self, "_" + op)
        node.disconnect_hooks.append(self._drop)
        node.services_fn = lambda: [] if self.paused else [self.service()]

    def _open(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        if self.paused:
            raise RuntimeError("this peer is not serving right now")
        x, y = header.get("start"), header.get("end")
        if header.get("model") != self.fingerprint:
            raise ValueError("this peer serves a different model")
        if not (isinstance(x, int) and isinstance(y, int) and self.start <= x < y <= self.end):
            raise ValueError(f"this peer serves layers {self.start}:{self.end}")
        with self._lock:
            self._drop_locked(state)
            self._expire()
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("busy: no free session slots")
            session = {"layers": nn.ModuleList(self._layers[x - self.start:y - self.start]),
                       "cache": None, "tokens": 0, "last": time.time()}
            self._sessions.append(session)
            state["session"] = session
        ch.send({"op": "open_ok", "variant": self.variant})

    def _step(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = state.get("session")
        if session is None or not any(session is s for s in self._sessions):
            raise RuntimeError("no open session (it may have timed out)")
        if tensor is None or tensor.dim() != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.hidden \
                or tensor.shape[1] < 1 or not tensor.is_floating_point():
            raise ProtocolError("hidden states must have shape [1, tokens, hidden size]")
        if session["tokens"] + tensor.shape[1] > self.max_tokens:
            raise RuntimeError(f"session is limited to {self.max_tokens} tokens")
        with self._lock, torch.no_grad():
            out, session["cache"] = _run(self._decoder, session["layers"],
                                         tensor.to(self._device, self._dtype), session["cache"])
            session["tokens"] += tensor.shape[1]
            session["last"] = time.time()
            self.steps_served += 1
            self.tokens_served += tensor.shape[1]
        ch.send({"op": "step_ok"}, tensor=out)

    def _close(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        self._drop(state)
        ch.send({"op": "close_ok"})

    def _drop_locked(self, state: dict) -> None:
        for name in ("session", "training"):
            session = state.pop(name, None)
            if session is not None:
                self._sessions[:] = [s for s in self._sessions if s is not session]

    # -- training: one example at a time, nothing kept ------------------------------
    # The person training owns the trained weights and sends them along for each calculation.
    # This device computes what it is asked and returns all of it, so its work can be
    # compared with another device's or recomputed by the trainer.
    def _t_session(self, state: dict) -> dict:
        session = state.get("training")
        if session is None or not any(session is s for s in self._sessions):
            raise RuntimeError("no open training session (it may have timed out)")
        session["last"] = time.time()
        return session

    def _t_open(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        from .lora import check_spec, shapes_of_layers
        if self.paused:
            raise RuntimeError("this peer is not serving right now")
        if not self.training:
            raise RuntimeError("this peer has not agreed to help with training")
        x, y = header.get("start"), header.get("end")
        if header.get("model") != self.fingerprint:
            raise ValueError("this peer serves a different model")
        if not (isinstance(x, int) and isinstance(y, int) and self.start <= x < y <= self.end):
            raise ValueError(f"this peer serves layers {self.start}:{self.end}")
        rank, alpha, targets = check_spec(header.get("rank"), header.get("alpha"), header.get("targets"))
        layers = self._layers[x - self.start:y - self.start]
        shapes = shapes_of_layers(layers, x, targets)
        if not shapes:
            raise ValueError(f"these layers have no modules named {targets}")
        with self._lock:
            self._drop_locked(state)
            self._expire()
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("busy: no free session slots")
            session = {"layers": nn.ModuleList(layers), "first": x, "rank": rank, "scale": alpha / rank,
                       "targets": targets, "shapes": shapes, "adapters": None, "graph": None,
                       "last": time.time()}
            self._sessions.append(session)
            state["training"] = session
        ch.send({"op": "t_open_ok", "variant": self.variant})

    def _t_adapters(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        from safetensors.torch import load as st_load

        from .lora import check_adapters
        session = self._t_session(state)
        blob = state.get("blob")
        if not blob or len(blob) > 1 << 30:
            raise ProtocolError("adapters missing or too large")
        adapters = st_load(blob)
        check_adapters(adapters, session["shapes"], session["rank"])
        with self._lock:
            session["adapters"] = {k: v.to(self._device, torch.float32).requires_grad_(True)
                                   for k, v in adapters.items()}
            session["graph"] = None
        ch.send({"op": "t_adapters_ok"})

    def _t_fwd(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        from .lora import Applied
        session = self._t_session(state)
        if session["adapters"] is None:
            raise RuntimeError("send adapters before asking for a calculation")
        if tensor is None or tensor.dim() != 3 or tensor.shape[0] != 1 or tensor.shape[2] != self.hidden \
                or not 1 <= tensor.shape[1] <= self.max_train_tokens or not tensor.is_floating_point():
            raise ProtocolError(f"hidden states must have shape [1, up to {self.max_train_tokens} tokens, hidden size]")
        with self._lock:
            for a in session["adapters"].values():
                a.grad = None
            h = tensor.to(self._device, self._dtype).requires_grad_(True)
            with Applied(session["layers"], session["first"], session["targets"], session["adapters"],
                         session["scale"]):
                out, _ = _run(self._decoder, session["layers"], h, None, keep_cache=False)
            session["graph"] = (h, out)
        ch.send({"op": "t_fwd_ok"}, tensor=out.detach())

    def _t_bwd(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = self._t_session(state)
        if session["graph"] is None:
            raise RuntimeError("backward asked with no forward calculation waiting")
        h, out = session["graph"]
        if tensor is None or tuple(tensor.shape) != tuple(out.shape) or not tensor.is_floating_point():
            raise ProtocolError("gradient does not match the result it belongs to")
        with self._lock:
            out.backward(tensor.to(out.device, out.dtype))
            session["graph"] = None
            self.examples_trained += 1
        ch.send({"op": "t_bwd_ok"}, tensor=h.grad)

    def _t_grads(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        from safetensors.torch import save as st_save
        session = self._t_session(state)
        if session["adapters"] is None or session["graph"] is not None:
            raise RuntimeError("no finished calculation to report")
        grads = {}
        for key, a in session["adapters"].items():
            if a.grad is None:
                raise RuntimeError("no finished calculation to report")
            grads[key] = a.grad.detach().float().cpu().contiguous()
            a.grad = None
        ch.send({"op": "t_grads_ok"}, blob=st_save(grads))

    def _t_reset(self, ch: Channel, header: dict, tensor, state: dict) -> None:
        session = self._t_session(state)
        with self._lock:
            session["graph"] = None
            if session["adapters"]:
                for a in session["adapters"].values():
                    a.grad = None
        ch.send({"op": "t_reset_ok"})

    def _drop(self, state: dict) -> None:
        with self._lock:
            self._drop_locked(state)


class LocalEnds:
    """The asking side's own part of the model: token table, first layers, final norm and output layer."""

    def __init__(self, model_dir: str, own_layers: int, work_dir: str, *, dtype: Optional[str] = None,
                 load_4bit: bool = False, device_map=None, max_memory: Optional[dict] = None,
                 allow_untested: bool = False):
        self.fingerprint = model_fingerprint(model_dir)
        _, self.num_layers = model_shape(model_dir)
        if not 1 <= own_layers <= self.num_layers:
            raise ValueError(f"keep between 1 and {self.num_layers} layers on your own device")
        self.own_layers = own_layers
        model = _load(_slice(model_dir, work_dir, 0, own_layers, True, allow_untested),
                      dtype, load_4bit, device_map, max_memory)
        self._decoder = _find_decoder(model)
        self._norm = self._decoder.norm
        self._decoder.norm = nn.Identity()
        self._layers = nn.ModuleList(list(self._decoder.layers))
        self._embed = self._decoder.embed_tokens
        self._head = model.get_output_embeddings()
        self.hidden = int(model.config.hidden_size)
        self.eos_token_id = model.config.eos_token_id

    @torch.no_grad()
    def head(self, new_ids: List[int], cache):
        ids = torch.tensor([new_ids], dtype=torch.long, device=self._embed.weight.device)
        return _run(self._decoder, self._layers, self._embed(ids), cache)

    @torch.no_grad()
    def logits(self, h_last: torch.Tensor) -> torch.Tensor:
        """Scores for the next token, given the last position's final hidden state."""
        device = self._head.weight.device
        h = h_last.to(device, self._head.weight.dtype if self._head.weight.is_floating_point() else None)
        return self._head(self._norm(h))[0, -1].float().cpu()


class LocalAuditor:
    """Settles disputes between peers by recomputing the disputed work on this device.

    The person asking has the whole model on disk. When peers disagree about what a run
    of layers produces, this loads just those layers here and computes the true result,
    so the answer does not depend on which side has more voices.

    It must run in the same precision the peers used; a 4-bit result cannot be checked
    against a full-precision recomputation.
    """

    def __init__(self, model_dir: str, work_dir: str, *, dtype: Optional[str] = None,
                 load_4bit: bool = False, device_map=None, max_memory: Optional[dict] = None,
                 allow_untested: bool = False, max_layers: int = 4):
        self._model_dir, self._work_dir = model_dir, work_dir
        self.max_layers = max(1, max_layers)      # how many layers this device loads at one time
        self._load = dict(dtype=dtype, load_4bit=load_4bit, device_map=device_map, max_memory=max_memory)
        self._allow_untested = allow_untested
        self.variant: Optional[str] = None
        self._loaded: Optional[tuple] = None      # (start, end, decoder, layers); one range kept at a time
        self.audits = 0

    def compute(self, start: int, end: int, blocks: List[torch.Tensor]) -> torch.Tensor:
        """True output of layers [start, end) for everything a peer has been sent so far.

        A long run of layers is worked through a few at a time, so this device never
        needs room for more than max_layers of them.
        """
        h = torch.cat(blocks, dim=1)
        for a in range(start, end, self.max_layers):
            h = self._some(a, min(a + self.max_layers, end), h)
        self.audits += 1
        return h.float().cpu()

    def train_example(self, start: int, end: int, adapters, rank_scale: float, targets,
                      h_in: torch.Tensor, g_out: Optional[torch.Tensor] = None):
        """The true result of one training calculation on layers [start, end).

        Returns (h_out, g_in, grads). Without g_out only the forward result is computed.
        Worked through max_layers at a time: forward keeping each piece's input, then
        backward through the pieces in reverse.
        """
        from .lora import Applied
        pieces = [(a, min(a + self.max_layers, end)) for a in range(start, end, self.max_layers)]
        inputs = []
        h = h_in
        for a, b in pieces:
            inputs.append(h)
            decoder, layers, device, dtype = self._piece(a, b)
            live = {k: v.to(device, torch.float32) for k, v in adapters.items() if a <= int(k.split(".", 1)[0]) < b}
            with torch.no_grad(), Applied(layers, a, targets, live, rank_scale):
                h, _ = _run(decoder, layers, h.to(device, dtype), None, keep_cache=False)
        h_out = h.float().cpu()
        self.audits += 1
        if g_out is None:
            return h_out, None, None
        grads = {}
        g = g_out
        for (a, b), x in zip(reversed(pieces), reversed(inputs)):
            decoder, layers, device, dtype = self._piece(a, b)
            live = {k: v.to(device, torch.float32).requires_grad_(True) for k, v in adapters.items()
                    if a <= int(k.split(".", 1)[0]) < b}
            x = x.to(device, dtype).detach().requires_grad_(True)
            with Applied(layers, a, targets, live, rank_scale):
                out, _ = _run(decoder, layers, x, None, keep_cache=False)
            out.backward(g.to(out.device, out.dtype))
            g = x.grad
            for k, v in live.items():
                grads[k] = v.grad.detach().float().cpu()
        return h_out, g.float().cpu(), grads

    def _piece(self, start: int, end: int):
        self._some(start, end, None)
        _, _, decoder, layers = self._loaded
        ref = next(p for p in layers[0].parameters())
        return decoder, layers, ref.device, (ref.dtype if ref.is_floating_point() else torch.float16)

    def _some(self, start: int, end: int, h: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self._loaded is None or self._loaded[:2] != (start, end):
            self._loaded = None                   # free the previous range before loading another
            model = _load(_slice(self._model_dir, self._work_dir, start, end, False, self._allow_untested),
                          **self._load)
            decoder = _find_decoder(model)
            decoder.norm = nn.Identity()
            self.variant = _variant(self._load["dtype"], self._load["load_4bit"], model)
            self._loaded = (start, end, decoder, nn.ModuleList(list(decoder.layers)))
        if h is None:
            return None
        _, _, decoder, layers = self._loaded
        ref = next(p for p in layers[0].parameters())
        dtype = ref.dtype if ref.is_floating_point() else torch.float16
        with torch.no_grad():
            out, _ = _run(decoder, layers, h.to(ref.device, dtype), None)
        return out
