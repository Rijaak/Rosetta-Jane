"""Running a training job from a device without PyTorch.

The same arrangement as on a desktop: this device keeps the training text, its own
first layers, the output layer, every trained weight and the optimizer. Peers are
sent the small trained matrices for their layers, compute one example forward and
backward, and hand everything back. Nothing trained lives anywhere else.

Checks available here: two or more peers doing the same work are compared, a strict
majority wins, a third opinion is called in for a tie, and a peer that leaves costs
one example. Known-answer tests and rechecking on this device are desktop-only so far.

A finished run is written to a folder as a standard LoRA adapter with a note on how
much of the work was checked. Nothing applies it. That step belongs to a person.
If you build on Rosetta-Jane, keep that step.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..arrays import agree, groups
from ..lan.protocol import ProtocolError, RemoteError
from ..routing import Disputed, Hop, NoRoute, _ip, plan_route
from . import st
from .ends import Ends
from .engine import f32

_NET_ERRORS = (ConnectionError, OSError, ProtocolError, RemoteError)
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
FACTS = "rosetta-jane-run.json"
Example = Tuple[List[int], List[int]]             # token ids, labels (-100 where not graded)


class _Redo(Exception):
    """Something went wrong partway through an example; put things right and do it again."""


def init_adapters(shapes: Dict[str, Tuple[int, int]], rank: int, seed: int) -> Dict[str, np.ndarray]:
    """Fresh adapters: A random, B zero, so training starts from the unchanged model."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in sorted(shapes):
        fan_in, fan_out = shapes[name]
        bound = math.sqrt(1.0 / fan_in)
        out[f"{name}.A"] = rng.uniform(-bound, bound, size=(rank, fan_in)).astype(f32)
        out[f"{name}.B"] = np.zeros((fan_out, rank), dtype=f32)
    return out


def _flat(tensors: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([tensors[k].ravel() for k in sorted(tensors)])[None] if tensors else np.zeros((1, 0), f32)


class Trainer:
    def __init__(self, node, ends: Ends, work_dir: str, *, total_steps: int, accum: int = 16, lr: float = 2e-4,
                 warmup_steps: int = 20, max_grad_norm: float = 1.0, rank: int = 16, alpha: float = 32.0,
                 targets: Sequence[str] = DEFAULT_TARGETS, seed: int = 0, save_every: int = 25,
                 run_id: str = "run", redundancy: Optional[int] = None, tolerance: float = 0.02,
                 find_wait: float = 15.0, rpc_timeout: float = 300.0, max_redo: int = 8,
                 init: Optional[Dict[str, np.ndarray]] = None):
        from .peer import check_spec
        self.node, self.ends = node, ends
        self.rank, self.alpha, self.targets = check_spec(rank, alpha, list(targets))
        self.scale = self.alpha / self.rank
        self.shapes = self._all_shapes()
        self.adapters = {k: np.array(v, dtype=f32) for k, v in (init or init_adapters(self.shapes, rank, seed)).items()}
        if sorted(self.adapters) != sorted(f"{n}.{p}" for n in self.shapes for p in ("A", "B")):
            raise ValueError("starting adapters do not fit this model")
        self._order = sorted(self.adapters)
        self._m = {k: np.zeros_like(v) for k, v in self.adapters.items()}       # AdamW's running averages
        self._v = {k: np.zeros_like(v) for k, v in self.adapters.items()}
        self._acc = {k: np.zeros_like(v) for k, v in self.adapters.items()}
        self.total_steps, self.accum, self.lr = total_steps, accum, lr
        self.warmup_steps, self.max_grad_norm, self.save_every = warmup_steps, max_grad_norm, save_every
        self.seed, self.run_id = seed, run_id
        self.dir = os.path.join(work_dir, f"train-{run_id}")
        os.makedirs(self.dir, exist_ok=True)
        self.redundancy, self.tolerance = redundancy, tolerance
        self.find_wait, self.rpc_timeout, self.max_redo = find_wait, rpc_timeout, max_redo
        self.route: List[Hop] = []
        self.step = 0
        self.last_loss: Optional[float] = None
        self.caught: List[str] = []
        self._bad: set = set()
        self._want_extra: set = set()
        self.checked = self.unchecked = self.disputes = self.redos = 0

    def _all_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Every adapter this model takes, read from the weight files without loading them."""
        from .engine import scan
        found = scan(self.ends.stack and self._model_dir())
        prefix = self.ends.stack.prefix
        out = {}
        for key, (_, shape, _) in found.items():
            if key.startswith(prefix) and key.endswith(".weight") and len(shape) == 2:
                index, _, path = key[len(prefix):-len(".weight")].partition(".")
                if index.isdigit() and path.split(".")[-1] in self.targets:
                    out[f"{int(index)}.{path}"] = (int(shape[1]), int(shape[0]))
        for f in {id(v[0]): v[0] for v in found.values()}.values():
            f.close()
        if not out:
            raise ValueError(f"no modules named {list(self.targets)} found in this model")
        return out

    def _model_dir(self) -> str:
        return os.path.dirname(self.ends._embed_file.path)

    def _keys(self, start: int, end: int) -> List[str]:
        return [k for k in self._order if start <= int(k.split(".", 1)[0]) < end]

    # -- saving progress (everything lives on this device) -----------------------------------
    def _ckpt(self) -> str:
        return os.path.join(self.dir, "progress")

    def save(self) -> None:
        tmp = self._ckpt() + ".partial"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        pack = {f"w.{k}": v for k, v in self.adapters.items()}
        pack.update({f"m.{k}": v for k, v in self._m.items()})
        pack.update({f"v.{k}": v for k, v in self._v.items()})
        with open(os.path.join(tmp, "state.safetensors"), "wb") as f:
            f.write(st.dumps(pack))
        with open(os.path.join(tmp, "done.json"), "w") as f:
            json.dump({"step": self.step, "rank": self.rank, "alpha": self.alpha, "targets": self.targets}, f)
        shutil.rmtree(self._ckpt(), ignore_errors=True)
        os.replace(tmp, self._ckpt())

    def resume(self) -> bool:
        done = os.path.join(self._ckpt(), "done.json")
        if not os.path.exists(done):
            return False
        with open(done) as f:
            meta = json.load(f)
        if (meta["rank"], meta["alpha"], meta["targets"]) != (self.rank, self.alpha, self.targets):
            raise ValueError("the saved run used different settings; use another name or the same settings")
        with open(os.path.join(self._ckpt(), "state.safetensors"), "rb") as f:
            pack = st.loads(f.read())
        for k in self._order:
            self.adapters[k], self._m[k], self._v[k] = pack[f"w.{k}"], pack[f"m.{k}"], pack[f"v.{k}"]
        self.step = int(meta["step"])
        return True

    # -- peers ----------------------------------------------------------------------------------
    def _excluded(self) -> set:
        return self._bad | self.node.banned

    def _shut(self, replica: Hop) -> None:
        if replica.ch is not None:
            try:
                replica.ch.send({"op": "bye"})
            except _NET_ERRORS:
                pass
            replica.ch.close()
            replica.ch = None

    def _block(self, replica: Hop) -> None:
        """This peer returned a wrong result. Never use it again while this app runs."""
        self._shut(replica)
        self.node.banned.update({replica.peer.node_id, replica.peer.addr})
        self.caught.append(replica.peer.node_id)

    def _blob(self, start: int, end: int) -> bytes:
        return st.dumps({k: self.adapters[k] for k in self._keys(start, end)})

    def _open(self, replica: Hop) -> None:
        ch = self.node.dial(replica.peer.addr, timeout=min(30.0, self.rpc_timeout))
        ch.sock.settimeout(self.rpc_timeout)
        try:
            ch.call({"op": "t_open", "model": self.ends.fingerprint, "start": replica.start, "end": replica.end,
                     "rank": self.rank, "alpha": self.alpha, "targets": self.targets})
            ch.call({"op": "t_adapters"}, blob=self._blob(replica.start, replica.end))
        except BaseException:
            ch.close()
            raise
        replica.ch = ch

    def _able(self) -> list:
        return [p for p in self.node.table.peers()
                if p.node_id not in self._excluded() and p.addr not in self._excluded()
                and any(s["model"] == self.ends.fingerprint and s.get("train") for s in p.services)]

    def _recruit(self, hop: Hop, count: int) -> List[Hop]:
        taken = [r for r in hop.replicas() if r.ch is not None] or [hop]
        used_ids = {r.peer.node_id for r in taken}
        used_ips = {_ip(r.peer.addr) for r in taken}
        options = [p for p in self._able() if p.node_id not in used_ids and any(
            s["model"] == self.ends.fingerprint and s.get("train") and s["variant"] == hop.variant
            and s["layers"][0] <= hop.start and hop.end <= s["layers"][1] for s in p.services)]
        options.sort(key=lambda p: (_ip(p.addr) not in used_ips, random.random()), reverse=True)
        got: List[Hop] = []
        for peer in options:
            if len(got) >= count:
                break
            extra = Hop(peer, hop.start, hop.end, hop.variant)
            try:
                self._open(extra)
            except _NET_ERRORS:
                self._bad.add(peer.node_id)
                continue
            got.append(extra)
        return got

    def _connect(self) -> List[Hop]:
        own, total = self.ends.own_layers, self.ends.num_layers
        if own >= total:
            return []
        deadline = time.time() + self.find_wait
        while True:
            wanted = (2 if len(self._able()) >= 3 else 1) if self.redundancy is None else self.redundancy
            try:
                hops = plan_route(self.node.table.peers(), self.ends.fingerprint, own, total, self._excluded(),
                                  None, training=True)
            except NoRoute as e:
                if time.time() >= deadline:
                    raise NoRoute(f"{e} (among peers that have agreed to help with training)") from None
                self.node.refresh(1)
                time.sleep(0.2)
                continue
            opened: List[Hop] = []
            try:
                for hop in hops:
                    self._open(hop)
                    opened.append(hop)
            except _NET_ERRORS:
                for hop in opened:
                    self._shut(hop)
                self._bad.add(hops[len(opened)].peer.node_id)
                continue
            for hop in hops:
                hop.witnesses = self._recruit(hop, wanted - 1) if wanted > 1 else []
            self._want_extra = set()
            return hops

    def close(self) -> None:
        for hop in self.route:
            for replica in hop.replicas():
                self._shut(replica)
        self.route = []

    def _live(self, i: int) -> List[Hop]:
        live = [r for r in self.route[i].replicas() if r.ch is not None]
        if not live:
            raise _Redo()
        live[0].witnesses = live[1:]
        self.route[i] = live[0]
        return live

    def _recover(self) -> None:
        self.redos += 1
        gone = lambda: any(not [r for r in hop.replicas() if r.ch is not None] for hop in self.route)
        if not gone():
            for i in sorted(self._want_extra):
                extra = self._recruit(self.route[i], 1)
                if not extra:
                    hop = self.route[i]
                    raise Disputed(f"peers disagree about layers {hop.start}-{hop.end - 1} and nobody is left "
                                   "to break the tie. Wait for more peers.")
                self.route[i].witnesses.extend(extra)
            self._want_extra = set()
            for hop in self.route:
                for replica in hop.replicas():
                    if replica.ch is not None:
                        try:
                            replica.ch.call({"op": "t_reset"})
                        except _NET_ERRORS:
                            self._shut(replica)
                            self._bad.add(replica.peer.node_id)
        if gone() or not self.route:
            self.close()
            self.route = self._connect()

    def _ask(self, i: int, op: str, tensor) -> list:
        outs = []
        for replica in self._live(i):
            try:
                _, out, blob = replica.ch.call({"op": op}, tensor=tensor)
                if op == "t_grads":
                    grads = st.loads(blob or b"")
                    want = self._keys(replica.start, replica.end)
                    if sorted(grads) != want or any(grads[k].shape != self.adapters[k].shape for k in want):
                        raise ProtocolError("peer returned corrections of the wrong shape")
                    outs.append((replica, grads))
                else:
                    if not isinstance(out, np.ndarray) or out.shape != tensor.shape or not np.isfinite(out).all():
                        raise ProtocolError("peer returned a result of the wrong shape")
                    outs.append((replica, out))
            except (st.FormatError, *_NET_ERRORS):
                self._shut(replica)               # gone or broken, not proven dishonest
                self._bad.add(replica.peer.node_id)
        if not outs:
            raise _Redo()
        return outs

    def _settle(self, i: int, outs: list, is_grads: bool):
        """Pick the result to trust for hop i at one stage of an example. Blocks peers that were outvoted."""
        view = [(r, _flat(v) if is_grads else v) for r, v in outs]
        if len(view) > 1:
            sets = groups(view, self.tolerance)
            if len(sets) > 1:
                self.disputes += 1
                if len(sets[0]) * 2 <= len(view):  # no strict majority: bring in another opinion
                    self._want_extra.add(i)
                    raise _Redo()
                keep = {id(r) for r, _ in sets[0]}
                for replica, _ in outs:
                    if id(replica) not in keep:
                        self._block(replica)
                view = sets[0]
        winner = view[0][0]
        self._live(i)
        return next(v for r, v in outs if r is winner)

    # -- one example ------------------------------------------------------------------------------
    def _attempt(self, ids: List[int], labels: List[int]) -> float:
        own = {k: self.adapters[k] for k in self._keys(0, self.ends.own_layers)}
        h0, graph = self.ends.train_forward(ids, own, self.scale)
        h = h0[None]
        for i in range(len(self.route)):
            h = self._settle(i, self._ask(i, "t_fwd", h), False)
        loss, g = self.ends.loss(h[0], labels)
        g = g[None]
        got: Dict[str, np.ndarray] = {}
        tally = []
        for i in reversed(range(len(self.route))):
            g_in = self._settle(i, self._ask(i, "t_bwd", g), False)
            got.update(self._settle(i, self._ask(i, "t_grads", None), True))
            tally.append(len([r for r in self.route[i].replicas() if r.ch is not None]) > 1)
            g = g_in
        got.update(self.ends.train_backward(graph, g[0]))
        for k, v in got.items():
            self._acc[k] += v
        self.checked += sum(tally)
        self.unchecked += len(tally) - sum(tally)
        return loss

    def _example(self, ids: List[int], labels: List[int]) -> float:
        for _ in range(self.max_redo):
            try:
                if not self.route and self.ends.own_layers < self.ends.num_layers:
                    self.route = self._connect()
                return self._attempt(ids, labels)
            except _Redo:
                self._recover()
        raise NoRoute("one training example kept failing; too many peers dropped out or were caught lying")

    # -- the optimizer (AdamW without weight decay, as on a desktop) ---------------------------------
    def lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        span = max(1, self.total_steps - self.warmup_steps)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - self.warmup_steps) / span)))

    def _optimizer_step(self) -> None:
        grads = {k: self._acc[k] / f32(self.accum) for k in self._order}
        for k in self._order:
            self._acc[k] = np.zeros_like(self._acc[k])
        if not all(np.isfinite(g).all() for g in grads.values()):
            return                                # a broken step is skipped, not applied
        norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values()))
        if self.max_grad_norm > 0 and norm > self.max_grad_norm:
            coef = f32(self.max_grad_norm / (norm + 1e-6))
            grads = {k: g * coef for k, g in grads.items()}
        lr, t = self.lr_at(self.step), self.step + 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self._order:
            self._m[k] = (b1 * self._m[k] + (1 - b1) * grads[k]).astype(f32)
            self._v[k] = (b2 * self._v[k] + (1 - b2) * grads[k] * grads[k]).astype(f32)
            denom = np.sqrt(self._v[k]) / math.sqrt(1 - b2 ** t) + eps
            self.adapters[k] = (self.adapters[k] - (lr / (1 - b1 ** t)) * self._m[k] / denom).astype(f32)
        self._push()                               # send everyone the updated weights

    def _push(self) -> None:
        for hop in self.route:
            for replica in hop.replicas():
                if replica.ch is not None:
                    try:
                        replica.ch.call({"op": "t_adapters"}, blob=self._blob(replica.start, replica.end))
                    except _NET_ERRORS:
                        self._shut(replica)
                        self._bad.add(replica.peer.node_id)

    # -- scoring: forward only, nothing is learned ------------------------------------------------
    def _hidden(self, ids: List[int]):
        """The final hidden states for one whole text, through the group, with the current adapters."""
        own = {k: self.adapters[k] for k in self._keys(0, self.ends.own_layers)}
        h0, _ = self.ends.train_forward(ids, own, self.scale)
        h, checked = h0[None], 0
        for i in range(len(self.route)):
            h = self._settle(i, self._ask(i, "t_fwd", h), False)
            live = self._live(i)
            checked += len(live) > 1
            for replica in live:                   # nothing will go backward; let the peer drop it
                try:
                    replica.ch.call({"op": "t_reset"})
                except _NET_ERRORS:
                    self._shut(replica)
                    self._bad.add(replica.peer.node_id)
        return h, checked, len(self.route) - checked

    def _score_once(self, ids: List[int], labels: List[int]):
        h, checked, unchecked = self._hidden(ids)
        loss, right, tokens = self.ends.grade(h[0], labels)
        return loss, right, tokens, checked, unchecked

    def _retry(self, work):
        for _ in range(self.max_redo):
            try:
                if not self.route and self.ends.own_layers < self.ends.num_layers:
                    self.route = self._connect()
                return work()
            except _Redo:
                self._recover()
        raise NoRoute("the calculation kept failing; too many peers dropped out or were caught lying")

    def answer(self, ids: Sequence[int], max_new: int = 48, stop_ids: Sequence[int] = (),
               with_adapter: bool = True) -> List[int]:
        """One side of a trial: continue a prompt with the adapters, or with the unchanged model."""
        if not ids:
            raise ValueError("empty prompt")
        use = self.adapters if with_adapter else \
            {k: (np.zeros_like(v) if k.endswith(".B") else v) for k, v in self.adapters.items()}
        mine, self.adapters = self.adapters, use
        stops = set(stop_ids)
        try:
            self._push()
            out: List[int] = []
            for _ in range(max_new):
                text = list(ids) + out
                h = self._retry(lambda: self._hidden(text))[0]
                token = int(np.argmax(self.ends.logits(h[:, -1:])))
                if token in stops:
                    break
                out.append(token)
            return out
        finally:
            self.adapters = mine

    def finish(self) -> None:
        self.close()

    def try_out(self, ids: Sequence[int], max_new: int = 48, stop_ids: Sequence[int] = ()):
        """Answer one prompt twice: with the unchanged model, and with these adapters. A trial, not a switch."""
        try:
            return (self.answer(ids, max_new, stop_ids, with_adapter=False),
                    self.answer(ids, max_new, stop_ids, with_adapter=True))
        finally:
            self.close()

    @classmethod
    def from_adapter(cls, node, ends: Ends, work_dir: str, adapter_dir: str, **kw) -> "Trainer":
        """A trainer holding a finished adapter read from a folder, for scoring it or trying it out."""
        try:
            with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
                config = json.load(f)
            with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "rb") as f:
                weights = st.loads(f.read())
            rank, alpha, targets = int(config["r"]), float(config["lora_alpha"]), list(config["target_modules"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"that folder does not hold a readable adapter: {e}") from None
        facts = os.path.join(adapter_dir, FACTS)
        if os.path.exists(facts):
            try:
                with open(facts) as f:
                    made_for = json.load(f).get("base_model_fingerprint")
            except (OSError, ValueError):
                made_for = None
            if made_for and made_for != ends.fingerprint:
                raise ValueError(f"this adapter was trained on a different model ({made_for}); "
                                 f"the model here is {ends.fingerprint}")
        t = cls(node, ends, work_dir, total_steps=1, rank=rank, alpha=alpha, targets=targets, **kw)
        head = "base_model.model." + ends.stack.prefix
        loaded = {}
        for key, value in weights.items():
            if key.startswith(head) and (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")):
                layer, _, rest = key[len(head):].partition(".")
                path, part = rest[:-len(".weight")].rsplit(".lora_", 1)
                loaded[f"{layer}.{path}.{part}"] = value
        if sorted(loaded) != t._order or any(loaded[k].shape != t.adapters[k].shape for k in loaded):
            raise ValueError("this adapter does not fit the model here")
        if not all(np.isfinite(v).all() for v in loaded.values()):
            raise ValueError("this adapter contains values that are not numbers")
        t.adapters = {k: np.array(loaded[k], dtype=f32) for k in t._order}
        return t

    def _score_with(self, adapters: Dict[str, np.ndarray], examples: List[Example]):
        from ..scoring import tally
        mine, self.adapters = self.adapters, adapters
        try:
            self._push()
            totals = [0.0, 0, 0, 0, 0]
            for ids, labels in examples:
                got = self._retry(lambda: self._score_once(ids, labels))
                totals = [a + b for a, b in zip(totals, got)]
            return tally(totals[0], totals[1], totals[2]), totals[2], totals[3], totals[4]
        finally:
            self.adapters = mine

    def compare(self, examples: List[Example], name: str = "", held_out: bool = False) -> dict:
        """A score card: the unchanged model against the model with these adapters, on the same pairs."""
        from ..scoring import card
        if not examples:
            return card(name, self.ends.fingerprint, 0, 0, {"loss": None}, {"loss": None}, held_out=held_out)
        plain = {k: (np.zeros_like(v) if k.endswith(".B") else v) for k, v in self.adapters.items()}
        try:
            if not self.route and self.ends.own_layers < self.ends.num_layers:
                self.route = self._connect()
            before, tokens, c1, u1 = self._score_with(plain, examples)
            after, _, c2, u2 = self._score_with(self.adapters, examples)
        finally:
            self.close()
        return card(name, self.ends.fingerprint, len(examples), tokens, before, after, held_out=held_out,
                    checked=c1 + c2, unchecked=u1 + u2)

    def train(self, examples: List[Example], on_step: Optional[Callable[[int], None]] = None) -> dict:
        self.resume()
        n = len(examples)
        try:
            while self.step < self.total_steps:
                for k in self._order:
                    self._acc[k] = np.zeros_like(self._acc[k])
                losses = []
                for m in range(self.accum):
                    position = self.step * self.accum + m
                    epoch, offset = divmod(position, n)
                    order = list(range(n))
                    random.Random(f"{self.seed}-{epoch}").shuffle(order)      # the desktop's order exactly
                    losses.append(self._example(*examples[order[offset]]))
                self._optimizer_step()
                self.step += 1
                self.last_loss = sum(losses) / len(losses)
                if self.step % self.save_every == 0 or self.step == self.total_steps:
                    self.save()
                if on_step is not None:
                    on_step(self.step)
        finally:
            self.close()
        return {"steps": self.step, "final_loss": self.last_loss, "checked": self.checked,
                "unchecked": self.unchecked, "disputes": self.disputes, "peers_blocked": len(self.caught),
                "redone": self.redos}

    def export(self, out_dir: str, pairs: int, base_model_name: str = "", score: Optional[dict] = None,
               kept_back: int = 0) -> str:
        """Write the result as a standard LoRA adapter, with the facts a person needs to judge it."""
        os.makedirs(out_dir, exist_ok=True)
        prefix = self.ends.stack.prefix
        weights = {}
        for key, value in self.adapters.items():
            layer, _, rest = key.partition(".")
            path, part = rest.rsplit(".", 1)
            weights[f"base_model.model.{prefix}{layer}.{path}.lora_{part}.weight"] = value
        with open(os.path.join(out_dir, "adapter_model.safetensors"), "wb") as f:
            f.write(st.dumps(weights))
        with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
            json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": self.rank, "lora_alpha": self.alpha,
                       "lora_dropout": 0.0, "bias": "none", "target_modules": list(self.targets),
                       "base_model_name_or_path": base_model_name, "fan_in_fan_out": False,
                       "inference_mode": True}, f, indent=2)
        with open(os.path.join(out_dir, FACTS), "w") as f:
            json.dump({"name": self.run_id, "base_model_fingerprint": self.ends.fingerprint,
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": pairs, "steps": self.step,
                       "final_loss": self.last_loss, "work_checked": self.checked, "work_unchecked": self.unchecked,
                       "peers_blocked": len(self.caught), "rank": self.rank, "alpha": self.alpha, "lr": self.lr,
                       "pairs_kept_back": kept_back, "score": score,
                       "base_model_name": os.path.basename(os.path.normpath(base_model_name or "")),
                       "trained_from": "a phone or other device without PyTorch"}, f, indent=2)
        return out_dir
