"""Training a model through the swarm.

The device doing the training keeps every trained weight and runs the optimizer.
It holds the training text, the first layers and the final output layer. Peers
hold middle layers and act as calculators: for one example at a time they are
sent the small trained matrices for their layers, run the example forward, run
the correction backward, and hand back everything they computed.

Because each peer's work is a pure function of what it was sent:
  * a peer that leaves takes nothing with it; the example is simply redone,
  * two peers given the same work can be compared,
  * the trainer can recompute any piece itself to see who is right,
  * a restart only needs the trainer's own checkpoint.

The price is data. Each example sends hidden states out and back, plus the
adapter gradients for each stretch of layers.

What peers can see: the hidden states for your training text (which can be
turned back into an approximation of it) and the trained matrices for their
own layers at that moment. Do not train private material with strangers.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load as st_load
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save as st_save
from safetensors.torch import save_file as st_save_file

from .lan.data import Example, Schedule
from .lan.protocol import ProtocolError, RemoteError
from .layers import LocalAuditor, LocalEnds, _run, model_shape
from .lora import DEFAULT_TARGETS, Applied, check_spec, flat, init_adapters, keys_for, shapes_from_files, to_peft
from .net import Node, Peer
from .swarm import Disputed, Generation, Hop, NoRoute, _groups, _ip, agree, plan_route

_NET_ERRORS = (ConnectionError, OSError, ProtocolError, RemoteError)


class _Redo(Exception):
    """Something went wrong partway through an example; put things right and do it again."""


def _log(msg: str) -> None:
    print(f"[jane train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SwarmTrainer:
    def __init__(self, node: Node, ends: LocalEnds, model_dir: str, work_dir: str, *,
                 total_steps: int, accum: int = 16, lr: float = 2e-4, warmup_steps: int = 20,
                 schedule: str = "cosine", max_grad_norm: float = 1.0, rank: int = 8, alpha: float = 16.0,
                 targets: Sequence[str] = DEFAULT_TARGETS, seed: int = 0, data_seed: int = 0,
                 save_every: int = 50, keep_checkpoints: int = 2, run_id: str = "run",
                 redundancy: Optional[int] = None, tolerance: float = 0.02,
                 auditor: Optional[LocalAuditor] = None, audit_rate: float = 0.0,
                 canaries=None, canary_rate: float = 0.2, variant: Optional[str] = None,
                 find_wait: float = 15.0, rpc_timeout: float = 300.0, loss_scale: float = 1.0,
                 max_redo: int = 8, quiet: bool = False, lock=None,
                 init: Optional[Dict[str, torch.Tensor]] = None, record_helpers: bool = True,
                 team: Optional[str] = None):
        self.node, self.ends = node, ends
        from . import identity
        identity.of(node, work_dir)                # the key helpers will ask for their points receipts
        self.team = team                          # train inside a team: only its signed-on devices do the work
        self.rank, self.alpha, self.targets = check_spec(rank, alpha, list(targets))
        self.scale = self.alpha / self.rank
        self.layer_prefix, self.num_layers = model_shape(model_dir)
        self.shapes = shapes_from_files(model_dir, self.targets)
        self.adapters = {k: v.requires_grad_(True) for k, v in init_adapters(self.shapes, self.rank, seed).items()}
        self._order = sorted(self.adapters)
        if init is not None:                      # carry on from a finished adapter instead of from nothing
            if sorted(init) != self._order or any(tuple(init[k].shape) != tuple(self.adapters[k].shape)
                                                  or not torch.isfinite(init[k]).all() for k in init):
                raise ValueError("the adapter to start from does not fit this model")
            with torch.no_grad():
                for k in self._order:
                    self.adapters[k].copy_(init[k])
        self.history: List[dict] = []             # the adapters this run stands on, oldest first
        self.opt = torch.optim.AdamW([self.adapters[k] for k in self._order], lr=0.0, weight_decay=0.0)
        self._acc = {k: torch.zeros_like(v) for k, v in self.adapters.items()}
        self.total_steps, self.accum, self.lr = total_steps, accum, lr
        self.warmup_steps, self.schedule, self.max_grad_norm = warmup_steps, schedule, max_grad_norm
        self.data_seed, self.save_every, self.keep = data_seed, save_every, keep_checkpoints
        self.dir = os.path.join(work_dir, f"train-{run_id}")
        os.makedirs(self.dir, exist_ok=True)
        # Who helped: each helping device gets a ticket, and its accepted work is counted, so the
        # finished adapter can be shared with the devices that worked on it (see helpers.py).
        from .helpers import RunBook
        self.book = RunBook(self.dir) if record_helpers else None
        self._pending: List[str] = []
        self.redundancy, self.tolerance = redundancy, tolerance
        self.auditor, self.audit_rate = auditor, (audit_rate if auditor is not None else 0.0)
        self.canaries, self.canary_rate = canaries, canary_rate
        self.variant, self.find_wait, self.rpc_timeout = variant, find_wait, rpc_timeout
        self.loss_scale, self.max_redo, self.quiet = loss_scale, max_redo, quiet
        self._vetter = Generation(node, ends, canaries=canaries, canary_rate=canary_rate, tolerance=tolerance,
                                  rpc_timeout=rpc_timeout)
        self._rng = random.Random(seed)
        self.route: List[Hop] = []
        self.step = 0
        self._bad: set = set()
        self._fresh: set = set()                  # hops that have not yet had an example rechecked
        self._want_extra: set = set()
        self.verified_examples = 0                # hop-examples with a second opinion or a recheck
        self.unverified_examples = 0
        self.disputes = self.spot_audits = self.redos = self.reroutes = self.skipped_steps = 0
        self.bytes = 0
        self.last_loss: Optional[float] = None
        self._lock = lock                         # shared with answering, so the two take turns

    @property
    def caught(self) -> List[str]:
        return self._vetter.caught

    # -- checkpoints (everything lives on this device) ------------------------------
    def _ckpt(self, step: int) -> str:
        return os.path.join(self.dir, f"step-{step:08d}")

    def checkpoints(self) -> List[int]:
        return sorted(int(n[5:]) for n in os.listdir(self.dir)
                      if n.startswith("step-") and n[5:].isdigit()
                      and os.path.exists(os.path.join(self.dir, n, "done.json")))

    def save(self) -> None:
        final, tmp = self._ckpt(self.step), self._ckpt(self.step) + ".partial"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        st_save_file({k: v.detach().contiguous() for k, v in self.adapters.items()},
                     os.path.join(tmp, "adapters.safetensors"))
        torch.save(self.opt.state_dict(), os.path.join(tmp, "optimizer.pt"))
        with open(os.path.join(tmp, "done.json"), "w") as f:
            json.dump({"step": self.step, "rank": self.rank, "alpha": self.alpha, "targets": self.targets}, f)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        for old in self.checkpoints()[:-self.keep]:
            shutil.rmtree(self._ckpt(old), ignore_errors=True)
        if self.book is not None:
            self.book.save()

    def resume(self) -> bool:
        have = self.checkpoints()
        if not have:
            return False
        d = self._ckpt(have[-1])
        with open(os.path.join(d, "done.json")) as f:
            meta = json.load(f)
        if (meta["rank"], meta["alpha"], meta["targets"]) != (self.rank, self.alpha, self.targets):
            raise ValueError("the saved run used different LoRA settings; use another run id or the same settings")
        saved = st_load_file(os.path.join(d, "adapters.safetensors"))
        with torch.no_grad():
            for k in self._order:
                self.adapters[k].copy_(saved[k])
        self.opt.load_state_dict(torch.load(os.path.join(d, "optimizer.pt"), weights_only=True))
        self.step = int(meta["step"])
        return True

    # -- peers ------------------------------------------------------------------------
    def _excluded(self) -> set:
        return self._bad | self.node.banned

    def _blob(self, start: int, end: int) -> bytes:
        return st_save({k: self.adapters[k].detach().contiguous() for k in keys_for(self.shapes, start, end)})

    def _shut(self, replica: Hop) -> None:
        if replica.ch is not None:
            self.bytes += replica.ch.bytes_sent + replica.ch.bytes_received
            try:
                replica.ch.send({"op": "bye"})
            except _NET_ERRORS:
                pass
            replica.ch.close()
            replica.ch = None

    def _block(self, replica: Hop) -> None:
        self._shut(replica)
        self._vetter._ban(replica.peer)

    def _open(self, replica: Hop) -> None:
        peer = replica.peer
        if self.canaries is not None and (peer.node_id not in self.node.vetted
                                          or self._rng.random() < self.canary_rate):
            if self._vetter._test(peer, replica.start, replica.end, replica.variant) is False:
                raise ProtocolError("peer failed a known-answer test")
        ch = self.node.dial(peer.addr, timeout=min(30.0, self.rpc_timeout))
        ch.sock.settimeout(self.rpc_timeout)
        try:
            ask = {"op": "t_open", "model": self.ends.fingerprint, "start": replica.start, "end": replica.end,
                   "rank": self.rank, "alpha": self.alpha, "targets": self.targets}
            if self.book is not None:
                ask.update(run=self.book.run, ticket=self.book.ticket(peer.node_id))
                if getattr(self.node, "identity", None) is not None:
                    ask["by"] = self.node.identity.key     # where the helper can ask for a points receipt later
            if self.team is not None:
                ask["team"] = self.team
            ch.call(ask)
            ch.call({"op": "t_adapters"}, blob=self._blob(replica.start, replica.end))
        except BaseException:
            ch.close()
            raise
        replica.ch = ch

    def _able(self) -> List[Peer]:
        return [p for p in self.node.table.peers()
                if p.node_id not in self._excluded() and p.addr not in self._excluded()
                and any(s["model"] == self.ends.fingerprint and s.get("train")
                        and (self.team is None or self.team in s.get("teams", ()))
                        and (self.variant is None or s["variant"] == self.variant) for s in p.services)]

    def _recruit(self, hop: Hop, count: int) -> List[Hop]:
        taken = [r for r in hop.replicas() if r.ch is not None] or [hop]
        used_ids = {r.peer.node_id for r in taken}
        used_ips = {_ip(r.peer.addr) for r in taken}
        options = []
        for p in self._able():
            if p.node_id in used_ids:
                continue
            for s in p.services:
                if s["model"] == self.ends.fingerprint and s.get("train") and s["variant"] == hop.variant \
                        and (self.team is None or self.team in s.get("teams", ())) \
                        and s["layers"][0] <= hop.start and hop.end <= s["layers"][1]:
                    options.append((p, s))
                    break
        options.sort(key=lambda ps: (_ip(ps[0].addr) not in used_ips, ps[1]["free"], random.random()), reverse=True)
        got: List[Hop] = []
        for peer, _ in options:
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
        own, total = self.ends.own_layers, self.num_layers
        if own >= total:
            return []
        deadline = time.time() + self.find_wait
        while True:
            if self.redundancy is None:
                wanted = 2 if len(self._able()) >= 3 else 1
            else:
                wanted = self.redundancy
            try:
                hops = plan_route(self.node.table.peers(), self.ends.fingerprint, own, total,
                                  self._excluded(), self.variant, training=True, team=self.team)
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
            self._fresh = set(range(len(hops)))
            self._want_extra = set()
            self._wanted = wanted
            return hops

    def _close_all(self) -> None:
        for hop in self.route:
            for replica in hop.replicas():
                self._shut(replica)
        self.route = []

    def _live(self, i: int) -> List[Hop]:
        live = [r for r in self.route[i].replicas() if r.ch is not None]
        if not live:
            raise _Redo()
        primary = live[0]
        primary.witnesses = live[1:]
        self.route[i] = primary
        return live

    def _recover(self) -> None:
        """After a failed attempt at an example: clear half-done work, replace what was lost."""
        self.redos += 1
        dead = any(not [r for r in hop.replicas() if r.ch is not None] for hop in self.route)
        if not dead:
            for i in sorted(self._want_extra):
                extra = self._recruit(self.route[i], 1)
                if not extra:
                    hop = self.route[i]
                    raise Disputed(
                        f"peers disagree about layers {hop.start}-{hop.end - 1} and nobody is left to break "
                        "the tie. Prepare known answers, turn on rechecking, or wait for more peers.")
                self.route[i].witnesses.extend(extra)
            self._want_extra = set()
            for hop in self.route:
                for replica in hop.replicas():
                    if replica.ch is None:
                        continue
                    try:
                        replica.ch.call({"op": "t_reset"})
                    except _NET_ERRORS:
                        self._shut(replica)
                        self._bad.add(replica.peer.node_id)
            dead = any(not [r for r in hop.replicas() if r.ch is not None] for hop in self.route)
        if dead or not self.route:
            self._close_all()
            self.reroutes += 1
            self.route = self._connect()

    # -- asking peers, and deciding whom to believe ---------------------------------------
    def _ask(self, i: int, op: str, tensor: Optional[torch.Tensor]) -> List[Tuple[Hop, object]]:
        outs = []
        for replica in self._live(i):
            try:
                reply, out, blob = replica.ch.call({"op": op}, tensor=tensor)
                if op == "t_grads":
                    grads = st_load(blob or b"")
                    want = keys_for(self.shapes, replica.start, replica.end)
                    if sorted(grads) != want or any(grads[k].shape != self.adapters[k].shape for k in want):
                        raise ProtocolError("peer returned gradients of the wrong shape")
                    outs.append((replica, grads))
                else:
                    if out is None or tuple(out.shape) != tuple(tensor.shape):
                        raise ProtocolError("peer returned a result of the wrong shape")
                    outs.append((replica, out))
            except _NET_ERRORS:
                self._shut(replica)               # gone or broken, not proven dishonest
                self._bad.add(replica.peer.node_id)
        if not outs:
            raise _Redo()
        return outs

    def _truth(self, i: int, h_in: torch.Tensor, g_out: Optional[torch.Tensor]) -> dict:
        hop = self.route[i]
        mine = {k: self.adapters[k].detach() for k in keys_for(self.shapes, hop.start, hop.end)}
        h_out, g_in, grads = self.auditor.train_example(hop.start, hop.end, mine, self.scale, self.targets,
                                                        h_in, g_out)
        return {"fwd": h_out, "bwd": g_in, "grads": None if grads is None else flat(grads).unsqueeze(0)}

    def _settle(self, i: int, outs: List[Tuple[Hop, object]], kind: str, h_in: torch.Tensor,
                g_out: Optional[torch.Tensor] = None):
        """Pick the result to trust for hop i at one stage of an example. Blocks proven liars."""
        as_tensor = (lambda v: flat(v).unsqueeze(0)) if kind == "grads" else (lambda v: v)
        view = [(r, as_tensor(v)) for r, v in outs]
        if len(view) > 1 and len(_groups(view, self.tolerance)) > 1:
            self.disputes += 1
            view = self._resolve(i, view, kind, h_in, g_out)
        kept = {id(r) for r, _ in view}
        for replica, _ in outs:
            if id(replica) not in kept and replica.ch is not None:
                self._block(replica)
        winner = view[0][0]
        self._live(i)
        return next(v for r, v in outs if r is winner)

    def _resolve(self, i: int, view, kind: str, h_in, g_out):
        hop = self.route[i]
        if self.canaries is not None:              # fresh work with a known answer for each side
            for replica, _ in list(view):
                try:
                    verdict = self._vetter._test(replica.peer, hop.start, hop.end, hop.variant)
                except _NET_ERRORS:
                    verdict = None
                if verdict is False:
                    self._shut(replica)
                    view = [(r, v) for r, v in view if r is not replica]
            if not view:
                raise _Redo()
            if len(_groups(view, self.tolerance)) == 1:
                return view
        if self.auditor is not None:               # recompute this example here: right however many lie
            truth = self._truth(i, h_in, g_out)[kind]
            good = [(r, v) for r, v in view if agree(v, truth, self.tolerance)]
            for replica, _ in view:
                if not any(replica is r for r, _ in good):
                    self._block(replica)
            if good:
                return good
            raise _Redo()
        groups = _groups(view, self.tolerance)
        if len(groups[0]) * 2 > len(view):         # a strict majority agrees
            return groups[0]
        self._want_extra.add(i)                    # bring in one more opinion and do the example again
        raise _Redo()

    # -- one example ------------------------------------------------------------------------
    def _forward(self, ids: torch.Tensor):
        own_keys = keys_for(self.shapes, 0, self.ends.own_layers)
        device = self.ends._embed.weight.device
        own = {k: self.adapters[k].detach().to(device).requires_grad_(True) for k in own_keys}
        emb = self.ends._embed(ids.to(device))
        with Applied(self.ends._layers, 0, self.targets, own, self.scale):
            h0, _ = _run(self.ends._decoder, self.ends._layers, emb, None, keep_cache=False)
        h = h0.detach()
        seen = []
        for i in range(len(self.route)):
            h_out = self._settle(i, self._ask(i, "t_fwd", h), "fwd", h)
            seen.append((h, h_out))
            h = h_out
        return own, h0, seen, h

    def _tail_loss(self, h_last: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        head = self.ends._head
        logits = head(self.ends._norm(h_last.to(head.weight.device)))
        return F.cross_entropy(logits[0, :-1].float(), labels[0, 1:].to(logits.device), ignore_index=-100)

    def _attempt(self, ids: torch.Tensor, labels: torch.Tensor) -> float:
        own, h0, seen, h = self._forward(ids)
        last = h.detach().to(self.ends._head.weight.device).requires_grad_(True)
        loss = self._tail_loss(last, labels)
        (loss * self.loss_scale).backward()
        g = last.grad
        got: Dict[str, torch.Tensor] = {}
        for i in reversed(range(len(self.route))):
            h_in, h_out = seen[i]
            g_in = self._settle(i, self._ask(i, "t_bwd", g), "bwd", h_in, g)
            grads = self._settle(i, self._ask(i, "t_grads", None), "grads", h_in, g)
            checked = len([r for r in self.route[i].replicas() if r.ch is not None]) > 1
            if self.audit_rate > 0 and (i in self._fresh or self._rng.random() < self.audit_rate):
                # Agreement proves nothing if they all tell the same lie. Recompute it here anyway.
                self.spot_audits += 1
                truth = self._truth(i, h_in, g)
                if not (agree(h_out, truth["fwd"], self.tolerance) and agree(g_in, truth["bwd"], self.tolerance)
                        and agree(flat(grads).unsqueeze(0), truth["grads"], self.tolerance)):
                    self.disputes += 1
                    for replica in self.route[i].replicas():
                        if replica.ch is not None:
                            self._block(replica)
                    raise _Redo()
                checked = True
            self._fresh.discard(i)
            if checked:
                self.verified_examples += 1
            else:
                self.unverified_examples += 1
            self._pending += [r.peer.node_id for r in self.route[i].replicas() if r.ch is not None]
            got.update(grads)
            g = g_in
        h0.backward(g.to(h0.device, h0.dtype))
        for k, v in own.items():
            got[k] = v.grad.detach().float().cpu()
        for k, v in got.items():
            self._acc[k] += v / self.loss_scale
        return float(loss.detach())

    def _example(self, ids: torch.Tensor, labels: torch.Tensor) -> float:
        if self._lock is None:
            return self._one(ids, labels)
        with self._lock:
            return self._one(ids, labels)

    def _one(self, ids: torch.Tensor, labels: torch.Tensor) -> float:
        for _ in range(self.max_redo):
            before = (self.verified_examples, self.unverified_examples)
            self._pending = []
            try:
                if not self.route and self.ends.own_layers < self.num_layers:
                    self.route = self._connect()
                began = time.time()
                loss = self._attempt(ids, labels)
                if self.book is not None:         # only work that went into the result counts as help
                    self.book.credit(self._pending, (time.time() - began) / max(1, len(self._pending)))
                return loss
            except _Redo:
                self.verified_examples, self.unverified_examples = before
                self._recover()
        raise NoRoute("one training example kept failing; too many peers dropped out or were caught lying")

    # -- optimizer -----------------------------------------------------------------------------
    def lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        if self.schedule == "constant":
            return self.lr
        span = max(1, self.total_steps - self.warmup_steps)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - self.warmup_steps) / span)))

    def _optimizer_step(self) -> dict:
        sq, finite = 0.0, True
        for k in self._order:
            g = self._acc[k] / self.accum
            self._acc[k] = torch.zeros_like(self._acc[k])
            if not torch.isfinite(g).all():
                finite = False
            sq += float(g.pow(2).sum())
            self.adapters[k].grad = g
        if not finite:
            self.opt.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            return {"skipped": True, "grad_norm": float("nan"), "lr": 0.0}
        norm = math.sqrt(sq)
        if self.max_grad_norm > 0 and norm > self.max_grad_norm:
            coef = self.max_grad_norm / (norm + 1e-6)
            for k in self._order:
                self.adapters[k].grad.mul_(coef)
        lr = self.lr_at(self.step)
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self._push()                               # send everyone the updated weights
        return {"skipped": False, "grad_norm": norm, "lr": lr}

    def _push(self) -> None:
        for hop in self.route:
            for replica in hop.replicas():
                if replica.ch is None:
                    continue
                try:
                    replica.ch.call({"op": "t_adapters"}, blob=self._blob(replica.start, replica.end))
                except _NET_ERRORS:
                    self._shut(replica)
                    self._bad.add(replica.peer.node_id)

    # -- scoring: forward only, nothing is learned ------------------------------------------------
    def _hidden(self, ids: torch.Tensor):
        """The final hidden states for one whole text, through the group, with the current adapters."""
        with torch.no_grad():
            own_keys = keys_for(self.shapes, 0, self.ends.own_layers)
            device = self.ends._embed.weight.device
            own = {k: self.adapters[k].detach().to(device) for k in own_keys}
            with Applied(self.ends._layers, 0, self.targets, own, self.scale):
                h, _ = _run(self.ends._decoder, self.ends._layers, self.ends._embed(ids.to(device)), None,
                            keep_cache=False)
            checked = 0
            for i in range(len(self.route)):
                h = self._settle(i, self._ask(i, "t_fwd", h), "fwd", h)
                live = self._live(i)
                checked += len(live) > 1
                for replica in live:               # the peer is holding the calculation for a backward pass
                    try:                           # that will never come; let it go
                        replica.ch.call({"op": "t_reset"})
                    except _NET_ERRORS:
                        self._shut(replica)
                        self._bad.add(replica.peer.node_id)
        return h, checked, len(self.route) - checked

    def _score_once(self, ids: torch.Tensor, labels: torch.Tensor):
        h, checked, unchecked = self._hidden(ids)
        with torch.no_grad():
            head = self.ends._head
            logits = head(self.ends._norm(h.to(head.weight.device)))[0, :-1].float()
            want = labels[0, 1:].to(logits.device)
            graded = want != -100
            loss = F.cross_entropy(logits[graded], want[graded], reduction="sum")
            right = int((logits[graded].argmax(-1) == want[graded]).sum())
        return float(loss), right, int(graded.sum()), checked, unchecked

    def _retry(self, work):
        for _ in range(self.max_redo):
            try:
                if not self.route and self.ends.own_layers < self.num_layers:
                    self.route = self._connect()
                return work()
            except _Redo:
                self._recover()
        raise NoRoute("the calculation kept failing; too many peers dropped out or were caught lying")

    def _continue(self, adapters: Dict[str, torch.Tensor], ids: List[int], max_new: int, stops) -> List[int]:
        mine, self.adapters = self.adapters, adapters
        try:
            self._push()
            out: List[int] = []
            for _ in range(max_new):
                text = torch.tensor([list(ids) + out], dtype=torch.long)
                h = self._retry(lambda: self._hidden(text))[0]
                with torch.no_grad():
                    head = self.ends._head
                    token = int(head(self.ends._norm(h[:, -1:].to(head.weight.device)))[0, -1].argmax())
                if token in stops:
                    break
                out.append(token)
            return out
        finally:
            self.adapters = mine

    def answer(self, ids: Sequence[int], max_new: int = 48, stop_ids: Sequence[int] = (),
               with_adapter: bool = True) -> List[int]:
        """One side of a trial: continue a prompt with the adapters, or with the unchanged model.

        Leaves the connection to the group open so several questions can follow; call finish() after.
        """
        if not ids:
            raise ValueError("empty prompt")
        use = self.adapters if with_adapter else \
            {k: (torch.zeros_like(v) if k.endswith(".B") else v.detach()) for k, v in self.adapters.items()}
        return self._continue(use, list(ids), max_new, set(stop_ids))

    def finish(self) -> None:
        self._close_all()

    def try_out(self, ids: Sequence[int], max_new: int = 48, stop_ids: Sequence[int] = ()) -> Tuple[List[int], List[int]]:
        """Answer one prompt twice: with the unchanged model, and with these adapters. Returns both.

        A trial, not a switch. It always produces both answers so a person can compare them, picks the
        likeliest word each time so the two differ only because of the adapters, and changes nothing.
        It runs on the training route (the whole reply is recomputed for every word), so it is slow.
        """
        if not ids:
            raise ValueError("empty prompt")
        plain = {k: (torch.zeros_like(v) if k.endswith(".B") else v.detach()) for k, v in self.adapters.items()}
        stops = set(stop_ids)
        try:
            without = self._continue(plain, list(ids), max_new, stops)
            with_it = self._continue(self.adapters, list(ids), max_new, stops)
        finally:
            self._close_all()
        return without, with_it

    @classmethod
    def from_adapter(cls, node, ends, model_dir: str, work_dir: str, adapter_dir: str, **kw) -> "SwarmTrainer":
        """A trainer holding a finished adapter read from a folder, for scoring it or trying it out."""
        from safetensors.torch import load_file

        from .lora import from_peft
        try:
            with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
                config = json.load(f)
            weights = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
            rank, alpha, targets = int(config["r"]), float(config["lora_alpha"]), list(config["target_modules"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"that folder does not hold a readable adapter: {e}") from None
        facts = os.path.join(adapter_dir, "rosetta-jane-run.json")
        if os.path.exists(facts):
            try:
                with open(facts) as f:
                    made_for = json.load(f).get("base_model_fingerprint")
            except (OSError, ValueError):
                made_for = None
            if made_for and made_for != ends.fingerprint:
                raise ValueError(f"this adapter was trained on a different model ({made_for}); "
                                 f"the model here is {ends.fingerprint}")
        t = cls(node, ends, model_dir, work_dir, total_steps=1, rank=rank, alpha=alpha, targets=targets, quiet=True,
                record_helpers=False, **kw)
        loaded = from_peft(weights, t.layer_prefix)
        if sorted(loaded) != t._order or any(tuple(loaded[k].shape) != tuple(t.adapters[k].shape) for k in loaded):
            raise ValueError("this adapter does not fit the model here")
        with torch.no_grad():
            for key in t._order:
                if not torch.isfinite(loaded[key]).all():
                    raise ValueError("this adapter contains values that are not numbers")
                t.adapters[key].copy_(loaded[key])
        return t

    @classmethod
    def continuing(cls, node, ends, model_dir: str, work_dir: str, adapter_dir: str, **kw) -> "SwarmTrainer":
        """A trainer that starts from a finished adapter, to train it further.

        Whoever holds an adapter can do this: one they trained, or one a leader shared with them.
        The rank, alpha and targets are the adapter's own. The new run keeps a record of what it
        was continued from.
        """
        from .team import adapter_fingerprint, continued, fitted
        prefix, _ = model_shape(model_dir)
        found = fitted(adapter_dir, prefix, ends.fingerprint, lambda targets: shapes_from_files(model_dir, targets))
        for key in ("rank", "alpha", "targets", "init"):
            kw.pop(key, None)
        t = cls(node, ends, model_dir, work_dir, rank=found["rank"], alpha=found["alpha"], targets=found["targets"],
                init={k: torch.from_numpy(v) for k, v in found["adapters"].items()}, **kw)
        t.history = continued(found["facts"], found["name"], adapter_fingerprint(found["adapters"]))
        return t

    def record(self) -> dict:
        """What to add to a finished adapter's facts: its run, how many devices helped, what it stands on."""
        out: dict = {}
        if self.book is not None:
            out.update(run=self.book.run, helpers=len(self.book.helpers))
        if self.history:
            out["history"] = self.history
        return out

    def _score_with(self, adapters: Dict[str, torch.Tensor], examples: List[Example]) -> Tuple[dict, int, int, int]:
        from .scoring import tally
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

    def losses(self, adapters: Dict[str, torch.Tensor], examples: List[Example]) -> List[float]:
        """The loss on each pair's answer, one number per pair, with the given adapters. Forward only."""
        mine, self.adapters = self.adapters, adapters
        try:
            self._push()
            out = []
            for ids, labels in examples:
                loss, _, tokens, _, _ = self._retry(lambda: self._score_once(ids, labels))
                out.append(loss / max(1, tokens))
            return out
        finally:
            self.adapters = mine

    def flattery(self, examples: List[Example]) -> dict:
        """The flattery check (see flattery.py): does the model, with these adapters, find replies that go
        along with false statements likelier than it did without them? Through the swarm, checked as usual."""
        from .flattery import card
        plain = {k: (torch.zeros_like(v) if k.endswith(".B") else v.detach()) for k, v in self.adapters.items()}
        try:
            if not self.route and self.ends.own_layers < self.num_layers:
                self.route = self._connect()
            return card(self.losses(plain, examples), self.losses(self.adapters, examples))
        finally:
            self._close_all()

    def compare(self, examples: List[Example], name: str = "", held_out: bool = False) -> dict:
        """A score card: the unchanged model against the model with these adapters, on the same pairs.

        Runs through the swarm, forward only. The unchanged model is the same adapters with their
        second halves set to zero, so both sides travel the same route and are checked the same way.
        """
        from .scoring import card
        if not examples:
            return card(name, self.ends.fingerprint, 0, 0, {"loss": None}, {"loss": None}, held_out=held_out)
        plain = {k: (torch.zeros_like(v) if k.endswith(".B") else v.detach()) for k, v in self.adapters.items()}
        try:
            if not self.route and self.ends.own_layers < self.num_layers:
                self.route = self._connect()
            before, tokens, c1, u1 = self._score_with(plain, examples)
            after, _, c2, u2 = self._score_with(self.adapters, examples)
        finally:
            self._close_all()
        return card(name, self.ends.fingerprint, len(examples), tokens, before, after, held_out=held_out,
                    checked=c1 + c2, unchecked=u1 + u2)

    # -- the run -------------------------------------------------------------------------------
    def train(self, examples: List[Example], on_step: Optional[Callable[[int], None]] = None) -> dict:
        if self.resume() and not self.quiet:
            _log(f"continuing from step {self.step}")
        order = Schedule(len(examples), self.data_seed)
        metrics = open(os.path.join(self.dir, "metrics.jsonl"), "a")
        last = float("nan")
        try:
            while self.step < self.total_steps:
                t0 = time.time()
                for k in self._order:              # a step interrupted earlier leaves nothing behind
                    self._acc[k].zero_()
                losses = [self._example(*examples[order.index(self.step * self.accum + m)])
                          for m in range(self.accum)]
                info = self._optimizer_step()
                self.step += 1
                last = self.last_loss = sum(losses) / len(losses)
                if self.step % self.save_every == 0 or self.step == self.total_steps:
                    self.save()
                record = {"step": self.step, "loss": last, "seconds": time.time() - t0,
                          "checked": self.verified_examples, "unchecked": self.unverified_examples,
                          "peers_blocked": len(self.caught), **info}
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                if not self.quiet:
                    _log(f"step {self.step}/{self.total_steps}  loss {last:.4f}  {record['seconds']:.1f}s")
                if on_step is not None:
                    on_step(self.step)
        finally:
            metrics.close()
            self._close_all()
        return {"steps": self.step, "final_loss": last, "checked_examples": self.verified_examples,
                "unchecked_examples": self.unverified_examples, "disputes": self.disputes,
                "spot_audits": self.spot_audits, "peers_blocked": len(self.caught), "redone": self.redos,
                "reroutes": self.reroutes, "skipped_steps": self.skipped_steps}

    def export(self, out_dir: str, base_model_name: str) -> str:
        """Write the result as a standard LoRA adapter for the whole model."""
        from peft import LoraConfig
        os.makedirs(out_dir, exist_ok=True)
        LoraConfig(r=self.rank, lora_alpha=self.alpha, lora_dropout=0.0, bias="none",
                   target_modules=list(self.targets), task_type="CAUSAL_LM",
                   base_model_name_or_path=base_model_name).save_pretrained(out_dir)
        st_save_file(to_peft(self.adapters, self.layer_prefix), os.path.join(out_dir, "adapter_model.safetensors"))
        if self.book is not None:
            self.book.copy_to(out_dir)             # lets the devices that helped be offered a copy later
        return out_dir
