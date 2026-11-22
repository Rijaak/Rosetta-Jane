"""Host side: owns the training data and both ends of the model, and drives the run.

One training example travels like this (two guests shown):

    host layers -> guest1 layers -> guest2 layers -> host output layer -> loss
    host layers <- guest1 layers <- guest2 layers <- gradient of the loss

The host relays between guests, so guests never talk to each other and never
see the text. After the chosen number of examples, the host asks every machine
for its gradient size, works out one shared clipping factor, and tells everyone
to take the same optimizer step. That makes the result match what a single
machine holding the whole model would compute.

Checkpoints are taken on every machine at the same step number. After any
failure the host reconnects, finds the newest step that every machine has
saved, rolls everyone back to it and carries on from there.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from safetensors.torch import save_file as st_save_file

from .data import Example, Schedule
from .protocol import AuthError, Channel, MismatchError, ProtocolError, RemoteError
from .slicing import make_slice, model_fingerprint, parse_addr, validate_plan
from .stage import LoraSettings, Stage, adapter_from_bytes

_RECOVERABLE = (ConnectionError, OSError, ProtocolError, RemoteError, socket.timeout)
# Wrong secret, wrong model, wrong plan: waiting and retrying cannot fix these, so stop
# at once and say why instead of looping.
_FATAL = (AuthError, MismatchError)


def _log(msg: str) -> None:
    print(f"[host {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class TrainConfig:
    total_steps: int = 100            # optimizer steps
    accum: int = 16                   # examples per optimizer step
    lr: float = 2e-4
    warmup_steps: int = 20
    schedule: str = "cosine"          # or "constant"
    max_grad_norm: float = 1.0        # 0 turns clipping off
    save_every: int = 200
    keep_checkpoints: int = 2
    fp16_scaling: bool = False        # loss scaling for half-precision cards
    init_scale: float = 65536.0
    growth_interval: int = 2000
    rpc_timeout: float = 900.0        # seconds to wait on one request to a guest
    retry_wait: float = 5.0           # seconds between reconnect attempts
    max_outage: float = 86400.0       # give up if guests stay unreachable this long
    max_same_step_failures: int = 3
    eval_every: int = 0               # optimizer steps between holdout checks; 0 = only at the end
    run_id: str = "run"
    data_seed: int = 0


class Remote:
    """The host's connection to one guest."""

    def __init__(self, info: dict, secret: bytes, fingerprint: str, settings: LoraSettings,
                 run_id: str, timeout: float, encrypt: bool = True):
        self._encrypt = encrypt
        self.info = info
        self.name = info["name"]
        self._secret = secret
        self._fingerprint = fingerprint
        self._settings = settings
        self._run_id = run_id
        self._timeout = timeout
        self.ch: Optional[Channel] = None
        self.checkpoints: List[int] = []
        self.bytes_total = 0
        self.handoffs = None                      # a HandoffLog; fingerprints of what crosses this link

    def connect(self) -> None:
        self.close()
        host, port = parse_addr(self.info["addr"])
        ch = Channel.connect(host, port, self._secret, timeout=min(30.0, self._timeout),
                             encrypt=self._encrypt)
        ch.sock.settimeout(self._timeout)
        if self.handoffs is not None:
            ch.tap = self.handoffs.tap("asking", self.info["addr"])
        try:
            hello, _, _ = ch.call({"op": "hello"})
            if hello.get("name") != self.name:
                raise MismatchError(f"{self.info['addr']} is {hello.get('name')!r}, plan expects {self.name!r}")
            if hello.get("fingerprint") != self._fingerprint:
                raise MismatchError(f"{self.name} has a different base model than the host")
            if hello.get("layers") != self.info["layers"]:
                raise MismatchError(f"{self.name} is using a different plan file than the host")
            reply, _, _ = ch.call({"op": "configure", "settings": self._settings.to_dict(),
                                   "run_id": self._run_id})
            self.checkpoints = [int(s) for s in reply.get("checkpoints", [])]
        except BaseException:
            ch.close()
            raise
        self.ch = ch

    def _call(self, header: dict, tensor=None):
        if self.ch is None:
            raise ConnectionError(f"not connected to {self.name}")
        before = self.ch.bytes_sent + self.ch.bytes_received
        try:
            return self.ch.call(header, tensor)
        finally:
            self.bytes_total += self.ch.bytes_sent + self.ch.bytes_received - before

    def forward(self, h: torch.Tensor, nograd: bool = False) -> torch.Tensor:
        return self._call({"op": "forward", "nograd": nograd}, h)[1]

    def backward(self, g: torch.Tensor) -> torch.Tensor:
        return self._call({"op": "backward"}, g)[1]

    def prepare(self, scale: float):
        r = self._call({"op": "prepare", "scale": scale})[0]
        return bool(r["finite"]), float(r["sqnorm"])

    def apply(self, clip_coef: float, lr: float) -> None:
        self._call({"op": "apply", "clip_coef": clip_coef, "lr": lr})

    def skip(self) -> None:
        self._call({"op": "skip"})

    def save(self, step: int) -> None:
        self._call({"op": "save", "step": step})

    def prune(self, keep: int) -> None:
        self._call({"op": "prune", "keep": keep})

    def load(self, step: int) -> None:
        self._call({"op": "load", "step": step})

    def reinit(self) -> None:
        self._call({"op": "reinit"})

    def export(self) -> Dict[str, torch.Tensor]:
        blob = self._call({"op": "export"})[2]
        if blob is None:
            raise ProtocolError(f"{self.name} sent no adapter")
        return adapter_from_bytes(blob)

    def close(self) -> None:
        if self.ch is not None:
            try:
                self.ch.send({"op": "bye"})
            except (OSError, ProtocolError):
                pass
            self.ch.close()
            self.ch = None


def merge_adapters(plan: dict, parts: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Join every machine's LoRA weights into one adapter for the whole model.

    Each machine numbers its layers from zero. This puts the real layer numbers back.
    """
    pattern = re.compile(re.escape(plan["layer_prefix"]) + r"(\d+)\.")
    merged: Dict[str, torch.Tensor] = {}
    for stage in plan["stages"]:
        start, end = stage["layers"]
        for key, value in parts[stage["name"]].items():
            m = pattern.search(key)
            if m is None:
                raise RuntimeError(f"adapter weight {key!r} is not inside a decoder layer")
            local = int(m.group(1))
            if not 0 <= local < end - start:
                raise RuntimeError(f"{stage['name']} sent a weight for a layer it does not hold")
            new_key = key[:m.start(1)] + str(local + start) + key[m.end(1):]
            if new_key in merged:
                raise RuntimeError(f"two machines sent the same weight {new_key!r}")
            merged[new_key] = value
    return merged


class Host:
    def __init__(self, model_dir: str, plan: dict, work_dir: str, secret: bytes,
                 settings: LoraSettings, cfg: TrainConfig, *, load_4bit: bool = False,
                 dtype: Optional[str] = None, device_map=None, max_memory: Optional[dict] = None,
                 encrypt: bool = True):
        validate_plan(plan)
        self.plan = plan
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.cfg = cfg
        self.settings = settings
        self.fingerprint = model_fingerprint(model_dir)
        self.ckpt_root = os.path.join(work_dir, "checkpoints", cfg.run_id)
        os.makedirs(self.ckpt_root, exist_ok=True)
        _log(f"cutting layers {plan['stages'][0]['layers']} plus both ends out of the local model copy")
        slice_dir = make_slice(model_dir, os.path.join(work_dir, "slice"), plan, "host")
        _log("loading layers and attaching LoRA")
        self.stage = Stage(slice_dir, True, settings, load_4bit=load_4bit, dtype=dtype,
                           device_map=device_map, max_memory=max_memory)
        self.remotes = [Remote(s, secret, self.fingerprint, settings, cfg.run_id, cfg.rpc_timeout, encrypt)
                        for s in plan["stages"][1:]]
        self.scale = cfg.init_scale if cfg.fp16_scaling else 1.0
        self._good_steps = 0
        self.skipped_steps = 0
        self.recoveries = 0
        self._metrics = open(os.path.join(work_dir, f"metrics-{cfg.run_id}.jsonl"), "a")
        from ..handoff import HandoffLog
        self.handoffs = HandoffLog(os.path.join(work_dir, "handoffs.jsonl"), "host")
        for r in self.remotes:
            r.handoffs = self.handoffs

    # -- connection and state sync -------------------------------------------
    def connect_all(self) -> None:
        for r in self.remotes:
            r.connect()
            _log(f"connected to {r.name} at {r.info['addr']}")

    def _sync(self) -> int:
        """Put every machine on the newest step they have all saved. Returns that step."""
        common = set(Stage.list_checkpoints(self.ckpt_root))
        for r in self.remotes:
            common &= set(r.checkpoints)
        if common:
            step = max(common)
            extra = self.stage.load(self.ckpt_root, step)
            for r in self.remotes:
                r.load(step)
            if self.cfg.fp16_scaling:
                self.scale = float(extra.get("scale", self.scale))
                self._good_steps = int(extra.get("good_steps", 0))
            _log(f"all machines restored to step {step}")
            return step
        self.stage.reinit()
        for r in self.remotes:
            r.reinit()
        _log("no shared checkpoint; starting from step 0")
        return 0

    def _recover(self) -> int:
        self.recoveries += 1
        self.stage.skip()
        deadline = time.time() + self.cfg.max_outage
        while True:
            try:
                self.connect_all()
                return self._sync()
            except _FATAL:
                raise
            except _RECOVERABLE as e:
                if time.time() > deadline:
                    raise RuntimeError(f"guests unreachable for too long; last error: {e}") from e
                _log(f"still waiting for guests ({e}); retrying in {self.cfg.retry_wait:.0f}s")
                time.sleep(self.cfg.retry_wait)

    # -- one step ------------------------------------------------------------
    def lr_at(self, step: int) -> float:
        c = self.cfg
        if c.warmup_steps > 0 and step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        if c.schedule == "constant":
            return c.lr
        span = max(1, c.total_steps - c.warmup_steps)
        progress = min(1.0, (step - c.warmup_steps) / span)
        return c.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _forward_chain(self, ids: torch.Tensor, nograd: bool) -> torch.Tensor:
        h = self.stage.head_forward(ids, keep_graph=not nograd)
        for r in self.remotes:
            h = r.forward(h, nograd=nograd)
        return h

    def _example(self, ids: torch.Tensor, labels: torch.Tensor) -> float:
        h = self._forward_chain(ids, nograd=False)
        loss, grad = self.stage.tail(h, labels, self.scale / self.cfg.accum)
        for r in reversed(self.remotes):
            grad = r.backward(grad)
        self.stage.head_backward(grad)
        return loss

    def _optimizer_step(self, step: int) -> dict:
        results = [self.stage.prepare(self.scale)] + [r.prepare(self.scale) for r in self.remotes]
        if not all(finite for finite, _ in results):
            self.stage.skip()
            for r in self.remotes:
                r.skip()
            self.skipped_steps += 1
            self._good_steps = 0
            if self.cfg.fp16_scaling:
                self.scale = max(self.scale / 2.0, 1.0)
            return {"skipped": True, "grad_norm": float("nan"), "lr": 0.0}
        norm = math.sqrt(sum(sq for _, sq in results))
        coef = 1.0
        if self.cfg.max_grad_norm > 0 and norm > self.cfg.max_grad_norm:
            coef = self.cfg.max_grad_norm / (norm + 1e-6)
        lr = self.lr_at(step)
        self.stage.apply(coef, lr)
        for r in self.remotes:
            r.apply(coef, lr)
        if self.cfg.fp16_scaling:
            self._good_steps += 1
            if self._good_steps >= self.cfg.growth_interval:
                self.scale *= 2.0
                self._good_steps = 0
        return {"skipped": False, "grad_norm": norm, "lr": lr}

    def _save(self, step: int) -> None:
        self.stage.save(self.ckpt_root, step, {"scale": self.scale, "good_steps": self._good_steps})
        for r in self.remotes:
            r.save(step)
        # Only once everyone has the new checkpoint is it safe to drop old ones.
        Stage.prune(self.ckpt_root, self.cfg.keep_checkpoints)
        for r in self.remotes:
            r.prune(self.cfg.keep_checkpoints)
            r.checkpoints = sorted(set(r.checkpoints) | {step})[-self.cfg.keep_checkpoints:]

    def evaluate(self, examples: List[Example]) -> float:
        total = 0.0
        for ids, labels in examples:
            total += self.stage.tail_eval(self._forward_chain(ids, nograd=True), labels)
        return total / max(1, len(examples))

    # -- the run ---------------------------------------------------------------
    def train(self, examples: List[Example], holdout: Optional[List[Example]] = None,
              on_step: Optional[Callable[[int], None]] = None) -> dict:
        cfg = self.cfg
        order = Schedule(len(examples), cfg.data_seed)
        try:
            self.connect_all()
            step = self._sync()
        except _FATAL:
            raise
        except _RECOVERABLE as e:
            _log(f"could not reach every guest ({e}); will keep trying")
            step = self._recover()
        failures_here, last_failed = 0, -1
        last_loss = float("nan")
        while step < cfg.total_steps:
            t0 = time.time()
            try:
                losses = []
                for micro in range(cfg.accum):
                    ids, labels = examples[order.index(step * cfg.accum + micro)]
                    losses.append(self._example(ids, labels))
                info = self._optimizer_step(step)
                step += 1
                if step % cfg.save_every == 0 or step == cfg.total_steps:
                    self._save(step)
                last_loss = sum(losses) / len(losses)
                record = {"step": step, "loss": last_loss, "seconds": time.time() - t0,
                          "net_mb": sum(r.bytes_total for r in self.remotes) / 1e6, **info}
                if holdout and cfg.eval_every and step % cfg.eval_every == 0:
                    record["holdout_loss"] = self.evaluate(holdout)
                self._metrics.write(json.dumps(record) + "\n")
                self._metrics.flush()
                _log("step {step}/{total}  loss {loss:.4f}  {seconds:.1f}s{skip}".format(
                    total=cfg.total_steps, skip="  (skipped: bad gradients)" if info["skipped"] else "",
                    **record))
                if on_step is not None:
                    on_step(step)
            except _FATAL:
                raise
            except _RECOVERABLE as e:
                failures_here = failures_here + 1 if step == last_failed else 1
                last_failed = step
                if failures_here >= cfg.max_same_step_failures:
                    raise RuntimeError(
                        f"step {step + 1} failed {failures_here} times in a row; last error: {e}") from e
                _log(f"step {step + 1} interrupted: {type(e).__name__}: {e}")
                step = self._recover()
        result = {"steps": step, "final_loss": last_loss, "skipped_steps": self.skipped_steps,
                  "recoveries": self.recoveries}
        if holdout:
            result["holdout_loss"] = self.evaluate(holdout)
        return result

    def export(self, out_dir: str, base_model_name: Optional[str] = None) -> str:
        """Write one standard LoRA adapter for the full model, built from every machine's part."""
        from peft import LoraConfig
        parts = {"host": self.stage.adapter_state()}
        for r in self.remotes:
            parts[r.name] = r.export()
        merged = merge_adapters(self.plan, parts)
        os.makedirs(out_dir, exist_ok=True)
        s = self.settings
        LoraConfig(r=s.rank, lora_alpha=s.alpha, lora_dropout=s.dropout, bias="none",
                   target_modules=list(s.target_modules), task_type="CAUSAL_LM",
                   base_model_name_or_path=base_model_name or self.model_dir).save_pretrained(out_dir)
        st_save_file(merged, os.path.join(out_dir, "adapter_model.safetensors"))
        return out_dir

    def close(self) -> None:
        for r in self.remotes:
            r.close()
        self._metrics.close()
