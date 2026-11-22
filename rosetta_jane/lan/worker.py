"""Guest side: hold a run of middle layers and do what the host asks.

A worker is a small server. It accepts one host at a time, checks the shared
secret, and then answers requests: run my layers forward, run them backward,
take an optimizer step, save or load a checkpoint. If the host goes away the
worker keeps its model loaded and waits for it to come back.

A worker only ever does work for a host that knows the secret, only while this
program is running, and stops the moment you press Ctrl-C.
"""
from __future__ import annotations

import os
import socket
import sys
import time
import traceback
from typing import Optional

import torch

from . import __version__
from .protocol import Channel, ProtocolError
from .slicing import load_plan, make_slice, model_fingerprint, parse_addr, stage_of
from .stage import LoraSettings, Stage


def _log(msg: str) -> None:
    print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Worker:
    def __init__(self, model_dir: str, plan_path: str, name: str, work_dir: str, secret: bytes, *,
                 load_4bit: bool = False, dtype: Optional[str] = None, device_map=None,
                 max_memory: Optional[dict] = None, encrypt: bool = True):
        self.encrypt = encrypt
        self.plan = load_plan(plan_path)
        self.name = name
        self.info = stage_of(self.plan, name)
        if name == "host":
            raise ValueError("'host' is the trainer's own stage; give this worker a guest name from the plan")
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.secret = secret
        self.load_kwargs = dict(load_4bit=load_4bit, dtype=dtype, device_map=device_map, max_memory=max_memory)
        self.ckpt_root = os.path.join(work_dir, "checkpoints")
        from ..handoff import HandoffLog
        self.handoffs = HandoffLog(os.path.join(work_dir, "handoffs.jsonl"), name)
        self.stage: Optional[Stage] = None
        self.run_id: Optional[str] = None
        os.makedirs(self.ckpt_root, exist_ok=True)
        _log(f"cutting layers {self.info['layers']} out of the local model copy (first run only)")
        self.slice_dir = make_slice(model_dir, os.path.join(work_dir, "slice"), self.plan, name)
        self.fingerprint = model_fingerprint(model_dir)

    # -- request handlers --------------------------------------------------
    def _run_root(self) -> str:
        return os.path.join(self.ckpt_root, self.run_id or "default")

    def _need_stage(self) -> Stage:
        if self.stage is None:
            raise RuntimeError("the host has not sent settings yet")
        return self.stage

    def handle(self, ch: Channel, header: dict, tensor, blob) -> None:
        op = header["op"]
        if op == "hello":
            ch.send({"op": "hello_ok", "name": self.name, "fingerprint": self.fingerprint,
                     "layers": self.info["layers"], "version": __version__,
                     "cuda": torch.cuda.is_available()})
        elif op == "configure":
            settings = LoraSettings.from_dict(header["settings"])
            run_id = str(header["run_id"])
            if not run_id.replace("-", "").replace("_", "").isalnum() or len(run_id) > 64:
                raise ValueError("bad run id")
            if self.stage is None or self.stage.settings != settings:
                _log("loading layers and attaching LoRA")
                self.stage = Stage(self.slice_dir, False, settings, **self.load_kwargs)
            self.run_id = run_id
            self.stage.skip()
            ch.send({"op": "configure_ok", "checkpoints": Stage.list_checkpoints(self._run_root())})
        elif op == "forward":
            if tensor is None:
                raise ProtocolError("forward needs a tensor")
            stage = self._need_stage()
            out = stage.forward(tensor.to(stage.act_dtype), keep_graph=not header.get("nograd", False))
            ch.send({"op": "forward_ok"}, tensor=out)
        elif op == "backward":
            if tensor is None:
                raise ProtocolError("backward needs a tensor")
            ch.send({"op": "backward_ok"}, tensor=self._need_stage().backward(tensor))
        elif op == "prepare":
            finite, sq = self._need_stage().prepare(float(header["scale"]))
            ch.send({"op": "prepare_ok", "finite": finite, "sqnorm": sq})
        elif op == "apply":
            self._need_stage().apply(float(header["clip_coef"]), float(header["lr"]))
            ch.send({"op": "apply_ok"})
        elif op == "skip":
            self._need_stage().skip()
            ch.send({"op": "skip_ok"})
        elif op == "save":
            self._need_stage().save(self._run_root(), int(header["step"]))
            ch.send({"op": "save_ok"})
        elif op == "prune":
            Stage.prune(self._run_root(), max(1, int(header["keep"])))
            ch.send({"op": "prune_ok"})
        elif op == "load":
            self._need_stage().load(self._run_root(), int(header["step"]))
            ch.send({"op": "load_ok"})
        elif op == "reinit":
            self._need_stage().reinit()
            ch.send({"op": "reinit_ok"})
        elif op == "export":
            ch.send({"op": "export_ok"}, blob=self._need_stage().adapter_bytes())
        else:
            raise ProtocolError(f"unknown request {op!r}")

    # -- server loop -------------------------------------------------------
    def serve(self, listen: str, ready_file: Optional[str] = None) -> None:
        host, port = parse_addr(listen)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        _log(f"{self.name} ready on {listen}, holding layers {self.info['layers']}, "
             f"{'encrypted' if self.encrypt else 'NOT encrypted'}. Ctrl-C to stop.")
        if ready_file:
            with open(ready_file, "w") as f:
                f.write(str(server.getsockname()[1]))
        try:
            while True:
                conn, peer = server.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(30)
                try:
                    ch = Channel.accept(conn, self.secret, encrypt=self.encrypt)
                    ch.tap = self.handoffs.tap("serving", peer[0])
                except (ProtocolError, ConnectionError, OSError) as e:
                    _log(f"refused connection from {peer[0]}: {e}")
                    conn.close()
                    continue
                ch.sock.settimeout(None)   # ch.sock, not conn: encryption replaces the socket object
                _log(f"host connected from {peer[0]}")
                self._serve_one(ch)
                _log("host disconnected; waiting for it to return")
        except KeyboardInterrupt:
            _log("stopped by user")
        finally:
            server.close()

    def _serve_one(self, ch: Channel) -> None:
        try:
            while True:
                try:
                    header, tensor, blob = ch.recv()
                except (ConnectionError, OSError):
                    return
                except ProtocolError as e:
                    _log(f"dropping host: {e}")
                    return
                if header["op"] == "bye":
                    return
                try:
                    self.handle(ch, header, tensor, blob)
                except (ConnectionError, OSError):
                    return
                except Exception as e:  # report to the host, keep serving
                    traceback.print_exc(file=sys.stderr)
                    if self.stage is not None:
                        self.stage.drop_graph()
                    try:
                        ch.send({"op": "error", "message": f"{type(e).__name__}: {e}"})
                    except (ConnectionError, OSError):
                        return
        finally:
            if self.stage is not None:
                self.stage.drop_graph()
            ch.close()
