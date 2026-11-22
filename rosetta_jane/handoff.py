"""A fingerprint of every handoff between devices.

Each time numbers cross from one device to another, both ends write one line to
a local file: what kind of work it was, which layers, and a fingerprint of the
numbers. Nothing is sent anywhere. When something goes wrong in a run across
several machines, the files from each machine can be laid side by side with
`rosetta_jane compare-handoffs` to see where the numbers stopped matching:

  * the same data has the same `hash` on the sending and the receiving device,
    so a handoff that was damaged or never arrived shows up as a missing match;
  * two peers given the same input should return the same output; the `norm`
    and `top` fields show whether a difference is rounding or something else.

A fingerprint cannot be turned back into text: it is a one-way hash, one overall
size, and the eight largest values at one position. The file does hold the
network addresses of the peers this device talked to, so treat it as private.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
except ImportError:                               # a phone has no PyTorch
    torch = None

TOP = 8
MAX_BYTES = 64 << 20          # per file; one older file is kept


def fingerprint(payload: bytes, tensor) -> dict:
    out: dict = {"hash": hashlib.sha256(payload).hexdigest()[:24], "bytes": len(payload)}
    if isinstance(tensor, np.ndarray):            # the same fields, worked out without PyTorch
        if tensor.size and tensor.dtype.kind == "f":
            t = tensor.astype(np.float32)
            out["shape"], out["dtype"] = list(t.shape), tensor.dtype.name
            norm = float(np.sqrt(np.sum(t.astype(np.float64) ** 2)))
            out["norm"] = norm if np.isfinite(norm) else None
            last = t.reshape(-1, t.shape[-1])[-1] if t.ndim >= 1 else t.reshape(1)
            k = min(TOP, last.size)
            idx = np.argsort(-np.abs(last), kind="stable")[:k]
            out["top"] = [[int(i), round(float(last[i]), 5)] for i in idx]
        return out
    if tensor is not None and tensor.numel() and tensor.is_floating_point():
        t = tensor.detach().float()
        out["shape"] = list(t.shape)
        out["dtype"] = str(tensor.dtype).replace("torch.", "")
        norm = float(t.norm())
        out["norm"] = norm if norm == norm and norm != float("inf") else None
        last = t.reshape(-1, t.shape[-1])[-1] if t.dim() >= 1 else t.reshape(1)
        k = min(TOP, last.numel())
        idx = torch.topk(last.abs(), k).indices
        out["top"] = [[int(i), round(float(last[i]), 5)] for i in idx]
    return out


class HandoffLog:
    def __init__(self, path: str, node_id: str = ""):
        self.path, self.node_id = path, node_id
        self._lock = threading.Lock()
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")
        self.written = 0

    def _write(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            if self._file.closed:
                return
            if self._file.tell() + len(line) > MAX_BYTES:
                self._file.close()
                os.replace(self.path, self.path + ".1")
                self._file = open(self.path, "a", encoding="utf-8")
            self._file.write(line)
            self._file.flush()
            self.written += 1

    def tap(self, role: str, peer: str) -> Callable:
        """A recorder for one connection. role: 'asking' (this device dialed) or 'serving'."""
        link = secrets.token_hex(4)
        ctx: Dict[str, object] = {"layers": None, "n": 0}

        def record(direction: str, header: dict, tensor, payload: bytes) -> None:
            op = str(header.get("op", ""))
            if op in ("open", "t_open") and isinstance(header.get("start"), int) and isinstance(header.get("end"), int):
                ctx["layers"] = [header["start"], header["end"]]
            if not payload:
                return
            try:
                ctx["n"] = int(ctx["n"]) + 1
                self._write({"t": round(time.time(), 3), "node": self.node_id[:12], "role": role, "peer": peer,
                             "link": link, "n": ctx["n"], "dir": direction, "op": op, "layers": ctx["layers"],
                             **fingerprint(payload, tensor)})
            except Exception:                      # a log must never break the work it describes
                pass
        return record

    def close(self) -> None:
        with self._lock:
            self._file.close()


# -- reading the files back ----------------------------------------------------------------
def read_log(path: str) -> List[dict]:
    rows = []
    for name in (path + ".1", path):
        if not os.path.exists(name):
            continue
        with open(name, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and "hash" in row and row.get("dir") in ("sent", "received"):
                    row["file"] = os.path.basename(path)
                    rows.append(row)
    return rows


def gap(a: dict, b: dict) -> dict:
    """How far apart two fingerprints are, from what little they hold."""
    if a["hash"] == b["hash"]:
        return {"same": True, "norm_gap": 0.0, "top_shared": TOP}
    na, nb = a.get("norm"), b.get("norm")
    norm_gap = abs(na - nb) / max(abs(na), abs(nb), 1e-12) if na is not None and nb is not None else None
    ta, tb = dict(map(tuple, a.get("top", []))), dict(map(tuple, b.get("top", [])))
    shared = set(ta) & set(tb)
    value_gap = max((abs(ta[i] - tb[i]) / max(abs(ta[i]), abs(tb[i]), 1e-12) for i in shared), default=None)
    return {"same": False, "norm_gap": norm_gap, "top_shared": len(shared), "top_value_gap": value_gap}


def compare(paths: Iterable[str], tolerance: float = 0.02) -> dict:
    """Lay several devices' handoff files side by side."""
    logs = {p: read_log(p) for p in paths}
    rows = [r for rs in logs.values() for r in rs]
    received = {}
    for r in rows:
        if r["dir"] == "received":
            received.setdefault(r["hash"], []).append(r)
    nodes = {r["node"] for r in rows}
    delivered, unmatched = 0, []
    used: Dict[int, bool] = {}
    for r in rows:                                 # each arrival can account for one sending only
        if r["dir"] != "sent":
            continue
        match = next((o for o in received.get(r["hash"], []) if o["node"] != r["node"] and id(o) not in used), None)
        if match is not None:
            used[id(match)] = True
            delivered += 1
        else:
            unmatched.append(r)

    # What each serving peer returned for each piece of work, as the asking device saw it.
    work: Dict[tuple, List[dict]] = {}
    by_link: Dict[tuple, List[dict]] = {}
    for r in rows:
        if r["role"] == "asking":
            by_link.setdefault((r["node"], r["link"]), []).append(r)
    for link_rows in by_link.values():
        link_rows.sort(key=lambda r: r["n"])
        pending = None
        for r in link_rows:
            if r["dir"] == "sent":
                pending = r
            elif pending is not None and r["op"] == pending["op"] + "_ok":
                key = (pending["op"], tuple(pending["layers"] or ()), pending["hash"])
                work.setdefault(key, []).append({"peer": r["peer"], "out": r, "t": r["t"]})
                pending = None
    disagreements = []
    compared = 0
    for (op, layers, _), answers in work.items():
        peers = {a["peer"] for a in answers}
        if len(peers) < 2:
            continue
        compared += 1
        first = answers[0]
        for other in answers[1:]:
            if other["peer"] == first["peer"]:
                continue
            g = gap(first["out"], other["out"])
            if g["same"]:
                continue
            close = g["norm_gap"] is not None and g["norm_gap"] <= tolerance and g["top_shared"] >= TOP // 2 \
                and (g["top_value_gap"] is None or g["top_value_gap"] <= tolerance * 2.5)
            disagreements.append({"op": op, "layers": list(layers), "t": first["t"], "peers": [first["peer"], other["peer"]],
                                  "kind": "rounding" if close else "different", **g})
    return {"files": {p: len(rs) for p, rs in logs.items()}, "devices": len(nodes), "handoffs_sent": delivered + len(unmatched),
            "delivered_intact": delivered, "unmatched": unmatched, "work_compared": compared,
            "rounding_differences": sum(d["kind"] == "rounding" for d in disagreements),
            "real_differences": [d for d in disagreements if d["kind"] == "different"]}
