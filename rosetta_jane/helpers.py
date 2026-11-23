"""Sharing a finished adapter with the devices that helped train it.

Training through a swarm leaves the helpers with nothing: they compute what they are
sent and keep none of it, and their own copy of the model is never changed. That stays
true. What this adds is a way for the person who led the training to hand the finished
adapter to the people whose devices did the work:

  * While training, the trainer gives each helping device a ticket and counts the
    pieces of work it accepted from that device.
  * When the adapter is finished, the leader can open it for sharing. Their device
    then announces the training run to the group.
  * A device holding a ticket for that run shows its owner an offer: who trained it,
    how it scored, how much of the work was checked. The owner can take a copy or
    ignore it.
  * A copy is stored beside the owner's other adapters, to try, to score, to lead a
    team with, or to train further. Storing it applies nothing.

Only devices whose work was accepted in that run can take a copy this way, so nobody
can use it to push files at strangers or to collect adapters they had no part in.

There are no accounts here, so "a device that helped" means: a device that still holds
the ticket it was given. The ticket file is in the work folder; lose it and the offer
cannot be taken.

Teaching points
---------------
A device in a teaching team gives its power to training and does not query, so teaching is
rewarded. The leader's device counts the time each helper spent on work that was accepted:
one point a minute. In a team that same leader runs for querying, a member can spend points
on a larger share of the team's computing power: each point is 1% more, for an hour (see
fair.py for how devices share their time out).

This file counts the points (RunBook) and keeps a helper's tickets (HelpLog). Turning them
into signed receipts that are good in any team is points.py.

Plain Python and NumPy, shared by the desktop program and the phone app.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from .lan.protocol import ProtocolError, RemoteError
from .mobile import st
from .mobile.peer import check_spec
from .team import FACTS, MAX_ADAPTER_BYTES, Refused, _send, _text, clean_facts, plain

BOOK = "rosetta-jane-helpers.json"        # the trainer's record of who helped; kept private
_RUN = re.compile(r"^[0-9a-f]{16}$")
_TICKET = re.compile(r"^[0-9a-f]{32}$")
_NODE = re.compile(r"^[0-9a-f]{8,64}$")
_NET = (ConnectionError, OSError, ProtocolError, RemoteError)
MAX_HELPERS = 5000
MAX_REMEMBERED = 200
SECONDS_PER_POINT = 60.0           # a minute of accepted teaching work earns one point
BOOST_SECONDS = 3600.0             # a spent point lasts an hour
MAX_BOOST = 100                    # points in force for one member at once: at most double a plain share
LEDGER = "teaching-points.json"


def _key(raw) -> str:
    return raw if isinstance(raw, str) and re.match(r"^[0-9a-f]{64}$", raw) else ""


def ticket(secret: bytes, node_id: str) -> str:
    return hmac.new(secret, b"helper:" + node_id.encode(), hashlib.sha256).hexdigest()[:32]


def _write(path: str, data: dict) -> None:
    tmp = path + ".partial"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class RunBook:
    """The trainer's record of one training run: its tag, its secret, and whose work was accepted."""

    def __init__(self, folder: str, create: bool = True):
        self.path = os.path.join(folder, BOOK)
        self.run = self.secret = None
        self.helpers: Dict[str, int] = {}
        self.seconds: Dict[str, float] = {}       # time each helper spent on work that was accepted
        try:
            with open(self.path) as f:
                data = json.load(f)
            run, secret, helpers = data["run"], bytes.fromhex(data["secret"]), data["helpers"]
            if _RUN.match(run) and len(secret) == 32 and isinstance(helpers, dict):
                self.run, self.secret = run, secret
                self.helpers = {k: int(v) for k, v in helpers.items()
                                if isinstance(k, str) and _NODE.match(k) and isinstance(v, int) and v > 0}
                timed = data.get("seconds") if isinstance(data.get("seconds"), dict) else {}
                self.seconds = {k: float(v) for k, v in timed.items()
                                if k in self.helpers and isinstance(v, (int, float)) and 0 <= v < 1e9}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if self.run is None:
            if not create:
                raise ValueError("this adapter has no record of who helped train it. It was trained before "
                                 "helpers were recorded, or it was received from someone else.")
            self.run, self.secret = secrets.token_hex(8), secrets.token_bytes(32)
            self.save()

    def ticket(self, node_id: str) -> str:
        return ticket(self.secret, node_id)

    def credit(self, node_ids: Iterable[str], seconds: float = 0.0) -> None:
        """Count one accepted piece of work for each device, and the time it took them."""
        for node_id in node_ids:
            if node_id in self.helpers or len(self.helpers) < MAX_HELPERS:
                self.helpers[node_id] = self.helpers.get(node_id, 0) + 1
                self.seconds[node_id] = self.seconds.get(node_id, 0.0) + max(0.0, seconds)

    def points(self, node_id: str) -> int:
        """Teaching points this run earned a device: one for each minute of its accepted work."""
        return int(self.seconds.get(node_id, 0.0) // SECONDS_PER_POINT)

    def save(self) -> None:
        _write(self.path, {"note": "Keep this file private. It lets the devices that helped train this adapter "
                                   "take a copy when you choose to share it with them.",
                           "run": self.run, "secret": self.secret.hex(), "helpers": self.helpers,
                           "seconds": dict(self.seconds)})      # exact, so the tally on disk and the one in memory agree

    def copy_to(self, folder: str) -> None:
        other = RunBook.__new__(RunBook)
        other.path, other.run, other.secret, other.helpers = os.path.join(folder, BOOK), self.run, self.secret, \
            dict(self.helpers)
        other.seconds = dict(self.seconds)
        other.save()


class HelpLog:
    """What a helping device remembers of the training runs it worked on: enough to be offered the result."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.runs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._unsaved = 0
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                for run, e in data.items() if isinstance(data, dict) else []:
                    if _RUN.match(run) and isinstance(e, dict) and _TICKET.match(str(e.get("ticket"))) \
                            and _NODE.match(str(e.get("as"))):
                        self.runs[run] = {"ticket": e["ticket"], "as": e["as"], "by": _key(e.get("by")),
                                          "examples": int(e.get("examples", 0)), "last": float(e.get("last", 0))}
            except (OSError, ValueError, TypeError):
                pass

    def note(self, run, given, node_id: str, by=None) -> Optional[str]:
        """A trainer has opened a training session here and handed over a ticket. Returns the run, if usable.
        `by` is the key the trainer says is theirs: where to ask for a receipt later."""
        if not (isinstance(run, str) and _RUN.match(run) and isinstance(given, str) and _TICKET.match(given)):
            return None
        with self._lock:
            entry = self.runs.get(run)
            if entry is None or entry["examples"] == 0:      # a ticket that has done work is the one to keep
                self.runs[run] = {"ticket": given, "as": node_id, "by": _key(by), "examples": 0,
                                  "last": time.time()}
            elif not entry.get("by") and _key(by):
                entry["by"] = _key(by)
            while len(self.runs) > MAX_REMEMBERED:
                # forget runs this device never finished any work for first, then the oldest
                del self.runs[min(self.runs, key=lambda r: (self.runs[r]["examples"] > 0, self.runs[r]["last"]))]
        return run

    def count(self, run: Optional[str]) -> None:
        if not run:
            return
        with self._lock:
            entry = self.runs.get(run)
            if entry is None:
                return
            entry["examples"] += 1
            entry["last"] = time.time()
            self._unsaved += 1
            due = entry["examples"] == 1 or self._unsaved >= 25
        if due:
            self.save()

    def tickets(self) -> List[dict]:
        """The tickets for runs this device did work in: what it shows a leader to use its teaching points."""
        with self._lock:
            return [{"run": run, "as": e["as"], "ticket": e["ticket"], "by": e.get("by", "")}
                    for run, e in self.runs.items() if e["examples"] > 0]

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = {run: dict(e) for run, e in self.runs.items() if e["examples"] > 0}
            self._unsaved = 0
        try:
            _write(self.path, data)
        except OSError:
            pass


class Ledger:
    """What a leader's training runs earned each helper: read from the run books in this work folder.
    Receipts for it, and the tally of what has been spent, are points.Bank's."""

    def __init__(self, work_dir: Optional[str]):
        self.work_dir = work_dir
        self._lock = threading.Lock()

    def _books(self) -> Dict[str, RunBook]:
        found: Dict[str, RunBook] = {}
        if not self.work_dir or not os.path.isdir(self.work_dir):
            return found
        places = [os.path.join(self.work_dir, n) for n in os.listdir(self.work_dir) if n.startswith("train-")]
        adapters = os.path.join(self.work_dir, "adapters")
        if os.path.isdir(adapters):
            places += [os.path.join(adapters, n) for n in os.listdir(adapters)]
        for folder in places:
            try:
                book = RunBook(folder, create=False)
            except ValueError:
                continue
            kept = found.get(book.run)              # the same run in two places: the fuller record counts
            if kept is None or sum(book.seconds.values()) > sum(kept.seconds.values()):
                found[book.run] = book
        return found

    def _valid(self, tickets) -> List[tuple]:
        books, out, seen = self._books(), [], set()
        for t in tickets if isinstance(tickets, list) else []:
            if not isinstance(t, dict):
                continue
            run, who, given = t.get("run"), t.get("as"), t.get("ticket")
            book = books.get(run) if isinstance(run, str) else None
            if book is None or not (isinstance(who, str) and _NODE.match(who) and isinstance(given, str)) \
                    or (run, who) in seen or not hmac.compare_digest(given.encode(), book.ticket(who).encode()):
                continue
            seen.add((run, who))
            out.append((f"{run}:{who}", book.points(who)))
        return out


def _config(folder: str) -> dict:
    with open(os.path.join(folder, "adapter_config.json")) as f:
        config = json.load(f)
    rank, alpha, targets = check_spec(config["r"], config["lora_alpha"], config["target_modules"])
    return {"r": rank, "lora_alpha": alpha, "target_modules": targets}


class Sharing:
    """Both ends of sharing with helpers. One per node."""

    def __init__(self, node, log: Optional[HelpLog] = None):
        self.node, self.log = node, log
        self._open: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def attach(self) -> "Sharing":
        self.node.handlers["gift_look"] = self._look
        self.node.handlers["gift_take"] = self._take
        self.node.notices["gifts"] = lambda: sorted(self._open)
        return self

    # -- the leader's side ---------------------------------------------------------------------------
    def open(self, adapter_dir: str, model: str) -> dict:
        """Start offering a finished adapter to the devices that helped train it."""
        book = RunBook(adapter_dir, create=False)
        if not book.helpers:
            raise ValueError("no other device's work went into this adapter, so there is nobody to share it "
                             "with this way. Share it as a file instead.")
        try:
            config = _config(adapter_dir)
            size = os.path.getsize(os.path.join(adapter_dir, "adapter_model.safetensors"))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"that folder does not hold a readable adapter: {e}") from None
        if size > MAX_ADAPTER_BYTES:
            raise ValueError("that adapter is too large to share this way")
        facts = {}
        try:
            with open(os.path.join(adapter_dir, FACTS)) as f:
                facts = json.load(f)
        except (OSError, ValueError):
            pass
        made_for = facts.get("base_model_fingerprint") if isinstance(facts, dict) else None
        entry = {"run": book.run, "book": book, "dir": adapter_dir, "config": config, "size": size,
                 "name": _text(os.path.basename(os.path.normpath(adapter_dir)), 60) or "adapter",
                 "model": made_for if isinstance(made_for, str) and _NODE.match(made_for) else model,
                 "facts": clean_facts(facts), "taken": set()}
        with self._lock:
            self._open[book.run] = entry
        return self._row(entry)

    def close(self, run: str) -> None:
        with self._lock:
            self._open.pop(run, None)

    @staticmethod
    def _row(e: dict) -> dict:
        return {"run": e["run"], "name": e["name"], "helpers": len(e["book"].helpers), "taken": len(e["taken"])}

    def report(self) -> List[dict]:
        with self._lock:
            return [self._row(e) for e in self._open.values()]

    def _entry(self, header: dict) -> dict:
        with self._lock:
            entry = self._open.get(header.get("run") if isinstance(header.get("run"), str) else "")
        if entry is None:
            raise KeyError("that adapter is not being shared from this device right now")
        return entry

    def _describe(self, e: dict) -> dict:
        return {"run": e["run"], "name": e["name"], "model": e["model"], "config": e["config"],
                "facts": e["facts"], "size": e["size"]}

    def _look(self, ch, header: dict, tensor, state: dict) -> None:
        ch.send({"op": "gift_look_ok", **self._describe(self._entry(header))})

    def _take(self, ch, header: dict, tensor, state: dict) -> None:
        e = self._entry(header)
        who, given = header.get("as"), header.get("ticket")
        if not (isinstance(who, str) and _NODE.match(who) and isinstance(given, str)
                and hmac.compare_digest(given.encode(), e["book"].ticket(who).encode())):
            raise Refused("that ticket is not from this training run")
        helped = e["book"].helpers.get(who, 0)
        if helped < 1:
            raise Refused("no finished work from that device went into this adapter")
        with open(os.path.join(e["dir"], "adapter_model.safetensors"), "rb") as f:
            blob = f.read()
        with self._lock:
            e["taken"].add(who)
        ch.send({"op": "gift_take_ok", **self._describe(e), "helped": helped}, blob=blob)

    # -- a helper's side -------------------------------------------------------------------------------
    def offers(self) -> List[dict]:
        """Finished adapters this device helped train that their leaders are sharing right now."""
        if self.log is None:
            return []
        out, seen = [], set()
        for peer in self.node.table.peers():
            if peer.node_id in self.node.banned or peer.addr in self.node.banned:
                continue
            for run in peer.gifts:
                entry = self.log.runs.get(run)
                if entry and entry["examples"] > 0 and run not in seen:
                    seen.add(run)
                    out.append({"run": run, "from": peer.addr, "examples": entry["examples"]})
        return out

    def _where(self, run: str) -> dict:
        for offer in self.offers():
            if offer["run"] == run:
                return offer
        raise KeyError("no adapter from a run this device helped with is being shared under that name right now")

    @staticmethod
    def _checked(reply: dict, run: str) -> dict:
        if reply.get("run") != run or not isinstance(reply.get("model"), str) or not _NODE.match(reply["model"]):
            raise ProtocolError("the reply could not be read")
        try:
            c = reply.get("config") or {}
            rank, alpha, targets = check_spec(c.get("r"), c.get("lora_alpha"), c.get("target_modules"))
        except (ValueError, AttributeError):
            raise ProtocolError("the adapter's settings could not be read") from None
        size = reply.get("size")
        return {"run": run, "name": _text(reply.get("name"), 60) or "adapter", "model": reply["model"],
                "config": {"r": rank, "lora_alpha": alpha, "target_modules": targets},
                "facts": clean_facts(reply.get("facts")),
                "size": size if isinstance(size, int) and not isinstance(size, bool) else 0}

    def look(self, run: str) -> dict:
        """What the leader says about the adapter on offer. Looking takes nothing."""
        offer = self._where(run)
        try:
            reply, _, _ = _send(self.node, offer["from"], {"op": "gift_look", "run": run})
        except RemoteError as e:
            raise KeyError(plain(e)) from None
        return {**self._checked(reply, run), "adapter": "shared with helpers", "examples": offer["examples"]}

    def take(self, run: str, work_dir: str, model: str, name: Optional[str] = None) -> str:
        """Take a copy of an adapter this device helped train. It is stored; nothing is applied."""
        from .control import keep_received
        offer = self._where(run)
        entry = self.log.runs[run]
        ch = self.node.dial(offer["from"], timeout=30.0)
        try:
            ch.sock.settimeout(600.0)
            reply, _, blob = ch.call({"op": "gift_take", "run": run, "as": entry["as"], "ticket": entry["ticket"]})
            try:
                ch.send({"op": "bye"})
            except _NET:
                pass
        except RemoteError as e:
            raise PermissionError(plain(e)) from None
        finally:
            ch.close()
        got = self._checked(reply, run)
        if got["model"] != model:
            raise ValueError(f"that adapter is for a different model ({got['model']}); the model here is {model}")
        if not blob or len(blob) > MAX_ADAPTER_BYTES:
            raise ProtocolError("the adapter did not arrive whole")
        try:
            weights = st.loads(blob)
        except st.FormatError as e:
            raise ProtocolError(f"the adapter could not be read: {e}") from None
        if not weights or any(v.dtype.kind != "f" or not np.isfinite(v).all() for v in weights.values()):
            raise ProtocolError("the adapter holds something other than numbers")
        facts = {**got["facts"], "base_model_fingerprint": got["model"], "received": True,
                 "received_on": time.strftime("%Y-%m-%d %H:%M:%S"), "received_for": "helping to train it",
                 "helped_examples": entry["examples"], "run": run}
        return keep_received(work_dir, name or got["name"], got["config"], blob, facts)
