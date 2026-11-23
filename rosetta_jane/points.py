"""Teaching points that are good in any team.

A device in a teaching team gives its power to training and asks no questions, so teaching
is rewarded with points: one for each minute of accepted work (helpers.py counts them).
In a team that queries the model, a member can spend points on a larger share of the
team's computing power: each point is 1% more, for an hour (fair.py shares the time out).

There is no server and there are no accounts, so there is no bank everyone trusts. This
is what stands in for one:

  * Every device has a key of its own (identity.py).
  * The leader whose teaching earned the points signs a RECEIPT with their key: this many
    points, earned in this training run, belong to that device's key. The device that
    earned them keeps the receipt. Anyone can check the signature without asking anybody.
  * To spend, the device signs a SPEND with its own key: this many points from that
    receipt, in this team, now. Only the device the receipt was made out to can do that.
  * The leader who signed a receipt is the bank for it. When points are spent in someone
    else's team, that team's leader asks the issuing leader's device, if it is in the
    group right now, and it keeps the tally exact: the same points cannot be spent twice.
  * If the issuing leader is not around, the team's leader can still take the points on
    the strength of the signatures (their choice), and passes the spend on when the
    issuer is next seen.

What this cannot do, and does not pretend to:

  * Spending twice. While the issuing leader is away, the same points could be spent in
    two teams that both take them unconfirmed. The issuer's tally shows it afterwards.
  * Made-up points. A leader can sign points nobody earned, for themselves or a friend.
    So each leader chooses whose points their teams honour: by default, only points issued
    by leaders whose teams this device has itself signed on to. That takes a choice by
    this device's owner; a stranger cannot arrange it.

The stakes are kept low on purpose: points buy a larger share of one team's computing
power for an hour, never more than double, among people who chose to team up. They are
not money and cannot be traded.

Plain Python, shared by the desktop program and the phone app.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from typing import Dict, List, Optional

from . import identity as ident
from .helpers import LEDGER, Ledger, _write
from .lan.protocol import ProtocolError, RemoteError
from .team import _send, plain

RECEIPT_DAYS = 90                  # a receipt can be spent for this long after it was issued
WALLET = "teaching-receipts.json"
MAX_RECEIPTS = 500
MAX_PAIRS = 16                     # receipts drawn on in one spend
MAX_PENDING = 500
_RUN = re.compile(r"^[0-9a-f]{16}$")
_NODE = re.compile(r"^[0-9a-f]{8,64}$")
_NONCE = re.compile(r"^[0-9a-f]{16}$")
_TEAM = re.compile(r"^[0-9a-f]{16}$")
_NET = (ConnectionError, OSError, ProtocolError, RemoteError)


def _whole(value, low: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= low


def receipt_id(by: str, run: str, who: str) -> str:
    return hashlib.sha256(f"{by}:{run}:{who}".encode()).hexdigest()[:32]


def make_receipt(me: ident.Identity, run: str, who: str, to: str, points: int, name: str = "") -> dict:
    issued = int(time.time())
    name = name[:24]
    return {"v": 1, "run": run, "as": who, "to": to, "points": points, "issued": issued, "by": me.key,
            "name": name, "sig": me.sign("receipt", run, who, to, points, issued, name)}


def check_receipt(raw, now: Optional[float] = None) -> dict:
    """A receipt as received, checked: well formed, signed by the key it names, and not out of date."""
    if not isinstance(raw, dict):
        raise ValueError("a receipt could not be read")
    run, who, to, by = raw.get("run"), raw.get("as"), ident.clean_key(raw.get("to")), ident.clean_key(raw.get("by"))
    points, issued, name = raw.get("points"), raw.get("issued"), raw.get("name", "")
    if not (isinstance(run, str) and _RUN.match(run) and isinstance(who, str) and _NODE.match(who) and to and by
            and _whole(points, 1) and points < 10 ** 7 and _whole(issued) and isinstance(name, str)
            and len(name) <= 24):
        raise ValueError("a receipt could not be read")
    if not ident.verify(by, raw.get("sig"), "receipt", run, who, to, points, issued, name):
        raise ValueError("a receipt's signature is not from the leader it names")
    now = time.time() if now is None else now
    if issued > now + ident.FRESH:
        raise ValueError("a receipt is dated in the future")
    if now - issued > RECEIPT_DAYS * 86400:
        raise ValueError(f"a receipt is more than {RECEIPT_DAYS} days old")
    return {"v": 1, "run": run, "as": who, "to": to, "points": points, "issued": issued, "by": by, "name": name,
            "sig": raw["sig"], "id": receipt_id(by, run, who)}


def make_spend(me: ident.Identity, receipt: dict, amount: int, team: str) -> dict:
    rid, at, n = receipt_id(receipt["by"], receipt["run"], receipt["as"]), int(time.time()), secrets.token_hex(8)
    return {"v": 1, "receipt": rid, "amount": amount, "team": team, "at": at, "n": n,
            "sig": me.sign("spend", rid, amount, team, at, n)}


def check_spend(raw, receipt: dict) -> dict:
    """A spend as received, checked against its receipt: signed by the device the receipt was made out to."""
    if not isinstance(raw, dict):
        raise ValueError("a spend could not be read")
    amount, team, at, n = raw.get("amount"), raw.get("team"), raw.get("at"), raw.get("n")
    if raw.get("receipt") != receipt["id"] or not (_whole(amount, 1) and isinstance(team, str) and _TEAM.match(team)
                                                   and _whole(at) and isinstance(n, str) and _NONCE.match(n)):
        raise ValueError("a spend could not be read")
    if not ident.verify(receipt["to"], raw.get("sig"), "spend", receipt["id"], amount, team, at, n):
        raise ValueError("a spend is not signed by the device its points belong to")
    return {"v": 1, "receipt": receipt["id"], "amount": amount, "team": team, "at": at, "n": n, "sig": raw["sig"]}


def _load(path: Optional[str]) -> dict:
    try:
        with open(path or "") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class Wallet:
    """The receipts this device holds for its own teaching, and what it has spent from each."""

    def __init__(self, work_dir: Optional[str], me: ident.Identity):
        self.path = os.path.join(work_dir, WALLET) if work_dir else None
        self.me = me
        self._lock = threading.Lock()
        data = _load(self.path)
        self.receipts: Dict[str, dict] = {}
        self.spent: Dict[str, int] = {}
        for raw in data.get("receipts", []) if isinstance(data.get("receipts"), list) else []:
            try:
                r = check_receipt(raw)
            except ValueError:
                continue
            if r["to"] == me.key:
                self.receipts[r["id"]] = r
        for rid, n in (data.get("spent") or {}).items() if isinstance(data.get("spent"), dict) else []:
            if rid in self.receipts and _whole(n):
                self.spent[rid] = n

    def _save(self) -> None:
        if not self.path:
            return
        try:
            _write(self.path, {"note": "Receipts for teaching points this device earned. They can only be spent "
                                       "with this device's own key (identity.json).",
                               "receipts": [{k: v for k, v in r.items() if k != "id"}
                                            for r in self.receipts.values()],
                               "spent": self.spent})
        except OSError:
            pass

    def add(self, raw, left: Optional[int] = None) -> bool:
        """Keep a receipt a leader sent, if it is good and made out to this device. `left` is the leader's
        own count of what remains on it, when they said."""
        try:
            r = check_receipt(raw)
        except ValueError:
            return False
        if r["to"] != self.me.key:
            return False
        with self._lock:
            old = self.receipts.get(r["id"])
            if old is None or (r["points"], r["issued"]) >= (old["points"], old["issued"]):
                self.receipts[r["id"]] = r
            if _whole(left) and left <= r["points"]:
                self.spent[r["id"]] = self.receipts[r["id"]]["points"] - left
            while len(self.receipts) > MAX_RECEIPTS:
                gone = min(self.receipts, key=lambda k: self.receipts[k]["issued"])
                del self.receipts[gone]
                self.spent.pop(gone, None)
            self._save()
        return True

    def rows(self) -> List[dict]:
        """Receipts with points left, soonest to run out first."""
        now, out = time.time(), []
        with self._lock:
            for rid, r in self.receipts.items():
                days = RECEIPT_DAYS - (now - r["issued"]) / 86400
                left = r["points"] - self.spent.get(rid, 0)
                if left > 0 and days > 0:
                    out.append({"id": rid, "by": r["by"], "from": ident.shown(r["name"], r["by"]), "left": left,
                                "days": int(days)})
        return sorted(out, key=lambda row: row["days"])

    def balance(self) -> int:
        return sum(row["left"] for row in self.rows())

    def spends(self, points: int, team: str, prefer: str = "") -> List[dict]:
        """Signed spends adding up to `points` (or to everything held, if that is less): the team leader's own
        receipts first, then the ones that run out soonest."""
        rows = sorted(self.rows(), key=lambda row: (row["by"] != prefer, row["days"]))
        out, need = [], points
        for row in rows[:MAX_PAIRS]:
            if need <= 0:
                break
            take = min(need, row["left"])
            with self._lock:
                receipt = self.receipts[row["id"]]
            out.append({"receipt": {k: v for k, v in receipt.items() if k != "id"},
                        "spend": make_spend(self.me, receipt, take, team)})
            need -= take
        return out

    def took(self, taken) -> None:
        """A team's leader took these spends: count them off."""
        with self._lock:
            for t in taken if isinstance(taken, list) else []:
                rid, amount = (t.get("receipt"), t.get("amount")) if isinstance(t, dict) else (None, None)
                if rid in self.receipts and _whole(amount, 1):
                    self.spent[rid] = min(self.receipts[rid]["points"], self.spent.get(rid, 0) + amount)
            self._save()

    def claim(self, node, helped, also: Optional[str] = None, timeout: float = 15.0) -> int:
        """Fetch receipts for the teaching this device did, from each leader who is in the group right now.
        `also` is one more address to ask (the leader of the team this device is in). Returns how many came."""
        tickets = helped.tickets() if helped is not None else []
        if not tickets:
            return 0
        def reachable():
            return [p for p in node.table.peers() if p.node_id not in node.banned and p.addr not in node.banned]
        peers = reachable()
        if any(t.get("by") and not any(p.key == t["by"] for p in peers) for t in tickets):
            node.refresh(1)                        # a leader this device has not heard from lately: look again
            peers = reachable()
        asks: Dict[str, list] = {}
        for t in tickets:
            # Every device showing the leader's key is asked (a leader may run more than one program), and
            # the team's own leader is asked about tickets that do not say whose they are.
            where = [p.addr for p in peers if t.get("by") and p.key == t["by"]]
            if also and also not in where and (not t.get("by") or not where):
                where.append(also)
            for addr in where:
                asks.setdefault(addr, []).append({"run": t["run"], "as": t["as"], "ticket": t["ticket"]})
        got = 0
        for addr, mine in asks.items():
            try:
                reply, _, _ = _send(node, addr, {"op": "points_claim", "tickets": mine[:64],
                                                 "id": self.me.here("claim")}, timeout=timeout)
            except _NET:
                continue
            for item in reply.get("receipts", []) if isinstance(reply.get("receipts"), list) else []:
                if isinstance(item, dict) and self.add(item.get("receipt"), item.get("left")):
                    got += 1
        return got


class Bank:
    """A leader's side of teaching points: issuing receipts for teaching done in this work folder, keeping
    the tally for them, and deciding which other leaders' receipts this device's teams take."""

    def __init__(self, node, work_dir: Optional[str], me: ident.Identity, known: Optional[ident.Known] = None,
                 name_fn=None):
        self.node, self.work_dir, self.me, self.known = node, work_dir, me, known
        self.name_fn = name_fn or (lambda: "")
        self.ledger = Ledger(work_dir)
        self.path = os.path.join(work_dir, LEDGER) if work_dir else None
        self.honours, self.unconfirmed = "teamed", True
        self._lock = threading.Lock()
        self._memory: dict = {}                    # the tally when there is no work folder to keep it in

    # -- the tally --------------------------------------------------------------------------------
    def _state(self) -> dict:
        data = _load(self.path) if self.path else self._memory
        out = {"claims": {}, "used": {}, "seen": {}, "here": {}, "pending": []}
        for key in ("claims", "used", "seen", "here"):
            if isinstance(data.get(key), dict):
                out[key] = data[key]
        if isinstance(data.get("pending"), list):
            out["pending"] = [p for p in data["pending"] if isinstance(p, dict)][:MAX_PENDING]
        return out

    def _keep(self, state: dict) -> None:
        if self.path:
            try:
                _write(self.path, {"note": "This leader's tally of teaching points: who claimed which, and what "
                                           "has been spent.", **state})
            except OSError:
                pass
        else:
            self._memory = state

    def attach(self) -> "Bank":
        self.node.handlers["points_claim"] = self._claim
        self.node.handlers["points_redeem"] = self._redeem
        threading.Thread(target=self._tend, daemon=True).start()
        return self

    def _tend(self) -> None:
        while not self.node._stop.wait(20.0):
            try:
                self.flush()
            except Exception:
                pass

    # -- as the leader who issued the points --------------------------------------------------------
    def _earned(self, run: str, who: str) -> int:
        book = self.ledger._books().get(run)
        return book.points(who) if book is not None else 0

    def issue(self, tickets, key: str) -> List[dict]:
        """Receipts for the holder of these tickets, made out to their key. A ticket goes with the first key
        that claims it, so a copied ticket file gets nobody else any points."""
        out = []
        with self._lock:
            state = self._state()
            for rkey, earned in self.ledger._valid(tickets):
                if earned < 1:
                    continue
                if state["claims"].setdefault(rkey, key) != key:
                    continue
                run, who = rkey.split(":", 1)
                rid = receipt_id(self.me.key, run, who)
                used = sum(v for v in state["used"].get(rid, {}).values() if _whole(v))
                out.append({"receipt": make_receipt(self.me, run, who, key, earned, self.name_fn()),
                            "left": max(0, earned - used)})
            self._keep(state)
        return out

    def redeem(self, receipt: dict, spend: dict) -> dict:
        """Take a spend off one of this leader's own receipts. {"ok", "left", "why"}. Asking twice about the
        same spend takes it off once."""
        if receipt["by"] != self.me.key:
            return {"ok": False, "left": 0, "why": "that receipt was not issued by this device"}
        rkey = f"{receipt['run']}:{receipt['as']}"
        with self._lock:
            state = self._state()
            if state["claims"].get(rkey) != receipt["to"]:
                return {"ok": False, "left": 0, "why": "those points were claimed by a different device"}
            used = state["used"].setdefault(receipt["id"], {})
            spent = sum(v for v in used.values() if _whole(v))
            left = max(0, self._earned(receipt["run"], receipt["as"]) - spent)
            if spend["n"] in used:
                return {"ok": True, "left": left, "why": ""}
            if spend["amount"] > left:
                return {"ok": False, "left": left, "why": f"only {left} of those points are left"}
            used[spend["n"]] = spend["amount"]
            self._keep(state)
            return {"ok": True, "left": left - spend["amount"], "why": ""}

    def _claim(self, ch, header: dict, tensor, state: dict) -> None:
        key = ident.proven(header.get("id"), "claim")
        if not key:
            raise ValueError("show this device's key to claim its teaching points")
        ch.send({"op": "points_claim_ok", "receipts": self.issue(header.get("tickets"), key)})

    def _redeem(self, ch, header: dict, tensor, state: dict) -> None:
        receipt = check_receipt(header.get("receipt"), now=None)
        spend = check_spend(header.get("spend"), receipt)
        got = self.redeem(receipt, spend)
        ch.send({"op": "points_redeem_ok", **got,
                 "sig": self.me.sign("redeemed", receipt["id"], spend["n"], spend["amount"], got["ok"], got["left"])})

    # -- as the leader of a team where points are being spent ------------------------------------------
    def honoured(self, issuer: str) -> Optional[str]:
        """None if this device's teams take points that key issued; otherwise why not."""
        if issuer == self.me.key or self.honours == "anyone":
            return None
        if self.honours == "mine":
            return "this team takes only points earned with its own leader"
        if self.known is not None and self.known.led(issuer):
            return None
        return ("this team's leader has not signed on to a team led by the device that issued those points, "
                "and takes points only from leaders they have")

    def _ask_issuer(self, receipt: dict, spend: dict) -> Optional[dict]:
        """Ask the leader who issued a receipt to take a spend off it. None if they cannot be reached."""
        for peer in self.node.table.peers():
            if peer.key != receipt["by"] or peer.node_id in self.node.banned or peer.addr in self.node.banned:
                continue
            try:
                reply, _, _ = _send(self.node, peer.addr, {
                    "op": "points_redeem", "receipt": {k: v for k, v in receipt.items() if k != "id"},
                    "spend": spend}, timeout=10.0)
            except RemoteError as e:
                if "unknown request" in str(e):    # a program of theirs that keeps no tally: try the next
                    continue
                return {"ok": False, "left": 0, "why": plain(e)}
            except _NET:
                continue
            ok, left = reply.get("ok") is True, reply.get("left") if _whole(reply.get("left")) else 0
            if not ident.verify(receipt["by"], reply.get("sig"), "redeemed", receipt["id"], spend["n"],
                                spend["amount"], ok, left):
                continue                           # not the issuer's answer: treat as not reached
            return {"ok": ok, "left": left, "why": reply.get("why") if isinstance(reply.get("why"), str) else ""}
        return None

    def accept(self, member_key: str, pairs, team: str) -> dict:
        """A member of this device's team spends points. Returns {"points": taken in all, "taken": [...],
        "refused": [why, ...]}. Each taken entry says whether the issuing leader confirmed it."""
        taken, refused = [], []
        if not ident.clean_key(member_key):
            return {"points": 0, "taken": [], "refused": ["this device showed no key, so it has no points to spend"]}
        now = time.time()
        for pair in (pairs if isinstance(pairs, list) else [])[:MAX_PAIRS]:
            try:
                if not isinstance(pair, dict):
                    raise ValueError("a spend could not be read")
                receipt = check_receipt(pair.get("receipt"), now)
                if receipt["to"] != member_key:
                    raise ValueError("those points belong to a different device")
                spend = check_spend(pair.get("spend"), receipt)
                if spend["team"] != team or abs(now - spend["at"]) > ident.FRESH:
                    raise ValueError("that spend was made out for another team or another time")
                why = self.honoured(receipt["by"])
                if why:
                    raise ValueError(why)
            except ValueError as e:
                refused.append(str(e))
                continue
            rid = receipt["id"]
            with self._lock:
                state = self._state()
                if spend["n"] in state["here"].get(rid, []):
                    refused.append("that spend was already used")
                    continue
            if receipt["by"] == self.me.key:
                got, confirmed = self.redeem(receipt, spend), True
            else:
                got, confirmed = self._ask_issuer(receipt, spend), True
                if got is None:
                    confirmed = False
                    with self._lock:
                        seen = self._state()["seen"].get(rid, 0)
                    if not self.unconfirmed:
                        got = {"ok": False, "why": "the leader who issued those points is not in the group right "
                                                   "now, and this team takes only points they confirm"}
                    elif seen + spend["amount"] > receipt["points"]:
                        got = {"ok": False, "why": "more has been spent from that receipt here than it holds"}
                    else:
                        got = {"ok": True, "why": ""}
            if not got["ok"]:
                refused.append(got["why"] or "the leader who issued those points refused the spend")
                continue
            with self._lock:
                state = self._state()
                state["here"].setdefault(rid, []).append(spend["n"])
                if not confirmed:
                    state["seen"][rid] = state["seen"].get(rid, 0) + spend["amount"]
                    state["pending"] = (state["pending"] + [{
                        "receipt": {k: v for k, v in receipt.items() if k != "id"}, "spend": spend}])[-MAX_PENDING:]
                self._keep(state)
            taken.append({"receipt": rid, "n": spend["n"], "amount": spend["amount"], "confirmed": confirmed,
                          "from": ident.shown(receipt["name"], receipt["by"])})
        return {"points": sum(t["amount"] for t in taken), "taken": taken, "refused": refused}

    def flush(self) -> int:
        """Pass on spends that were taken while their issuing leader was away, to any who are back."""
        with self._lock:
            pending = self._state()["pending"]
        if not pending:
            return 0
        keys = {p.key for p in self.node.table.peers() if p.key}
        settled = []
        for item in pending:
            try:
                receipt = check_receipt(item.get("receipt"), now=None)
                spend = check_spend(item.get("spend"), receipt)
            except ValueError:
                settled.append(item)               # out of date by now: nothing more to do with it
                continue
            if receipt["by"] in keys and self._ask_issuer(receipt, spend) is not None:
                settled.append(item)
        if settled:
            with self._lock:
                state = self._state()
                state["pending"] = [p for p in state["pending"] if p not in settled]
                self._keep(state)
        return len(settled)

    def pending(self) -> int:
        with self._lock:
            return len(self._state()["pending"])
