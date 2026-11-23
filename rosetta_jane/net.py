"""The peer network. Every copy of Rosetta is the same program: there is no server.

How peers find each other:
  * You start a node knowing at least one other peer's address (or none, on a
    local network, where nodes announce themselves).
  * Every so often each node dials a few peers it knows and they swap lists.
    Within a few rounds everybody knows everybody. This is the same idea
    LimeWire and BitTorrent peer exchange used.
  * A node only passes on addresses it has itself connected to, so nobody can
    flood the network with made-up peers through one honest node.

Who can join: nodes sharing a group code form a private network. The code
"public" is the open network anyone can join. Connections are encrypted with a
key made from the code (see lan/protocol.py).

Identity: each run picks a fresh random node ID. There are no accounts and no
names. Peers you connect to do see your IP address.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import re
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .lan.protocol import (AuthError, Channel, ProtocolError, RemoteError, load_secret,
                              secret_from_group_code)

from . import __version__

DEFAULT_PORT = 7700
BEACON_PORT = 47700
_BEACON_MAGIC = b"RSTA1"
_ADDR_RE = re.compile(r"^[0-9A-Za-z.\-]{1,64}:\d{1,5}$")
_HEX_RE = re.compile(r"^[0-9a-f]{8,64}$")
_ID_RE = re.compile(r"^[0-9a-f]{16}$")
MAX_SERVICES = 8
MAX_TEAMS = 8                  # teams one serving device may be signed on to
MAX_NOTICES = 32               # team and sharing announcements one node may make
MAX_PEERS = 2000
MAX_CANDIDATES = 500
MAX_EXCHANGE = 200


def swarm_secret(group_code: Optional[str] = None, secret_file: Optional[str] = None) -> bytes:
    """Key for a Rosetta network. Kept separate from trainer keys so the two never mix."""
    if bool(group_code) == bool(secret_file):
        raise ValueError("give exactly one of a group code or a secret file")
    base = load_secret(secret_file) if secret_file else secret_from_group_code(group_code)
    return hmac.new(base, b"rosetta-swarm-v1", hashlib.sha256).digest()


def _log(msg: str) -> None:
    print(f"[jane {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_ids(raw, limit: int = MAX_TEAMS) -> List[str]:
    """Validate a list of announced ids (teams, shared training runs) received from another peer."""
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > limit \
            or not all(isinstance(v, str) and _ID_RE.match(v) for v in raw):
        raise ValueError("bad id list")
    return list(dict.fromkeys(raw))


def clean_name(raw) -> str:
    """A name someone chose for team screens: short, printable, one line. A label; nothing checks whose it is."""
    if not isinstance(raw, str):
        return ""
    return " ".join("".join(ch if ch.isprintable() else " " for ch in raw).split())[:24]


def clean_presence(raw) -> dict:
    """Whether a peer's owner can be invited to teams. Anything but a plain yes counts as no."""
    raw = raw if isinstance(raw, dict) else {}
    return {"invites": raw.get("invites") is True, "busy": raw.get("busy") is True,
            "name": clean_name(raw.get("name"))}


def clean_services(raw) -> List[dict]:
    """Validate a service list received from another peer. Raises ValueError on anything odd."""
    if not isinstance(raw, list) or len(raw) > MAX_SERVICES:
        raise ValueError("bad service list")
    out = []
    for s in raw:
        if not isinstance(s, dict):
            raise ValueError("bad service entry")
        model, variant, layers = s.get("model"), s.get("variant", ""), s.get("layers")
        hidden, free = s.get("hidden"), s.get("free", 0)
        if not isinstance(s.get("train", False), bool):
            raise ValueError("bad training flag")
        if not (isinstance(model, str) and _HEX_RE.match(model)):
            raise ValueError("bad model id")
        if not (isinstance(variant, str) and len(variant) <= 32 and variant.replace("-", "").isalnum() or variant == ""):
            raise ValueError("bad variant")
        if not (isinstance(layers, list) and len(layers) == 2 and all(isinstance(v, int) for v in layers)
                and 0 <= layers[0] < layers[1] <= 4096):
            raise ValueError("bad layer range")
        if not (isinstance(hidden, int) and 0 < hidden <= 1 << 20 and isinstance(free, int) and 0 <= free <= 4096):
            raise ValueError("bad service numbers")
        out.append({"model": model, "variant": variant, "layers": [layers[0], layers[1]],
                    "hidden": hidden, "free": free, "train": bool(s.get("train", False)),
                    "teams": clean_ids(s.get("teams"))})
    return out


@dataclass
class Peer:
    node_id: str
    addr: str
    services: List[dict] = field(default_factory=list)
    seen: float = 0.0          # local clock, last time we connected to it ourselves
    failures: int = 0
    leads: List[str] = field(default_factory=list)   # teams this node says it leads
    gifts: List[str] = field(default_factory=list)   # training runs whose result it is sharing with helpers
    invites: bool = False      # its owner sees invitations to teams, and has not chosen to be invisible
    busy: bool = False         # it is in a team right now
    name: str = ""             # what its owner chose to be called on team screens; a label, never checked
    key: str = ""              # the public key it says is its own; proven only when it signs something


class PeerTable:
    """Peers this node has reached itself (shared with others) and candidates it has only heard of."""

    def __init__(self, ttl: float = 300.0):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._peers: Dict[str, Peer] = {}        # by node id, verified by our own connection
        self._candidates: Dict[str, float] = {}  # addr -> time first heard
        self._self_addrs: set = set()

    def confirm(self, node_id: str, addr: str, services: List[dict], leads: Optional[List[str]] = None,
                gifts: Optional[List[str]] = None, invites: bool = False, busy: bool = False,
                name: str = "", key: str = "") -> None:
        with self._lock:
            self._candidates.pop(addr, None)
            for other_id, p in list(self._peers.items()):   # an address now belongs to a new node
                if p.addr == addr and other_id != node_id:
                    del self._peers[other_id]
            if len(self._peers) < MAX_PEERS or node_id in self._peers:
                self._peers[node_id] = Peer(node_id, addr, services, time.time(), 0, list(leads or []),
                                            list(gifts or []), invites, busy, name, key)

    def suggest(self, addr: str) -> None:
        if not isinstance(addr, str) or not _ADDR_RE.match(addr):
            return
        port = int(addr.rsplit(":", 1)[1])
        if not 1 <= port <= 65535:
            return
        with self._lock:
            if addr in self._self_addrs or any(p.addr == addr for p in self._peers.values()):
                return
            if len(self._candidates) < MAX_CANDIDATES:
                self._candidates.setdefault(addr, time.time())

    def mark_self(self, addr: str) -> None:
        with self._lock:
            self._self_addrs.add(addr)
            self._candidates.pop(addr, None)

    def fail(self, addr: str) -> None:
        with self._lock:
            self._candidates.pop(addr, None)
            for node_id, p in list(self._peers.items()):
                if p.addr == addr:
                    p.failures += 1
                    if p.failures >= 2:
                        del self._peers[node_id]

    def prune(self) -> None:
        cutoff = time.time() - self.ttl
        with self._lock:
            for node_id, p in list(self._peers.items()):
                if p.seen < cutoff:
                    del self._peers[node_id]
            for addr, heard in list(self._candidates.items()):
                if heard < cutoff:
                    del self._candidates[addr]

    def peers(self) -> List[Peer]:
        with self._lock:
            return [Peer(p.node_id, p.addr, [dict(s) for s in p.services], p.seen, p.failures, list(p.leads),
                         list(p.gifts), p.invites, p.busy, p.name, p.key) for p in self._peers.values()]

    def candidates(self) -> List[str]:
        with self._lock:
            return list(self._candidates)

    def addresses_to_share(self) -> List[str]:
        with self._lock:
            addrs = [p.addr for p in self._peers.values() if p.failures == 0]
        random.shuffle(addrs)
        return addrs[:MAX_EXCHANGE]


Handler = Callable[[Channel, dict, object, dict], None]


class Node:
    def __init__(self, secret: bytes, listen: str = f"0.0.0.0:{DEFAULT_PORT}", *, encrypt: bool = True,
                 bootstrap: Optional[List[str]] = None, peers_file: Optional[str] = None,
                 gossip_interval: float = 20.0, peer_ttl: float = 300.0, beacon: bool = False,
                 beacon_port: int = BEACON_PORT, beacon_targets: Optional[List[Tuple[str, int]]] = None,
                 max_connections: int = 64, quiet: bool = False):
        self.secret = secret
        self.encrypt = encrypt
        self.node_id = os.urandom(16).hex()      # fresh every run: no lasting identity
        self.table = PeerTable(peer_ttl)
        self.banned: set = set()                 # node ids and addresses caught returning wrong results
        self.vetted: set = set()                 # node ids that have passed a known-answer test this run
        self.handlers: Dict[str, Handler] = {}
        self.disconnect_hooks: List[Callable[[dict], None]] = []
        self.services_fn: Callable[[], List[dict]] = lambda: []
        self.notices: Dict[str, Callable[[], List[str]]] = {}   # "leads" and "gifts": ids announced to the group
        self.presence: Optional[Callable[[], dict]] = None       # {"invites": bool, "busy": bool}, for invite lists
        self.identity = None                                     # this device's key (identity.Identity), once it has one
        self.gossip_interval = gossip_interval
        self.quiet = quiet
        self._bootstrap = list(bootstrap or [])
        self._peers_file = peers_file
        self.handoffs = None                      # a HandoffLog, when fingerprints of handoffs are kept
        self._stop = threading.Event()
        self._conn_slots = threading.Semaphore(max_connections)
        host, _, port = listen.rpartition(":")
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, int(port)))
        self._server.listen(64)
        self.port = self._server.getsockname()[1]
        self._beacon = beacon
        self._beacon_port = beacon_port
        self._beacon_targets = beacon_targets or [("255.255.255.255", beacon_port)]
        self._beacon_tag = hashlib.sha256(secret + b"rosetta-beacon").digest()[:8]
        self._threads: List[threading.Thread] = []
        if peers_file and os.path.exists(peers_file):
            try:
                with open(peers_file) as f:
                    for addr in json.load(f)[:MAX_CANDIDATES]:
                        self.table.suggest(addr)
            except (OSError, ValueError):
                pass
        for addr in self._bootstrap:
            self.table.suggest(addr)

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "Node":
        for target in (self._accept_loop, self._gossip_loop):
            self._spawn(target)
        if self._beacon:
            self._spawn(self._beacon_listen)
            self._spawn(self._beacon_send)
        return self

    def _spawn(self, target, *args) -> None:
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        if self._peers_file:
            try:
                with open(self._peers_file, "w") as f:
                    json.dump([p.addr for p in self.table.peers()], f)
            except OSError:
                pass

    def info(self) -> dict:
        out = {"node_id": self.node_id, "port": self.port, "version": __version__,
               "services": self.services_fn()}
        for kind in ("leads", "gifts"):
            said = self.notices[kind]()[:MAX_NOTICES] if kind in self.notices else []
            if said:
                out[kind] = said
        if self.identity is not None:
            out["key"] = self.identity.key
        if self.presence is not None:
            here = self.presence()
            out["presence"] = {"invites": bool(here.get("invites")), "busy": bool(here.get("busy")),
                               "name": clean_name(here.get("name"))}
        return out

    # -- serving -------------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, peer = self._server.accept()
            except OSError:
                return
            if not self._conn_slots.acquire(blocking=False):
                conn.close()                      # full: refuse rather than queue without limit
                continue
            self._spawn(self._serve_conn, conn, peer[0])

    def _serve_conn(self, conn: socket.socket, peer_ip: str) -> None:
        state: dict = {"peer_ip": peer_ip}
        ch = None
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(30)
            try:
                ch = Channel.accept(conn, self.secret, encrypt=self.encrypt)
            except (ProtocolError, ConnectionError, OSError):
                conn.close()
                return
            ch.sock.settimeout(600)
            if self.handoffs is not None:
                ch.tap = self.handoffs.tap("serving", peer_ip)
            while not self._stop.is_set():
                try:
                    header, tensor, blob = ch.recv()
                except (ProtocolError, ConnectionError, OSError):
                    return
                state["blob"] = blob              # raw attachment, for the few requests that carry one
                op = header["op"]
                try:
                    if op == "hello":
                        port = header.get("port")
                        if isinstance(port, int) and 1 <= port <= 65535 and header.get("node_id") != self.node_id:
                            self.table.suggest(f"{peer_ip}:{port}")   # verified later by dialing back
                        ch.send({"op": "hello_ok", **self.info(), "your_ip": peer_ip})
                    elif op == "peers":
                        ch.send({"op": "peers_ok", "peers": self.table.addresses_to_share()})
                    elif op == "bye":
                        return
                    elif op in self.handlers:
                        self.handlers[op](ch, header, tensor, state)
                    else:
                        ch.send({"op": "error", "message": f"unknown request {op!r}"})
                except (ConnectionError, OSError):
                    return
                except Exception as e:           # tell the caller, keep the node alive
                    try:
                        ch.send({"op": "error", "message": f"{type(e).__name__}: {e}"[:300]})
                    except (ConnectionError, OSError):
                        return
        finally:
            for hook in self.disconnect_hooks:
                try:
                    hook(state)
                except Exception:
                    pass
            if ch is not None:
                ch.close()
            self._conn_slots.release()

    # -- finding peers -------------------------------------------------------
    def dial(self, addr: str, timeout: float = 15.0) -> Channel:
        host, _, port = addr.rpartition(":")
        ch = Channel.connect(host, int(port), self.secret, timeout=timeout, encrypt=self.encrypt)
        ch.sock.settimeout(timeout)
        if self.handoffs is not None:
            ch.tap = self.handoffs.tap("asking", addr)
        return ch

    def visit(self, addr: str) -> bool:
        """Connect to one address, learn who it is and whom it knows. True if it is a live peer."""
        try:
            ch = self.dial(addr)
        except AuthError:
            self.table.fail(addr)                 # a different group; not ours to talk to
            return False
        except (ProtocolError, ConnectionError, OSError):
            self.table.fail(addr)
            return False
        try:
            hello, _, _ = ch.call({"op": "hello", "node_id": self.node_id, "port": self.port})
            node_id = hello.get("node_id")
            if not (isinstance(node_id, str) and _HEX_RE.match(node_id)):
                raise ProtocolError("bad node id")
            if node_id == self.node_id:
                self.table.mark_self(addr)
                return False
            self.table.confirm(node_id, addr, clean_services(hello.get("services", [])),
                               clean_ids(hello.get("leads"), MAX_NOTICES),
                               clean_ids(hello.get("gifts"), MAX_NOTICES), **clean_presence(hello.get("presence")),
                               key=hello.get("key") if isinstance(hello.get("key"), str)
                               and re.match(r"^[0-9a-f]{64}$", hello["key"]) else "")
            reply, _, _ = ch.call({"op": "peers"})
            shared = reply.get("peers", [])
            if isinstance(shared, list):
                for other in shared[:MAX_EXCHANGE]:
                    self.table.suggest(other)
            ch.send({"op": "bye"})
            return True
        except (ProtocolError, RemoteError, ConnectionError, OSError, ValueError):
            self.table.fail(addr)
            return False
        finally:
            ch.close()

    def gossip_once(self, fanout: int = 3) -> None:
        self.table.prune()
        known = [p.addr for p in self.table.peers()]
        fresh = self.table.candidates()
        random.shuffle(known)
        random.shuffle(fresh)
        targets = known[:fanout] + fresh[:fanout * 2]
        if not targets:
            targets = list(self._bootstrap)       # lost everyone: go back to the starting addresses
        for addr in targets:
            if self._stop.is_set():
                return
            self.visit(addr)

    def _gossip_loop(self) -> None:
        delay = 0.2                               # quick first rounds, then settle
        while not self._stop.wait(delay):
            try:
                self.gossip_once()
            except Exception as e:
                if not self.quiet:
                    _log(f"peer exchange error: {e}")
            delay = min(self.gossip_interval, delay * 2)

    def refresh(self, rounds: int = 3) -> None:
        for _ in range(rounds):
            self.gossip_once()

    # -- local network announcements -------------------------------------------
    def _beacon_packet(self) -> bytes:
        return _BEACON_MAGIC + self._beacon_tag + bytes.fromhex(self.node_id)[:8] + struct.pack("!H", self.port)

    def _beacon_send(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            return
        while not self._stop.wait(0.5 if not self.table.peers() else 30.0):
            for target in self._beacon_targets:
                try:
                    sock.sendto(self._beacon_packet(), target)
                except OSError:
                    pass
        sock.close()

    def _beacon_listen(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self._beacon_port))
            sock.settimeout(1.0)
        except OSError:
            return
        size = len(_BEACON_MAGIC) + 8 + 8 + 2
        while not self._stop.is_set():
            try:
                data, (ip, _) = sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != size or not data.startswith(_BEACON_MAGIC):
                continue
            tag = data[len(_BEACON_MAGIC):len(_BEACON_MAGIC) + 8]
            if not hmac.compare_digest(tag, self._beacon_tag):
                continue                           # a different group's announcement
            if data[len(_BEACON_MAGIC) + 8:len(_BEACON_MAGIC) + 16] == bytes.fromhex(self.node_id)[:8]:
                continue                           # our own
            port = struct.unpack("!H", data[-2:])[0]
            self.table.suggest(f"{ip}:{port}")
        sock.close()
