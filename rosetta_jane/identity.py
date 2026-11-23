"""A device's own key, and the names that travel with keys.

There are no accounts and no server here, so nobody can promise that a name belongs to one
person. What a device can do is keep a secret key of its own and sign with it. Then:

  * a name shown on a team screen comes with a short tag made from the device's public key,
    like "Sam [3f9a2c]". Anyone can type the name Sam; only Sam's device can show that tag,
    because joining a team means signing with the key behind it;
  * a device remembers the keys it has been in teams with, so its owner is told "you have
    teamed with this device before", and is warned when a name they know turns up with a
    different key;
  * a leader can sign a receipt for teaching points that another leader can check
    (see points.py).

The key is made the first time it is needed and kept in the device's work folder
(identity.json). It never leaves the device. Lose the file and the device is a stranger
again: its name has a new tag and its receipts can no longer be spent. Copy the file to
another device and that device is, to everyone else, the same one.

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
from typing import Dict, Optional

from . import sign as _sign

IDENTITY = "identity.json"
KNOWN = "known-devices.json"
MAX_KNOWN = 2000
FRESH = 600.0                      # a signed "this is me, now" is good for ten minutes either way
_KEY = re.compile(r"^[0-9a-f]{64}$")
_SIG = re.compile(r"^[0-9a-f]{128}$")


def clean_key(raw) -> str:
    return raw if isinstance(raw, str) and _KEY.match(raw) else ""


def tag(key: str) -> str:
    """The short tag shown after a name: six characters made from the device's public key."""
    return hashlib.sha256(bytes.fromhex(key)).hexdigest()[:6] if clean_key(key) else ""


def shown(name: str, key: str) -> str:
    """A name with its key's tag, as team screens show it."""
    mark = tag(key)
    if not mark:
        return name or ""
    return f"{name} [{mark}]" if name else f"[{mark}]"


def message(*parts) -> bytes:
    """The bytes that get signed: what the signature is for, then its details, in one fixed form."""
    return b"rosetta-jane/1 " + json.dumps(list(parts), separators=(",", ":"), sort_keys=True,
                                           ensure_ascii=True).encode()


def verify(key, signature, *parts) -> bool:
    if not (clean_key(key) and isinstance(signature, str) and _SIG.match(signature)):
        return False
    try:
        return _sign.verify(bytes.fromhex(key), message(*parts), bytes.fromhex(signature))
    except Exception:
        return False


class Identity:
    """This device's key. With a work folder it is kept there; without one it lasts as long as the program."""

    def __init__(self, work_dir: Optional[str] = None):
        self.path = os.path.join(work_dir, IDENTITY) if work_dir else None
        seed = None
        if self.path:
            try:
                with open(self.path) as f:
                    seed = bytes.fromhex(json.load(f)["secret"])
                if len(seed) != 32:
                    seed = None
            except (OSError, ValueError, KeyError, TypeError):
                seed = None
        if seed is None:
            seed = secrets.token_bytes(32)
            if self.path:
                try:
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                    tmp = self.path + ".partial"
                    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "w") as f:
                        json.dump({"note": "This device's secret key. Keep it private and do not copy it to "
                                           "another device. Without it this device's name has a new tag and its "
                                           "teaching points can no longer be spent.",
                                   "secret": seed.hex()}, f, indent=2)
                    os.replace(tmp, self.path)
                except OSError:
                    self.path = None                 # nowhere to keep it: a key for this run only
        self._seed = seed
        self.key = _sign.public_key(seed).hex()
        self.tag = tag(self.key)

    def sign(self, *parts) -> str:
        return _sign.sign(self._seed, message(*parts)).hex()

    def here(self, purpose: str, *about) -> dict:
        """A signed "this is my key, right now, for this": what a device sends to show a key is its own."""
        at = int(time.time())
        return {"key": self.key, "at": at, "sig": self.sign(purpose, *about, at)}


def proven(raw, purpose: str, *about, now: Optional[float] = None) -> str:
    """The key in a signed "this is me", if the signature is good and recent. Otherwise ''."""
    if not isinstance(raw, dict):
        return ""
    key, at = clean_key(raw.get("key")), raw.get("at")
    if not key or isinstance(at, bool) or not isinstance(at, int):
        return ""
    if abs((time.time() if now is None else now) - at) > FRESH:
        return ""
    return key if verify(key, raw.get("sig"), purpose, *about, at) else ""


def of(node, work_dir: Optional[str] = None) -> Identity:
    """The identity a node goes by. Made on first use; given a work folder, it is the one kept there."""
    found = getattr(node, "identity", None)
    if found is None or (work_dir and found.path is None):
        found = Identity(work_dir)
        node.identity = found
    return found


class Known:
    """The keys this device has been in teams with, and the names they went by.

    `led` counts the teams this device signed on to that the key led: that took this device's
    owner choosing to join them. `joined` counts the times the key signed on to a team this
    device led: that only took them choosing to join. The difference matters for whose
    teaching points a leader honours.
    """

    def __init__(self, work_dir: Optional[str] = None):
        self.path = os.path.join(work_dir, KNOWN) if work_dir else None
        self.keys: Dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path:
            try:
                with open(self.path) as f:
                    data = json.load(f)
                for key, e in data.items() if isinstance(data, dict) else []:
                    if clean_key(key) and isinstance(e, dict):
                        self.keys[key] = {"name": str(e.get("name", ""))[:24], "led": int(e.get("led", 0)),
                                          "joined": int(e.get("joined", 0)), "last": float(e.get("last", 0))}
            except (OSError, ValueError, TypeError):
                pass

    def _save(self) -> None:
        if not self.path:
            return
        try:
            tmp = self.path + ".partial"
            with open(tmp, "w") as f:
                json.dump(self.keys, f, indent=1)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def note(self, key: str, name: str = "", led: bool = False) -> None:
        """This device has just been in a team with that key: as its member (`led`), or as its leader."""
        if not clean_key(key):
            return
        with self._lock:
            e = self.keys.setdefault(key, {"name": "", "led": 0, "joined": 0, "last": 0.0})
            e["led" if led else "joined"] += 1
            e["last"] = time.time()
            if name:
                e["name"] = name[:24]
            while len(self.keys) > MAX_KNOWN:
                del self.keys[min(self.keys, key=lambda k: self.keys[k]["last"])]
            self._save()

    def teamed(self, key: str) -> bool:
        with self._lock:
            return key in self.keys

    def led(self, key: str) -> bool:
        """Has this device signed on to a team that key led?"""
        with self._lock:
            return self.keys.get(key, {}).get("led", 0) > 0

    def borrowed(self, name: str, key: str) -> bool:
        """Is this a name this device knows from a different key?"""
        if not name or not clean_key(key):
            return False
        with self._lock:
            return key not in self.keys and any(e["name"] == name for e in self.keys.values())

    def about(self, name: str, key: str) -> str:
        """A few words for a team screen about the device behind a name."""
        if not clean_key(key):
            return "no key shown; the name is only a label"
        if self.teamed(key):
            return "you have teamed with this device before"
        if self.borrowed(name, key):
            return "a different device used this name with you before"
        return "new to this device"
