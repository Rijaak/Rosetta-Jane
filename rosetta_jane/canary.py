"""Known-answer tests for peers.

A peer never sees a question. It receives hidden states for some layer and
returns hidden states for a later layer. So a test with a known answer is a
pair of hidden states: what goes into a run of layers, and what must come out.

Preparing the answers ("calibration"): this device takes a set of private test
prompts and computes, by itself, the hidden states at every layer boundary. It
cannot hold the whole model at once, so it loads a few layers at a time. That is
slow, and it is done ahead of time.

Using them: before a peer is trusted, and at random afterwards, it is given the
recorded input for the layers it serves, as an ordinary session. Its output
must match the recorded output. To the peer a test is indistinguishable from
the start of a real answer: both are just numbers.

Limits: a test covers the first steps of a session, so a peer that only lies
deep into long answers outlasts it. Each test is shown to a given address once,
since a repeated input would give the test away; when the tests run out,
calibrate again.

Keeping the answers from being passed around
--------------------------------------------
Anyone holding a device's answers could pass that device's tests. Three things
stand in the way of answers being shared, on purpose or for convenience:

  * Every device invents its own test prompts at random. No two devices have
    the same tests, so someone else's answers are no use against yours, and
    yours are no use against theirs.
  * The answers are stored encrypted with a key that exists only on the device
    that made them and is tied to that machine. Copied to another machine, the
    file cannot be read and is refused.
  * A device only ever trusts answers it worked out itself.

What this cannot do: stop the owner of a device from digging out their own
data if they are determined. Doing so weakens only their own protection.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import random
import uuid
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from .lan.stage import _find_decoder
from .layers import LocalEnds, _load, _run, _slice, _variant

# Varied on purpose: arithmetic, logic, recall, instructions. The peer cannot see which
# kind it is being tested with, and the numbers are randomised per device.
_SEEDS = [
    "What is {a} multiplied by {b}? Show each step.",
    "If a train leaves at {a} minutes past noon and travels for {b} minutes, when does it arrive?",
    "Is {a} a prime number? Explain how you checked.",
    "Sort these numbers from smallest to largest: {a}, {b}, {c}, {d}.",
    "All birds in the garden are sparrows. Some sparrows are ringed. Does it follow that some birds are ringed?",
    "A rope {a} metres long is cut into pieces of {b} centimetres. How many whole pieces are there?",
    "Explain in two sentences why the sky looks blue.",
    "Translate into plain words: the quotient of {a} and {b}, increased by {c}.",
    "Write a short checklist for packing a first aid kit for {a} people.",
    "Which is heavier, {a} kilograms of feathers or {b} kilograms of iron, and by how much?",
    "Summarise how a heat pump moves warmth from outside to inside.",
    "What is the remainder when {a} is divided by {b}?",
    "Three friends share {a} apples so that each gets a whole number. What is the most each can get?",
    "Give the next number in the sequence {a}, {b}, {c} if the rule is add the same amount each time.",
    "Describe how to purify water using only a pot and a fire.",
    "If today is the {a}th day of a 365 day year, how many days remain?",
]


def _machine() -> bytes:
    """Something that differs from one computer to the next and stays put on each."""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(path, "rb") as f:
                value = f.read().strip()
            if value:
                return value
        except OSError:
            pass
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography") as key:
                return str(winreg.QueryValueEx(key, "MachineGuid")[0]).encode()
        except OSError:
            pass
    return f"{uuid.getnode()}-{platform.node()}".encode()


def device_key(work_dir: str, create: bool) -> bytes:
    """The key that locks this device's known answers. Never leaves the device."""
    path = os.path.join(work_dir, "device.key")
    if not os.path.exists(path):
        if not create:
            raise ValueError("this device has no key for known answers yet; run calibrate here")
        os.makedirs(work_dir, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    with open(path, "rb") as f:
        seed = f.read()
    return hmac.new(seed, b"rosetta-jane-known-answers|" + _machine(), hashlib.sha256).digest()


def make_prompts(count: int, rng: random.Random) -> List[str]:
    out = []
    for _ in range(count):
        seed = rng.choice(_SEEDS)
        a, b = rng.randint(12, 9999), rng.randint(3, 997)
        out.append(seed.format(a=a, b=b, c=a + b, d=rng.randint(2, 99999)))
    return out


def calibrate(model_dir: str, work_dir: str, ends: LocalEnds, prompt_ids: Sequence[Sequence[int]], *,
              chunk: int = 4, dtype: Optional[str] = None, load_4bit: bool = False, device_map=None,
              max_memory: Optional[dict] = None, allow_untested: bool = False,
              progress=None) -> "CanaryBook":
    """Compute the true hidden states at every layer boundary for each test prompt, a few layers at a time."""
    if not prompt_ids or any(len(ids) < 4 for ids in prompt_ids):
        raise ValueError("each test prompt needs at least 4 tokens")
    own, total = ends.own_layers, ends.num_layers
    states: List[Dict[int, torch.Tensor]] = []
    current: List[torch.Tensor] = []
    for ids in prompt_ids:
        h, _ = ends.head(list(ids), None)
        current.append(h)
        states.append({own: h})
    variant = None
    for a in range(own, total, chunk):
        b = min(a + chunk, total)
        if progress:
            progress(a, total)
        model = _load(_slice(model_dir, work_dir, a, b, False, allow_untested), dtype, load_4bit,
                      device_map, max_memory)
        decoder = _find_decoder(model)
        decoder.norm = nn.Identity()
        variant = _variant(dtype, load_4bit, model)
        ref = next(p for p in decoder.layers[0].parameters())
        run_dtype = ref.dtype if ref.is_floating_point() else (getattr(torch, dtype) if dtype else torch.float16)
        layers = list(decoder.layers)
        with torch.no_grad():
            for j, layer in enumerate(layers):       # one layer at a time, to record every boundary
                one = nn.ModuleList([layer])
                for i, h in enumerate(current):
                    current[i], _ = _run(decoder, one, h.to(ref.device, run_dtype), None)
                    states[i][a + j + 1] = current[i]
        del model, decoder, layers
    if variant is None:                               # the asking device already holds every layer
        variant = "local"
    store = torch.float32 if variant in ("float32", "local") else torch.float16
    tensors, lengths = {}, []
    for i, per_layer in enumerate(states):
        lengths.append(int(per_layer[own].shape[1]))
        for layer, h in per_layer.items():
            tensors[f"{i}.{layer}"] = h.detach().to("cpu", store).contiguous()
    os.makedirs(work_dir, exist_ok=True)
    path = CanaryBook.path(work_dir, variant)
    nonce = os.urandom(12)
    sealed = AESGCM(device_key(work_dir, create=True)).encrypt(
        nonce, st_save(tensors), f"{ends.fingerprint}|{variant}".encode())
    with open(path, "wb") as f:
        f.write(nonce + sealed)
    meta = {"fingerprint": ends.fingerprint, "variant": variant, "own_layers": own, "num_layers": total,
            "lengths": lengths, "used": {}}
    with open(path + ".json", "w") as f:
        json.dump(meta, f)
    return CanaryBook(work_dir, variant)


class CanaryBook:
    """Known answers for one precision. A device can keep one book per precision it can run."""

    @staticmethod
    def path(work_dir: str, variant: str) -> str:
        return os.path.join(work_dir, f"known-answers-{variant}.bin")

    @staticmethod
    def variants(work_dir: str) -> List[str]:
        found = []
        if os.path.isdir(work_dir):
            for name in sorted(os.listdir(work_dir)):
                if name.startswith("known-answers-") and name.endswith(".bin") \
                        and os.path.exists(os.path.join(work_dir, name + ".json")):
                    found.append(name[len("known-answers-"):-len(".bin")])
        return found

    @classmethod
    def exists(cls, work_dir: str) -> bool:
        return bool(cls.variants(work_dir))

    def __init__(self, work_dir: str, variant: Optional[str] = None):
        if variant is None:
            have = self.variants(work_dir)
            if len(have) != 1:
                raise ValueError(f"say which precision; this folder has known answers for: {have or 'none'}")
            variant = have[0]
        self._file = self.path(work_dir, variant)
        with open(self._file + ".json") as f:
            self.meta = json.load(f)
        with open(self._file, "rb") as f:
            raw = f.read()
        try:
            plain = AESGCM(device_key(work_dir, create=False)).decrypt(
                raw[:12], raw[12:], f"{self.meta['fingerprint']}|{self.meta['variant']}".encode())
        except InvalidTag:
            raise ValueError(
                "these known answers were made on a different device, or the file is damaged. Known "
                "answers cannot be shared between devices: run calibrate on this one.") from None
        self._states = st_load(plain)
        self.fingerprint = self.meta["fingerprint"]
        self.variant = self.meta["variant"]
        self.count = len(self.meta["lengths"])
        self._rng = random.Random()

    def _save(self) -> None:
        tmp = self._file + ".json.tmp"
        with open(tmp, "w") as f:
            json.dump(self.meta, f)
        os.replace(tmp, self._file + ".json")

    def can_test(self, start: int, end: int, variant: str) -> bool:
        return variant == self.variant and f"0.{start}" in self._states and f"0.{end}" in self._states

    def remaining(self, ip: str) -> int:
        return self.count - len(self.meta["used"].get(ip, []))

    def take(self, ip: str) -> Optional[int]:
        """A test this address has not been shown before, or None if they are all spent."""
        used = self.meta["used"].setdefault(ip, [])
        fresh = [i for i in range(self.count) if i not in used]
        if not fresh:
            return None
        pick = self._rng.choice(fresh)
        used.append(pick)
        self._save()
        return pick

    def steps(self, index: int, start: int, end: int, tail: int):
        """(input, expected output) pairs: the prompt as one block, then `tail` single tokens."""
        x, y = self._states[f"{index}.{start}"], self._states[f"{index}.{end}"]
        length = x.shape[1]
        tail = max(0, min(tail, length - 2))
        head = length - tail
        cuts = [(0, head)] + [(head + k, head + k + 1) for k in range(tail)]
        return [(x[:, a:b], y[:, a:b]) for a, b in cuts]


class CanaryShelf:
    """Every book this device has prepared for one model, picked by the precision a peer runs.

    Results from a compressed 4-bit copy and an uncompressed copy differ by about as much
    as a deliberate lie could, so they are never compared with each other. The safe way
    to check peers of several precisions is to prepare a book for each.
    """

    def __init__(self, work_dir: str, fingerprint: str, same: Sequence[Sequence[str]] = ()):
        self.fingerprint = fingerprint
        self.books: Dict[str, CanaryBook] = {}
        self.refused: List[str] = []              # precisions whose answers came from another device
        # Groups of precisions the owner has measured to be close enough to share one book.
        self.same = [set(group) for group in same]
        for variant in CanaryBook.variants(work_dir):
            try:
                book = CanaryBook(work_dir, variant)
            except ValueError:
                self.refused.append(variant)
                continue
            if book.fingerprint == fingerprint:
                self.books[variant] = book

    def for_variant(self, variant: str) -> Optional[CanaryBook]:
        if variant in self.books:
            return self.books[variant]
        for group in self.same:
            if variant in group:
                for other in sorted(group):
                    if other in self.books:
                        return self.books[other]
        return None

    def __bool__(self) -> bool:
        return bool(self.books)


def measure_precisions(model_dir: str, work_dir: str, ends_for, prompt_ids: Sequence[Sequence[int]],
                       a: dict, b: dict, chunk: int = 4, allow_untested: bool = False) -> Dict[int, float]:
    """How far apart two precisions are at every layer boundary, as a share of the result's size.

    `a` and `b` are loading options such as {"dtype": "float32"} or {"load_4bit": True, "dtype": "float16"}.
    Both are given the same inputs, so the difference is down to precision alone.
    """
    books = []
    for tag, opts in (("a", a), ("b", b)):
        folder = os.path.join(work_dir, f"measure-{tag}")
        ends = ends_for(folder, opts)
        books.append(calibrate(model_dir, folder, ends, prompt_ids, chunk=chunk, dtype=opts.get("dtype"),
                               load_4bit=bool(opts.get("load_4bit")), device_map=opts.get("device_map"),
                               max_memory=opts.get("max_memory"), allow_untested=allow_untested))
    first, second = books
    gaps: Dict[int, float] = {}
    for key, x in first._states.items():
        layer = int(key.split(".")[1])
        x, y = x.float(), second._states[key].float()
        gaps[layer] = max(gaps.get(layer, 0.0), float((x - y).norm() / (x.norm() + 1e-6)))
    return dict(sorted(gaps.items()))
