"""Teams: a group choosing, together, to pool its devices, with or without a trained adapter.

Modelled on co-op in video games, and in particular on the team screen of BBSC: Yharnam
Together, the community co-op project for Bloodborne's PC port: a leader, a team code, teams
anyone can see and ask to join, invitations, and a leader's settings that are remembered.
No code of theirs is used here; the design is the debt.

  * Anyone can lead a team. A team is for one model, and either for one adapter held by
    the leader, or for no adapter at all: then it simply pools its members' computing
    power to run the unchanged model. The model and the adapter are shown to everyone
    before they join, because everyone in a team has to be using the same ones.
  * The leader chooses who can sign on, and can change it at any time. The choice is
    remembered for their next team:
      open       anyone in the group sees the team and can sign on;
      ask        anyone in the group sees the team and can ask to join; the leader
                 approves or declines each request;
      invisible  the team is not listed. Its code or an invitation is needed.
    The team's code and the leader's invitations work in every mode. A code is new for
    each team, like DAF H2D.
  * The leader sees the devices in the group whose owners can be invited, and invites
    them. A device's owner can turn that off and be left out of those lists.
  * Nobody is put in a team. An invitation waits until its owner says yes; a request is
    something its owner sent. When either is settled, the adapter is supplied by the
    leader's device at once, with nothing more to do.
  * Devices that serve layers sign on the same way. Only then do they apply the adapter,
    and only to that team's calculations. Everything else they compute is the unchanged
    model, exactly as before.
  * The leader begins. From then on the team's answers come from the model with the
    adapter, at normal speed, with the same checks as any other answer. Training can run
    inside a team too: then only the team's devices do the work.
  * The leader can remove a device. Leave, or have the leader end the team, and it is the
    unchanged model again. A member holds the leader's adapter in memory only. Nothing is
    written to their storage, and nothing is ever applied on a device whose owner did not
    sign it on.

Whoever signs on is sent the adapter, so who can sign on is who can obtain it. An open
team hands it to anyone in the group who asks; on the open 'public' network that is
anyone at all. A code is six characters, which keeps out people who were not given it
but is not strong secrecy. Being invisible keeps a team or a device off other people's
lists; it does not hide a device's address from the peers it connects to.

If you build on Rosetta-Jane, keep the signing on. A leader starts a team; nobody is put
in one.

Plain Python and NumPy, shared by the desktop program and the phone app.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .lan.protocol import ProtocolError, RemoteError
from .mobile import st
from .mobile.peer import check_adapters, check_spec

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"     # no I, O, 0 or 1: a code read out loud still works
FACTS = "rosetta-jane-run.json"
MAX_MEMBERS = 64
JOINING = ("open", "ask", "invisible")
JOINING_WORDS = {"open": "open: anyone in the group can sign on",
                 "ask": "ask to join: anyone in the group can ask, and the leader approves",
                 "invisible": "invisible: not listed; the team's code or an invitation is needed"}
SETTINGS = "team-settings.json"
DEFAULT_SETTINGS = {"joining": "open", "limit": 16, "hidden": False, "name": ""}
MAX_REQUESTS = 32
MIN_MARKS = 5                     # a community rating is not averaged in front of people on fewer marks than this
# The community rating: several questions, each marked 1 (rarely) to 5 (nearly always), about what a
# flatterer does badly. The rating is their average. People who used the adapter in a team answer them.
QUESTIONS = {"right": "How often was it right, where you could check?",
             "unsure": "When it did not know, or could not be sure, how often did it say so?",
             "pushback": "When you were wrong about something, how often did it tell you?"}
QUESTION_WORDS = {"right": "right where they could check", "unsure": "said so when it was not sure",
                  "pushback": "told them when they were wrong"}
PICKS = ("adapter", "unchanged", "same", "neither")      # blind comparisons: which answer was right
MAX_PICKS = 50                    # blind comparisons counted from one membership
KINDS = ("session", "teaching")   # two lobbies: teams that query the model, and teams that train an adapter
_PASS = re.compile(r"^[0-9a-f]{32}$")
MAX_STRIKES = 10                # wrong codes from one address before it is refused outright
MAX_ADAPTER_BYTES = 512 << 20
MAX_HISTORY = 20
_ID = re.compile(r"^[0-9a-f]{16}$")
_NODE = re.compile(r"^[0-9a-f]{8,64}$")
_KEY = re.compile(r"^(\d+)\.([A-Za-z0-9_.]+)\.(A|B)$")
_NET = (ConnectionError, OSError, ProtocolError, RemoteError)


class NoTeam(Exception):
    """There is no such team to be found, or it has ended."""


class Refused(Exception):
    """What a leader's device answers when the caller may not do what it asked.
    (Not PermissionError: that is an OSError, which the node takes for a broken connection.)"""


# -- codes --------------------------------------------------------------------------------------
def new_code() -> str:
    raw = "".join(secrets.choice(ALPHABET) for _ in range(6))
    return raw[:3] + " " + raw[3:]


def tidy(code) -> str:
    """A code as typed or read out loud, in the one form it is written: 'DAF H2D'."""
    raw = "".join(ch for ch in str(code).upper() if ch not in " -") if isinstance(code, str) else ""
    if len(raw) != 6 or any(ch not in ALPHABET for ch in raw):
        raise ValueError("a team code is six letters and digits, like DAF H2D")
    return raw[:3] + " " + raw[3:]


def keys(code: str) -> Tuple[str, str]:
    """(the id a team is announced under, the key that shows you were given the code).

    Both come from the code through a slow step, so the announced id cannot simply be
    tried against every possible code.
    """
    raw = hashlib.scrypt(tidy(code).replace(" ", "").encode(), salt=b"rosetta-jane-team-v1", n=1 << 14, r=8, p=1,
                         maxmem=1 << 26, dklen=32)
    return raw[:8].hex(), raw[8:].hex()


# -- adapters as plain arrays --------------------------------------------------------------------
def adapter_fingerprint(adapters: Dict[str, np.ndarray]) -> str:
    """One short name for exactly these numbers, the same on a phone and on a PC."""
    h = hashlib.sha256()
    for key in sorted(adapters):
        a = np.ascontiguousarray(adapters[key], dtype="<f4")
        h.update(key.encode() + b"\0" + ",".join(str(d) for d in a.shape).encode() + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def pick(adapters: Dict[str, np.ndarray], ranges: Sequence[Sequence[int]]) -> Dict[str, np.ndarray]:
    """The adapters belonging to the given stretches of layers."""
    return {k: v for k, v in adapters.items()
            if any(a <= int(k.split(".", 1)[0]) < b for a, b in ranges)}


def read_adapter(adapter_dir: str, layer_prefix: str) -> dict:
    """An adapter folder as plain arrays under their layer numbers, with its settings and its record."""
    try:
        with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
            config = json.load(f)
        path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if os.path.getsize(path) > MAX_ADAPTER_BYTES:
            raise ValueError("it is too large")
        with open(path, "rb") as f:
            weights = st.loads(f.read())
        rank, alpha, targets = check_spec(config["r"], config["lora_alpha"], config["target_modules"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"that folder does not hold a readable adapter: {e}") from None
    pattern = re.compile(r"^base_model\.model\." + re.escape(layer_prefix)
                         + r"(\d+)\.(.+)\.lora_(A|B)(?:\.[^.]+)?\.weight$")
    adapters = {}
    for key, value in weights.items():
        m = pattern.match(key)
        if m and value.dtype.kind == "f":
            adapters[f"{int(m.group(1))}.{m.group(2)}.{m.group(3)}"] = np.ascontiguousarray(value, dtype=np.float32)
    facts = {}
    try:
        with open(os.path.join(adapter_dir, FACTS)) as f:
            read = json.load(f)
        facts = read if isinstance(read, dict) else {}
    except (OSError, ValueError):
        pass
    return {"name": os.path.basename(os.path.normpath(adapter_dir)), "adapters": adapters, "rank": rank,
            "alpha": alpha, "targets": targets, "facts": facts}


def fitted(adapter_dir: str, layer_prefix: str, model: str, shapes_fn: Callable[[List[str]], dict]) -> dict:
    """read_adapter, refused unless it was made for this model and fits every layer of it."""
    found = read_adapter(adapter_dir, layer_prefix)
    made_for = found["facts"].get("base_model_fingerprint")
    if isinstance(made_for, str) and made_for and made_for != model:
        raise ValueError(f"this adapter was trained on a different model ({made_for[:40]}); "
                         f"the model here is {model}")
    try:
        check_adapters(found["adapters"], shapes_fn(found["targets"]), found["rank"])
    except ValueError:
        raise ValueError("this adapter does not fit the model here") from None
    return found


# -- what a person is shown before signing on -----------------------------------------------------
_TEXT = ("name", "base_model_fingerprint", "base_model_name", "finished", "trained_from", "received_on")
_NUMBERS = ("pairs", "pairs_kept_back", "steps", "final_loss", "work_checked", "work_unchecked",
            "peers_blocked", "helpers", "helped_examples")


def _text(value, limit: int = 120) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return "".join(ch for ch in value if ch.isprintable())[:limit]


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def clean_card(raw) -> Optional[dict]:
    """A score card received from someone else, reduced to its numbers. The verdict is worked out again here."""
    from .scoring import verdict
    if not isinstance(raw, dict):
        return None
    sides = {}
    for side in ("before", "after"):
        got = raw.get(side) if isinstance(raw.get(side), dict) else {}
        sides[side] = {"loss": _number(got.get("loss")), "right_next_word": _number(got.get("right_next_word"))}
    if sides["before"]["loss"] is None or sides["after"]["loss"] is None \
            or sides["before"]["right_next_word"] is None or sides["after"]["right_next_word"] is None:
        return None
    pairs = _number(raw.get("pairs"))
    pairs = int(pairs) if pairs is not None and 0 <= pairs <= 10 ** 9 else 0
    return {"adapter": _text(raw.get("adapter")) or "", "base_model_fingerprint": _text(raw.get("base_model_fingerprint")) or "",
            "scored": _text(raw.get("scored"), 40) or "", "pairs": pairs,
            "answer_tokens": int(_number(raw.get("answer_tokens")) or 0),
            "never_trained_on": raw.get("never_trained_on") is True, **sides,
            "work_checked": int(_number(raw.get("work_checked")) or 0),
            "work_unchecked": int(_number(raw.get("work_unchecked")) or 0),
            "verdict": verdict(sides["before"], sides["after"], pairs)}


def clean_facts(raw) -> dict:
    """An adapter's record, cut down to plain known fields. Used on anything that came from another device."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for key in _TEXT:
        value = _text(raw.get(key))
        if value:
            out[key] = value
    for key in _NUMBERS:
        value = _number(raw.get(key))
        if value is not None:
            out[key] = value
    if raw.get("received") is True:
        out["received"] = True
    for key in ("score", "my_score"):
        card = clean_card(raw.get(key))
        if card:
            out[key] = card
    if isinstance(raw.get("community"), dict):
        out["community"] = tally_ratings(raw["community"])
    from .flattery import clean as clean_flattery
    checked = clean_flattery(raw.get("flattery"))
    if checked:
        out["flattery"] = checked
    history = []
    for step in raw.get("history") if isinstance(raw.get("history"), list) else []:
        if isinstance(step, dict) and isinstance(step.get("adapter"), str) and _ID.match(step["adapter"]):
            history.append({"name": _text(step.get("name"), 60) or "", "adapter": step["adapter"],
                            "finished": _text(step.get("finished"), 40) or ""})
    if history:
        out["history"] = history[-MAX_HISTORY:]
    return out


def empty_ratings() -> dict:
    return {"marks": {q: [0] * 5 for q in QUESTIONS}, "blind": {p: 0 for p in PICKS},
            "trick": {p: 0 for p in PICKS}}


def clean_ratings(raw) -> dict:
    """Ratings as counts only, whoever they came from: how many people gave each mark to each question,
    and how blind comparisons came out (with questions of the person's own, and with trick questions)."""
    out = empty_ratings()
    if not isinstance(raw, dict):
        return out

    def count(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 10 ** 9 else 0
    marks = raw.get("marks") if isinstance(raw.get("marks"), dict) else {}
    for q in QUESTIONS:
        if isinstance(marks.get(q), list) and len(marks[q]) == 5:
            out["marks"][q] = [count(c) for c in marks[q]]
    for kind in ("blind", "trick"):
        got = raw.get(kind) if isinstance(raw.get(kind), dict) else {}
        out[kind] = {p: count(got.get(p)) for p in PICKS}
    return out


def add_ratings(*many) -> dict:
    total = empty_ratings()
    for one in many:
        one = clean_ratings(one)
        for q in QUESTIONS:
            total["marks"][q] = [a + b for a, b in zip(total["marks"][q], one["marks"][q])]
        for kind in ("blind", "trick"):
            for p in PICKS:
                total[kind][p] += one[kind][p]
    return total


def tally_ratings(raw) -> dict:
    """A community rating from the counts: an average for each question, and one over all of them.
    Every average is worked out here, never taken from someone else's say-so."""
    counts = clean_ratings(raw)
    each, marks, total = {}, 0, 0
    for q in QUESTIONS:
        n = sum(counts["marks"][q])
        points = sum((i + 1) * c for i, c in enumerate(counts["marks"][q]))
        each[q] = {"ratings": n, "average": round(points / n, 2) if n else None}
        marks, total = marks + n, total + points
    return {**counts, "questions": each, "people": max(e["ratings"] for e in each.values()),
            "ratings": marks, "average": round(total / marks, 2) if marks else None}


def continued(facts: dict, name: str, fingerprint: str) -> List[dict]:
    """The chain of adapters a new training run stands on, ending with the one it starts from."""
    earlier = clean_facts(facts).get("history", [])
    step = {"name": _text(name, 60) or "", "adapter": fingerprint, "finished": _text(facts.get("finished"), 40) or ""}
    return (earlier + [step])[-MAX_HISTORY:]


def fact_lines(desc: dict) -> List[str]:
    """What to read before signing on to a team, or before taking a copy of an adapter.

    The model and the adapter come first, each with where a person gets it, because everyone in a
    team has to be using the same ones. Then the two ratings, kept apart: the teacher rating (the
    score from the run that trained it) and the community rating (marks from people who used it).
    """
    from .scoring import describe
    facts = desc.get("facts") or {}
    named = desc.get("model_name") or facts.get("base_model_name")
    model = f"Model: {named + ' ' if named else ''}({desc.get('model', 'unknown')})."
    if "fits" in desc:
        model += " You have this model on this device." if desc["fits"] else \
            " It is NOT the model on this device, so this device cannot join. Models are not sent through " \
            "a team: get the same model files first."
    lines = [model]
    in_team = "team" in desc
    if in_team and desc.get("kind") == "teaching":
        lines.append("This is a teaching team: its devices work together on training an adapter. "
                     "Answers are not changed by it.")
    elif in_team and desc.get("adapter") is None:
        lines.append("Adapter: none. This team pools its devices' computing power to run the unchanged model.")
    if desc.get("adapter") is not None or not in_team:
        adapter = f"Adapter: \"{desc.get('name', '')}\" ({desc.get('adapter', 'unknown')})."
        if in_team:
            adapter += " Available to download for the session: the leader's device sends it when you join, " \
                       "and it is not kept afterwards."
            if desc.get("held_as"):
                adapter += f" You also hold this adapter yourself, as \"{desc['held_as']}\"."
        lines.append(adapter)
        checked, unchecked = facts.get("work_checked"), facts.get("work_unchecked")
        if checked is not None or unchecked is not None:
            lines.append(f"Training work double-checked: {checked if checked is not None else 'unknown'}. "
                         f"Resting on one device's word: {unchecked if unchecked is not None else 'unknown'}.")
        else:
            lines.append("How much of its training was double-checked is not recorded.")
        if facts.get("received"):
            lines.append("The person offering it received it from someone else; they did not train it.")
        for step in facts.get("history", []):
            lines.append(f"Continued from \"{step['name']}\" ({step['adapter']}).")
        if facts.get("score"):
            lines += ["Teacher rating (scored by the run that trained it):"] + describe(facts["score"])
        else:
            lines.append("Teacher rating: none. The run that trained it did not score it.")
        if facts.get("my_score"):
            lines += ["Scored by the person offering it, on their own questions:"] + describe(facts["my_score"])
        from .flattery import describe as describe_flattery
        lines += describe_flattery(facts.get("flattery"))
        lines += community_lines(facts.get("community"))
        lines.append("Ratings are evidence, not proof: people tend to like being agreed with. The teacher rating "
                     "and the flattery check are measured; the community rating is opinion. None of them decides "
                     "how adapters are listed.")
    theirs = desc.get("version")
    if theirs and theirs != __version__:
        lines.append(f"The leader runs Rosetta-Jane {theirs}; this device runs {__version__}. "
                     "Different versions may not work together.")
    return lines


def community_lines(got: Optional[dict]) -> List[str]:
    """The community rating as lines a person can read: the average over several questions, each
    question's own average, and how blind comparisons came out."""
    got = tally_ratings(got)
    asked = "People who use it in a team mark how often it was right, said so when unsure, and told them " \
            "when they were wrong."
    if got["people"] >= MIN_MARKS:
        lines = [f"Community rating: {got['average']} out of 5, the average of {got['ratings']} marks from about "
                 f"{got['people']} people who used it in a team."]
        lines += [f"  {QUESTION_WORDS[q].capitalize()}: {e['average']} out of 5 ({e['ratings']} marks)."
                  for q, e in got["questions"].items() if e["ratings"]]
    elif got["ratings"]:
        lines = [f"Community rating: {got['ratings']} mark{'s' if got['ratings'] != 1 else ''} so far, from fewer "
                 f"than {MIN_MARKS} people; too few to rely on. {asked}"]
    else:
        lines = [f"Community rating: none yet. {asked}"]
    for kind, what in (("blind", "on people's own questions"), ("trick", "on questions built on a false statement")):
        tried = sum(got[kind].values())
        if tried:
            lines.append(f"  Blind comparisons {what} ({tried}): the adapter's answer was picked as right "
                         f"{got[kind]['adapter']} times, the unchanged model's {got[kind]['unchanged']}, "
                         f"no difference {got[kind]['same']}, neither {got[kind]['neither']}.")
    return lines


def plain(error: Exception) -> str:
    """An error from the other device, without the programmer's label in front."""
    text = str(error)
    head, sep, rest = text.partition(": ")
    return rest if sep and head.isidentifier() else text


# -- applying it -----------------------------------------------------------------------------------
def tag_of(member: str) -> str:
    """The name a member's work goes under on the team's devices. It does not reveal the membership itself."""
    return hashlib.sha256(b"rosetta-team-tag:" + member.encode()).hexdigest()[:16]


def proof_of(member: str, node_id: str) -> str:
    """Shows one serving device that a piece of work really is from the member whose tag it carries.
    It is good at that device only, so a device that sees it cannot use it anywhere else."""
    return hmac.new(member.encode(), b"at:" + node_id.encode(), hashlib.sha256).hexdigest()[:32]


class TeamUse:
    """What one calculation needs to apply a team's adapter: plain arrays and the settings."""

    def __init__(self, team_id: str, adapters: Dict[str, np.ndarray], rank: int, alpha: float, targets: List[str],
                 member: Optional[str] = None):
        self.member = member                       # this device's membership; None for the team's leader
        self.id, self.adapters = team_id, adapters
        self.plain = not adapters                  # no adapter: the team only pools its devices
        self.rank, self.alpha, self.targets = rank, float(alpha), list(targets)
        self.scale = self.alpha / self.rank
        self._made: dict = {}
        self._lock = threading.Lock()

    def memo(self, key, make):
        """Something built from these arrays once and kept (their PyTorch form, say) for as long as the team lasts."""
        with self._lock:
            if key not in self._made:
                self._made[key] = make()
            return self._made[key]

    def lora(self) -> dict:
        """The form the NumPy engine takes: {"<layer>.<module>": (A, B, scale)}."""
        return self.memo("lora", lambda: {k[:-2]: (v, self.adapters[k[:-2] + ".B"], self.scale)
                                          for k, v in self.adapters.items() if k.endswith(".A")})

    def blob(self, ranges: Sequence[Sequence[int]]) -> Optional[bytes]:
        return None if self.plain else st.dumps(pick(self.adapters, ranges))

    def ident(self, node_id: str) -> dict:
        """What to send a serving device with this member's work, so its share of that device is theirs."""
        if self.member is None:
            return {}
        return {"as": tag_of(self.member), "proof": proof_of(self.member, node_id)}


def _range(value, total: int) -> Optional[List[int]]:
    if value is None:
        return None
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool)
                                                                 for v in value) and 0 <= value[0] < value[1] <= total):
        raise ValueError("bad layer range")
    return [value[0], value[1]]


def _send(node, addr: str, header: dict, timeout: float = 20.0):
    ch = node.dial(addr, timeout=timeout)
    try:
        ch.sock.settimeout(timeout)
        got = ch.call(header)
        try:
            ch.send({"op": "bye"})
        except _NET:
            pass
        return got
    finally:
        ch.close()


# -- the leader's settings, remembered for the next team ------------------------------------------------
def check_settings(joining=None, limit=None, hidden=None, name=None) -> dict:
    out = {}
    if name is not None:
        from .net import clean_name
        if not isinstance(name, str):
            raise ValueError("a name is text")
        out["name"] = clean_name(name)
    if hidden is not None:
        if not isinstance(hidden, bool):
            raise ValueError("'hidden' must be true or false")
        out["hidden"] = hidden
    if joining is not None:
        if joining not in JOINING:
            raise ValueError("who can sign on is one of: open, ask, invisible")
        out["joining"] = joining
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 2 <= limit <= MAX_MEMBERS:
            raise ValueError(f"a team holds from 2 to {MAX_MEMBERS} devices")
        out["limit"] = limit
    return out


def load_settings(work_dir: Optional[str]) -> dict:
    """The choices a leader made for their last team; they are the starting point for the next one."""
    out = dict(DEFAULT_SETTINGS)
    try:
        with open(os.path.join(work_dir or "", SETTINGS)) as f:
            read = json.load(f)
        out.update(check_settings(read.get("joining"), read.get("limit"), read.get("hidden"), read.get("name")))
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return out


def save_settings(work_dir: Optional[str], settings: dict) -> None:
    if not work_dir:
        return
    try:
        with open(os.path.join(work_dir, SETTINGS), "w") as f:
            json.dump({k: settings[k] for k in DEFAULT_SETTINGS if k in settings}, f)
    except OSError:
        pass


class TeamHost:
    """The leader's side of a team."""
    role = "leader"

    def __init__(self, node, *, model: str, total_layers: int, name: Optional[str] = None,
                 adapters: Optional[Dict[str, np.ndarray]] = None, rank: Optional[int] = None,
                 alpha: Optional[float] = None, targets: Optional[List[str]] = None,
                 facts: Optional[dict] = None, code: Optional[str] = None, joining: str = "open",
                 limit: int = 16, model_name: Optional[str] = None, kind: str = "session",
                 away_after: float = 20.0, gone_after: float = 90.0):
        """With no adapters the team has no adapter: it pools its devices to run the unchanged model.
        kind 'session' is a team that queries the model; 'teaching' is one that trains an adapter."""
        if kind not in KINDS:
            raise ValueError("a team is for sessions or for teaching")
        if kind == "teaching" and adapters:
            raise ValueError("a teaching team is not led with an adapter; start the training run from one instead")
        self.kind = kind
        self.ledger = None                        # a helpers.Ledger: this leader's tally of teaching points
        self._boosts: Dict[str, list] = {}        # membership -> [(points, until)] spent for a larger share
        self._ever: set = set()                   # every membership this team has given out
        self._marks: Dict[str, dict] = {}         # community ratings: each membership's mark for each question
        self._picks: Dict[str, list] = {}         # blind comparisons: each membership's (trick, pick) results
        self.on_rating: Optional[Callable[[dict], None]] = None
        self.node, self.model, self.total = node, model, total_layers
        self.model_name = _text(model_name, 60)
        self.code = tidy(code) if code else new_code()
        self.id, self._key = keys(self.code)
        self.plain = not adapters
        self.name = "no adapter" if self.plain else (_text(name, 60) or "adapter")
        self.adapter = None if self.plain else adapter_fingerprint(adapters)
        self._use = TeamUse(self.id, adapters or {}, rank or 1, alpha or 1.0, targets or [])
        self.facts = {} if self.plain else clean_facts(facts)
        self._asks: Dict[str, dict] = {}          # requests to join, by request token
        self._barred: set = set()                 # devices the leader removed; they do not come back in
        self.joining, self.limit = "open", 16
        self.set(joining, limit)
        self.begun = False
        self.ended: Optional[str] = None
        self.serves: Optional[List[int]] = None   # layers this device itself serves for the team
        self.away_after, self.gone_after = away_after, gone_after
        self._members: Dict[str, dict] = {}
        self._passes: Dict[str, str] = {}         # invitation pass -> address it was sent to
        self._strikes: Dict[str, int] = {}
        self._lock = threading.Lock()

    def attach(self) -> "TeamHost":
        current = getattr(self.node, "team_host", None)
        if current is not None and current is not self and current.ended is None:
            raise ValueError("this device already leads a team; end that one first")
        self.node.team_host = self
        for op in ("look", "join", "beat", "leave", "ask", "asked", "rate", "points", "boost", "check"):
            self.node.handlers["team_" + op] = getattr(self, "_" + op)
        self.node.notices["leads"] = lambda: [] if self.ended else [self.id]
        return self

    # -- what the leader sees and does ------------------------------------------------------------
    def use(self) -> Optional[TeamUse]:
        return self._use

    def active(self) -> bool:
        return self.begun and self.ended is None and self.kind == "session"

    def boost_of(self, token: str) -> int:
        """Teaching points in force for a member right now (each is 1% more of the team's computing power)."""
        from .helpers import MAX_BOOST
        now = time.time()
        live = [(p, until) for p, until in self._boosts.get(token, []) if until > now]
        if live:
            self._boosts[token] = live
        else:
            self._boosts.pop(token, None)
        return min(MAX_BOOST, sum(p for p, _ in live))

    def weights(self) -> Dict[str, int]:
        """{member tag: weight} for members whose share is raised right now. Everyone else's weight is 100."""
        with self._lock:
            return {tag_of(t): 100 + self.boost_of(t) for t in list(self._boosts) if t in self._members
                    and self.boost_of(t) > 0}

    def weight_of(self, tag, proof, node_id: str) -> int:
        """The weight a serving device should give work carrying this tag and proof."""
        with self._lock:
            token = next((t for t in self._members if tag_of(t) == tag), None)
            if token is None or not isinstance(proof, str) \
                    or not hmac.compare_digest(proof.encode(), proof_of(token, node_id).encode()):
                return 100
            return 100 + self.boost_of(token)

    def ratings(self) -> dict:
        """This team's community ratings as counts: marks for each question, and blind comparisons."""
        out = empty_ratings()
        with self._lock:
            for given in self._marks.values():
                for q, mark in given.items():
                    out["marks"][q][mark - 1] += 1
            for results in self._picks.values():
                for trick, pick in results:
                    out["trick" if trick else "blind"][pick] += 1
        return out

    def set(self, joining=None, limit=None) -> dict:
        """Change who can sign on, or how many devices the team holds. The leader's choice is the whole team's."""
        for key, value in check_settings(joining, limit).items():
            setattr(self, key, value)
        return {"joining": self.joining, "limit": self.limit}

    def describe(self) -> dict:
        use = self._use
        spec = {} if self.plain else {"rank": use.rank, "alpha": use.alpha, "targets": use.targets}
        return {"team": self.id, "kind": self.kind, "model": self.model, "model_name": self.model_name,
                "layers": self.total, "adapter": self.adapter, "name": self.name, **spec,
                "facts": self.facts, "begun": self.begun, "members": len(self._members) + 1,
                "joining": self.joining, "limit": self.limit, "version": __version__}

    def _sweep(self) -> None:
        cutoff = time.time() - self.gone_after
        for token in [t for t, m in self._members.items() if m["seen"] < cutoff]:
            del self._members[token]

    def roster(self) -> List[dict]:
        """One card per device in the team, the leader first."""
        now = time.time()
        cards = [{"who": "This device", "leader": True, "asks": True, "serves": self.serves, "state": "Leader"}]
        with self._lock:
            self._sweep()
            members = sorted(self._members.items(), key=lambda tm: tm[1]["joined"])
        for i, (token, m) in enumerate(members):
            state = "Away" if now - m["seen"] > self.away_after else ("In the team" if self.begun else "Ready")
            cards.append({"who": f"{m['name']} ({m['ip']})" if m.get("name") else f"Device {i + 2} ({m['ip']})",
                          "leader": False, "asks": m["asks"],
                          "serves": m["serves"], "state": state, "id": token[:8], "boost": self.boost_of(token)})
        return cards

    def remove(self, card_id: str) -> None:
        """Take a device out of the team. It drops the adapter at its next check-in and cannot rejoin."""
        with self._lock:
            token = next((t for t in self._members if isinstance(card_id, str) and len(card_id) == 8
                          and t.startswith(card_id)), None)
            if token is None:
                raise ValueError("no such device in the team")
            gone = self._members.pop(token)
            if gone["node"]:
                self._barred.add(gone["node"])

    def requests(self) -> List[dict]:
        """Requests to join that are waiting for the leader, oldest first."""
        now = time.time()
        with self._lock:
            waiting = sorted(((t, a) for t, a in self._asks.items() if a["state"] == "waiting"),
                             key=lambda ta: ta[1]["at"])
        return [{"id": t[:8], "who": f"{a['name']} ({a['ip']})" if a.get("name") else a["ip"], "asks": a["asks"],
                 "serves": a["serves"], "waited": int(now - a["at"])} for t, a in waiting]

    def _request(self, request_id: str) -> dict:
        with self._lock:
            found = next((a for t, a in self._asks.items() if isinstance(request_id, str) and len(request_id) == 8
                          and t.startswith(request_id) and a["state"] == "waiting"), None)
        if found is None:
            raise ValueError("that request is no longer waiting")
        return found

    def approve(self, request_id: str) -> None:
        """Let a device that asked to join in. Its next check brings it in and supplies the adapter."""
        asked = self._request(request_id)
        with self._lock:
            token = secrets.token_hex(16)
            self._passes[token] = "asked:" + asked["ip"]
            asked.update(state="approved", **{"pass": token})

    def decline(self, request_id: str) -> None:
        self._request(request_id)["state"] = "declined"

    def devices(self) -> List[dict]:
        """Devices in the group the leader could invite, with what each serves and whether it was invited."""
        invited = set(self._passes.values())
        with self._lock:
            here = {m["node"] for m in self._members.values() if m["node"]}
        out = []
        for peer in sorted(self.node.table.peers(), key=lambda p: p.addr):
            if peer.node_id in self.node.banned or peer.addr in self.node.banned:
                continue
            if not peer.invites and peer.node_id not in here:
                continue                           # its owner is invisible, or it shows nobody invitations
            serves = [s["layers"] for s in peer.services if s["model"] == self.model]
            out.append({"addr": peer.addr, "name": peer.name, "serves": serves[0] if serves else None,
                        "in_team": peer.node_id in here, "invited": peer.addr in invited, "busy": peer.busy})
        return out

    def invite(self, addr: str) -> None:
        """Send one device an invitation. Its owner sees it and decides. May be sent again."""
        if self.ended is not None:
            raise ValueError("this team has ended")
        if not any(p.addr == addr for p in self.node.table.peers()):
            raise ValueError("no device at that address is in this group right now")
        with self._lock:
            token = next((t for t, a in self._passes.items() if a == addr), None) or secrets.token_hex(16)
            if len(self._passes) >= 4 * MAX_MEMBERS and token not in self._passes:
                raise ValueError("too many invitations are out; end the team and start a new one")
            self._passes[token] = addr
        try:
            _send(self.node, addr, {"op": "team_invite", "team": self.id, "pass": token, "port": self.node.port})
        except RemoteError as e:
            raise ValueError(f"that device did not take the invitation: {plain(e)}") from None
        except _NET:
            raise ValueError("that device could not be reached") from None

    def missing(self, own_layers: int, for_leader: bool = False) -> List[int]:
        """Layers past the leader's own that no signed-on device is serving for the team.

        A device never routes its own questions through the layers it serves itself, so for the
        leader's own questions (for_leader=True) only other devices count.
        """
        cover = set(range(*self.serves)) if self.serves and not for_leader else set()
        now = time.time()
        with self._lock:
            for m in self._members.values():
                if m["serves"] and now - m["seen"] <= self.away_after:
                    cover.update(range(*m["serves"]))
        return [layer for layer in range(own_layers, self.total) if layer not in cover]

    def begin(self, own_layers: int, force: bool = False) -> None:
        if self.ended is not None:
            raise ValueError("this team has ended")
        if self.kind == "teaching":
            raise ValueError("a teaching team has nothing to begin: start the training run once its devices "
                             "have signed on")
        gaps = self.missing(own_layers)
        if gaps and not force:
            raise ValueError(f"no device serving layer{'s' if len(gaps) > 1 else ''} {_spans(gaps)} has signed on "
                             "to this team yet. Devices that serve layers have to sign on too.")
        self.begun = True

    def end(self, reason: str = "the leader ended the team") -> None:
        if self.ended is None:
            with self._lock:
                self._untold = set(self._members)
            self.ended, self.begun = reason, False

    def settle(self, timeout: float = 8.0) -> None:
        """After ending: give each member's next check-in time to hear it, so they are told why
        rather than left to notice the leader has gone. Worth calling before the program exits."""
        deadline = time.time() + timeout
        while getattr(self, "_untold", None) and time.time() < deadline:
            time.sleep(0.1)

    # -- requests from members ------------------------------------------------------------------------
    def _mine(self, header: dict) -> None:
        if header.get("team") != self.id:
            raise NoTeam("this device is not leading that team")
        if self.ended is not None:
            raise NoTeam("that team has ended")

    def _shown(self, header: dict, state: dict) -> bool:
        """Did this caller show the team's code, or an invitation the leader sent?"""
        key, token = header.get("key"), header.get("pass")
        if isinstance(token, str) and _PASS.match(token) and token in self._passes:
            return True
        if key is None:
            return False
        ip = str(state.get("peer_ip", ""))
        if self._strikes.get(ip, 0) >= MAX_STRIKES:
            raise Refused("too many wrong team codes from this address")
        if isinstance(key, str) and hmac.compare_digest(key.encode(), self._key.encode()):
            return True
        self._strikes[ip] = self._strikes.get(ip, 0) + 1
        raise Refused("wrong team code")

    def _look(self, ch, header: dict, tensor, state: dict) -> None:
        self._mine(header)
        if not self._shown(header, state) and self.joining == "invisible":
            raise NoTeam("that team is invisible: its code or an invitation is needed")
        ch.send({"op": "team_look_ok", **self.describe()})

    def _ask(self, ch, header: dict, tensor, state: dict) -> None:
        """Someone who can see the team asks to join. The request waits for the leader."""
        self._mine(header)
        if self.joining == "invisible" and not self._shown(header, state):
            raise NoTeam("that team is invisible: its code or an invitation is needed")
        node_id = header.get("node") if isinstance(header.get("node"), str) and _NODE.match(header["node"]) else ""
        if node_id in self._barred:
            raise Refused("the leader removed this device from the team")
        ip, now = str(state.get("peer_ip", "")), time.time()
        with self._lock:
            for old in [t for t, a in self._asks.items() if now - a["at"] > 1800]:
                del self._asks[old]
            token = next((t for t, a in self._asks.items()
                          if a["state"] == "waiting" and (a["ip"], a["node"]) == (ip, node_id)), None)
            if token is None:
                if sum(a["state"] == "waiting" for a in self._asks.values()) >= MAX_REQUESTS:
                    raise RuntimeError("too many requests are waiting for this team's leader")
                token = secrets.token_hex(16)
                from .net import clean_name
                self._asks[token] = {"node": node_id, "ip": ip, "name": clean_name(header.get("name")),
                                     "asks": header.get("asks") is True,
                                     "serves": _range(header.get("serves"), self.total), "at": now,
                                     "state": "waiting", "pass": None}
        ch.send({"op": "team_ask_ok", "request": token})

    def _asked(self, ch, header: dict, tensor, state: dict) -> None:
        if header.get("team") != self.id:
            raise NoTeam("this device is not leading that team")
        with self._lock:
            asked = self._asks.get(header.get("request") if isinstance(header.get("request"), str) else "")
        if self.ended is not None:
            ch.send({"op": "team_asked_ok", "state": "ended"})
        elif asked is None:
            ch.send({"op": "team_asked_ok", "state": "declined"})
        else:
            ch.send({"op": "team_asked_ok", "state": asked["state"],
                     **({"pass": asked["pass"]} if asked["state"] == "approved" else {})})

    def _join(self, ch, header: dict, tensor, state: dict) -> None:
        self._mine(header)
        if not self._shown(header, state) and self.joining != "open":
            raise Refused("this team is invisible: its code or an invitation is needed"
                          if self.joining == "invisible" else
                          "this team takes requests: ask to join and its leader decides")
        serves = _range(header.get("serves"), self.total)
        want = header.get("want")
        if not (isinstance(want, list) and 1 <= len(want) <= 4):
            raise ValueError("say which layers' adapters are needed")
        want = [_range(r, self.total) for r in want]
        if any(r is None for r in want):
            raise ValueError("bad layer range")
        node_id = header.get("node") if isinstance(header.get("node"), str) and _NODE.match(header["node"]) else ""
        if node_id in self._barred:
            raise Refused("the leader removed this device from the team")
        now = time.time()
        with self._lock:
            self._sweep()
            if len(self._members) + 1 >= self.limit:
                raise RuntimeError("this team is full")
            token = secrets.token_hex(16)
            self._ever.add(token)
            from .net import clean_name
            self._members[token] = {"node": node_id, "ip": str(state.get("peer_ip", "")),
                                    "name": clean_name(header.get("name")),
                                    "asks": header.get("asks") is True, "serves": serves, "joined": now, "seen": now}
        ch.send({"op": "team_join_ok", "member": token, **self.describe()}, blob=self._use.blob(want))

    def _beat(self, ch, header: dict, tensor, state: dict) -> None:
        if header.get("team") != self.id:
            raise NoTeam("this device is not leading that team")
        if self.ended is not None:
            ch.send({"op": "team_beat_ok", "ended": self.ended})
            getattr(self, "_untold", set()).discard(header.get("member"))
            return
        with self._lock:
            member = self._members.get(header.get("member") if isinstance(header.get("member"), str) else "")
            if member is not None:
                member["seen"] = time.time()
                if "serves" in header:
                    member["serves"] = _range(header.get("serves"), self.total)
            count = len(self._members) + 1
        if member is None:
            ch.send({"op": "team_beat_ok", "ended": "this device is no longer in the team"})
        else:
            ch.send({"op": "team_beat_ok", "begun": self.begun, "members": count, "joining": self.joining,
                     "weights": self.weights() if member["serves"] else {}})

    def _member(self, header: dict) -> str:
        if header.get("team") != self.id:
            raise NoTeam("this device is not leading that team")
        token = header.get("member")
        with self._lock:
            if not isinstance(token, str) or token not in self._members:
                raise Refused("only a member of the team can do that")
        return token

    def _tally(self, token: str, tickets) -> dict:
        from .helpers import BOOST_SECONDS
        now = time.time()
        with self._lock:
            in_force = self.boost_of(token)
            left = max((until - now for _, until in self._boosts.get(token, [])), default=0.0)
        return {"balance": self.ledger.balance(tickets) if self.ledger is not None else 0, "boost": in_force,
                "minutes": int(left // 60) if in_force else 0, "hour": int(BOOST_SECONDS // 60)}

    def _points(self, ch, header: dict, tensor, state: dict) -> None:
        """How many teaching points a member has with this leader, and what share they have bought."""
        token = self._member(header)
        ch.send({"op": "team_points_ok", **self._tally(token, header.get("tickets"))})

    def _boost(self, ch, header: dict, tensor, state: dict) -> None:
        """A member spends teaching points: each is 1% more of the team's computing power, for an hour."""
        from .helpers import BOOST_SECONDS, MAX_BOOST
        token = self._member(header)
        points = header.get("points")
        if self.kind != "session":
            raise ValueError("points are spent in a team that queries the model, not in a teaching team")
        if self.ledger is None:
            raise ValueError("this leader keeps no tally of teaching points")
        if isinstance(points, bool) or not isinstance(points, int) or points < 1:
            raise ValueError("spend a whole number of points, 1 or more")
        with self._lock:
            if self.boost_of(token) + points > MAX_BOOST:
                raise ValueError(f"at most {MAX_BOOST} points can be in force at once (a doubled share); "
                                 f"{self.boost_of(token)} are in force now")
        self.ledger.spend(header.get("tickets"), points)
        with self._lock:
            self._boosts.setdefault(token, []).append((points, time.time() + BOOST_SECONDS))
        ch.send({"op": "team_boost_ok", **self._tally(token, header.get("tickets"))})

    def _check(self, ch, header: dict, tensor, state: dict) -> None:
        """A serving device in the team asks what weight a piece of work carries."""
        if header.get("team") != self.id:
            raise NoTeam("this device is not leading that team")
        node_id = header.get("node") if isinstance(header.get("node"), str) else ""
        ch.send({"op": "team_check_ok", "weight": self.weight_of(header.get("tag"), header.get("proof"), node_id)})

    def _rate(self, ch, header: dict, tensor, state: dict) -> None:
        """A member's part in the adapter's community rating: marks from 1 to 5 for any of the questions,
        or the result of one blind comparison. Taken while they are in the team and after it, for as long
        as this device still answers for it. Marks replace that member's earlier ones; comparisons add up."""
        if header.get("team") != self.id:
            raise NoTeam("this device is not leading that team")
        token, marks, pick = header.get("member"), header.get("marks"), header.get("blind")
        if self.plain:
            raise ValueError("this team has no adapter to rate")
        if token not in self._ever:
            raise Refused("only people who were in the team can rate its adapter")
        if marks is None and pick is None:
            raise ValueError("give a mark from 1 to 5 for a question, or the result of a blind comparison")
        if marks is not None:
            if not isinstance(marks, dict) or not marks or any(
                    q not in QUESTIONS or isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= 5
                    for q, m in marks.items()):
                raise ValueError("a rating is a whole number from 1 to 5 for one of: " + ", ".join(QUESTIONS))
        if pick is not None and pick not in PICKS:
            raise ValueError("a blind comparison ends in one of: " + ", ".join(PICKS))
        with self._lock:
            if marks is not None:
                self._marks.setdefault(token, {}).update(marks)
            if pick is not None and len(self._picks.setdefault(token, [])) < MAX_PICKS:
                self._picks[token].append((header.get("trick") is True, pick))
        if self.on_rating is not None:
            try:
                self.on_rating(self.ratings())
            except Exception:
                pass
        ch.send({"op": "team_rate_ok"})

    def _leave(self, ch, header: dict, tensor, state: dict) -> None:
        if header.get("team") != self.id:
            raise NoTeam("this device is not leading that team")
        with self._lock:
            self._members.pop(header.get("member") if isinstance(header.get("member"), str) else "", None)
        ch.send({"op": "team_leave_ok"})


def _spans(layers: List[int]) -> str:
    """[4, 5, 6, 9] as '4-6, 9'."""
    out, i = [], 0
    while i < len(layers):
        j = i
        while j + 1 < len(layers) and layers[j + 1] == layers[j] + 1:
            j += 1
        out.append(str(layers[i]) if i == j else f"{layers[i]}-{layers[j]}")
        i = j + 1
    return ", ".join(out)


def _described(reply: dict, team_id: str) -> dict:
    """A leader's description of a team, checked field by field before anything is shown or used."""
    if reply.get("team") != team_id:
        raise ProtocolError("the leader answered for a different team")
    model, layers, adapter = reply.get("model"), reply.get("layers"), reply.get("adapter")
    if not (isinstance(model, str) and _NODE.match(model) and isinstance(layers, int) and not isinstance(layers, bool)
            and 1 <= layers <= 4096 and (adapter is None or isinstance(adapter, str) and _ID.match(adapter))):
        raise ProtocolError("the leader's description of the team could not be read")
    spec = {}
    if adapter is not None:
        try:
            rank, alpha, targets = check_spec(reply.get("rank"), reply.get("alpha"), reply.get("targets"))
        except ValueError:
            raise ProtocolError("the leader's description of the adapter could not be read") from None
        spec = {"rank": rank, "alpha": alpha, "targets": targets}

    def whole(value, default):
        return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 4096 else default
    return {"team": team_id, "kind": reply.get("kind") if reply.get("kind") in KINDS else "session",
            "model": model, "model_name": _text(reply.get("model_name"), 60), "layers": layers,
            "adapter": adapter, "name": "no adapter" if adapter is None else (_text(reply.get("name"), 60) or "adapter"),
            **spec, "facts": clean_facts(reply.get("facts")) if adapter is not None else {},
            "begun": reply.get("begun") is True, "version": _text(reply.get("version"), 20),
            "members": whole(reply.get("members"), 0), "limit": whole(reply.get("limit"), MAX_MEMBERS),
            "joining": reply.get("joining") if reply.get("joining") in JOINING else "invisible"}


def teams_in_group(node, model: Optional[str] = None, kind: Optional[str] = None) -> List[dict]:
    """The teams anyone in this group can see right now: open ones and ones that take requests, as
    their leaders describe them. Invisible teams are not listed. Each entry says where its leader is."""
    out = []
    for peer in node.table.peers():
        if peer.node_id in node.banned or peer.addr in node.banned:
            continue
        for team_id in peer.leads:
            try:
                reply, _, _ = _send(node, peer.addr, {"op": "team_look", "team": team_id})
                desc = _described(reply, team_id)
            except _NET:
                continue                           # private, ended, or gone
            if (model is None or desc["model"] == model) and (kind is None or desc["kind"] == kind):
                out.append({**desc, "from": peer.addr})
    return sorted(out, key=lambda d: (d["joining"] != "open", not d["begun"], -d["members"]))


def open_teams(node, model: str, kind: Optional[str] = "session") -> List[dict]:
    """Teams for this model that can be signed on to right now with no code and no invitation, best first.
    Session teams unless told otherwise: that is what quick match is for."""
    return [t for t in teams_in_group(node, model, kind) if t["joining"] == "open" and t["members"] < t["limit"]]


class Invitations:
    """Invitations this device has been sent. They wait here for its owner; nothing signs the device on."""

    def __init__(self, node, limit: int = 16, ttl: float = 3600.0):
        self.node, self.limit, self.ttl = node, limit, ttl
        self.closed = False                        # the owner chose to be invisible: no invitations are taken
        self._held: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def attach(self) -> "Invitations":
        self.node.handlers["team_invite"] = self._invite
        return self

    def _invite(self, ch, header: dict, tensor, state: dict) -> None:
        team_id, token, port = header.get("team"), header.get("pass"), header.get("port")
        if self.closed:
            raise Refused("that device's owner is not taking invitations")
        if not (isinstance(team_id, str) and _ID.match(team_id) and isinstance(token, str) and _PASS.match(token)
                and isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535):
            raise ValueError("that invitation could not be read")
        with self._lock:
            cutoff = time.time() - self.ttl
            for old in [t for t, e in self._held.items() if e["at"] < cutoff]:
                del self._held[old]
            if team_id not in self._held and len(self._held) >= self.limit:
                raise RuntimeError("this device is holding as many invitations as it will take")
            self._held[team_id] = {"team": team_id, "leader": f"{state.get('peer_ip', '')}:{port}",
                                   "pass": token, "at": time.time()}
        ch.send({"op": "team_invite_ok"})

    def get(self, team_id: str) -> dict:
        with self._lock:
            held = self._held.get(team_id)
        if held is None:
            raise NoTeam("there is no invitation to that team here")
        return dict(held)

    def drop(self, team_id: str) -> None:
        with self._lock:
            self._held.pop(team_id, None)

    def list(self, model: Optional[str] = None) -> List[dict]:
        """Each waiting invitation with its leader's description of the team. Ended teams fall off the list."""
        with self._lock:
            held = [dict(e) for e in self._held.values()]
        out = []
        for e in sorted(held, key=lambda e: -e["at"]):
            try:
                reply, _, _ = _send(self.node, e["leader"], {"op": "team_look", "team": e["team"], "pass": e["pass"]})
                desc = _described(reply, e["team"])
            except _NET:
                self.drop(e["team"])
                continue
            if model is None or desc["model"] == model:
                out.append({**desc, "from": e["leader"]})
        return out


class TeamLink:
    """A member's side of a team: sign on, keep in touch with the leader, leave.

    Three ways in, matching the leader's choice of who can sign on: the team's code; an
    invitation the leader sent this device (team, leader and invitation given); or, for an
    open team seen in the group's list, just the team and where its leader is.
    """
    role = "member"

    def __init__(self, node, *, model: str, code: Optional[str] = None, team: Optional[str] = None,
                 leader: Optional[str] = None, invitation: Optional[str] = None, asks: bool = True,
                 serves: Optional[List[int]] = None, want: Optional[List[List[int]]] = None, beat: float = 5.0,
                 patience: int = 4, find_wait: float = 15.0, on_end: Optional[Callable[[str], None]] = None,
                 called: str = ""):
        self.node, self.model, self.called = node, model, called
        self.code = self._key = None
        if code is not None:
            self.code = tidy(code)
            self.id, self._key = keys(self.code)
        elif isinstance(team, str) and _ID.match(team):
            self.id = team
        else:
            raise ValueError("give a team code, or a team from the list")
        self.invitation = invitation
        self.asks, self.serves, self.want = asks, serves, want
        self.beat, self.patience, self.find_wait, self.on_end = beat, patience, find_wait, on_end
        self.leader: Optional[str] = leader
        self.member: Optional[str] = None
        self.weights: Dict[str, int] = {}          # for a serving device: members whose share is raised
        self._checked: Dict[tuple, tuple] = {}
        self.rated: Optional[dict] = None          # this person's own marks, by question
        self.desc: dict = {}
        self.begun = False
        self.ended: Optional[str] = None
        self.members = 0
        self._use: Optional[TeamUse] = None
        self._stop = threading.Event()

    @property
    def name(self) -> str:
        return self.desc.get("name", "")

    def _shown(self) -> dict:
        out = {}
        if self._key is not None:
            out["key"] = self._key
        if self.invitation is not None:
            out["pass"] = self.invitation
        return out

    def _find(self) -> str:
        deadline = time.time() + self.find_wait
        while True:
            for peer in self.node.table.peers():
                if self.id in peer.leads and peer.node_id not in self.node.banned \
                        and peer.addr not in self.node.banned:
                    return peer.addr
            if time.time() >= deadline:
                raise NoTeam("no team with that code can be found in this group right now")
            self.node.refresh(1)
            time.sleep(0.2)

    def look(self) -> dict:
        """What the leader says about the team and its adapter. Looking signs nobody on."""
        if self.leader is None:
            self.leader = self._find()
        try:
            reply, _, _ = _send(self.node, self.leader, {"op": "team_look", "team": self.id, **self._shown()})
        except RemoteError as e:
            raise (PermissionError if "wrong team code" in str(e) else NoTeam)(plain(e)) from None
        except _NET:
            raise NoTeam("the team's leader could not be reached") from None
        self.desc = _described(reply, self.id)
        return self.desc

    def ask(self) -> None:
        """Ask the leader of a team this device can see to let it in. The answer is fetched with asked()."""
        desc = self.look()
        if desc["model"] != self.model:
            raise ValueError(f"that team is for a different model ({desc['model']}); the model here is {self.model}")
        try:
            reply, _, _ = _send(self.node, self.leader, {"op": "team_ask", "team": self.id, "asks": self.asks,
                                                         "node": self.node.node_id, "serves": self.serves,
                                                         "name": self.called,
                                                         **self._shown()})
        except RemoteError as e:
            raise (PermissionError if str(e).startswith("Refused") else NoTeam)(plain(e)) from None
        self.request = reply.get("request") if isinstance(reply.get("request"), str) else None

    def asked(self) -> str:
        """'waiting', 'approved', 'declined' or 'ended'. Once approved, join() gets in."""
        try:
            reply, _, _ = _send(self.node, self.leader, {"op": "team_asked", "team": self.id,
                                                         "request": getattr(self, "request", None)})
        except RemoteError:
            return "ended"
        state = reply.get("state")
        if state == "approved" and isinstance(reply.get("pass"), str) and _PASS.match(reply["pass"]):
            self.invitation = reply["pass"]
            return "approved"
        return state if state in ("waiting", "declined", "ended") else "declined"

    def join(self) -> "TeamLink":
        """Sign on. The adapter arrives from the leader and is held in memory, never written down."""
        if self.member is not None:
            return self
        desc = self.look()
        if desc["model"] != self.model:
            raise ValueError(f"that team's adapter is for a different model ({desc['model']}); "
                             f"the model here is {self.model}")
        want = self.want or [[0, desc["layers"]]]
        try:
            ch = self.node.dial(self.leader, timeout=30.0)
        except _NET:
            raise NoTeam("the team's leader could not be reached") from None
        try:
            ch.sock.settimeout(300.0)
            reply, _, blob = ch.call({"op": "team_join", "team": self.id, "node": self.node.node_id,
                                      "asks": self.asks, "serves": self.serves, "want": want,
                                      "name": self.called, **self._shown()})
            try:
                ch.send({"op": "bye"})
            except _NET:
                pass
        except RemoteError as e:
            raise (PermissionError if str(e).startswith("Refused") else NoTeam)(plain(e)) from None
        finally:
            ch.close()
        self.desc = _described(reply, self.id)
        member = reply.get("member")
        if not (isinstance(member, str) and len(member) == 32):
            raise ProtocolError("the leader's reply could not be read")
        if self.desc["adapter"] is None:           # a team with no adapter: nothing to receive
            self._use = TeamUse(self.id, {}, 1, 1.0, [], member=member)
            self.member, self.begun, self.members = member, self.desc["begun"], self.desc["members"]
            threading.Thread(target=self._loop, daemon=True).start()
            return self
        if not blob or len(blob) > MAX_ADAPTER_BYTES:
            raise ProtocolError("the leader's reply could not be read")
        try:
            adapters = st.loads(blob)
        except st.FormatError as e:
            raise ProtocolError(f"the adapter from the leader could not be read: {e}") from None
        rank = self.desc["rank"]
        for key, value in adapters.items():
            m = _KEY.match(key)
            if not m or value.dtype.kind != "f" or value.ndim != 2 or not np.isfinite(value).all() \
                    or value.shape[0 if m.group(3) == "A" else 1] != rank \
                    or not any(a <= int(m.group(1)) < b for a, b in want):
                raise ProtocolError("the adapter from the leader is not what was asked for")
        if not adapters or any(k[:-2] + (".B" if k.endswith(".A") else ".A") not in adapters for k in adapters):
            raise ProtocolError("the adapter from the leader is incomplete")
        self._use = TeamUse(self.id, {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in adapters.items()},
                            rank, self.desc["alpha"], self.desc["targets"], member=member)
        self.member, self.begun, self.members = member, self.desc["begun"], self.desc["members"]
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def use(self) -> Optional[TeamUse]:
        return self._use

    @property
    def kind(self) -> str:
        return self.desc.get("kind", "session")

    def active(self) -> bool:
        return self.begun and self.ended is None and self._use is not None and self.kind == "session"

    def weight_of(self, tag, proof) -> int:
        """For a serving device: the weight to give a piece of the team's work. 100 unless its member has
        spent teaching points, which the leader confirms (the answer is kept for a few minutes)."""
        raised = self.weights.get(tag) if isinstance(tag, str) else None
        if raised is None or not isinstance(proof, str):
            return 100
        known = self._checked.get((tag, proof))
        if known is None or time.time() - known[1] > 300:
            try:
                reply, _, _ = _send(self.node, self.leader, {"op": "team_check", "team": self.id, "tag": tag,
                                                             "proof": proof, "node": self.node.node_id}, timeout=10.0)
                known = (reply.get("weight") if isinstance(reply.get("weight"), int) else 100, time.time())
            except _NET:
                known = (100, time.time())
            if len(self._checked) > 256:
                self._checked.clear()
            self._checked[(tag, proof)] = known
        return max(100, min(known[0], raised))

    def _points(self, op: str, tickets: list, **more) -> dict:
        if self.member is None or self.ended is not None:
            raise ValueError("only a member of the team can do that")
        try:
            reply, _, _ = _send(self.node, self.leader, {"op": op, "team": self.id, "member": self.member,
                                                         "tickets": tickets, **more}, timeout=20.0)
        except RemoteError as e:
            raise ValueError(plain(e)) from None
        except _NET:
            raise ValueError("the team's leader could not be reached") from None
        whole = lambda v: v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0   # noqa: E731
        return {k: whole(reply.get(k)) for k in ("balance", "boost", "minutes", "hour")}

    def points(self, tickets: list) -> dict:
        """This device's teaching points with the team's leader, and any share it has bought with them."""
        return self._points("team_points", tickets)

    def boost(self, tickets: list, points: int) -> dict:
        """Spend teaching points: each is 1% more of the team's computing power for an hour."""
        return self._points("team_boost", tickets, points=points)

    def rate(self, marks: Optional[dict] = None, blind: Optional[str] = None, trick: bool = False) -> None:
        """This person's part in the adapter's community rating: marks from 1 to 5 for any of QUESTIONS,
        or the result of a blind comparison. Works while in the team and after it, for as long as the
        leader's device can still be reached."""
        if self.member is None:
            raise ValueError("only people who were in the team can rate its adapter")
        try:
            _send(self.node, self.leader, {"op": "team_rate", "team": self.id, "member": self.member,
                                           "marks": marks, "blind": blind, "trick": trick}, timeout=10.0)
        except RemoteError as e:
            raise ValueError(plain(e)) from None
        except _NET:
            raise ValueError("the leader's device can no longer be reached, so the rating could not be given") \
                from None
        if marks:
            self.rated = {**(self.rated or {}), **marks}

    def _loop(self) -> None:
        misses = 0
        while not self._stop.wait(self.beat):
            try:
                reply, _, _ = _send(self.node, self.leader, {"op": "team_beat", "team": self.id,
                                                             "member": self.member, "serves": self.serves})
            except _NET:
                misses += 1
                if misses >= self.patience:
                    self._finish("the leader can no longer be reached")
                    return
                continue
            misses = 0
            if reply.get("ended"):
                self._finish(_text(reply["ended"], 200) or "the team has ended")
                return
            self.begun = reply.get("begun") is True
            if isinstance(reply.get("members"), int):
                self.members = reply["members"]
            if reply.get("joining") in JOINING:
                self.desc["joining"] = reply["joining"]
            raised = reply.get("weights")
            self.weights = {t: w for t, w in raised.items() if isinstance(t, str) and isinstance(w, int)
                            and 100 <= w <= 1000} if isinstance(raised, dict) else {}

    def _finish(self, reason: str) -> None:
        if self.ended is not None:
            return
        self.ended, self.begun = reason, False
        self._use = None                           # the adapter was only ever held in memory; let it go
        self._stop.set()
        if self.on_end is not None:
            try:
                self.on_end(reason)
            except Exception:
                pass

    def leave(self) -> None:
        self._stop.set()
        if self.member is not None and self.ended is None:
            try:
                _send(self.node, self.leader, {"op": "team_leave", "team": self.id, "member": self.member},
                      timeout=10.0)
            except _NET:
                pass
        self._finish("you left the team")


class TeamService:
    """Keeps a device that only serves layers signed on to the teams its owner chose.

    The owner names teams by their code, accepts an invitation, or says the device may serve
    any open team for its model. Signing on is what makes the device apply a team's adapter,
    and then only to that team's work. When a team ends, its adapter is dropped.
    """

    def __init__(self, node, server, *, codes: Sequence[str] = (), any_open: bool = False,
                 any_invited: bool = False, say: Optional[Callable[[str], None]] = None):
        self.node, self.server, self.any_open = node, server, any_open
        # With any_invited the owner has said yes in advance: an invitation from a leader signs the device on.
        self.inbox = Invitations(node).attach() if any_invited else None
        if any_invited:
            node.presence = lambda: {"invites": True, "busy": bool(self.links)}
        self.say = say or (lambda text: None)
        self._waiting = {tidy(c) for c in codes}   # codes whose team has not been found yet
        self.links: Dict[str, TeamLink] = {}
        self._refused: set = set()                 # open teams that did not fit; not tried again

    def join(self, **how) -> TeamLink:
        """Sign the device on: code=..., or team= and leader= (with invitation= if there is one)."""
        s = self.server
        wait = how.pop("find_wait", 5.0)
        link = TeamLink(self.node, model=s.fingerprint, asks=False, serves=[s.start, s.end],
                        want=[[s.start, s.end]], find_wait=wait, **how)
        if link.id in self.links:
            return self.links[link.id]
        link.on_end = lambda reason: self._over(link, reason)
        link.join()
        use = link.use()
        try:
            s.join_team(link.id, use.blob([[s.start, s.end]]), use.rank, use.alpha, use.targets,
                        weight_of=link.weight_of)
        except (ValueError, RuntimeError) as e:
            link.on_end = None
            link.leave()
            raise ValueError(f"this device could not take that team's adapter: {e}") from None
        self.links[link.id] = link
        what = "serve that team, with no adapter" if use.plain else "apply that adapter for the team's work only"
        self.say(f"Signed on to the team using \"{link.name}\". Layers {s.start}-{s.end - 1} here {what}.\n  "
                 + "\n  ".join(fact_lines(link.desc)))
        return link

    def _over(self, link: TeamLink, reason: str) -> None:
        self.server.leave_team(link.id)
        self.links.pop(link.id, None)
        self.say(f"The team using \"{link.name}\" is over ({reason}). Its adapter has been dropped here.")

    def leave(self, team_id: str) -> None:
        link = self.links.get(team_id)
        if link is not None:
            link.leave()

    def leave_all(self) -> None:
        for link in list(self.links.values()):
            link.leave()

    def tick(self) -> None:
        """Call every few seconds: finds teams whose code was given, and open teams if those were asked for."""
        for code in list(self._waiting):
            try:
                self.join(code=code, find_wait=0.0)
                self._waiting.discard(code)
            except NoTeam:
                pass                               # not started yet, or out of reach: keep looking
            except (PermissionError, ValueError, ProtocolError) as e:
                self._waiting.discard(code)
                self.say(f"Team code {code}: {e}")
        for held in self.inbox.list(self.server.fingerprint) if self.inbox is not None else []:
            if held["team"] not in self.links and held["team"] not in self._refused:
                try:
                    pass_ = self.inbox.get(held["team"])
                    self.join(team=held["team"], leader=pass_["leader"], invitation=pass_["pass"])
                except (NoTeam, PermissionError, ValueError, ProtocolError) as e:
                    self._refused.add(held["team"])
                    self.say(f"Invitation to the team using \"{held['name']}\": {e}")
        if self.any_open:
            for found in open_teams(self.node, self.server.fingerprint, kind=None):
                if found["team"] in self.links or found["team"] in self._refused:
                    continue
                try:
                    self.join(team=found["team"], leader=found["from"])
                except (NoTeam, PermissionError, ValueError, ProtocolError) as e:
                    self._refused.add(found["team"])
                    self.say(f"Open team using \"{found['name']}\": {e}")

    def report(self) -> List[dict]:
        return [{"team": l.id, "name": l.name, "begun": l.begun, "members": l.members}
                for l in self.links.values()]


class TeamDesk:
    """Everything one device does about teams, behind a few calls a screen can press.

    The desktop gateway and the phone app both use this, so a team behaves the same on each.
    `ends` is the device's own first layers; `service` its served layers, if it serves any;
    `shapes(targets)` gives the model's adapter shapes; `whole` says the device needs a team's
    whole adapter (it rechecks other devices' work itself).
    """

    def __init__(self, node, ends, *, layer_prefix, shapes: Callable[[List[str]], dict],
                 work_dir: Optional[str] = None, service=None, whole: bool = False, find_wait: float = 15.0,
                 model_name: Optional[str] = None, helped=None):
        self.helped = helped                       # a helpers.HelpLog: this device's tickets from teaching
        self.points: Optional[dict] = None         # what the leader last said about this device's points
        self.node, self.ends, self.service = node, ends, service
        self.layer_prefix, self.shapes, self.work_dir = layer_prefix, shapes, work_dir
        self.whole, self.find_wait, self.model_name = whole, find_wait, model_name
        self.team = None                           # a TeamHost (this device leads) or a TeamLink (it signed on)
        self.pending: Optional[dict] = None        # a request to join that is waiting for a leader
        self.invitations = Invitations(node).attach()
        self.hidden = load_settings(work_dir)["hidden"]
        self.name = load_settings(work_dir)["name"]     # what this device's owner is called on team screens
        self.invitations.closed = self.hidden
        node.presence = lambda: {"invites": not self.hidden, "name": self.name,
                                 "busy": self.team is not None and self.team.ended is None}

    def leading(self) -> bool:
        return self.team is not None and self.team.role == "leader" and self.team.ended is None

    def settings(self) -> dict:
        """The leader's choices (theirs to change while leading), and whether this device can be invited."""
        kept = load_settings(self.work_dir)
        if self.leading():
            kept.update(joining=self.team.joining, limit=self.team.limit)
        return {**kept, "hidden": self.hidden, "name": self.name}

    def use(self) -> Optional[TeamUse]:
        """The team to answer in right now, once it has begun. Otherwise none: the unchanged model, any peer."""
        return self.team.use() if self.team is not None and self.team.active() else None

    def waiting(self) -> Optional[dict]:
        p = self.pending
        return None if p is None else {"team": p["link"].id, "name": p["link"].name, "state": p["state"]}

    def report(self) -> Optional[dict]:
        t = self.team
        if t is None:
            return None
        leads = t.role == "leader"
        desc = t.describe() if leads else t.desc
        own = self.ends.own_layers
        out = {"role": t.role, "team": t.id, "kind": t.kind, "code": t.code, "name": t.name,
               "adapter": desc.get("adapter"), "model": desc.get("model"), "begun": t.begun,
               "active": t.active(), "ended": t.ended,
               "points": None if leads else self.points,
               "can_rate": not leads and desc.get("adapter") is not None and t.member is not None,
               "rated": None if leads else t.rated, "community": tally_ratings(t.ratings()) if leads else None,
               "questions": QUESTIONS,
               "joining": desc.get("joining"), "limit": desc.get("limit"),
               "members": desc.get("members") if leads else t.members,
               "lines": fact_lines({**desc, "fits": True}),
               "serving": bool(self.service is not None and t.id in self.service.teams())}
        if leads:
            out.update(roster=t.roster(), requests=t.requests(), missing=t.missing(own),
                       missing_for_you=t.missing(own, for_leader=True))
        return out

    def _free(self) -> None:
        if self.team is not None and self.team.ended is None:
            raise ValueError("this device is already in a team; leave it first")
        self.pending = None                        # a new choice replaces a request that was still waiting

    def _over(self, reason: str) -> None:
        """A team this device signed on to has ended: stop applying its adapter anywhere on this device."""
        if self.team is not None and self.service is not None:
            self.service.leave_team(self.team.id)

    def _held(self, fingerprint: Optional[str]) -> Optional[str]:
        """The name this device holds an adapter under, if it has those exact numbers already."""
        from .control import past_runs
        if not fingerprint or not self.work_dir:
            return None
        prefix = None
        for name in past_runs(self.work_dir):
            folder = os.path.join(self.work_dir, "adapters", name)
            try:
                with open(os.path.join(folder, FACTS)) as f:
                    facts = json.load(f)
                if not isinstance(facts, dict):
                    continue
                if not isinstance(facts.get("fingerprint"), str):      # worked out once, then remembered
                    prefix = prefix or (self.layer_prefix() if callable(self.layer_prefix) else self.layer_prefix)
                    facts["fingerprint"] = adapter_fingerprint(read_adapter(folder, prefix)["adapters"])
                    with open(os.path.join(folder, FACTS), "w") as f:
                        json.dump(facts, f, indent=2)
                if facts["fingerprint"] == fingerprint:
                    return name
            except (OSError, ValueError):
                continue
        return None

    def _rated(self, adapter_dir: str, team_id: str, counts: dict) -> None:
        """Keep a team's marks with the adapter they are about, so its community rating travels with it."""
        path = os.path.join(adapter_dir, FACTS)
        try:
            with open(path) as f:
                facts = json.load(f)
        except (OSError, ValueError):
            facts = {}
        by_team = facts.get("community_by_team") if isinstance(facts.get("community_by_team"), dict) else {}
        by_team[team_id] = clean_ratings(counts)
        by_team = dict(list(by_team.items())[-200:])
        facts.update(community_by_team=by_team, community=tally_ratings(add_ratings(*by_team.values())))
        with open(path, "w") as f:
            json.dump(facts, f, indent=2)

    def blind_put(self, with_adapter: str, unchanged: str, trick: bool) -> dict:
        """Hold the two answers of a blind comparison and hand them back as A and B, in an order only
        this device knows, until the person says which was right."""
        import random
        first = random.random() < 0.5
        self._blind = {"id": secrets.token_hex(4), "adapter": "a" if first else "b", "trick": trick,
                       "team": self.team.id if self.team is not None else None}
        return {"id": self._blind["id"], "a": with_adapter if first else unchanged,
                "b": unchanged if first else with_adapter}

    def blind_pick(self, comparison: str, choice: str) -> dict:
        """The person's answer to a blind comparison: 'a', 'b', 'both' or 'neither'. It is sent to the
        team's leader towards the adapter's community rating, and which answer was which is revealed."""
        held = getattr(self, "_blind", None)
        t = self.team
        if held is None or held["id"] != comparison or t is None or t.role == "leader" or held["team"] != t.id:
            raise ValueError("there is no blind comparison waiting for that answer")
        if choice not in ("a", "b", "both", "neither"):
            raise ValueError("say which answer was right: a, b, both or neither")
        pick = "same" if choice == "both" else "neither" if choice == "neither" else \
            "adapter" if choice == held["adapter"] else "unchanged"
        t.rate(blind=pick, trick=held["trick"])
        self._blind = None
        return {"adapter_was": held["adapter"], "picked": pick}

    def lead(self, adapter_dir: Optional[str] = None, name: Optional[str] = None, joining=None, limit=None,
             code=None, serve: bool = True, kind: str = "session") -> TeamHost:
        """Create a team. This device is its leader. With no adapter the team pools its devices
        for the unchanged model; with one, held on this device, the team's answers use it."""
        self._free()
        found = {"adapters": None, "rank": None, "alpha": None, "targets": None, "facts": None}
        if adapter_dir is not None:
            prefix = self.layer_prefix() if callable(self.layer_prefix) else self.layer_prefix
            found = fitted(adapter_dir, prefix, self.ends.fingerprint, self.shapes)
        kept = load_settings(self.work_dir)
        kept.update(check_settings(joining, limit))
        host = TeamHost(self.node, model=self.ends.fingerprint, total_layers=self.ends.num_layers, name=name,
                        adapters=found["adapters"], rank=found["rank"], alpha=found["alpha"],
                        targets=found["targets"], facts=found["facts"], code=code, joining=kept["joining"],
                        limit=kept["limit"], model_name=self.model_name, kind=kind)
        if adapter_dir is not None:
            host.on_rating = lambda counts: self._rated(adapter_dir, host.id, counts)
        from .helpers import Ledger
        host.ledger = Ledger(self.work_dir)        # teaching points earned in this leader's training runs
        use = host.use()
        if not use.plain:
            self.ends.fit_team(use)
        s = self.service
        if s is not None and serve:
            s.join_team(host.id, use.blob([[s.start, s.end]]), use.rank, use.alpha, use.targets,
                        weight_of=lambda tag, proof: host.weight_of(tag, proof, self.node.node_id))
            host.serves = [s.start, s.end]
        try:
            host.attach()
        except ValueError:
            if s is not None:
                s.leave_team(host.id)
            raise
        save_settings(self.work_dir, {**kept, "hidden": self.hidden, "name": self.name})
        self.team = host
        return host

    def _link(self, req: dict, serve: bool) -> TeamLink:
        s, own, total = self.service, self.ends.own_layers, self.ends.num_layers
        want = [[0, total]] if self.whole else [[0, own]] + ([[s.start, s.end]] if serve else [])
        common = dict(model=self.ends.fingerprint, asks=True, serves=[s.start, s.end] if serve else None,
                      want=want, find_wait=self.find_wait, on_end=self._over, called=self.name)
        if req.get("code") is not None:
            return TeamLink(self.node, code=req["code"], **common)
        team = req.get("team")
        try:                                       # an invitation, if the leader sent this device one
            held = self.invitations.get(team if isinstance(team, str) else "")
            return TeamLink(self.node, team=team, leader=held["leader"], invitation=held["pass"], **common)
        except NoTeam:
            return TeamLink(self.node, team=team, **common)

    def look(self, req: dict) -> dict:
        """What a team's leader says about it: its model and adapter first. Looking signs nobody on."""
        desc = self._link(req, False).look()
        desc["fits"] = desc["model"] == self.ends.fingerprint
        desc["held_as"] = self._held(desc["adapter"])
        return {**desc, "lines": fact_lines(desc)}

    def join(self, req: dict) -> Optional[TeamLink]:
        """Sign this device on to a team: by its code, by saying yes to an invitation, or from the list.

        A listed team that takes requests is asked instead, and this returns None: when its leader
        approves, the device is signed on and supplied with the adapter without anything more to press.
        """
        self._free()
        serve = self.service is not None and req.get("serve", True) is not False
        link = self._link(req, serve)
        desc = link.look()
        if desc["joining"] == "ask" and link._key is None and link.invitation is None:
            link.ask()
            self.pending = waiting = {"link": link, "serve": serve, "state": "waiting", "since": time.time()}
            threading.Thread(target=self._wait, args=(waiting,), daemon=True).start()
            return None
        return self._enter(link, serve)

    def _wait(self, waiting: dict, every: float = 2.0, patience: float = 1800.0) -> None:
        link = waiting["link"]
        while self.pending is waiting and time.time() - waiting["since"] < patience:
            time.sleep(every)
            try:
                state = link.asked()
            except _NET:
                continue
            if self.pending is not waiting:
                return
            if state == "approved":
                try:
                    self._enter(link, waiting["serve"])
                    self.pending = None
                except (ValueError, NoTeam, PermissionError, RuntimeError, ProtocolError) as e:
                    waiting["state"] = f"could not sign on: {e}"
                return
            if state != "waiting":
                waiting["state"] = "the leader declined" if state == "declined" else "the team has ended"
                return
        if self.pending is waiting:
            waiting["state"] = "the leader did not answer"

    def _enter(self, link: TeamLink, serve: bool) -> TeamLink:
        link.join()
        self.team = link
        try:
            use = link.use()
            if not use.plain:
                self.ends.fit_team(use)
                if self.whole:                     # rechecking a team's work here needs its whole adapter to fit
                    check_adapters(use.adapters, self.shapes(use.targets), use.rank)
            if serve:
                s = self.service
                s.join_team(link.id, use.blob([[s.start, s.end]]), use.rank, use.alpha, use.targets,
                            weight_of=link.weight_of)
        except (ValueError, RuntimeError) as e:
            link.leave()
            raise ValueError(f"signed off again: {e}") from None
        self.points = None
        self.invitations.drop(link.id)
        return link

    def leave(self) -> None:
        """Leave the team (a member), or end it for everyone (its leader). The unchanged model is back."""
        self.pending = None
        t = self.team
        if t is None or t.ended is not None:
            return
        if t.role == "leader":
            t.end()
            if self.service is not None:
                self.service.leave_team(t.id)
        else:
            t.leave()

    def control(self, what: str, req: dict) -> dict:
        """One button press from a front end. Raises ValueError with a message to show."""
        try:
            return self._control(what, req)
        except (NoTeam, PermissionError, RuntimeError, ProtocolError) as e:
            raise ValueError(str(e)) from None

    def _control(self, what: str, req: dict) -> dict:
        from .control import _NAME, past_runs
        t, fp = self.team, self.ends.fingerprint
        if what == "create":
            name = req.get("name")
            kind = req.get("kind", "session")
            if name is None:                       # no adapter: a team that pools its devices, or a teaching team
                self.lead(None, None, req.get("joining"), req.get("limit"), req.get("code"),
                          req.get("serve", True) is not False, kind)
            else:
                if not self.work_dir or not isinstance(name, str) or not _NAME.match(name) \
                        or name not in past_runs(self.work_dir):
                    raise ValueError("no adapter by that name")
                self.lead(os.path.join(self.work_dir, "adapters", name), name, req.get("joining"),
                          req.get("limit"), req.get("code"), req.get("serve", True) is not False)
        elif what == "list":
            self.node.refresh(1)

            def shown(found):
                out = []
                for d in found:
                    d = {**d, "fits": True, "held_as": self._held(d["adapter"])}
                    out.append({**d, "lines": fact_lines(d)})
                return out
            found = teams_in_group(self.node, fp)
            return {"teams": shown(d for d in found if d["kind"] == "session"),           # the two lobbies
                    "teaching": shown(d for d in found if d["kind"] == "teaching"),
                    "invitations": shown(self.invitations.list(fp))}
        elif what == "look":
            return {"look": self.look(req)}
        elif what in ("join", "ask"):
            self.join(req)
        elif what == "cancel":
            self.pending = None
        elif what == "quick":                      # quick match: any open team for this model
            self._free()
            self.node.refresh(1)
            found = open_teams(self.node, fp)
            if found:
                self.join({"team": found[0]["team"], "serve": req.get("serve", True)})
            elif "name" in req:
                return self._control("create", {**req, "joining": "open"})
            else:
                raise ValueError("no open team for this model in the group right now. Lead one yourself, "
                                 "or ask a leader for their code.")
        elif what == "settings":
            change = check_settings(req.get("joining"), req.get("limit"), req.get("hidden"), req.get("name"))
            if "hidden" in change:                 # this device's own: whether leaders see it to invite
                self.hidden = self.invitations.closed = change.pop("hidden")
            if "name" in change:                   # and what its owner is called on team screens
                self.name = change.pop("name")
            if change and t is not None and t.ended is None and not self.leading():
                raise ValueError("in a team its leader chooses these for everyone")
            if change and self.leading():
                change = t.set(**change)
            save_settings(self.work_dir, {**load_settings(self.work_dir), **change, "hidden": self.hidden,
                                          "name": self.name})
        elif what == "leave":
            self.leave()
        elif what in ("points", "boost"):          # teaching points: a larger share of the team's computing power
            if t is None or t.ended is not None or t.role == "leader":
                raise ValueError("teaching points are spent by a member of a team, with that team's leader")
            tickets = self.helped.tickets() if self.helped is not None else []
            self.points = t.points(tickets) if what == "points" else t.boost(tickets, req.get("points"))
            return {"points": self.points, "team": self.report()}
        elif what == "rate":                       # the community rating: a member's own mark, 1 to 5
            if t is None or t.role == "leader":
                raise ValueError("community ratings come from the people who used the adapter in a team, "
                                 "not from its leader")
            if t.desc.get("adapter") is None:
                raise ValueError("that team had no adapter to rate")
            marks = req.get("marks")
            if marks is None and req.get("question") is not None:      # one button: one question, one mark
                marks = {req.get("question"): req.get("stars")}
            if not isinstance(marks, dict) or not marks or any(
                    q not in QUESTIONS or isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= 5
                    for q, m in marks.items()):
                raise ValueError("a rating is a whole number from 1 to 5 for one of: " + ", ".join(QUESTIONS))
            t.rate(marks=marks)
        elif what == "pick":                       # the answer to a blind comparison
            return {"revealed": self.blind_pick(req.get("id"), req.get("choice")), "team": self.report()}
        elif not self.leading():
            raise ValueError("only a team's leader can do that; this device is not leading a team")
        elif what == "begin":
            t.begin(self.ends.own_layers, force=req.get("force") is True)
        elif what == "invite":
            t.invite(req.get("addr") if isinstance(req.get("addr"), str) else "")
        elif what == "devices":
            return {"devices": t.devices()}
        elif what == "approve":
            t.approve(req.get("request"))
        elif what == "decline":
            t.decline(req.get("request"))
        elif what == "remove":
            t.remove(req.get("member"))
        else:
            raise KeyError(what)
        return {"team": self.report(), "team_settings": self.settings(), "team_waiting": self.waiting()}
