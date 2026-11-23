"""Controls for a front end (Project NOMAD's settings page, or a phone app).

The gateway already answers questions like an Ollama server. This adds the few
things a screen with buttons needs: see the swarm, switch serving on and off, and
start, watch and stop a training run. Changing anything requires the control
token, a random secret kept in a file on this device, because the gateway's port
may be reachable by other programs.
"""
from __future__ import annotations

import hmac
import io
import json
import os
import re
import secrets
import threading
import time
from typing import List, Optional  # noqa: F401

from .swarm import NoRoute

MAX_PAIRS = 50000
FACTS = "rosetta-jane-run.json"
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


def load_or_make_token(path: str) -> str:
    if os.path.exists(path):
        with open(path) as f:
            token = f.read().strip()
        if len(token) >= 24:
            return token
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token


def token_ok(expected: Optional[str], header: Optional[str]) -> bool:
    if not expected or not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(expected.encode(), header[7:].strip().encode())


class _Stopped(Exception):
    pass


def _number(req: dict, key: str, default, low, high, kind=float):
    value = req.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"'{key}' must be a number from {low} to {high}")
    return kind(value)


class TrainJob:
    """One training run in the background. A gateway runs at most one at a time."""

    def __init__(self, gateway, model_dir: str, work_dir: str, req: dict):
        from .lan.data import tokenize_pairs
        pairs = req.get("pairs")
        if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_PAIRS:
            raise ValueError(f"'pairs' must be a list of 1 to {MAX_PAIRS} question/answer pairs")
        for row in pairs:
            if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip() \
                    or not isinstance(row.get("answer"), str) or not row["answer"].strip():
                raise ValueError("every pair needs a non-empty 'question' and 'answer'")
        self.name = req.get("name", "run")
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise ValueError("'name' may use lowercase letters, digits and dashes, up to 40 characters")
        self.accum = _number(req, "accum", 16, 1, 256, int)
        epochs = _number(req, "epochs", 3.0, 0.01, 100.0)
        max_tokens = _number(req, "max_tokens", 1024, 8, gateway.max_prompt_tokens, int)
        self.examples, self.cut_short = tokenize_pairs(
            [{"question": r["question"], "answer": r["answer"]} for r in pairs], gateway.tokenizer, max_tokens)
        from .scoring import split_holdout
        self.examples, self.held = split_holdout(self.examples)      # kept back: scored on, never trained on
        self.score: Optional[dict] = None
        self.flattery: Optional[dict] = None
        steps = req.get("steps") or max(1, round(len(self.examples) * epochs / self.accum))
        self.total_steps = _number({"steps": steps}, "steps", steps, 1, 1_000_000, int)
        self.settings = dict(
            lr=_number(req, "lr", 2e-4, 1e-7, 1.0), rank=_number(req, "rank", 16, 1, 256, int),
            alpha=_number(req, "alpha", 32.0, 0.01, 4096.0),
            warmup_steps=min(20, max(1, self.total_steps // 10)),
            save_every=_number(req, "save_every", 25, 1, 100000, int))
        self.gateway, self.model_dir, self.work_dir = gateway, model_dir, work_dir
        self.in_team = req.get("team") is True     # train inside the team this device is in
        self.start_from = req.get("from")          # carry on from an adapter held here, instead of from nothing
        if self.start_from is not None:
            if not isinstance(self.start_from, str) or not _NAME.match(self.start_from) \
                    or self.start_from not in past_runs(work_dir):
                raise ValueError("'from' must name an adapter held on this device")
            if self.start_from == self.name:
                raise ValueError("give the continued adapter a new name, so the one it starts from is kept")
        self.out = os.path.join(work_dir, "adapters", self.name)
        self.state, self.error = "starting", None
        self.step, self.loss = 0, None
        self.started, self.finished = time.time(), None
        self.stats: dict = {}
        self.trainer = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "TrainJob":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def running(self) -> bool:
        return self.state in ("starting", "running", "stopping", "scoring")

    def _on_step(self, step: int) -> None:
        self.step = step
        if self._stop.is_set():
            self.trainer.save()
            raise _Stopped()

    def _run(self) -> None:
        from .train import SwarmTrainer
        g = self.gateway
        try:
            if self.in_team:                       # only the team's signed-on devices do the work
                t = g.desk.team
                if t is None or t.ended is not None:
                    raise ValueError("this device is not in a team; start or join one, or train without 'team'")
                if t.kind != "teaching":
                    raise ValueError("training together happens in a teaching team; this one is for sessions")
                self.settings["team"] = t.id
            common = dict(total_steps=self.total_steps, accum=self.accum, run_id=self.name,
                          redundancy=g.redundancy, tolerance=g.tolerance, auditor=g.auditor,
                          audit_rate=g.audit_rate, canaries=g.canaries, canary_rate=g.canary_rate,
                          variant=g.variant, find_wait=g.find_wait, lock=g._busy, quiet=True, **self.settings)
            if self.start_from is None:
                self.trainer = SwarmTrainer(g.node, g.ends, self.model_dir, self.work_dir, **common)
            else:                                  # the adapter's own rank and alpha are kept
                self.trainer = SwarmTrainer.continuing(
                    g.node, g.ends, self.model_dir, self.work_dir,
                    os.path.join(self.work_dir, "adapters", self.start_from), **common)
                self.settings.update(rank=self.trainer.rank, alpha=self.trainer.alpha)
            self.trainer.resume()
            self.step = self.trainer.step
            self.state = "running"
            self.stats = self.trainer.train(self.examples, on_step=self._on_step)
            self.loss = self.stats.get("final_loss")
            if self.held:                          # is the model better with it, on pairs it never saw?
                self.state = "scoring"
                with g._busy:
                    self.score = self.trainer.compare(self.held, name=self.name, held_out=True)
            self.state = "scoring"                 # and does it go along with false statements more than before?
            with g._busy:
                self.flattery = self.trainer.flattery(flattery_examples(g.tokenizer, g.max_prompt_tokens))
            self.trainer.export(self.out, os.path.abspath(self.model_dir))
            self.state = "done"
            with open(os.path.join(self.out, FACTS), "w") as f:      # what a person needs to judge this adapter
                json.dump({"name": self.name, "base_model_fingerprint": g.ends.fingerprint,
                           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": len(self.examples),
                           "pairs_cut_short": self.cut_short, "steps": self.stats.get("steps"),
                           "final_loss": self.loss, "work_checked": self.stats.get("checked_examples"),
                           "work_unchecked": self.stats.get("unchecked_examples"),
                           "base_model_name": os.path.basename(os.path.normpath(self.model_dir)),
                           "peers_blocked": self.stats.get("peers_blocked"), "pairs_kept_back": len(self.held),
                           "score": self.score, "flattery": self.flattery,
                           **{k: v for k, v in self.settings.items() if k != "team"},
                           "trained_in_team": self.in_team, **self.trainer.record()}, f, indent=2)
        except _Stopped:
            self.state = "stopped"
        except NoRoute as e:
            self.state, self.error = "failed", f"{e}. Progress is saved; start again with the same name to carry on."
        except Exception as e:                     # shown to the person, never raised into the server
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            self.finished = time.time()

    def report(self) -> dict:
        t = self.trainer
        if self._stop.is_set() and self.running():
            self.state = "stopping"
        return {"name": self.name, "state": self.state, "error": self.error,
                "step": t.step if t is not None and self.running() else self.step,
                "total_steps": self.total_steps, "pairs": len(self.examples), "pairs_cut_short": self.cut_short,
                "loss": self.loss if self.loss is not None else (t.last_loss if t is not None else None),
                "checked": t.verified_examples if t else 0, "unchecked": t.unverified_examples if t else 0,
                "peers_blocked": len(t.caught) if t else 0, "redone": t.redos if t else 0,
                "seconds": round((self.finished or time.time()) - self.started, 1),
                "pairs_kept_back": len(self.held), "score": self.score, "flattery": self.flattery,
                "adapter": self.out if self.state == "done" else None,
                # When it is done, offering it to the devices that helped is the leader's choice to make.
                "helpers": len(t.book.helpers) if t is not None and t.book is not None else 0}


def adapter_zip(work_dir: str, name: str) -> bytes:
    """One finished adapter as a zip, for a person to download, look over and apply themselves.

    Nothing in Rosetta-Jane applies a trained adapter by itself. A model trained by
    strangers' hardware should be a deliberate choice, made one adapter at a time.
    If you build on Rosetta-Jane, keep this step.
    """
    import zipfile

    from .scoring import describe
    if not isinstance(name, str) or not _NAME.match(name) or name not in past_runs(work_dir):
        raise KeyError(name)
    folder = os.path.join(work_dir, "adapters", name)
    facts = {}
    try:
        with open(os.path.join(folder, FACTS)) as f:
            facts = json.load(f)
    except (OSError, ValueError):
        pass
    unchecked = facts.get("work_unchecked")
    note = [f"Adapter '{name}', trained through a Rosetta-Jane swarm.", "",
            "Before you use it:", "",
            "1. Other people's devices did part of this training. Checks catch peers that return wrong",
            "   numbers, but no check can tell you the result is good. Try it on questions you know the",
            "   answers to before relying on it.",
            f"2. Work that was double-checked: {facts.get('work_checked', 'unknown')}. "
            f"Not checked: {unchecked if unchecked is not None else 'unknown'}."
            + ("  Unchecked work rests on one peer's word." if unchecked else ""),
            "3. It only fits the exact base model it was trained on (fingerprint "
            f"{facts.get('base_model_fingerprint', 'unknown')}).", "",
            "How it scored:", *describe(facts.get("score")),
            "A score is one piece of evidence. Score it again on your own questions before deciding.", "",
            "To apply it: this is a standard LoRA adapter. Load it over the base model with the PEFT library",
            "(PeftModel.from_pretrained), or merge it and convert the result for your AI server. Nothing",
            "applies it for you; that step is yours on purpose.", "",
            "If you build on Rosetta-Jane, keep this step. Removing it trades a minute of someone's",
            "attention for the one safeguard that does not depend on the network behaving.", "",
            "Run details are in " + FACTS + "."]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}/READ-ME-FIRST.txt", "\n".join(note) + "\n")
        for item in ("adapter_config.json", "adapter_model.safetensors", FACTS):
            path = os.path.join(folder, item)
            if os.path.exists(path):
                z.write(path, f"{name}/{item}")
    return out.getvalue()


MAX_ADAPTER_BYTES = 512 << 20
MAX_TRIAL_TOKENS = 512


def import_adapter(work_dir: str, name: str, blob: bytes) -> str:
    """Store an adapter someone else shared (the zip they downloaded), so it can be tried and scored here.

    The zip is treated as untrusted: only the three files an adapter consists of are taken, by fixed
    name, whatever paths the zip claims; sizes are capped; the settings must be plain and sane. The
    weights are a safetensors file, which holds numbers only and cannot carry code. Whether the
    adapter fits this model is checked when it is first used. Storing it applies nothing.
    """
    import zipfile
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError("the name may use lowercase letters, digits and dashes, up to 40 characters")
    target = os.path.join(work_dir, "adapters", name)
    if os.path.exists(target):
        raise ValueError(f"there is already an adapter called {name}; choose another name")
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except (zipfile.BadZipFile, ValueError):
        raise ValueError("that file is not a zip of an adapter") from None
    found = {}
    for info in z.infolist():
        base = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if base in ("adapter_config.json", "adapter_model.safetensors", FACTS) and not info.is_dir():
            if base in found:
                raise ValueError("that zip holds more than one adapter")
            if info.file_size > MAX_ADAPTER_BYTES:
                raise ValueError("that adapter is too large")
            found[base] = info
    if "adapter_config.json" not in found or "adapter_model.safetensors" not in found:
        raise ValueError("that zip does not hold an adapter (adapter_config.json and adapter_model.safetensors)")
    try:
        config = json.loads(z.read(found["adapter_config.json"]))
        rank, alpha, targets = config["r"], config["lora_alpha"], config["target_modules"]
        ok = isinstance(rank, int) and not isinstance(rank, bool) and 1 <= rank <= 256 \
            and isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and 0 < alpha <= 4096 \
            and isinstance(targets, list) and 1 <= len(targets) <= 16 \
            and all(isinstance(t, str) and t.replace("_", "").isalnum() and len(t) <= 40 for t in targets)
    except (ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise ValueError("that adapter's settings could not be read")
    facts = {}
    if FACTS in found:
        try:
            read = json.loads(z.read(found[FACTS]))
            facts = read if isinstance(read, dict) else {}
        except ValueError:
            facts = {}
    facts.update(name=name, received=True, received_on=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = target + ".partial"
    os.makedirs(tmp)
    try:
        with open(os.path.join(tmp, "adapter_config.json"), "w") as f:
            json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": alpha,
                       "lora_dropout": 0.0, "bias": "none", "target_modules": targets,
                       "fan_in_fan_out": False, "inference_mode": True}, f, indent=2)
        with z.open(found["adapter_model.safetensors"]) as src, \
                open(os.path.join(tmp, "adapter_model.safetensors"), "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
        with open(os.path.join(tmp, FACTS), "w") as f:
            json.dump(facts, f, indent=2)
        os.replace(tmp, target)
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return name


def keep_received(work_dir: str, name: str, config: dict, weights: bytes, facts: dict) -> str:
    """Store an adapter that arrived from another device over the network. Returns the name it was given.

    The same rules as a shared zip: the settings are rewritten from checked values, the weights are a
    safetensors file (numbers only), and storing it applies nothing. A free name is chosen from the
    one asked for.
    """
    rank, alpha, targets = config["r"], config["lora_alpha"], config["target_modules"]
    if len(weights) > MAX_ADAPTER_BYTES:
        raise ValueError("that adapter is too large")
    base = re.sub(r"[^a-z0-9-]+", "-", str(name).lower()).strip("-")[:36] or "shared"
    folder = os.path.join(work_dir, "adapters")
    name, n = base, 1
    while os.path.exists(os.path.join(folder, name)):
        n += 1
        name = f"{base}-{n}"
    target = os.path.join(folder, name)
    tmp = target + ".partial"
    os.makedirs(tmp)
    try:
        with open(os.path.join(tmp, "adapter_config.json"), "w") as f:
            json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": alpha,
                       "lora_dropout": 0.0, "bias": "none", "target_modules": targets,
                       "fan_in_fan_out": False, "inference_mode": True}, f, indent=2)
        with open(os.path.join(tmp, "adapter_model.safetensors"), "wb") as f:
            f.write(weights)
        with open(os.path.join(tmp, FACTS), "w") as f:
            json.dump({**facts, "name": name}, f, indent=2)
        os.replace(tmp, target)
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return name


def flattery_examples(tokenizer, max_tokens: int = 1024) -> list:
    """The flattery check's questions and replies, made ready for this model (see flattery.py)."""
    from .flattery import examples
    from .lan.data import tokenize_pairs

    def make(question: str, reply: str):
        made, cut = tokenize_pairs([{"question": question, "answer": reply}], tokenizer, max_tokens)
        return made[0] if made and not cut else None
    return examples(make)


def save_fact(folder: str, key: str, value) -> None:
    """Add one thing to the record kept beside an adapter."""
    path = os.path.join(folder, FACTS)
    try:
        with open(path) as f:
            facts = json.load(f)
        facts = facts if isinstance(facts, dict) else {}
    except (OSError, ValueError):
        facts = {"name": os.path.basename(os.path.normpath(folder))}
    facts[key] = value
    with open(path, "w") as f:
        json.dump(facts, f, indent=2)


def adapters_info(work_dir: str) -> List[dict]:
    """Every adapter held here: trained on this device, or received from someone else."""
    out = []
    for name in past_runs(work_dir):
        facts = {}
        try:
            with open(os.path.join(work_dir, "adapters", name, FACTS)) as f:
                facts = json.load(f)
        except (OSError, ValueError):
            pass
        out.append({"name": name, "received": bool(facts.get("received")), "scored": bool(facts.get("score")),
                    "my_score": bool(facts.get("my_score")),
                    "helpers": facts.get("helpers") if isinstance(facts.get("helpers"), int) else 0,
                    "continued": bool(facts.get("history")),
                    "community": (facts.get("community") or {}).get("average")
                    if isinstance(facts.get("community"), dict) else None,
                    "flattery": (facts.get("flattery") or {}).get("verdict")
                    if isinstance(facts.get("flattery"), dict) else None})
    return out


def save_my_score(work_dir: str, name: str, result: dict) -> None:
    """Keep the latest score this device's owner gave an adapter, beside the adapter."""
    path = os.path.join(work_dir, "adapters", name, FACTS)
    try:
        with open(path) as f:
            facts = json.load(f)
    except (OSError, ValueError):
        facts = {"name": name}
    facts["my_score"] = result
    with open(path, "w") as f:
        json.dump(facts, f, indent=2)


def trial_items(req: dict) -> List[dict]:
    """The questions of a trial. Each item carries the conversation so far on both sides, so a list of
    separate questions and a continuing chat are the same request."""
    items = req.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 50:
        raise ValueError("a trial takes 1 to 50 questions")

    def clean(messages):
        if not isinstance(messages, list) or not 1 <= len(messages) <= 100:
            raise ValueError("each trial question needs its messages")
        out = []
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") \
                    or not isinstance(m.get("content"), str) or not m["content"].strip():
                raise ValueError("each message needs a role and some text")
            out.append({"role": m["role"], "content": m["content"]})
        if out[-1]["role"] != "user":
            raise ValueError("a trial question must end with what the person asked")
        return out
    return [{"without": clean(i.get("without") if isinstance(i, dict) else None),
             "with": clean(i.get("with") if isinstance(i, dict) else None)} for i in items]


class AdapterJob:
    """Scoring or trying an adapter in the background. Shares the gateway's one-job-at-a-time rule.

    kind 'score': compare the unchanged model with the adapter on the person's own pairs.
    kind 'try':   answer each question twice, without and with the adapter. Nothing is applied.
    """

    def __init__(self, kind: str, name: str, work, total: int):
        self.kind, self.name, self.total = kind, name, total
        self.state, self.error, self.done_count = "running", None, 0
        self.result = None
        self._work = work
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def running(self) -> bool:
        return self.state == "running"

    def stop(self) -> None:
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()

    def _run(self) -> None:
        try:
            self.result = self._work(self)
            self.state = "stopped" if self._stop.is_set() else "done"
        except NoRoute as e:
            self.state, self.error = "failed", f"The group cannot do this right now: {e}"
        except (ValueError, OSError) as e:
            self.state, self.error = "failed", str(e)
        except Exception as e:                     # shown to the person, never raised into the server
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"

    def report(self) -> dict:
        return {"kind": self.kind, "name": self.name, "state": self.state, "error": self.error,
                "done": self.done_count, "total": self.total, "result": self.result}


def score_card(work_dir: str, name: str) -> dict:
    """The score card saved with a finished adapter, with a version ready to share."""
    from .scoring import describe, share_text
    if not isinstance(name, str) or not _NAME.match(name) or name not in past_runs(work_dir):
        raise KeyError(name)
    try:
        with open(os.path.join(work_dir, "adapters", name, FACTS)) as f:
            found = json.load(f).get("score")
    except (OSError, ValueError):
        found = None
    mine = None
    try:
        with open(os.path.join(work_dir, "adapters", name, FACTS)) as f:
            mine = json.load(f).get("my_score")
    except (OSError, ValueError):
        pass
    if mine:                                       # the owner's own scoring is the one they would share
        return {"name": name, "score": mine, "lines": describe(mine), "share_text": share_text(mine),
                "trainer_score": found}
    return {"name": name, "score": found, "lines": describe(found), "share_text": share_text(found)}


def past_runs(work_dir: str) -> List[str]:
    folder = os.path.join(work_dir, "adapters")
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder)
                  if os.path.exists(os.path.join(folder, n, "adapter_model.safetensors")))
