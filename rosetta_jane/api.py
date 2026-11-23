"""A local door for other programs: the same HTTP interface Ollama offers.

Project NOMAD (or anything else that can talk to an Ollama server) points its
"remote AI server" address at this, and its questions are answered through the
swarm. It listens on this computer only unless told otherwise.

Covered: /api/version, /api/tags, /api/show, /api/ps, /api/chat, /api/generate,
and the OpenAI-style /v1/models and /v1/chat/completions, with streaming.
/api/rosetta/status reports what the swarm looks like from this device.
/api/rosetta/team/... is the co-op side: lead a team with an adapter, sign on to
one, begin, leave (see team.py). /api/rosetta/share/... shares a finished adapter
with the devices that helped train it (see helpers.py).
Not covered: embeddings, model downloads, images. Those return a clear error;
NOMAD keeps embeddings on its own machine.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional

from . import __version__
from .layers import LocalEnds
from .net import Node
from .swarm import Generation, NoRoute
from .team import NoTeam

MAX_BODY = 32 << 20
MAX_UPLOAD = 600 << 20         # an adapter someone shared, as a zip


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


class Gateway:
    def __init__(self, node: Node, ends: LocalEnds, tokenizer, model_name: str, *,
                 variant: Optional[str] = None, max_new_tokens: int = 512, max_prompt_tokens: int = 4096,
                 find_wait: float = 15.0, redundancy: Optional[int] = None, auditor=None, require_verified: bool = False,
                 tolerance: float = 0.02, audit_rate: float = 0.0, canaries=None, canary_rate: float = 0.2,
                 service=None, control_token: Optional[str] = None, model_dir: Optional[str] = None,
                 work_dir: Optional[str] = None):
        self.service, self.control_token = service, control_token
        self.model_dir, self.work_dir = model_dir, work_dir
        self.job = None                           # the current or most recent training run
        self.side = None                          # the current or most recent scoring or trial of an adapter
        self.audit_rate = audit_rate
        self.canaries, self.canary_rate = canaries, canary_rate
        self.find_wait = find_wait
        self.redundancy, self.auditor = redundancy, auditor
        self.require_verified, self.tolerance = require_verified, tolerance
        self.node, self.ends, self.tokenizer = node, ends, tokenizer
        self.model_name = model_name
        self.variant = variant
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self._busy = threading.Lock()             # one answer at a time through this gateway
        self.last: Optional[dict] = None          # how the most recent answer went
        from .helpers import Sharing
        from .team import TeamDesk

        def shapes(targets):
            from .lora import shapes_from_files
            if not self.model_dir:
                raise ValueError("this needs the model folder; this gateway was started without it")
            return shapes_from_files(self.model_dir, targets)

        def prefix():
            from .layers import model_shape
            if not self.model_dir:
                raise ValueError("leading a team needs the model folder; this gateway was started without it")
            return model_shape(self.model_dir)[0]
        self.desk = TeamDesk(node, ends, layer_prefix=prefix, shapes=shapes, work_dir=work_dir, service=service,
                             whole=auditor is not None, find_wait=find_wait,
                             helped=service.helped if service is not None else self._tickets(work_dir),
                             model_name=os.path.basename(os.path.normpath(model_dir)) if model_dir else None)
        self.sharing = Sharing(node, service.helped if service is not None else None).attach()

    @staticmethod
    def _tickets(work_dir):
        """This device's tickets from teaching, kept in its work folder even when it serves nothing now."""
        from .helpers import HelpLog
        return HelpLog(os.path.join(work_dir, "helped.json")) if work_dir else None

    def context_length(self) -> int:
        limit = getattr(self.tokenizer, "model_max_length", None)
        if not isinstance(limit, int) or limit <= 0 or limit > 1 << 20:
            limit = self.max_prompt_tokens + self.max_new_tokens
        return min(limit, self.max_prompt_tokens + self.max_new_tokens)

    def status(self) -> dict:
        """What the swarm looks like from here, for a settings page to show."""
        own, total = self.ends.own_layers, self.ends.num_layers
        cover = [0] * total
        serving = 0
        for peer in self.node.table.peers():
            if peer.node_id in self.node.banned or peer.addr in self.node.banned:
                continue
            mine = [s for s in peer.services if s["model"] == self.ends.fingerprint
                    and (self.variant is None or s["variant"] == self.variant)]
            serving += bool(mine)
            for s in mine:
                for layer in range(max(0, s["layers"][0]), min(total, s["layers"][1])):
                    cover[layer] += 1
        served = cover[own:]
        return {"rosetta_jane": __version__, "model": self.model_name, "fingerprint": self.ends.fingerprint,
                "layers": total, "own_layers": own, "peers_known": len(self.node.table.peers()),
                "peers_serving": serving, "layers_missing": [own + i for i, n in enumerate(served) if n == 0],
                "thinnest_cover": min(served) if served else 0, "can_answer": all(n > 0 for n in served),
                "peers_blocked": len({b for b in self.node.banned if ":" not in b}),
                "checks": {"compare_peers": "auto" if self.redundancy is None else self.redundancy,
                           "recheck_here": self.auditor is not None,
                           "known_answers": self.canaries is not None,
                           "require_checked": self.require_verified},
                "last_answer": self.last, "controls": bool(self.control_token),
                "service": self.service_report(), "training": self.job.report() if self.job else None,
                "adapters": self.adapters(), "adapter_details": self.adapter_details(),
                "adapter_job": self.side.report() if self.side else None,
                "team": self.team_report(), "team_settings": self.team_settings(),
                "team_waiting": self.desk.waiting(),
                "invitations": len(self.invitations._held),
                "sharing": self.sharing.report(), "offers": self.sharing.offers()}

    def service_report(self) -> Optional[dict]:
        s = self.service
        if s is None:
            return None
        return {"on": not s.paused, "layers": [s.start, s.end], "training": s.training,
                "steps_helped": s.steps_served, "examples_helped": s.examples_trained}

    def adapters(self) -> list:
        from .control import past_runs
        return past_runs(self.work_dir) if self.work_dir else []

    def adapter_details(self) -> list:
        from .control import adapters_info
        return adapters_info(self.work_dir) if self.work_dir else []

    # -- teams: an adapter applied because people signed on (the work is in team.TeamDesk) --------------
    @property
    def team(self):
        """The team this device leads or has signed on to, if any."""
        return self.desk.team

    @property
    def invitations(self):
        return self.desk.invitations

    def team_settings(self) -> dict:
        return self.desk.settings()

    def team_report(self) -> Optional[dict]:
        return self.desk.report()

    def lead_team(self, adapter_dir=None, name=None, joining=None, limit=None, code=None, serve: bool = True):
        return self.desk.lead(adapter_dir, name, joining, limit, code, serve)

    def look_team(self, req: dict) -> dict:
        return self.desk.look(req)

    def join_team(self, req: dict):
        return self.desk.join(req)

    def leave_team(self) -> None:
        self.desk.leave()

    def team_control(self, what: str, req: dict) -> dict:
        return self.desk.control(what, req)

    def share_control(self, what: str, req: dict) -> dict:
        """Share a finished adapter with the devices that helped train it, or take a copy of one offered."""
        from .control import _NAME, past_runs
        from .team import fact_lines
        run = req.get("run") if isinstance(req.get("run"), str) else ""
        if not self.work_dir:
            raise ValueError("adapters are not set up on this gateway")
        if what == "open":
            name = req.get("name")
            if not isinstance(name, str) or not _NAME.match(name) or name not in past_runs(self.work_dir):
                raise ValueError("no adapter by that name")
            self.sharing.open(os.path.join(self.work_dir, "adapters", name), self.ends.fingerprint)
        elif what == "close":
            self.sharing.close(run)
        elif what == "look":
            try:
                found = self.sharing.look(run)
            except KeyError as e:
                raise ValueError(str(e.args[0])) from None
            return {"look": {**found, "lines": fact_lines(found)[1:]}}
        elif what == "take":
            try:
                kept = self.sharing.take(run, self.work_dir, self.ends.fingerprint, req.get("name"))
            except (KeyError, PermissionError) as e:
                raise ValueError(str(e.args[0])) from None
            return {"taken": kept, "adapter_details": self.adapter_details()}
        else:
            raise KeyError(what)
        return {"sharing": self.sharing.report(), "offers": self.sharing.offers()}

    def _idle(self) -> None:
        if (self.job is not None and self.job.running()) or (self.side is not None and self.side.running()):
            raise ValueError("this gateway is busy with a training run, a scoring or a trial; wait or stop it first")

    def _holding(self, name) -> "object":
        """A trainer holding the named adapter, checked against this model. For scoring and trials only."""
        from .control import _NAME, past_runs
        from .train import SwarmTrainer
        if not (self.model_dir and self.work_dir):
            raise ValueError("adapters are not set up on this gateway")
        if not isinstance(name, str) or not _NAME.match(name) or name not in past_runs(self.work_dir):
            raise ValueError("no adapter by that name")
        return SwarmTrainer.from_adapter(
            self.node, self.ends, self.model_dir, os.path.join(self.work_dir, "scoring"),
            os.path.join(self.work_dir, "adapters", name), redundancy=self.redundancy, tolerance=self.tolerance,
            auditor=self.auditor, canaries=self.canaries, canary_rate=self.canary_rate, variant=self.variant,
            find_wait=self.find_wait)

    def _stops(self) -> list:
        return [t for t in {self.ends.eos_token_id, getattr(self.tokenizer, "eos_token_id", None)} if isinstance(t, int)]

    def score_adapter(self, req: dict):
        """Score an adapter on the person's own pairs: the unchanged model against the model with it."""
        from .control import MAX_PAIRS, AdapterJob, save_my_score
        from .lan.data import tokenize_pairs
        from .scoring import describe, share_text
        self._idle()
        pairs = req.get("pairs")
        if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_PAIRS or not all(
                isinstance(r, dict) and isinstance(r.get("question"), str) and r["question"].strip()
                and isinstance(r.get("answer"), str) and r["answer"].strip() for r in pairs):
            raise ValueError("'pairs' must be a list of question/answer pairs")
        name = req.get("name")
        trainer = self._holding(name)
        examples, _ = tokenize_pairs([{"question": r["question"], "answer": r["answer"]} for r in pairs],
                                     self.tokenizer, min(1024, self.max_prompt_tokens))

        def work(job):
            with self._busy:
                result = trainer.compare(examples, name=name, held_out=False)
            job.done_count = len(examples)
            save_my_score(self.work_dir, name, result)
            return {"score": result, "lines": describe(result), "share_text": share_text(result)}
        self.side = AdapterJob("score", name, work, len(examples))
        return {"adapter_job": self.side.report()}

    def check_flattery(self, req: dict):
        """Run the flattery check on an adapter held here: measured against the unchanged model."""
        from .control import AdapterJob, flattery_examples, save_fact
        from .flattery import describe
        self._idle()
        name = req.get("name")
        trainer = self._holding(name)
        examples = flattery_examples(self.tokenizer, min(1024, self.max_prompt_tokens))

        def work(job):
            with self._busy:
                result = trainer.flattery(examples)
            job.done_count = len(examples)
            save_fact(os.path.join(self.work_dir, "adapters", name), "flattery", result)
            return {"flattery": result, "lines": describe(result)}
        self.side = AdapterJob("flattery", name, work, len(examples))
        return {"adapter_job": self.side.report()}

    def blind(self, req: dict) -> dict:
        """A blind comparison for a member of a team with an adapter: the same question answered by the
        team (with the adapter) and by the unchanged model, returned as A and B with nothing to say which
        is which. The member then says which was right ('pick'), and that joins the community rating."""
        from .flattery import trick_question
        t = self.team
        if t is None or not t.active() or t.use().plain:
            raise ValueError("blind comparisons are for a team that has begun and uses an adapter")
        if t.role == "leader":
            raise ValueError("blind comparisons are made by the team's members; a leader does not rate their "
                             "own adapter")
        trick = req.get("messages") is None
        messages = [{"role": "user", "content": trick_question()}] if trick else req["messages"]
        ids = self.chat_ids(messages)
        options = {"temperature": 0, "num_predict": req.get("max_new", 96)}
        with_adapter = self.answer(ids, options, lambda text: None)["text"]
        unchanged = self.answer(ids, options, lambda text: None, plain=True)["text"]
        shown = self.desk.blind_put(with_adapter, unchanged, trick)
        return {"blind": {**shown, "question": messages[-1]["content"]}}

    def try_adapter(self, req: dict):
        """Answer each question twice, without the adapter and with it. A trial: nothing is applied."""
        from .control import MAX_TRIAL_TOKENS, AdapterJob, _number, trial_items
        self._idle()
        items = trial_items(req)
        limit = _number(req, "max_new", 48, 1, MAX_TRIAL_TOKENS, int)
        name = req.get("name")
        trainer = self._holding(name)
        prompts = [(self.chat_ids(i["without"]), self.chat_ids(i["with"])) for i in items]
        if any(len(p) > self.max_prompt_tokens for pair in prompts for p in pair):
            raise ValueError(f"a trial question is longer than {self.max_prompt_tokens} tokens")
        stops = self._stops()

        def work(job):
            answers = []
            try:
                for plain_ids, adapter_ids in prompts:
                    if job.stopped():
                        break
                    with self._busy:
                        without = trainer.answer(plain_ids, limit, stops, with_adapter=False)
                        with_it = trainer.answer(adapter_ids, limit, stops, with_adapter=True)
                    answers.append({"without": self.tokenizer.decode(without, skip_special_tokens=True),
                                    "with": self.tokenizer.decode(with_it, skip_special_tokens=True)})
                    job.done_count = len(answers)
                    job.result = {"answers": list(answers)}        # visible as each question finishes
            finally:
                trainer.finish()
            return {"answers": answers}
        self.side = AdapterJob("try", name, work, len(prompts))
        return {"adapter_job": self.side.report()}

    def control(self, path: str, req: dict) -> dict:
        """Change something on behalf of a front end. Raises ValueError with a message to show."""
        if path == "/api/rosetta/service":
            if self.service is None:
                raise ValueError("this device was started without layers to serve (see --serve)")
            for key in ("on", "training"):
                if key in req and not isinstance(req[key], bool):
                    raise ValueError(f"'{key}' must be true or false")
            if "on" in req:
                self.service.paused = not req["on"]
            if "training" in req:
                self.service.training = req["training"]
            return {"service": self.service_report()}
        if path == "/api/rosetta/train":
            from .control import TrainJob
            if not (self.model_dir and self.work_dir):
                raise ValueError("training is not set up on this gateway")
            if self.job is not None and self.job.running():
                raise ValueError("a training run is already going; stop it first")
            self._idle()
            self.job = TrainJob(self, self.model_dir, self.work_dir, req).start()
            return {"training": self.job.report()}
        if path == "/api/rosetta/train/stop":
            if self.job is not None:
                self.job.stop()
            return {"training": self.job.report() if self.job else None}
        if path == "/api/rosetta/adapters/score":
            return self.score_adapter(req)
        if path == "/api/rosetta/adapters/try":
            return self.try_adapter(req)
        if path == "/api/rosetta/adapters/flattery":
            return self.check_flattery(req)
        if path == "/api/rosetta/team/blind":
            try:
                return self.blind(req)
            except NoRoute as e:
                raise ValueError(f"the group cannot answer right now: {e}") from None
        if path == "/api/rosetta/adapters/stop":
            if self.side is not None:
                self.side.stop()
            return {"adapter_job": self.side.report() if self.side else None}
        for prefix, handle in (("/api/rosetta/team/", self.team_control), ("/api/rosetta/share/", self.share_control)):
            if path.startswith(prefix):
                try:
                    return handle(path[len(prefix):], req)
                except (NoTeam, PermissionError, RuntimeError) as e:
                    raise ValueError(str(e)) from None
        raise KeyError(path)

    # -- prompts -----------------------------------------------------------------
    def chat_ids(self, messages: list) -> List[int]:
        clean = []
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
                raise ValueError("each message needs a role (system, user, assistant) and text content")
            content = m.get("content")
            if isinstance(content, list):          # OpenAI-style parts: keep the text, drop images
                content = "".join(part.get("text", "") for part in content
                                  if isinstance(part, dict) and part.get("type") == "text"
                                  and isinstance(part.get("text"), str))
            if not isinstance(content, str):
                raise ValueError("each message needs a role (system, user, assistant) and text content")
            clean.append({"role": m["role"], "content": content})
        if not clean:
            raise ValueError("no messages")
        ids = self.tokenizer.apply_chat_template(clean, tokenize=True, add_generation_prompt=True)
        return list(getattr(ids, "input_ids", ids))

    def text_ids(self, prompt: str) -> List[int]:
        return list(self.tokenizer(prompt)["input_ids"])

    # -- answering -----------------------------------------------------------------
    def answer(self, ids: List[int], options: dict, emit: Callable[[str], None], plain: bool = False) -> dict:
        """plain=True answers with the unchanged model whatever team this device is in (for comparisons)."""
        if len(ids) > self.max_prompt_tokens:
            raise ValueError(f"prompt is {len(ids)} tokens; this gateway accepts up to {self.max_prompt_tokens}")
        temperature = float(options.get("temperature", 0.8))
        top_p = float(options.get("top_p", 0.9))
        limit = int(options.get("num_predict", self.max_new_tokens))
        limit = self.max_new_tokens if limit <= 0 else min(limit, self.max_new_tokens)
        seed = options.get("seed")
        stop_texts = options.get("stop") or []
        if isinstance(stop_texts, str):
            stop_texts = [stop_texts]
        stop_ids = [t for t in {self.ends.eos_token_id, getattr(self.tokenizer, "eos_token_id", None)}
                    if isinstance(t, int)]
        produced: List[int] = []
        shown = ""
        reason = "length"

        def on_token(token: int) -> bool:
            nonlocal shown, reason
            produced.append(token)
            text = self.tokenizer.decode(produced, skip_special_tokens=True)
            for stop in stop_texts:
                cut = text.find(stop) if stop else -1
                if cut >= 0:
                    if cut > len(shown):
                        emit(text[len(shown):cut])
                    shown = text[:cut]
                    reason = "stop"
                    return False
            if text.endswith("�"):           # half of a multi-byte character; wait for the rest
                return True
            if len(text) > len(shown):
                emit(text[len(shown):])
                shown = text
            return True

        t = self.team
        if t is not None and t.ended is None and t.kind == "teaching":
            raise ValueError("this device is in a teaching team, which gives its power to training. "
                             "Leave the teaching team to ask questions.")
        started = time.time()
        with self._busy:
            # In a team that has begun, the answer comes from the model with the team's adapter.
            # In every other case it is the unchanged model.
            team = None if plain else self.desk.use()
            gen = Generation(self.node, self.ends, variant=self.variant, find_wait=self.find_wait,
                             redundancy=self.redundancy, auditor=self.auditor,
                             require_verified=self.require_verified, tolerance=self.tolerance,
                             audit_rate=self.audit_rate, canaries=self.canaries,
                             canary_rate=self.canary_rate, team=team)
            out = gen.run(ids, limit, temperature=temperature, top_p=top_p,
                          seed=int(seed) if seed is not None else None, stop_ids=stop_ids, on_token=on_token)
        if reason == "length" and len(out) < limit:
            reason = "stop"
        self.last = {"at": _now(), "peers_used": sum(len(h.replicas()) for h in gen.route),
                     "checked_steps": gen.verified_steps, "unchecked_steps": gen.unverified_steps,
                     "peers_blocked": len(gen.caught), "reroutes": gen.reroutes,
                     "team": team.id if team is not None else None}
        return {"done": True, "done_reason": reason, "total_duration": int((time.time() - started) * 1e9),
                "prompt_eval_count": len(ids), "eval_count": len(produced), "text": shown,
                "peers_used": sum(len(h.replicas()) for h in gen.route), "reroutes": gen.reroutes,
                "checked_steps": gen.verified_steps, "unchecked_steps": gen.unverified_steps,
                "disputes": gen.disputes, "peers_blocked": len(gen.caught), "spot_audits": gen.spot_audits,
                "redundancy": gen.redundancy, "known_answer_tests": gen.known_answer_tests,
                "known_answers_spent": gen.tests_spent, "team": team.id if team is not None else None}

    # -- HTTP ------------------------------------------------------------------------
    def serve(self, listen: str = "127.0.0.1:11435") -> ThreadingHTTPServer:
        gateway = self
        host, _, port = listen.rpartition(":")

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _json(self, code: int, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", ""):
                    body = b"Rosetta-Jane is running"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/api/version":
                    self._json(200, {"version": f"rosetta-{__version__}", "rosetta_jane": __version__})
                elif self.path == "/api/rosetta/status":
                    self._json(200, gateway.status())
                elif self.path == "/api/rosetta/train":
                    self._json(200, {"training": gateway.job.report() if gateway.job else None})
                elif self.path == "/api/rosetta/model-adapter-score":
                    from .control import token_ok
                    from .scorebook import build
                    if not token_ok(gateway.control_token, self.headers.get("Authorization")):
                        self._json(401 if gateway.control_token else 403, {"error": "scores need the control token"})
                        return
                    body = build(gateway.work_dir or "").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/api/rosetta/adapters/"):
                    from .control import adapter_zip, score_card, token_ok
                    if not token_ok(gateway.control_token, self.headers.get("Authorization")):
                        self._json(401 if gateway.control_token else 403,
                                   {"error": "trained adapters need the control token"})
                        return
                    if self.path.endswith("/score-card"):
                        try:
                            self._json(200, score_card(gateway.work_dir or "", self.path.split("/")[-2]))
                        except KeyError:
                            self._json(404, {"error": "no finished adapter by that name"})
                        return
                    try:
                        body = adapter_zip(gateway.work_dir or "", self.path.rsplit("/", 1)[1])
                    except KeyError:
                        self._json(404, {"error": "no finished adapter by that name"})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/api/ps":
                    self._json(200, {"models": []})
                elif self.path == "/v1/models":
                    self._json(200, {"object": "list", "data": [{
                        "id": gateway.model_name, "object": "model", "created": 0, "owned_by": "rosetta-jane"}]})
                elif self.path == "/api/tags":
                    self._json(200, {"models": [{
                        "name": gateway.model_name, "model": gateway.model_name, "modified_at": _now(),
                        "size": 0, "digest": gateway.ends.fingerprint,
                        "details": {"family": "rosetta", "format": "swarm",
                                    "parameter_size": "", "quantization_level": gateway.variant or ""}}]})
                else:
                    self._json(404, {"error": f"{self.path} is not available through Rosetta-Jane"})

            def _read(self) -> bool:
                """Take the request body off the connection, so the next request on it starts clean."""
                self._raw = b""
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                limit = MAX_UPLOAD if self.path.startswith("/api/rosetta/adapters/import") else MAX_BODY
                if not 0 <= length <= limit:
                    self.close_connection = True
                    self._json(413, {"error": "missing or oversized request body"})
                    return False
                self._raw = self.rfile.read(length)
                return True

            def _body(self) -> dict:
                if not self._raw:
                    raise ValueError("missing request body")
                req = json.loads(self._raw)
                if not isinstance(req, dict):
                    raise ValueError("request must be a JSON object")
                return req

            def _openai(self) -> None:
                try:
                    req = self._body()
                    ids = gateway.chat_ids(req.get("messages") or [])
                    options = {k: req[k] for k in ("temperature", "top_p", "seed", "stop") if req.get(k) is not None}
                    limit = req.get("max_completion_tokens") or req.get("max_tokens")
                    if limit:
                        options["num_predict"] = int(limit)
                    stream = req.get("stream") is True
                except (ValueError, TypeError) as e:
                    self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                    return
                ident, created = f"chatcmpl-{int(time.time() * 1000)}", int(time.time())

                def chunk(delta: dict, finish=None) -> dict:
                    return {"id": ident, "object": "chat.completion.chunk", "created": created,
                            "model": gateway.model_name,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

                def usage(final: dict) -> dict:
                    return {"prompt_tokens": final["prompt_eval_count"], "completion_tokens": final["eval_count"],
                            "total_tokens": final["prompt_eval_count"] + final["eval_count"]}
                if not stream:
                    try:
                        final = gateway.answer(ids, options, lambda t: None)
                    except NoRoute as e:
                        self._json(503, {"error": {"message": f"the swarm cannot answer right now: {e}",
                                                   "type": "server_error"}})
                        return
                    except ValueError as e:
                        self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                        return
                    self._json(200, {"id": ident, "object": "chat.completion", "created": created,
                                     "model": gateway.model_name, "usage": usage(final),
                                     "choices": [{"index": 0, "finish_reason": final["done_reason"],
                                                  "message": {"role": "assistant", "content": final["text"]}}]})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def write(obj) -> None:
                    line = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
                    self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                    self.wfile.flush()
                try:
                    write(chunk({"role": "assistant", "content": ""}))
                    final = gateway.answer(ids, options, lambda t: write(chunk({"content": t})))
                    last = chunk({}, final["done_reason"])
                    last["usage"] = usage(final)
                    write(last)
                    write(b"[DONE]")
                except (NoRoute, ValueError) as e:
                    write({"error": {"message": f"the swarm cannot answer right now: {e}", "type": "server_error"}})
                except (BrokenPipeError, ConnectionError):
                    return
                try:
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionError):
                    pass

            def do_POST(self):
                if not self._read():
                    return
                if self.path == "/v1/chat/completions":
                    self._openai()
                    return
                if self.path.startswith("/api/rosetta/"):
                    from .control import token_ok
                    if not gateway.control_token:
                        self._json(403, {"error": "controls are switched off on this gateway"})
                    elif not token_ok(gateway.control_token, self.headers.get("Authorization")):
                        self._json(401, {"error": "wrong or missing control token"})
                    elif self.path.startswith("/api/rosetta/adapters/import"):
                        from urllib.parse import parse_qs, urlsplit

                        from .control import import_adapter
                        try:
                            name = (parse_qs(urlsplit(self.path).query).get("name") or [""])[0]
                            if not gateway.work_dir:
                                raise ValueError("adapters are not set up on this gateway")
                            self._json(200, {"imported": import_adapter(gateway.work_dir, name, self._raw),
                                             "adapter_details": gateway.adapter_details()})
                        except ValueError as e:
                            self._json(400, {"error": str(e)})
                    else:
                        try:
                            self._json(200, gateway.control(self.path, self._body() if self._raw else {}))
                        except KeyError:
                            self._json(404, {"error": f"{self.path} is not available through Rosetta-Jane"})
                        except (ValueError, TypeError) as e:
                            self._json(400, {"error": str(e)})
                    return
                if self.path == "/api/show":
                    self._json(200, {
                        "details": {"family": "rosetta-jane", "format": "swarm", "parameter_size": "",
                                    "quantization_level": gateway.variant or ""},
                        "capabilities": ["completion"], "parameters": "",
                        "model_info": {"general.architecture": "rosetta",
                                       "rosetta.context_length": gateway.context_length(),
                                       "rosetta.block_count": gateway.ends.num_layers}})
                    return
                if self.path in ("/api/embed", "/api/embeddings", "/v1/embeddings"):
                    self._json(501, {"error": "Rosetta-Jane does not make embeddings. Keep knowledge-base "
                                              "indexing on this machine (in NOMAD: Settings > Models > "
                                              "'Index the Knowledge Base on This Machine')."})
                    return
                if self.path not in ("/api/chat", "/api/generate"):
                    self._json(404, {"error": f"{self.path} is not available through Rosetta-Jane "
                                              "(downloads stay on your local AI server)"})
                    return
                try:
                    req = self._body()
                    chat = self.path == "/api/chat"
                    if not chat and not str(req.get("prompt") or ""):
                        # Ollama's "load" and "unload" hints: a generate call with no prompt.
                        self._json(200, {"model": gateway.model_name, "created_at": _now(), "response": "",
                                         "done": True, "done_reason": "load"})
                        return
                    ids = gateway.chat_ids(req.get("messages") or []) if chat \
                        else gateway.text_ids(str(req.get("prompt") or ""))
                    options = req.get("options") if isinstance(req.get("options"), dict) else {}
                    stream = req.get("stream", True) is not False
                except (ValueError, TypeError) as e:
                    self._json(400, {"error": str(e)})
                    return

                def piece(text: str, done: Optional[dict] = None) -> dict:
                    out = {"model": gateway.model_name, "created_at": _now()}
                    out["message" if chat else "response"] = {"role": "assistant", "content": text} if chat else text
                    out["done"] = bool(done)
                    if done:
                        out.update({k: v for k, v in done.items() if k != "text"})
                    return out

                if not stream:
                    try:
                        final = gateway.answer(ids, options, lambda t: None)
                    except NoRoute as e:
                        self._json(503, {"error": f"the swarm cannot answer right now: {e}"})
                        return
                    except ValueError as e:
                        self._json(400, {"error": str(e)})
                        return
                    self._json(200, piece(final["text"], final))
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def write(obj: dict) -> None:
                    line = json.dumps(obj).encode() + b"\n"
                    self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                    self.wfile.flush()

                try:
                    final = gateway.answer(ids, options, lambda t: write(piece(t)))
                    write(piece("", final))
                except (NoRoute, ValueError) as e:
                    write({"error": f"the swarm cannot answer right now: {e}", "done": True})
                except (BrokenPipeError, ConnectionError):
                    return                         # the program asking went away
                try:
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionError):
                    pass

        server = ThreadingHTTPServer((host, int(port)), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
