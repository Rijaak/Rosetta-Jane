"""What the Android app calls. One p2p service per app, started and stopped from the screen."""
import json
import os
import threading
import time
import traceback

_lock = threading.Lock()
_state = {"node": None, "server": None, "error": None, "started": None, "phase": "stopped", "layers": None}


def models(folders):
    """Model folders found in the places the app looks: [{"name", "path", "family", "layers", "ok", "why"}]."""
    from rosetta_jane.mobile.engine import Unsupported, layout, read_config, scan
    out = []
    for base in folders:
        if not base or not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if not os.path.isfile(os.path.join(path, "config.json")):
                continue
            item = {"name": name, "path": path, "family": "", "layers": 0, "ok": False, "why": ""}
            try:
                cfg = read_config(path)
                item["family"] = str(cfg.get("model_type"))
                item["layers"] = layout(scan(path).keys())[1]
                item["ok"] = True
            except (Unsupported, ValueError, OSError) as e:
                item["why"] = str(e)
            out.append(item)
    return json.dumps(out)


def start(model_dir, group_code, peers, layers, span, training, work_dir, port):
    """Join the network and start the p2p service. Returns at once; status() reports progress."""
    with _lock:
        if _state["phase"] in ("starting", "running"):
            return "already running"
        _state.update(error=None, phase="starting", started=time.time(), layers=None)
    threading.Thread(target=_run, args=(model_dir, group_code, list(peers), layers, int(span), bool(training),
                                        work_dir, int(port)), daemon=True).start()
    return "starting"


def _run(model_dir, group_code, peers, layers, span, training, work_dir, port):
    node = None
    try:
        from rosetta_jane.handoff import HandoffLog
        from rosetta_jane.mobile.__main__ import choose
        from rosetta_jane.mobile.engine import fingerprint, layout, read_config, scan
        from rosetta_jane.mobile.peer import NumpyLayerServer
        from rosetta_jane.net import Node, swarm_secret
        read_config(model_dir)
        found = scan(model_dir)
        total = layout(found.keys())[1]
        model_id = fingerprint(model_dir, found)
        os.makedirs(work_dir, exist_ok=True)
        node = Node(swarm_secret(group_code), f"0.0.0.0:{port}", bootstrap=[p for p in peers if p],
                    peers_file=os.path.join(work_dir, "peers.json"), quiet=True)
        node.handoffs = HandoffLog(os.path.join(work_dir, "handoffs.jsonl"), node.node_id)
        node.start()
        if layers == "auto":
            node.refresh(3)
            first, last = choose(node.table.peers(), model_id, total, span)
        else:
            first, last = (int(v) for v in layers.split(":"))
        server = NumpyLayerServer(model_dir, first, last, training=training, work_dir=work_dir)
        server.attach(node)
        with _lock:
            if _state["phase"] != "starting":      # stopped while the model was loading
                node.stop()
                return
            _state.update(node=node, server=server, phase="running", layers=[first, last])
    except BaseException as e:                     # shown on the screen; never crashes the app
        if node is not None:
            try:
                node.stop()
            except Exception:
                pass
        with _lock:
            _state.update(node=None, server=None, phase="failed",
                          error=f"{e}" if isinstance(e, (ValueError, SystemExit, OSError)) else traceback.format_exc(limit=3))


def stop():
    _team_reset("the p2p service was switched off")     # a team does not outlive the connection it was made on
    with _lock:
        node, server = _state["node"], _state["server"]
        _state.update(node=None, server=None, phase="stopped", layers=None)
    if server is not None:
        server.helped.save()
    if node is not None:
        node.stop()
    return "stopped"


def set_training(on):
    with _lock:
        if _state["server"] is not None:
            _state["server"].training = bool(on)
    return status()


def status():
    with _lock:
        node, server = _state["node"], _state["server"]
        out = {"phase": _state["phase"], "error": _state["error"], "layers": _state["layers"]}
    if node is not None and server is not None:
        peers = node.table.peers()
        out.update(port=node.port, peers=len(peers), model=server.fingerprint, training=server.training,
                   serving_peers=sum(1 for p in peers if any(s["model"] == server.fingerprint for s in p.services)),
                   steps=server.steps_served, tokens=server.tokens_served, examples=server.examples_trained)
    return json.dumps(out)


def self_check():
    """Proves the pieces the service depends on exist on this device. Returns a JSON report."""
    import hashlib
    import ssl
    import sys

    import numpy as np
    report = {"python": sys.version.split()[0], "numpy": np.__version__, "openssl": ssl.OPENSSL_VERSION}
    report["encryption"] = hasattr(ssl.SSLContext, "set_psk_server_callback")
    try:
        hashlib.scrypt(b"x", salt=b"y", n=1 << 15, r=8, p=1, maxmem=1 << 27, dklen=32)
        report["group_codes"] = True
    except Exception as e:
        report["group_codes"] = f"{type(e).__name__}: {e}"
    a = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    t = time.time()
    for _ in range(20):
        a @ a
    report["matmul_ms"] = round((time.time() - t) / 20 * 1000, 2)
    return json.dumps(report)


# -- asking and training from this device ---------------------------------------------------
_ask = {"text": "", "done": True, "error": None, "stats": None, "stop": False}
_train = {"state": "idle", "error": None, "step": 0, "total": 0, "loss": None, "pairs": 0, "skipped": 0,
          "checked": 0, "unchecked": 0, "blocked": 0, "name": "", "stop": False, "trainer": None, "score_lines": []}
_own = {"key": None, "ends": None, "tok": None, "node": None, "node_key": None}


def _session(model_dir, group_code, peers, own_layers):
    """This device's own layers, its tokenizer, and a connection to the group, loaded once and kept."""
    from rosetta_jane.mobile.ends import Ends
    from rosetta_jane.mobile.tok import Tokenizer
    from rosetta_jane.net import Node, swarm_secret
    key = (model_dir, int(own_layers))
    if _own["key"] != key:
        _own.update(key=None, ends=None, tok=None)
        _own.update(ends=Ends(model_dir, int(own_layers)), tok=Tokenizer(model_dir), key=key)
    with _lock:
        node = _state["node"]                       # the p2p service's connection, if it is on
    if node is None:
        node_key = (group_code, tuple(peers))
        if _own["node"] is None or _own["node_key"] != node_key:
            if _own["node"] is not None:
                _own["node"].stop()
            _own["node"] = Node(swarm_secret(group_code), "127.0.0.1:0", bootstrap=[p for p in peers if p],
                                quiet=True).start()
            _own["node_key"] = node_key
        node = _own["node"]
    node.refresh(2)
    return node, _own["ends"], _own["tok"]


def _explain(e):
    from rosetta_jane.routing import NoRoute
    if isinstance(e, NoRoute):
        return f"The group cannot do this right now: {e}"
    if isinstance(e, (ValueError, OSError)):
        return str(e)
    return traceback.format_exc(limit=3)


def ask_start(model_dir, group_code, peers, messages_json, max_new, temperature, own_layers):
    if not _ask["done"]:
        return "busy"
    if _train["state"] in ("starting", "running"):
        return "training"
    _ask.update(text="", done=False, error=None, stats=None, stop=False)
    threading.Thread(target=_ask_run, args=(model_dir, group_code, list(peers), messages_json, int(max_new),
                                            float(temperature), int(own_layers)), daemon=True).start()
    return "started"


def _ask_run(model_dir, group_code, peers, messages_json, max_new, temperature, own_layers):
    try:
        from rosetta_jane.swarm import Generation
        node, ends, tok = _session(model_dir, group_code, peers, own_layers)
        ids = tok.chat_ids(json.loads(messages_json), True)
        stops = {t for t in (tok.eos_token_id, ends.eos_token_id) if isinstance(t, int)}
        if isinstance(ends.eos_token_id, list):
            stops.update(t for t in ends.eos_token_id if isinstance(t, int))
        produced = []

        def on_token(token):
            produced.append(token)
            text = tok.decode(produced)
            if not text.endswith("�"):        # half of a character; wait for the rest
                _ask["text"] = text
            return not _ask["stop"]
        desk = _team["desk"]                       # in a team that has begun, answer with its adapter
        if desk is not None and desk.team is not None and desk.team.ended is None and desk.team.kind == "teaching":
            raise ValueError("This phone is in a teaching team, which gives its power to training. Leave the "
                             "teaching team to ask questions.")
        use = desk.use() if desk is not None and desk.node is node and desk.ends is ends else None
        gen = Generation(node, ends, find_wait=10.0, redundancy=None, team=use)
        started = time.time()
        gen.run(ids, max_new, temperature=temperature, top_p=0.9, stop_ids=sorted(stops), on_token=on_token)
        _ask["text"] = tok.decode(produced)
        _ask["stats"] = {"tokens": len(produced), "seconds": round(time.time() - started, 1),
                         "peers": sum(len(h.replicas()) for h in gen.route), "checked": gen.verified_steps,
                         "unchecked": gen.unverified_steps, "blocked": len(gen.caught),
                         "team": (desk.team.name if not use.plain else "no adapter") if use is not None else None}
    except BaseException as e:
        _ask["error"] = _explain(e)
    finally:
        _ask["done"] = True


def ask_stop():
    _ask["stop"] = True
    return "stopping"


def ask_poll():
    return json.dumps({k: _ask[k] for k in ("text", "done", "error", "stats")})


def _read_pairs(path):
    pairs, skipped = [], 0
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().strip()
    rows = json.loads(text) if text.startswith("[") else None
    if rows is None:
        rows = []
        for line in text.splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    skipped += 1
    for row in rows:
        q = row.get("question") or row.get("prompt") or row.get("instruction") if isinstance(row, dict) else None
        a = row.get("answer") or row.get("response") or row.get("output") if isinstance(row, dict) else None
        if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
            pairs.append((q.strip(), a.strip()))
        else:
            skipped += 1
    return pairs, skipped


def train_start(model_dir, group_code, peers, pairs_path, name, epochs, own_layers, files_dir, start_from=""):
    """start_from names an adapter held here to carry on from, instead of starting from nothing."""
    if _train["state"] in ("starting", "running", "stopping", "scoring"):
        return "busy"
    if not _ask["done"]:
        return "asking"
    _train.update(state="starting", error=None, step=0, total=0, loss=None, pairs=0, skipped=0, checked=0,
                  unchecked=0, blocked=0, name=name, stop=False, trainer=None, score_lines=[])
    threading.Thread(target=_train_run, args=(model_dir, group_code, list(peers), pairs_path, name, float(epochs),
                                              int(own_layers), files_dir, str(start_from or "")), daemon=True).start()
    return "started"


class _Stopped(Exception):
    pass


def _train_run(model_dir, group_code, peers, pairs_path, name, epochs, own_layers, files_dir, start_from=""):
    try:
        from rosetta_jane.control import _NAME, past_runs
        from rosetta_jane.mobile.train import Trainer
        if not _NAME.match(name):
            raise ValueError("The run name may use lowercase letters, digits and dashes, up to 40 characters.")
        if start_from and (not _NAME.match(start_from) or start_from not in past_runs(os.path.join(files_dir, "work"))):
            raise ValueError("The adapter to continue from is not on this device.")
        if start_from == name:
            raise ValueError("Give the continued adapter a new name, so the one it starts from is kept.")
        node, ends, tok = _session(model_dir, group_code, peers, own_layers)
        pairs, skipped = _read_pairs(pairs_path)
        examples = []
        for q, a in pairs:
            example = tok.pair_example(q, a, 512)
            if example is None:
                skipped += 1
            else:
                examples.append(example)
        if not examples:
            raise ValueError("No usable question and answer pairs were found in that file.")
        from rosetta_jane.scoring import split_holdout
        examples, held = split_holdout(examples)   # kept back: scored on, never trained on
        accum = min(16, len(examples))
        total = max(1, round(len(examples) * epochs / accum))
        settings = dict(total_steps=total, accum=accum, warmup_steps=min(20, max(1, total // 10)), run_id=name,
                        find_wait=10.0, save_every=5)
        desk = _team["desk"]                       # in a teaching team, only its devices do the work
        teaching = desk is not None and desk.node is node and desk.team is not None \
            and desk.team.ended is None and desk.team.kind == "teaching"
        if teaching:
            settings["team"] = desk.team.id
        if start_from:
            trainer = Trainer.continuing(node, ends, os.path.join(files_dir, "work"),
                                         os.path.join(files_dir, "work", "adapters", start_from), **settings)
        else:
            trainer = Trainer(node, ends, os.path.join(files_dir, "work"), **settings)
        _train.update(state="running", total=total, pairs=len(examples), skipped=skipped, trainer=trainer)

        def on_step(step):
            _train.update(step=step, loss=trainer.last_loss, checked=trainer.checked, unchecked=trainer.unchecked,
                          blocked=len(trainer.caught))
            if _train["stop"]:
                trainer.save()
                raise _Stopped()
        stats = trainer.train(examples, on_step=on_step)
        score = None
        if held:
            _train["state"] = "scoring"
            score = trainer.compare(held, name=name, held_out=True)
            from rosetta_jane.scoring import describe
            _train["score_lines"] = describe(score)
        from rosetta_jane import flattery
        from rosetta_jane.control import save_fact
        checked = trainer.flattery(flattery.examples(lambda q, a: tok.pair_example(q, a, 512)))
        _train["score_lines"] = list(_train["score_lines"]) + flattery.describe(checked)
        out_dir = trainer.export(os.path.join(files_dir, "work", "adapters", name), len(examples),
                                 os.path.basename(model_dir), score, len(held))
        save_fact(out_dir, "flattery", checked)
        _train.update(state="done", step=stats["steps"], loss=stats["final_loss"], checked=stats["checked"],
                      unchecked=stats["unchecked"], blocked=stats["peers_blocked"])
    except _Stopped:
        _train["state"] = "stopped"
    except BaseException as e:
        _train.update(state="failed", error=_explain(e))
    finally:
        _train["trainer"] = None


def train_stop():
    _train["stop"] = True
    if _train["state"] == "running":
        _train["state"] = "stopping"
    return "stopping"


def train_poll(files_dir):
    from rosetta_jane.control import past_runs
    out = {k: v for k, v in _train.items() if k not in ("trainer", "stop")}
    out["adapters"] = past_runs(os.path.join(files_dir, "work"))
    return json.dumps(out)


def score_card(files_dir, name):
    """The score card saved with a finished adapter: lines to show, and text its owner may choose to share."""
    from rosetta_jane.control import score_card as read
    return json.dumps(read(os.path.join(files_dir, "work"), name))


def score_document(files_dir, out_path):
    """Write the Model Adapter Score document: every score held on this device, side by side."""
    from rosetta_jane.scorebook import build
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build(os.path.join(files_dir, "work")))
    return out_path


def adapter_zip(files_dir, name, out_path):
    """Write one finished adapter as a zip for the person to keep. Nothing here applies it."""
    from rosetta_jane.control import adapter_zip as make
    with open(out_path, "wb") as f:
        f.write(make(os.path.join(files_dir, "work"), name))
    return out_path


# -- trying and scoring adapters, your own or ones other people shared ---------------------------
_side = {"job": None}


def _busy():
    job = _side["job"]
    return (job is not None and job.running()) or not _ask["done"] \
        or _train["state"] in ("starting", "running", "stopping", "scoring")


def adapters_list(files_dir):
    from rosetta_jane.control import adapters_info
    return json.dumps(adapters_info(os.path.join(files_dir, "work")))


def adapter_import(files_dir, name, zip_path):
    """Store the zip of an adapter someone shared. Returns "" or a message saying why it was refused."""
    from rosetta_jane.control import import_adapter
    try:
        with open(zip_path, "rb") as f:
            import_adapter(os.path.join(files_dir, "work"), name, f.read())
        return ""
    except (ValueError, OSError) as e:
        return str(e)


def _holding(model_dir, group_code, peers, own_layers, files_dir, name):
    from rosetta_jane.control import _NAME, past_runs
    from rosetta_jane.mobile.train import Trainer
    work = os.path.join(files_dir, "work")
    if not _NAME.match(name) or name not in past_runs(work):
        raise ValueError("no adapter by that name")
    node, ends, tok = _session(model_dir, group_code, peers, own_layers)
    trainer = Trainer.from_adapter(node, ends, os.path.join(work, "scoring"), os.path.join(work, "adapters", name),
                                   find_wait=10.0)
    return trainer, ends, tok


def adapter_try(model_dir, group_code, peers, own_layers, files_dir, name, items_json, max_new):
    """Answer each question twice, without the adapter and with it. A trial: nothing is applied."""
    from rosetta_jane.control import MAX_TRIAL_TOKENS, AdapterJob, trial_items
    if _busy():
        return "busy"
    try:
        items = trial_items({"items": json.loads(items_json)})
    except ValueError as e:
        return str(e)
    limit = max(1, min(int(max_new), MAX_TRIAL_TOKENS))
    peers = list(peers)

    def work(job):
        trainer, ends, tok = _holding(model_dir, group_code, peers, own_layers, files_dir, name)
        stops = [t for t in (tok.eos_token_id, ends.eos_token_id) if isinstance(t, int)]
        answers = []
        try:
            for item in items:
                if job.stopped():
                    break
                without = trainer.answer(tok.chat_ids(item["without"], True), limit, stops, with_adapter=False)
                with_it = trainer.answer(tok.chat_ids(item["with"], True), limit, stops, with_adapter=True)
                answers.append({"without": tok.decode(without), "with": tok.decode(with_it)})
                job.done_count = len(answers)
                job.result = {"answers": list(answers)}
        finally:
            trainer.finish()
        return {"answers": answers}
    _side["job"] = AdapterJob("try", name, work, len(items))
    return "started"


def adapter_score(model_dir, group_code, peers, own_layers, files_dir, name, pairs_path):
    """Score an adapter on your own pairs: the unchanged model against the model with it."""
    from rosetta_jane.control import AdapterJob, save_my_score
    from rosetta_jane.scoring import describe, share_text
    if _busy():
        return "busy"
    peers = list(peers)

    def work(job):
        trainer, ends, tok = _holding(model_dir, group_code, peers, own_layers, files_dir, name)
        pairs, _ = _read_pairs(pairs_path)
        examples = [e for e in (tok.pair_example(q, a, 512) for q, a in pairs) if e is not None]
        if not examples:
            raise ValueError("No usable question and answer pairs were found in that file.")
        job.total = len(examples)
        result = trainer.compare(examples, name=name, held_out=False)
        job.done_count = len(examples)
        save_my_score(os.path.join(files_dir, "work"), name, result)
        return {"score": result, "lines": describe(result), "share_text": share_text(result)}
    _side["job"] = AdapterJob("score", name, work, 0)
    return "started"


def adapter_flattery(model_dir, group_code, peers, own_layers, files_dir, name):
    """Measure whether an adapter goes along with false statements more than the unchanged model does."""
    from rosetta_jane import flattery
    from rosetta_jane.control import AdapterJob, save_fact
    if _busy():
        return "busy"
    peers = list(peers)

    def work(job):
        trainer, ends, tok = _holding(model_dir, group_code, peers, own_layers, files_dir, name)
        examples = flattery.examples(lambda q, a: tok.pair_example(q, a, 512))
        job.total = len(examples)
        result = trainer.flattery(examples)
        job.done_count = len(examples)
        save_fact(os.path.join(files_dir, "work", "adapters", name), "flattery", result)
        return {"flattery": result, "lines": flattery.describe(result)}
    _side["job"] = AdapterJob("flattery", name, work, 0)
    return "started"


def adapter_poll():
    job = _side["job"]
    return json.dumps(job.report() if job is not None else None)


def adapter_stop():
    if _side["job"] is not None:
        _side["job"].stop()
    return "stopping"


# -- teams, and sharing with the devices that helped ------------------------------------------------
# The screen is drawn from what team_poll() returns: some lines of text and a list of buttons. Each
# button names something for panel_do() to do. All the deciding happens here, where it can be tested.
_team = {"desk": None, "sharing": None}
_panel = {"message": "", "looked": None, "found": None, "devices": None, "offer": None}


def _team_reset(reason=""):
    desk = _team["desk"]
    if desk is not None and desk.team is not None and desk.team.ended is None:
        desk.leave()
        if desk.team.role == "leader":
            desk.team.settle(6.0)
        _panel["message"] = f"Your team is over: {reason}." if reason else _panel["message"]
    _team.update(desk=None, sharing=None)
    _panel.update(looked=None, found=None, devices=None, offer=None)


def _desk(model_dir, group_code, peers, own_layers, files_dir):
    """The team desk for the current connection. A new connection (p2p switched on or off) starts fresh."""
    from rosetta_jane.helpers import HelpLog, Sharing
    from rosetta_jane.mobile.train import all_shapes
    from rosetta_jane.team import TeamDesk
    node, ends, _ = _session(model_dir, group_code, peers, own_layers)
    with _lock:
        server = _state["server"] if _state["node"] is node else None
    desk = _team["desk"]
    if desk is None or desk.node is not node or desk.ends is not ends or desk.service is not server:
        _team_reset("the connection changed")
        desk = TeamDesk(node, ends, layer_prefix=ends.stack.prefix, service=server, find_wait=10.0,
                        shapes=lambda targets: all_shapes(model_dir, ends.stack.prefix, targets),
                        work_dir=os.path.join(files_dir, "work"), model_name=os.path.basename(model_dir),
                        helped=server.helped if server is not None else
                        HelpLog(os.path.join(files_dir, "work", "helped.json")))
        _team.update(desk=desk, sharing=Sharing(node, server.helped if server is not None else None).attach())
    return desk, server


def panel_do(model_dir, group_code, peers, own_layers, files_dir, kind, what, req_json):
    """One button press on the Teams part of the screen. Returns "" or a message; team_poll() shows the rest."""
    from rosetta_jane.team import NoTeam, fact_lines, open_teams
    try:
        req = json.loads(req_json or "{}")
        called = req.pop("_called", None)          # the name box on the screen travels with every press
        typed = req.pop("_question", "")           # and so does whatever is in the Ask box
        if what == "forget":
            _panel.update(looked=None, found=None, devices=None, offer=None, message="")
            return ""
        if _busy() and what in ("create", "join", "leave", "begin", "remove"):
            raise ValueError("Wait for the question, training or trial in progress to finish first.")
        desk, server = _desk(model_dir, group_code, list(peers), own_layers, files_dir)
        sharing = _team["sharing"]
        if isinstance(called, str) and called.strip() != desk.name:
            desk.control("settings", {"name": called})
        reachable = server is not None             # others can only connect to this phone while its p2p service is on
        _panel["message"] = ""
        if kind == "share":
            run = str(req.get("run", ""))
            if what == "open":
                if not reachable:
                    raise ValueError("Switch the p2p service on first: the devices that helped have to be able "
                                     "to reach this phone to take a copy.")
                row = sharing.open(os.path.join(files_dir, "work", "adapters", str(req.get("name"))),
                                   desk.ends.fingerprint)
                _panel["message"] = (f"Offering \"{row['name']}\" to the {row['helpers']} device(s) that helped "
                                     "train it, for as long as the p2p service stays on. Each owner decides.")
            elif what == "close":
                sharing.close(run)
            elif what == "look":
                found = sharing.look(run)
                _panel["offer"] = {"run": run, "lines": [f"Adapter \"{found['name']}\", offered to this device "
                                                         "because it helped train it."] + fact_lines(found)[1:]}
            elif what == "take":
                kept = sharing.take(run, os.path.join(files_dir, "work"), desk.ends.fingerprint)
                _panel.update(offer=None, message=f"Stored as \"{kept}\" with your adapters. Nothing has been "
                                                   "applied. You can try it, score it, lead a team with it or "
                                                   "train it further.")
            return ""
        if what == "blind":                        # two unlabelled answers; the person says which was right
            if _busy() or _blind["state"] == "running":
                raise ValueError("Wait for the question, training or trial in progress to finish first.")
            use = desk.use()
            if use is None or use.plain:
                raise ValueError("Blind comparisons are for a team that has begun and uses an adapter.")
            own = bool(req.get("own"))
            if own and not str(typed).strip():
                raise ValueError("Type your question in the Ask box above first.")
            from rosetta_jane.flattery import trick_question
            question = str(typed).strip() if own else trick_question()
            _blind.update(state="running", question=question, a="", b="", id="", error=None)
            threading.Thread(target=_blind_run, args=(desk, question, not own), daemon=True).start()
            return ""
        if what == "pick":
            shown = desk.control("pick", req)["revealed"]
            _blind.update(state="idle")
            _panel["message"] = (f"Answer {shown['adapter_was'].upper()} was the adapter's. Your answer joins its "
                                 "community rating.")
            return ""
        if what == "skip":
            _panel.setdefault("skipped", set()).add(req.get("question"))
            return ""
        if what in ("create", "invite", "devices") and not reachable:
            raise ValueError("Switch the p2p service on first: a leader's device has to be reachable by its team.")
        if what == "quick":                        # find an open team and show it; signing on is the next press
            desk.node.refresh(1)
            found = open_teams(desk.node, desk.ends.fingerprint)
            if not found:
                raise ValueError("No open team for this model in your group right now. Lead one with an adapter "
                                 "of yours, or ask a leader for their code.")
            what, req = "look", {"team": found[0]["team"]}
        if what == "look":
            got = desk.control("look", req)["look"]
            invited = True
            try:
                desk.invitations.get(got["team"])
            except NoTeam:
                invited = False
            _panel["looked"] = {"req": req, "lines": got["lines"], "name": got["name"], "fits": got["fits"],
                                "asks": got["joining"] == "ask" and "code" not in req and not invited}
        elif what == "list":
            got = desk.control("list", req)
            _panel["found"] = got
            if not got["teams"] and not got["teaching"] and not got["invitations"]:
                _panel["message"] = ("No teams are listed in your group right now. Invisible teams take their "
                                     "code or an invitation.")
        elif what == "devices":
            _panel["devices"] = desk.control("devices", req)["devices"]
        else:
            desk.control(what, req)
            if what == "join":
                _panel.update(looked=None, found=None)
                if desk.pending is not None:
                    _panel["message"] = ("Asked. If the leader approves, this phone is signed on and supplied "
                                         "with the adapter at once.")
            if what == "invite":
                _panel["devices"] = desk.control("devices", {})["devices"]
            if what == "leave":
                _panel.update(devices=None)
            if what == "rate":
                _panel["message"] = "Thank you. Your mark joins the adapter's community rating."
            if what == "join":
                _panel["skipped"] = set()
        return ""
    except (ValueError, NoTeam, PermissionError) as e:
        _panel["message"] = str(e)
    except KeyError as e:
        _panel["message"] = str(e.args[0]) if e.args else "That is no longer there."
    except Exception as e:                         # shown on the screen; never crashes the app
        _panel["message"] = _explain(e)
    return _panel["message"]


_blind = {"state": "idle", "question": "", "a": "", "b": "", "id": "", "error": None}


def _blind_run(desk, question, trick):
    """Answer one question twice, with the team's adapter and with the unchanged model, for a blind comparison."""
    try:
        from rosetta_jane.swarm import Generation
        node, ends, tok = desk.node, desk.ends, _own["tok"]
        ids = tok.chat_ids([{"role": "user", "content": question}], True)
        stops = {t for t in (tok.eos_token_id, ends.eos_token_id) if isinstance(t, int)}
        texts = []
        for use in (desk.use(), None):
            gen = Generation(node, ends, find_wait=10.0, redundancy=None, team=use)
            texts.append(tok.decode(gen.run(ids, 96, temperature=0.0, stop_ids=sorted(stops))))
        shown = desk.blind_put(texts[0], texts[1], trick)
        _blind.update(state="ready", a=shown["a"], b=shown["b"], id=shown["id"])
    except BaseException as e:
        _blind.update(state="idle", error=_explain(e))
        _panel["message"] = "The blind comparison could not be made: " + _explain(e)


def _rating_buttons(r, lines, button):
    """The community rating's questions, one at a time, each with five marks and a way to skip it."""
    from rosetta_jane.team import QUESTIONS
    given = r["rated"] or {}
    left = [q for q in QUESTIONS if q not in given and q not in _panel.get("skipped", set())]
    if given:
        lines.append("Your marks: " + ", ".join(f"{given[q]} out of 5" for q in QUESTIONS if q in given) + ".")
    if not left:
        return
    lines.append(f"{QUESTIONS[left[0]]} 1 is rarely, 5 is nearly always. Your mark joins the adapter's community "
                 "rating, apart from its teacher rating.")
    for mark in range(1, 6):
        button(f"{mark} out of 5", "rate", {"question": left[0], "stars": mark})
    button("Skip this question", "skip", {"question": left[0]})


def team_poll():
    """What the Teams part of the screen shows right now: {"text": ..., "buttons": [...]}. Uses no network."""
    from rosetta_jane.team import JOINING_WORDS
    lines, buttons = [], []

    def button(label, what, req=None, kind="team", needs_code=False):
        buttons.append({"label": label, "kind": kind, "what": what, "req": req or {}, "needs_code": needs_code})
    if _panel["message"]:
        lines += [_panel["message"], ""]
    desk, sharing = _team["desk"], _team["sharing"]
    r = desk.report() if desk is not None else None
    if r is not None and r["ended"] is None and r["role"] == "leader":
        lines += [f"You lead a team with the adapter \"{r['name']}\"." if r["adapter"] else
                  "You lead a team with no adapter: it pools its devices to run the unchanged model.",
                  f"Team code: {r['code']}", f"Who can sign on: {JOINING_WORDS[r['joining']]}.",
                  "A teaching team: start a training run above and only this team's devices do the work. When it "
                  "is done, offering the adapter to the devices that helped is your choice." if r["kind"] == "teaching"
                  else ("Begun: the team's answers use the adapter." if r["adapter"] else
                        "Begun: the team's answers run on the team's devices.") if r["begun"] else
                  "Not begun yet: answers are still from the unchanged model, through any device."]
        if r["kind"] == "teaching":
            lines[0] = "You lead a teaching team: its devices work together on training an adapter."
        elif r["adapter"] and r["community"]["ratings"]:
            lines.append(f"Marks from this team so far: {r['community']['average']} out of 5 from "
                         f"{r['community']['ratings']}.")
        for card in r["roster"]:
            role = ", ".join(x for x in ("asks" if card["asks"] else "",
                                         f"serves layers {card['serves'][0]}-{card['serves'][1] - 1}"
                                         if card["serves"] else "",
                                         f"+{card['boost']}% share from teaching points" if card.get("boost") else "") if x)
            lines.append(f"  {card['who']}: {card['state']}" + (f" ({role})" if role else ""))
        if r["missing"]:
            lines.append(f"No signed-on device serves layers {r['missing'][0]}-{r['missing'][-1]} yet. Devices that "
                         "serve layers have to sign on too.")
        elif r["missing_for_you"]:
            lines.append("For your own questions another device has to sign on for layers "
                         f"{r['missing_for_you'][0]}-{r['missing_for_you'][-1]}: a device does not route through "
                         "the layers it serves itself.")
        for q in r["requests"]:
            lines.append(f"A device at {q['who']} asks to join.")
            button(f"Approve the device at {q['who']}", "approve", {"request": q["id"]})
            button(f"Decline the device at {q['who']}", "decline", {"request": q["id"]})
        if not r["begun"] and r["kind"] == "session":
            button("Begin anyway" if r["missing"] else "Begin", "begin", {"force": bool(r["missing"])})
        for mode, label in (("open", "Let anyone in the group sign on"), ("ask", "People ask to join; I approve"),
                            ("invisible", "Invisible: code or invitation only")):
            if mode != r["joining"]:
                button(label, "settings", {"joining": mode})
        button("See who I can invite…", "devices")
        for d in _panel["devices"] or []:
            if not d["in_team"]:
                what = f"serves layers {d['serves'][0]}-{d['serves'][1] - 1}" if d["serves"] else "not serving"
                button(("Invited. Invite again: " if d["invited"] else "Invite ")
                       + (f"{d['name']} at " if d.get("name") else "") + f"{d['addr']} ({what}"
                       + (", in another team" if d.get("busy") else "") + ")", "invite", {"addr": d["addr"]})
        for card in r["roster"]:
            if card.get("id"):
                button(f"Remove {card['who']} from the team", "remove", {"member": card["id"]})
        button("End the team", "leave")
    elif r is not None and r["ended"] is None:
        lines += [f"You are signed on to the team using \"{r['name']}\"." if r["adapter"] else
                  "You are signed on to a team with no adapter.",
                  ("Begun: your answers use the adapter." if r["adapter"] else
                   "Begun: your answers run on the team's devices.") if r["begun"] else
                  "Waiting for the leader to begin. Until then answers are from the unchanged model.",
                  f"{r['members']} devices in the team."
                  + (" This phone's served layers work for the team." if r["serving"] else ""), ""]
        if r["kind"] == "teaching":
            lines[:2] = ["You are signed on to a teaching team: this phone's devices help train an adapter.",
                         "Your answers are not changed by it."]
        lines += r["lines"]
        if r["kind"] == "session":
            got = r["points"]
            lines.append("Teaching points with this leader: " + ("press the button to see them." if got is None else
                         f"{got['balance']}." + (f" In force: {got['boost']}, so your share of the team's computing "
                                                 f"power is {100 + got['boost']} to everyone else's 100, for "
                                                 f"{got['minutes']} more minutes." if got["boost"] else "")))
            button("See my teaching points", "points")
            for spend in (5, 10, 25):
                if got is not None and got["balance"] >= spend:
                    button(f"Spend {spend} points: {spend}% more of the team's computing power for an hour",
                           "boost", {"points": spend})
        if r["can_rate"] and r["begun"]:
            if _blind["state"] == "running":
                lines.append("Blind comparison: getting two answers…")
            elif _blind["state"] == "ready":
                lines += ["", f"Blind comparison. Question: {_blind['question']}", f"Answer A: {_blind['a']}",
                          f"Answer B: {_blind['b']}", "One is the adapter's, one the unchanged model's. Which was right?"]
                for choice, label in (("a", "A was right"), ("b", "B was right"), ("both", "Both were"),
                                      ("neither", "Neither was")):
                    button(label, "pick", {"id": _blind["id"], "choice": choice})
            else:
                button("Blind comparison: a trick question", "blind")
                button("Blind comparison: the question in the Ask box", "blind", {"own": True})
            _rating_buttons(r, lines, button)
        button("Leave the team", "leave")
    else:
        if r is not None:
            lines.append(f"Your last team is over: {r['ended']}. Answers are from the unchanged model.")
            if r["can_rate"]:
                _rating_buttons(r, lines, button)
        else:
            lines.append("You are not in a team. Answers are from the unchanged model.")
        waiting = desk.waiting() if desk is not None else None
        looked = _panel["looked"]
        if waiting is not None:
            lines += ["", f"You asked to join the team using \"{waiting['name']}\": "
                          + ("waiting for its leader." if waiting["state"] == "waiting" else waiting["state"] + ".")]
            button("Cancel the request" if waiting["state"] == "waiting" else "Clear", "cancel")
        elif looked:
            lines += ["", "Before you sign on:"] + looked["lines"]
            if looked["fits"]:
                button((f"Ask to join the team using \"{looked['name']}\"" if looked["asks"] else
                        f"Sign on to the team using \"{looked['name']}\""), "join", looked["req"])
            button("Never mind", "forget")
        else:
            button("Look at the team with this code", "look", needs_code=True)
            button("Find teams in my group", "list")
            button("Quick match: find an open team", "quick")
            button("Lead a team with no adapter (pool devices for the unchanged model)", "create")
            button("Lead a teaching team (train an adapter together)", "create", {"kind": "teaching"})
        found = _panel["found"] or {}
        for kind, label in (("invitations", "You are invited"), ("teams", "Session team"),
                            ("teaching", "Teaching team")):
            for t in found.get(kind, []):
                lines.append(f"{label}: model {t.get('model_name') or t['model'][:8]}, "
                             + (f"adapter \"{t['name']}\"" if t["adapter"] else "no adapter")
                             + f", {t['members']}/{t['limit']} devices, {'begun' if t['begun'] else 'gathering'}, "
                             f"{JOINING_WORDS[t['joining']]}.")
                button(f"Look at \"{t['name']}\"", "look", {"team": t["team"]})
    if desk is not None:
        button("Invite lists: you are hidden. Press to be shown." if desk.hidden else
               "Invite lists: leaders can see this phone. Press to hide.", "settings", {"hidden": not desk.hidden})
    waiting_invites = len(desk.invitations._held) if desk is not None else 0
    if waiting_invites and not (_panel["found"] or {}).get("invitations"):
        lines.append(f"You have {waiting_invites} invitation{'s' if waiting_invites != 1 else ''}. "
                     "Press 'Find teams in my group'.")
    for offer in sharing.offers() if sharing is not None else []:
        lines += ["", f"An adapter this phone helped train is on offer to you ({offer['examples']} pieces of work "
                      "done here)."]
        shown = _panel["offer"]
        if shown and shown["run"] == offer["run"]:
            lines += shown["lines"]
            button("Take a copy (nothing is applied)", "take", {"run": offer["run"]}, kind="share")
        else:
            button("Look at the adapter on offer", "look", {"run": offer["run"]}, kind="share")
    for row in sharing.report() if sharing is not None else []:
        lines.append(f"Offering \"{row['name']}\" to its helpers: {row['taken']} of {row['helpers']} have taken a copy.")
        button(f"Stop offering \"{row['name']}\"", "close", {"run": row["run"]}, kind="share")
    return json.dumps({"text": "\n".join(lines), "buttons": buttons})
