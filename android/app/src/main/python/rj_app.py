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
        server = NumpyLayerServer(model_dir, first, last, training=training)
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
    with _lock:
        node = _state["node"]
        _state.update(node=None, server=None, phase="stopped", layers=None)
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
        gen = Generation(node, ends, find_wait=10.0, redundancy=None)
        started = time.time()
        gen.run(ids, max_new, temperature=temperature, top_p=0.9, stop_ids=sorted(stops), on_token=on_token)
        _ask["text"] = tok.decode(produced)
        _ask["stats"] = {"tokens": len(produced), "seconds": round(time.time() - started, 1),
                         "peers": sum(len(h.replicas()) for h in gen.route), "checked": gen.verified_steps,
                         "unchecked": gen.unverified_steps, "blocked": len(gen.caught)}
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


def train_start(model_dir, group_code, peers, pairs_path, name, epochs, own_layers, files_dir):
    if _train["state"] in ("starting", "running", "stopping", "scoring"):
        return "busy"
    if not _ask["done"]:
        return "asking"
    _train.update(state="starting", error=None, step=0, total=0, loss=None, pairs=0, skipped=0, checked=0,
                  unchecked=0, blocked=0, name=name, stop=False, trainer=None, score_lines=[])
    threading.Thread(target=_train_run, args=(model_dir, group_code, list(peers), pairs_path, name, float(epochs),
                                              int(own_layers), files_dir), daemon=True).start()
    return "started"


class _Stopped(Exception):
    pass


def _train_run(model_dir, group_code, peers, pairs_path, name, epochs, own_layers, files_dir):
    try:
        from rosetta_jane.control import _NAME
        from rosetta_jane.mobile.train import Trainer
        if not _NAME.match(name):
            raise ValueError("The run name may use lowercase letters, digits and dashes, up to 40 characters.")
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
        trainer = Trainer(node, ends, os.path.join(files_dir, "work"), total_steps=total, accum=accum,
                          warmup_steps=min(20, max(1, total // 10)), run_id=name, find_wait=10.0, save_every=5)
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
        trainer.export(os.path.join(files_dir, "work", "adapters", name), len(examples), os.path.basename(model_dir),
                       score, len(held))
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


def adapter_poll():
    job = _side["job"]
    return json.dumps(job.report() if job is not None else None)


def adapter_stop():
    if _side["job"] is not None:
        _side["job"].stop()
    return "stopping"
