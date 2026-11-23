"""The controls a front end uses: status, serving on and off, and training runs, over HTTP."""
import json
import os
import time
import urllib.error
import urllib.request

import pytest

from rosetta_jane.api import Gateway
from rosetta_jane.control import load_or_make_token
from rosetta_jane.layers import LayerServer, LocalEnds

from test_swarm import SECRET, _tokenizer, addr, model_dir, node, wait_for  # noqa: F401
from test_train import serving_peer

PAIRS = [{"question": f"w{i} w{i + 3} w{i + 9}", "answer": f"w{i + 1} w{i + 2} w{i + 5}"} for i in range(8)]


def call(base, path, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base + path, None if body is None else json.dumps(body).encode(), headers,
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


@pytest.fixture()
def rig(model_dir, tmp_path):
    """One device that asks, serves layers 2-7 and has controls on, plus one outside serving peer and one visitor."""
    tok = _tokenizer(model_dir)
    hub = node()
    boot = [addr(hub)]
    other, other_server = serving_peer(model_dir, tmp_path, 2, 8, boot)
    me = node(bootstrap=boot)
    mine = LayerServer(model_dir, 2, 8, str(tmp_path / "mine"))
    mine.attach(me)
    visitor = node(bootstrap=boot)
    ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
    ends.eos_token_id = None
    tok.eos_token = None
    work = str(tmp_path / "work")
    os.makedirs(work)
    token = load_or_make_token(os.path.join(work, "control-token"))
    gateway = Gateway(me, ends, tok, "rosetta-test", max_new_tokens=8, find_wait=3.0, redundancy=1, service=mine,
                      control_token=token, model_dir=model_dir, work_dir=work)
    http = gateway.serve("127.0.0.1:0")
    assert wait_for(lambda: any(p.services for p in me.table.peers()))
    yield {"base": f"http://127.0.0.1:{http.server_address[1]}", "token": token, "gateway": gateway, "me": me,
           "visitor": visitor, "other": other_server, "work": work}
    http.shutdown()
    for n in (hub, other, me, visitor):
        n.stop()


def test_the_token_is_kept_and_required(rig, tmp_path):
    base, token = rig["base"], rig["token"]
    assert load_or_make_token(os.path.join(rig["work"], "control-token")) == token and len(token) >= 24
    assert oct(os.stat(os.path.join(rig["work"], "control-token")).st_mode & 0o777) == "0o600"
    assert call(base, "/api/rosetta/service", {"on": False})[0] == 401
    assert call(base, "/api/rosetta/service", {"on": False}, "wrong")[0] == 401
    assert call(base, "/api/rosetta/train", {"pairs": PAIRS}, token + "x")[0] == 401
    assert rig["gateway"].service.paused is False and rig["gateway"].job is None
    status, body = call(base, "/api/rosetta/status")           # looking needs no token
    assert status == 200 and body["controls"] and body["service"]["on"] and body["training"] is None
    rig["gateway"].control_token = None
    assert call(base, "/api/rosetta/service", {"on": False}, token)[0] == 403


def test_the_p2p_service_can_be_switched_off_and_on(rig):
    base, token, visitor = rig["base"], rig["token"], rig["visitor"]
    mine = lambda: [p for p in visitor.table.peers() if p.node_id == rig["me"].node_id and p.services]
    assert wait_for(lambda: mine())
    status, body = call(base, "/api/rosetta/service", {"on": False}, token)
    assert status == 200 and body["service"]["on"] is False
    assert wait_for(lambda: not mine())                        # the swarm stops seeing the layers
    ch = visitor.dial(addr(rig["me"]))
    with pytest.raises(Exception, match="not serving"):
        ch.call({"op": "open", "model": rig["gateway"].ends.fingerprint, "start": 2, "end": 8})
    ch.close()
    status, body = call(base, "/api/rosetta/service", {"on": True, "training": True}, token)
    assert body["service"] == {"on": True, "layers": [2, 8], "training": True, "steps_helped": 0,
                               "examples_helped": 0}
    assert wait_for(lambda: mine() and mine()[0].services[0]["train"])
    assert call(base, "/api/rosetta/service", {"on": "yes"}, token)[0] == 400
    assert call(base, "/api/rosetta/nothing", {}, token)[0] == 404


def test_a_training_run_from_start_to_adapter(rig):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base, token = rig["base"], rig["token"]
    for bad in ({}, {"pairs": []}, {"pairs": [{"question": "x"}]}, {"pairs": PAIRS, "name": "../up"},
                {"pairs": PAIRS, "rank": 100000}, {"pairs": PAIRS, "lr": "fast"}, {"pairs": PAIRS, "epochs": True}):
        status, body = call(base, "/api/rosetta/train", bad, token)
        assert status == 400 and body["error"], bad
    assert rig["gateway"].job is None
    run = {"pairs": PAIRS, "name": "first-run", "steps": 6, "accum": 2, "rank": 4, "alpha": 8, "lr": 5e-3,
           "save_every": 2}
    status, body = call(base, "/api/rosetta/train", run, token)
    assert status == 200 and body["training"]["state"] in ("starting", "running")
    assert body["training"]["total_steps"] == 6 and body["training"]["pairs"] == 7
    assert body["training"]["pairs_kept_back"] == 1          # one pair is never trained on, to score with
    assert call(base, "/api/rosetta/train", run, token)[0] == 400          # one run at a time
    # Answers still come while training is going: the two take turns.
    status, reply = call(base, "/api/chat", {"messages": [{"role": "user", "content": "w3 w9"}], "stream": False,
                                             "options": {"temperature": 0, "num_predict": 4}})
    assert status == 200 and reply["eval_count"] == 4
    before_training = reply["message"]["content"]
    state = lambda: call(base, "/api/rosetta/train")[1]["training"]
    assert wait_for(lambda: state()["state"] not in ("starting", "running", "scoring"), 120)
    final = state()
    score = final["score"]                         # scored by comparison on the pair kept back
    assert score["pairs"] == 1 and score["never_trained_on"] is True and score["answer_tokens"] > 0
    assert score["before"]["loss"] > 0 and score["after"]["loss"] > 0 and "too few to rely on" in score["verdict"]
    assert final["state"] == "done" and final["step"] == 6 and final["error"] is None, final
    assert final["loss"] > 0 and final["unchecked"] == 12 and final["adapter"].endswith("first-run")
    assert call(base, "/api/rosetta/status")[1]["adapters"] == ["first-run"]
    assert rig["other"].examples_trained == 12
    PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(rig["gateway"].model_dir), final["adapter"])
    # A finished adapter is handed over as a download, with the facts needed to judge it. Nothing applies it.
    import io
    import zipfile
    for path, tok, code in (("first-run", None, 401), ("first-run", "wrong", 401), ("nope", token, 404),
                            ("..%2f..%2fcontrol-token", token, 404)):
        req = urllib.request.Request(base + "/api/rosetta/adapters/" + path,
                                     headers={"Authorization": f"Bearer {tok}"} if tok else {})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=30)
        assert e.value.code == code
    req = urllib.request.Request(base + "/api/rosetta/adapters/first-run", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    assert sorted(z.namelist()) == ["first-run/READ-ME-FIRST.txt", "first-run/adapter_config.json",
                                    "first-run/adapter_model.safetensors", "first-run/rosetta-jane-run.json"]
    facts = json.loads(z.read("first-run/rosetta-jane-run.json"))
    assert facts["work_unchecked"] == 12 and facts["work_checked"] == 0 and facts["pairs"] == 7 and facts["rank"] == 4
    assert facts["pairs_kept_back"] == 1 and facts["score"] == score
    note = z.read("first-run/READ-ME-FIRST.txt").decode()
    assert "How it scored:" in note and "never saw in training" in note and "Score it again on your own" in note
    assert "Not checked: 12" in note and "one peer's word" in note and "Nothing\napplies it for you" in note
    # The score card can be read back as text to share. It needs the token, and carries no pairs.
    assert call(base, "/api/rosetta/adapters/first-run/score-card")[0] == 401
    assert call(base, "/api/rosetta/adapters/nope/score-card", token=token)[0] == 404
    status, shared = call(base, "/api/rosetta/adapters/first-run/score-card", token=token)
    assert status == 200 and shared["score"] == score and shared["lines"][0].startswith("Scored on 1 pairs")
    assert shared["share_text"].startswith("Rosetta-Jane score card: first-run")
    assert all(row["question"] not in shared["share_text"] and row["answer"] not in shared["share_text"] for row in PAIRS)
    status, reply = call(base, "/api/chat", {"messages": [{"role": "user", "content": "w3 w9"}], "stream": False,
                                             "options": {"temperature": 0, "num_predict": 4}})
    assert reply["message"]["content"] == before_training     # answers are unchanged by the finished run


def test_a_run_can_be_stopped_and_carried_on(rig):
    base, token = rig["base"], rig["token"]
    run = {"pairs": PAIRS, "name": "long", "steps": 400, "accum": 2, "rank": 4, "alpha": 8, "save_every": 1000}
    call(base, "/api/rosetta/train", run, token)
    state = lambda: call(base, "/api/rosetta/train")[1]["training"]
    assert wait_for(lambda: state()["step"] >= 2, 120)
    status, body = call(base, "/api/rosetta/train/stop", {}, token)
    assert status == 200
    assert wait_for(lambda: state()["state"] == "stopped", 60)
    reached = state()["step"]
    assert 2 <= reached < 400 and state()["adapter"] is None
    call(base, "/api/rosetta/train", {**run, "steps": reached + 2}, token)
    assert wait_for(lambda: state()["state"] == "done", 180)
    assert state()["step"] == reached + 2 and rig["gateway"].job.trainer.redos == 0
    # Only the two new steps were computed: the saved progress was used.
    assert rig["other"].examples_trained == (reached + 2) * 2


def test_training_with_nobody_to_help_says_so(rig):
    base, token = rig["base"], rig["token"]
    rig["other"].training = False
    rig["gateway"].find_wait = 1.0
    time.sleep(1.0)
    call(base, "/api/rosetta/train", {"pairs": PAIRS, "steps": 2, "accum": 2, "rank": 4}, token)
    state = lambda: call(base, "/api/rosetta/train")[1]["training"]
    assert wait_for(lambda: state()["state"] == "failed", 60)
    assert "agreed to help with training" in state()["error"] and "saved" in state()["error"]


def _nomad_admin():
    import shutil
    if not shutil.which("node"):
        return None
    for folder in (os.environ.get("NOMAD_ADMIN_DIR"),
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nomad", "repo", "admin")):
        if folder and os.path.exists(os.path.join(folder, "tests", "standalone", "rosetta_jane_live.standalone.ts")) \
                and os.path.isdir(os.path.join(folder, "node_modules", "ts-node-maintained")):
            return os.path.abspath(folder)
    return None


@pytest.mark.skipif(_nomad_admin() is None, reason="needs node and the NOMAD fork's admin folder with the "
                                                   "Rosetta-Jane page (set NOMAD_ADMIN_DIR)")
def test_nomads_settings_page_code_can_run_the_gateway(rig):
    """NOMAD's own client code does everything its Rosetta-Jane page does, against a live gateway."""
    import subprocess
    admin = _nomad_admin()
    r = subprocess.run(["node", "--import", "ts-node-maintained/register/esm",
                        "tests/standalone/rosetta_jane_live.standalone.ts", rig["base"], rig["token"]],
                       cwd=admin, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["model"] == "rosetta-test" and got["health"] in ("ok", "warning")
    assert got["service_before"] == {"on": True, "layers": [2, 8], "training": False, "stepsHelped": 0,
                                     "examplesHelped": 0}
    assert "token is wrong" in got["without_token"] and "token is wrong" in got["wrong_token"]
    assert got["service_off"]["on"] is False and got["service_on"]["on"] and got["service_on"]["training"]
    assert got["pairs"] == 8 and got["finished"]["pairs"] == 7 and got["started"] in ("starting", "running")
    assert "already going" in got["second_run"]
    done = got["finished"]
    assert done["state"] == "done" and done["step"] == done["totalSteps"] >= 1 and done["error"] is None, done
    assert got["adapters"] == ["from-nomad"] and got["zip_magic"] == "PK" and got["zip_bytes"] > 500
    assert got["score_card"]["shareText"].startswith("Rosetta-Jane score card: from-nomad")
    assert got["finished"]["score"]["pairs"] == 1 and got["finished"]["pairsKeptBack"] == 1
    assert "refused" in got["zip_without_token"] and "no longer on the gateway" in got["zip_missing"]
    assert rig["gateway"].service.training is True and rig["other"].examples_trained > 0
    # The same page code adds someone else's adapter, tries it both ways, and scores it.
    assert "already an adapter" in got["import_twice"]
    assert {"name": "shared-copy", "received": True, "scored": True, "myScore": False} in got["details"]
    assert got["trial_list"]["state"] == "done" and len(got["trial_list"]["answers"]) == 2
    assert all(isinstance(a["without"], str) and isinstance(a["with"], str) for a in got["trial_list"]["answers"])
    assert got["trial_chat"]["state"] == "done" and len(got["trial_chat"]["answers"]) == 1
    assert got["scored"]["state"] == "done" and got["scored"]["kind"] == "score"
    assert got["scored"]["shareText"].startswith("Rosetta-Jane score card: shared-copy")
    assert got["my_card"]["lines"][0].startswith("Scored on 8 pairs chosen by the person scoring")


# -- trying and scoring each other's adapters ------------------------------------------------
def post_raw(base, path, blob, token):
    req = urllib.request.Request(base + path, blob, {"Authorization": f"Bearer {token}",
                                                     "Content-Type": "application/zip"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def shared_adapter(rig, tmp_path, strength=0.6):
    """The zip a friend would send: an adapter for this model that changes its answers, made elsewhere."""
    import torch
    from rosetta_jane.control import adapter_zip
    from rosetta_jane.train import SwarmTrainer
    g = rig["gateway"]
    friend = SwarmTrainer(g.node, g.ends, g.model_dir, str(tmp_path / "friend"), total_steps=1, rank=4, alpha=8.0,
                          targets=["q_proj", "v_proj"], quiet=True)
    gen = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for key in friend._order:
            if key.endswith(".B"):
                friend.adapters[key].copy_(torch.randn(friend.adapters[key].shape, generator=gen) * strength)
    folder = str(tmp_path / "friend" / "adapters" / "from-a-friend")
    friend.export(folder, g.model_dir)
    json.dump({"name": "from-a-friend", "base_model_fingerprint": g.ends.fingerprint, "work_unchecked": 5},
              open(os.path.join(folder, "rosetta-jane-run.json"), "w"))
    return adapter_zip(str(tmp_path / "friend"), "from-a-friend"), friend


def test_an_adapter_someone_shared_can_be_added_tried_and_scored(rig, tmp_path):
    from test_train import ALPHA, RANK, TARGETS, direct_answers, direct_scores
    from rosetta_jane.lan.data import tokenize_pairs
    base, token, g = rig["base"], rig["token"], rig["gateway"]
    rig["other"].training = True
    blob, friend = shared_adapter(rig, tmp_path)
    assert post_raw(base, "/api/rosetta/adapters/import?name=friends-model", blob, "wrong")[0] == 401
    status, body = post_raw(base, "/api/rosetta/adapters/import?name=friends-model", blob, token)
    assert status == 200 and body["imported"] == "friends-model"
    assert {"name": "friends-model", "received": True, "scored": False, "my_score": False, "helpers": 0,
            "continued": False, "community": None, "flattery": None} in body["adapter_details"]
    assert post_raw(base, "/api/rosetta/adapters/import?name=friends-model", blob, token)[0] == 400   # no overwriting
    job = lambda: call(base, "/api/rosetta/status")[1]["adapter_job"]

    # Try it: a list of separate questions, each answered both ways.
    questions = ["w3 w9 w20 w41", "w5 w7"]
    items = [{"without": [{"role": "user", "content": q}], "with": [{"role": "user", "content": q}]} for q in questions]
    status, body = call(base, "/api/rosetta/adapters/try", {"name": "friends-model", "items": items, "max_new": 6}, token)
    assert status == 200 and body["adapter_job"]["total"] == 2
    assert call(base, "/api/rosetta/adapters/try", {"name": "friends-model", "items": items}, token)[0] == 400  # busy
    assert wait_for(lambda: job()["state"] != "running", 120)
    done = job()
    assert done["state"] == "done" and done["kind"] == "try" and done["done"] == 2, done
    for q, got in zip(questions, done["result"]["answers"]):
        ids = g.chat_ids([{"role": "user", "content": q}])
        want_without, want_with = direct_answers(g.model_dir, friend.adapters, ids, 6)
        assert got["without"] == g.tokenizer.decode(want_without, skip_special_tokens=True)
        assert got["with"] == g.tokenizer.decode(want_with, skip_special_tokens=True)
    first = done["result"]["answers"][0]
    assert first["without"] != first["with"]

    # Try it: a chat. Each side carries its own earlier reply forward.
    turn = [{"without": [{"role": "user", "content": questions[0]}, {"role": "assistant", "content": first["without"]},
                         {"role": "user", "content": "w11 w12"}],
             "with": [{"role": "user", "content": questions[0]}, {"role": "assistant", "content": first["with"]},
                      {"role": "user", "content": "w11 w12"}]}]
    call(base, "/api/rosetta/adapters/try", {"name": "friends-model", "items": turn, "max_new": 5}, token)
    assert wait_for(lambda: job()["state"] != "running", 120)
    second = job()["result"]["answers"][0]
    want = direct_answers(g.model_dir, friend.adapters, g.chat_ids(turn[0]["with"]), 5)[1]
    assert second["with"] == g.tokenizer.decode(want, skip_special_tokens=True)
    want = direct_answers(g.model_dir, friend.adapters, g.chat_ids(turn[0]["without"]), 5)[0]
    assert second["without"] == g.tokenizer.decode(want, skip_special_tokens=True)

    # Trying applied nothing: ordinary answers are unchanged.
    status, reply = call(base, "/api/chat", {"messages": [{"role": "user", "content": questions[0]}], "stream": False,
                                             "options": {"temperature": 0, "num_predict": 6}})
    assert reply["message"]["content"] == first["without"]

    # Score it on my own pairs, and get a card of my own to share.
    status, body = call(base, "/api/rosetta/adapters/score", {"name": "friends-model", "pairs": PAIRS}, token)
    assert status == 200
    assert wait_for(lambda: job()["state"] != "running", 120)
    scored = job()
    assert scored["state"] == "done" and scored["kind"] == "score", scored
    card = scored["result"]["score"]
    examples, _ = tokenize_pairs(PAIRS, g.tokenizer, 1024)
    (want_before, want_after), _ = direct_scores(g.model_dir, friend.adapters, examples)
    assert abs(card["before"]["loss"] - want_before["loss"]) < 1e-4 and abs(card["after"]["loss"] - want_after["loss"]) < 1e-4
    assert card["pairs"] == 8 and card["never_trained_on"] is False
    assert scored["result"]["share_text"].startswith("Rosetta-Jane score card: friends-model")
    status, shown = call(base, "/api/rosetta/adapters/friends-model/score-card", token=token)
    assert shown["score"] == card                  # my own score is the one offered for sharing
    assert {"name": "friends-model", "received": True, "scored": False, "my_score": True, "helpers": 0,
            "continued": False, "community": None, "flattery": None} in \
        call(base, "/api/rosetta/status")[1]["adapter_details"]

    for bad in ({"name": "nope", "items": items}, {"name": "friends-model", "items": []},
                {"name": "friends-model", "items": [{"without": [], "with": []}]},
                {"name": "friends-model", "items": items, "max_new": 100000},
                {"name": "friends-model", "items": [{"without": [{"role": "assistant", "content": "x"}],
                                                     "with": [{"role": "user", "content": "x"}]}]}):
        assert call(base, "/api/rosetta/adapters/try", bad, token)[0] == 400, bad


def test_a_shared_zip_is_treated_as_untrusted(tmp_path):
    import io
    import zipfile
    from rosetta_jane.control import import_adapter
    work = str(tmp_path / "work")
    os.makedirs(work)

    def zipped(files):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)
        return out.getvalue()
    config = json.dumps({"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "base_model_name_or_path": "/etc/passwd",
                         "auto_mapping": {"evil": True}})
    good = {"x/adapter_config.json": config, "x/adapter_model.safetensors": b"weights"}
    # Whatever paths the zip claims, only the adapter's own files are written, inside the adapter's folder.
    sneaky = zipped({"../../outside/adapter_config.json": config, "/abs/adapter_model.safetensors": b"weights",
                     "x/evil.py": "import os", "../../../escape.txt": "x", "x/rosetta-jane-run.json": json.dumps({"pairs": 3})})
    assert import_adapter(work, "careful", sneaky) == "careful"
    assert sorted(os.listdir(os.path.join(work, "adapters", "careful"))) == [
        "adapter_config.json", "adapter_model.safetensors", "rosetta-jane-run.json"]
    assert sorted(os.listdir(tmp_path)) == ["work"] and os.listdir(work) == ["adapters"]
    kept = json.load(open(os.path.join(work, "adapters", "careful", "adapter_config.json")))
    assert "auto_mapping" not in kept and "base_model_name_or_path" not in kept and kept["r"] == 4
    facts = json.load(open(os.path.join(work, "adapters", "careful", "rosetta-jane-run.json")))
    assert facts["received"] is True and facts["pairs"] == 3 and facts["name"] == "careful"
    for name, blob, why in (
            ("../up", zipped(good), "name"), ("ok", b"not a zip", "not a zip"), ("ok", zipped({"a.txt": "x"}), "does not hold"),
            ("ok", zipped({**good, "y/adapter_config.json": config}), "more than one"),
            ("ok", zipped({"adapter_config.json": "{", "adapter_model.safetensors": b"w"}), "settings"),
            ("ok", zipped({"adapter_config.json": json.dumps({"r": 9999, "lora_alpha": 8, "target_modules": ["q"]}),
                           "adapter_model.safetensors": b"w"}), "settings"),
            ("ok", zipped({"adapter_config.json": json.dumps({"r": 4, "lora_alpha": 8, "target_modules": ["../x"]}),
                           "adapter_model.safetensors": b"w"}), "settings"),
            ("careful", zipped(good), "already")):
        with pytest.raises(ValueError, match=why):
            import_adapter(work, name, blob)
    assert os.listdir(os.path.join(work, "adapters")) == ["careful"]      # nothing half-written is left behind
