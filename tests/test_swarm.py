"""Tests on a tiny random model over real sockets on one machine (CPU).

The key claim: text produced through a chain of peers is token-for-token what
the whole model produces on one machine, including when a peer dies mid-answer.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
import torch

from rosetta_jane.layers import LayerServer, LocalEnds
from rosetta_jane.net import Node, Peer, clean_services, swarm_secret
from rosetta_jane.swarm import Generation, NoRoute, choose_layers, plan_route

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SECRET = swarm_secret("test group code 123")
VOCAB, LAYERS = 96, 8


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM
    path = str(tmp_path_factory.mktemp("model"))
    torch.manual_seed(5)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    LlamaForCausalLM(cfg).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def reference(model_dir):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()

    def greedy(ids, n):
        with torch.no_grad():
            out = model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False, pad_token_id=0,
                                 eos_token_id=None)
        return out[0, len(ids):].tolist()

    def logits(ids):
        with torch.no_grad():
            return model(torch.tensor([ids])).logits[0, -1]
    return greedy, logits


def record_scores(ends):
    """Capture the next-token scores the asking side computes at every step."""
    seen, real = [], ends.logits

    def spy(h):
        seen.append(real(h))
        return seen[-1]
    ends.logits = spy
    return seen


def assert_scores_match(seen, prompt, produced, logits, tol=1e-4):
    """Every step's scores must equal what the whole model computes for the same text so far."""
    assert len(seen) >= len(produced) > 0
    for t in range(len(produced)):
        diff = float((seen[t] - logits(list(prompt) + list(produced[:t]))).abs().max())
        assert diff < tol, f"step {t}: scores differ from the whole model by {diff:.2e}"


def node(secret=SECRET, bootstrap=None, **kw):
    return Node(secret, "127.0.0.1:0", bootstrap=bootstrap or [], gossip_interval=0.3, quiet=True, **kw).start()


def addr(n):
    return f"127.0.0.1:{n.port}"


def serving(model_dir, tmp_path, start, end, bootstrap=None, **kw):
    n = Node(SECRET, "127.0.0.1:0", bootstrap=bootstrap or [], gossip_interval=0.3, quiet=True)
    server = LayerServer(model_dir, start, end, str(tmp_path / f"srv-{start}-{end}-{n.port}"), **kw)
    server.attach(n)
    return n.start(), server


def wait_for(cond, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


PROMPT = [1, 17, 44, 8, 61, 23, 90, 5, 33]


# -- finding each other -------------------------------------------------------------
def test_peers_find_each_other_from_one_address_each():
    """Six nodes, each told about just one other. Everyone ends up knowing everyone, with no server."""
    nodes = [node()]
    for _ in range(5):
        nodes.append(node(bootstrap=[addr(nodes[-1])]))
    try:
        assert wait_for(lambda: all(len(n.table.peers()) == 5 for n in nodes)), \
            [len(n.table.peers()) for n in nodes]
        ids = {n.node_id for n in nodes}
        for n in nodes:
            assert {p.node_id for p in n.table.peers()} == ids - {n.node_id}
        gone = nodes.pop()                       # a peer that leaves is forgotten
        gone.stop()
        assert wait_for(lambda: all(gone.node_id not in {p.node_id for p in n.table.peers()} for n in nodes))
    finally:
        for n in nodes:
            n.stop()


def test_groups_with_different_codes_never_mix():
    a1, a2 = node(swarm_secret("alpha team code")), None
    b1 = node(swarm_secret("bravo team code"), bootstrap=[addr(a1)])
    pub = node(swarm_secret("public"), bootstrap=[addr(a1)])
    a2 = node(swarm_secret("Alpha Team Code"), bootstrap=[addr(a1)])   # same code, typed differently
    try:
        assert wait_for(lambda: len(a1.table.peers()) == 1 and len(a2.table.peers()) == 1)
        time.sleep(1.0)
        assert {p.node_id for p in a1.table.peers()} == {a2.node_id}
        assert b1.table.peers() == [] and pub.table.peers() == []
    finally:
        for n in (a1, a2, b1, pub):
            n.stop()


def test_a_node_only_passes_on_peers_it_reached_itself():
    from rosetta_jane.net import PeerTable
    table = PeerTable()
    table.suggest("10.1.1.1:7700")               # heard of, never connected to
    table.confirm("ab" * 16, "10.2.2.2:7700", [])   # connected to in person
    assert table.candidates() == ["10.1.1.1:7700"]
    assert table.addresses_to_share() == ["10.2.2.2:7700"]
    for junk in ("not an address", "10.1.1.1:99999", "10.1.1.1:0", "x" * 200 + ":80", 12345, "a b:80"):
        table.suggest(junk)                      # malformed rumours are dropped outright
    assert table.candidates() == ["10.1.1.1:7700"]
    table.fail("10.2.2.2:7700")
    table.fail("10.2.2.2:7700")                  # unreachable twice: forgotten and no longer shared
    assert table.addresses_to_share() == [] and table.peers() == []


def test_local_network_announcements_bring_nodes_together():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        udp = s.getsockname()[1]
    listener = Node(SECRET, "127.0.0.1:0", gossip_interval=0.3, quiet=True, beacon=True, beacon_port=udp,
                    beacon_targets=[]).start()
    announcer = Node(SECRET, "127.0.0.1:0", gossip_interval=0.3, quiet=True, beacon=True,
                     beacon_port=udp + 1 if udp < 65000 else udp - 1, beacon_targets=[("127.0.0.1", udp)]).start()
    stranger = Node(swarm_secret("some other group"), "127.0.0.1:0", gossip_interval=0.3, quiet=True, beacon=True,
                    beacon_port=udp + 2 if udp < 65000 else udp - 2, beacon_targets=[("127.0.0.1", udp)]).start()
    try:
        assert wait_for(lambda: {p.node_id for p in listener.table.peers()} == {announcer.node_id})
        time.sleep(0.8)
        assert {p.node_id for p in listener.table.peers()} == {announcer.node_id}   # stranger ignored
    finally:
        for n in (listener, announcer, stranger):
            n.stop()


def test_bad_announcements_are_rejected():
    good = [{"model": "ab" * 8, "variant": "float32", "layers": [2, 5], "hidden": 32, "free": 3, "train": False,
             "teams": ["0123456789abcdef"]}]
    assert clean_services(good) == good
    older = [{k: v for k, v in good[0].items() if k != "teams"}]      # a peer from before teams existed
    assert clean_services(older) == [{**older[0], "teams": []}]
    for bad in ([{"model": "ab" * 8, "layers": [5, 2], "hidden": 32}],
                [{"model": "../../etc", "layers": [1, 2], "hidden": 32}],
                [{"model": "ab" * 8, "layers": [1, 2], "hidden": -1}],
                [{"model": "ab" * 8, "layers": [1, 2], "hidden": 32, "variant": "x" * 99}],
                [{**good[0], "teams": ["not a team id"]}], [{**good[0], "teams": ["0123456789abcdef"] * 99}],
                [{**good[0], "teams": "0123456789abcdef"}],
                "nope", [good[0]] * 50):
        with pytest.raises(ValueError):
            clean_services(bad)


# -- routing ----------------------------------------------------------------------------
def test_a_new_serving_peer_takes_the_layers_the_swarm_is_shortest_of():
    import random
    def peer(i, a, b, model="m"):
        return Peer(f"10.0.0.{i}:1", f"id{i}", [{"model": model, "variant": "float32", "layers": [a, b],
                                               "hidden": 8, "free": 1, "train": False}])
    rng = random.Random(0)
    first = choose_layers([], "m", 24, 8, rng)
    assert first[1] - first[0] == 8 and first[0] >= 1 and first[1] <= 24
    assert choose_layers([peer(1, 1, 16)], "m", 24, 8, rng) == (16, 24)        # the uncovered end
    assert choose_layers([peer(1, 1, 10), peer(2, 14, 24)], "m", 24, 4, rng) == (10, 14)   # the hole in the middle
    # Everything covered once, one stretch covered twice: go where there is only one.
    start, end = choose_layers([peer(1, 1, 24), peer(2, 1, 12)], "m", 24, 6, rng)
    assert start >= 12 and end <= 24
    assert choose_layers([peer(1, 1, 16, model="other")], "m", 24, 30, rng) == (1, 24)   # never layer 0, never past the end
    # Serving peers arriving one after another end up covering the whole model.
    peers, covered = [], set()
    for i in range(4):
        a, b = choose_layers(peers, "m", 25, 6, rng)
        peers.append(peer(i, a, b))
        covered |= set(range(a, b))
    assert covered == set(range(1, 25))


def test_route_planning():
    def peer(i, a, b, free=2, variant="float32"):
        return Peer(f"{i:032x}", f"h{i}:1", [{"model": "m" * 16, "variant": variant, "layers": [a, b],
                                              "hidden": 32, "free": free}])
    peers = [peer(1, 2, 5), peer(2, 4, 8), peer(3, 5, 6), peer(4, 2, 4)]
    route = plan_route(peers, "m" * 16, 2, 8)
    assert [(h.peer.node_id[-1], h.start, h.end) for h in route] == [("1", 2, 5), ("2", 5, 8)]   # uses part of 2
    with pytest.raises(NoRoute, match="layer 6"):
        plan_route(peers, "m" * 16, 2, 8, exclude={peers[1].node_id})
    with pytest.raises(NoRoute):
        plan_route(peers, "x" * 16, 2, 8)
    with pytest.raises(NoRoute):
        plan_route(peers, "m" * 16, 2, 8, variant="4bit")


# -- answering ----------------------------------------------------------------------------
@pytest.mark.parametrize("own,ranges,hops", [(2, [(2, 8)], 1), (2, [(2, 5), (5, 8)], 2),
                                             (1, [(1, 4), (3, 6), (5, 8)], 3), (3, [(2, 7), (6, 8)], 2),
                                             (8, [], 0)])
def test_swarm_text_matches_single_machine(model_dir, reference, tmp_path, own, ranges, hops):
    greedy, logits = reference
    hub = node()
    servers = [serving(model_dir, tmp_path, a, b, bootstrap=[addr(hub)]) for a, b in ranges]
    asker = node(bootstrap=[addr(hub)])
    try:
        ends = LocalEnds(model_dir, own, str(tmp_path / "ends"))
        assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == len(ranges))
        seen = record_scores(ends)
        gen = Generation(asker, ends)
        got = gen.run(PROMPT, 20)
        assert got == greedy(PROMPT, 20)
        assert len(gen.route) == hops
        assert_scores_match(seen, PROMPT, got, logits)   # the scores match too, not just the choices
        assert all(s.service()["free"] == s.max_sessions for _, s in servers)   # sessions were released
    finally:
        for n in [hub, asker] + [n for n, _ in servers]:
            n.stop()


FAMILIES = {"qwen2": ("Qwen2Config", "Qwen2ForCausalLM"), "qwen3": ("Qwen3Config", "Qwen3ForCausalLM"),
            "mistral": ("MistralConfig", "MistralForCausalLM"), "phi3": ("Phi3Config", "Phi3ForCausalLM"),
            "gemma": ("GemmaConfig", "GemmaForCausalLM"), "gemma3": ("Gemma3TextConfig", "Gemma3ForCausalLM"),
            "mixtral": ("MixtralConfig", "MixtralForCausalLM"),
            "starcoder2": ("Starcoder2Config", "Starcoder2ForCausalLM")}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_other_model_families_answer_identically(tmp_path, family):
    """Each verified family, through three peers with overlapping ranges, scored at every step."""
    import transformers
    cfg_cls, model_cls = (getattr(transformers, n) for n in FAMILIES[family])
    torch.manual_seed(9)
    extra = {"head_dim": 8} if family.startswith("gemma") else {}
    path = str(tmp_path / "model")
    cfg_cls_kwargs = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                          pad_token_id=0, bos_token_id=1, eos_token_id=2, **extra)
    model_cls(cfg_cls(**cfg_cls_kwargs)).save_pretrained(path)
    whole = transformers.AutoModelForCausalLM.from_pretrained(path).eval()

    def logits(ids):
        with torch.no_grad():
            return whole(torch.tensor([ids])).logits[0, -1]

    want, ids = [], list(PROMPT)
    for _ in range(12):
        want.append(int(logits(ids).argmax()))
        ids.append(want[-1])
    hub = node()
    servers = [serving(path, tmp_path, a, b, bootstrap=[addr(hub)]) for a, b in [(1, 4), (3, 6), (5, 8)]]
    asker = node(bootstrap=[addr(hub)])
    try:
        ends = LocalEnds(path, 1, str(tmp_path / "ends"))
        assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == 3)
        seen = record_scores(ends)
        gen = Generation(asker, ends)
        got = gen.run(PROMPT, 12)
        assert [(h.start, h.end) for h in gen.route] == [(1, 4), (4, 6), (6, 8)]
        assert got == want
        assert_scores_match(seen, PROMPT, got, logits)
    finally:
        for n in [hub, asker] + [n for n, _ in servers]:
            n.stop()


def test_peer_dying_mid_answer_is_replaced_and_text_is_unchanged(model_dir, reference, tmp_path):
    greedy, logits = reference
    hub = node()
    first = serving(model_dir, tmp_path, 2, 5, bootstrap=[addr(hub)])
    victim = serving(model_dir, tmp_path, 5, 8, bootstrap=[addr(hub)])
    asker = node(bootstrap=[addr(hub)])
    spare = []
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == 2)
        seen = record_scores(ends)
        gen = Generation(asker, ends)
        count = [0]

        def on_token(_):
            count[0] += 1
            if count[0] == 7:                     # mid-answer: the second peer vanishes,
                victim[0].stop()                  # and two other peers that can cover its layers appear
                for hop in gen.route:
                    if hop.peer.node_id == victim[0].node_id:
                        hop.ch.sock.close()
                spare.append(serving(model_dir, tmp_path, 4, 7, bootstrap=[addr(hub)]))
                spare.append(serving(model_dir, tmp_path, 6, 8, bootstrap=[addr(hub)]))
            return True

        got = gen.run(PROMPT, 18, on_token=on_token)
        assert got == greedy(PROMPT, 18)
        assert_scores_match(seen, PROMPT, got, logits)   # including every step after the replacement
        assert gen.reroutes == 1 and len(gen.route) == 3
        assert [(h.start, h.end) for h in gen.route] == [(2, 5), (5, 7), (7, 8)]
    finally:
        for n in [hub, asker, first[0], victim[0]] + [n for n, _ in spare]:
            n.stop()


def test_no_peers_for_some_layers_gives_a_clear_error(model_dir, tmp_path):
    hub = node()
    only = serving(model_dir, tmp_path, 2, 5, bootstrap=[addr(hub)])
    asker = node(bootstrap=[addr(hub)])
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert wait_for(lambda: any(p.services for p in asker.table.peers()))
        with pytest.raises(NoRoute, match="layer 5"):
            Generation(asker, ends, find_wait=1.0).run(PROMPT, 4)
    finally:
        for n in (hub, only[0], asker):
            n.stop()


def test_server_limits_and_bad_requests(model_dir, tmp_path):
    n, server = serving(model_dir, tmp_path, 2, 6, max_sessions=1, max_tokens=12)
    try:
        fp = server.fingerprint
        ch = n.dial(addr(n))
        from rosetta_jane.lan.protocol import RemoteError
        with pytest.raises(RemoteError, match="different model"):
            ch.call({"op": "open", "model": "0" * 16, "start": 2, "end": 6})
        with pytest.raises(RemoteError, match="serves layers 2:6"):
            ch.call({"op": "open", "model": fp, "start": 1, "end": 6})
        with pytest.raises(RemoteError, match="no open session"):
            ch.call({"op": "step"}, tensor=torch.zeros(1, 1, 32))
        ch.call({"op": "open", "model": fp, "start": 3, "end": 5})
        for bad in (torch.zeros(1, 2, 31), torch.zeros(2, 2, 32), torch.zeros(1, 32),
                    torch.zeros(1, 2, 32, dtype=torch.int64)):
            with pytest.raises(RemoteError):
                ch.call({"op": "step"}, tensor=bad)
        ch.call({"op": "step"}, tensor=torch.zeros(1, 10, 32))
        with pytest.raises(RemoteError, match="limited to 12 tokens"):
            ch.call({"op": "step"}, tensor=torch.zeros(1, 3, 32))
        other = n.dial(addr(n))                   # the single slot is taken
        with pytest.raises(RemoteError, match="busy"):
            other.call({"op": "open", "model": fp, "start": 2, "end": 6})
        ch.close()                                # disconnecting frees the slot
        assert wait_for(lambda: server.service()["free"] == 1)
        other.call({"op": "open", "model": fp, "start": 2, "end": 6})
        other.close()
        with pytest.raises(ValueError, match="layer 0 always stays"):
            LayerServer(model_dir, 0, 3, str(tmp_path / "x"))
    finally:
        n.stop()


# -- the door NOMAD uses --------------------------------------------------------------------
def _tokenizer(model_dir):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    words = ["[UNK]", "<s>", "</s>", "<|", "|>", "user", "assistant", "system"] + [f"w{i}" for i in range(VOCAB - 8)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", bos_token="<s>", eos_token="</s>",
                                   pad_token="[UNK]")
    fast.chat_template = ("{% for m in messages %}<| {{ m['role'] }} |> {{ m['content'] }} {% endfor %}"
                          "{% if add_generation_prompt %}<| assistant |>{% endif %}")
    fast.save_pretrained(model_dir)
    return fast


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read().decode()


def test_ollama_style_api(model_dir, reference, tmp_path):
    from rosetta_jane.api import Gateway
    greedy, _ = reference
    tok = _tokenizer(model_dir)
    hub = node()
    servers = [serving(model_dir, tmp_path, 2, 5, bootstrap=[addr(hub)]),
               serving(model_dir, tmp_path, 5, 8, bootstrap=[addr(hub)])]
    asker = node(bootstrap=[addr(hub)])
    http = None
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        ends.eos_token_id = None                  # the random model has no real stop token
        tok.eos_token_id_backup, tok.eos_token = tok.eos_token_id, None
        assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == 2)
        gateway = Gateway(asker, ends, tok, "rosetta-test", max_new_tokens=16, find_wait=1.0)
        http = gateway.serve("127.0.0.1:0")
        base = f"http://127.0.0.1:{http.server_address[1]}"
        messages = [{"role": "user", "content": "w3 w9 w20 w41"}]
        want_ids = greedy(gateway.chat_ids(messages), 10)
        want_text = tok.decode(want_ids, skip_special_tokens=True)

        with urllib.request.urlopen(base + "/api/tags", timeout=10) as r:
            assert json.load(r)["models"][0]["name"] == "rosetta-test"
        with urllib.request.urlopen(base + "/api/version", timeout=10) as r:
            assert "rosetta" in json.load(r)["version"]

        opts = {"temperature": 0, "num_predict": 10}
        status, body = _post(base + "/api/chat", {"model": "rosetta-test", "messages": messages,
                                                  "stream": False, "options": opts})
        reply = json.loads(body)
        assert status == 200 and reply["done"] and reply["message"]["content"] == want_text
        assert reply["eval_count"] == 10 and reply["peers_used"] == 2

        status, body = _post(base + "/api/chat", {"model": "rosetta-test", "messages": messages, "options": opts})
        lines = [json.loads(line) for line in body.strip().split("\n")]
        assert len(lines) > 2 and lines[-1]["done"] and not lines[0]["done"]
        assert "".join(line["message"]["content"] for line in lines) == want_text

        status, body = _post(base + "/api/generate", {"prompt": "w3 w9", "stream": False, "options": opts})
        assert json.loads(body)["response"] == tok.decode(greedy(gateway.text_ids("w3 w9"), 10),
                                                         skip_special_tokens=True)
        stop_word = want_text.split()[4]
        status, body = _post(base + "/api/chat", {"messages": messages, "stream": False,
                                                  "options": {**opts, "stop": [stop_word]}})
        cut = json.loads(body)
        assert cut["done_reason"] == "stop" and stop_word not in cut["message"]["content"]
        assert want_text.startswith(cut["message"]["content"])

        for path, payload in (("/api/chat", {"messages": [{"role": "robot", "content": "x"}]}),
                              ("/api/embed", {"input": "x"})):
            with pytest.raises(urllib.error.HTTPError) as e:
                _post(base + path, payload)
            assert e.value.code in (400, 404, 501) and "error" in json.loads(e.value.read())

        for n, _ in servers:                      # the swarm goes away: a clear error, not a hang
            n.stop()
        asker.table._peers.clear()
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/api/chat", {"messages": messages, "stream": False, "options": opts})
        assert e.value.code == 503 and "cannot answer" in json.loads(e.value.read())["error"]
    finally:
        if http:
            http.shutdown()
        for n in [hub, asker] + [n for n, _ in servers]:
            n.stop()


def _nomad_modules():
    """A folder with the `ollama` and `openai` npm packages, the client libraries NOMAD uses."""
    import shutil
    if not shutil.which("node"):
        return None
    for folder in (os.environ.get("NOMAD_ADMIN_DIR"), os.path.join(ROOT, "..", "nomad", "repo", "admin")):
        if folder and all(os.path.isdir(os.path.join(folder, "node_modules", m)) for m in ("ollama", "openai")):
            return os.path.abspath(folder)
    return None


@pytest.mark.skipif(_nomad_modules() is None, reason="needs node and NOMAD's admin folder with npm packages "
                                                      "installed (set NOMAD_ADMIN_DIR)")
def test_nomads_own_client_libraries_can_talk_to_the_gateway(model_dir, reference, tmp_path):
    from rosetta_jane.api import Gateway
    greedy, _ = reference
    tok = _tokenizer(model_dir)
    hub = node()
    servers = [serving(model_dir, tmp_path, 2, 8, bootstrap=[addr(hub)])]
    asker = node(bootstrap=[addr(hub)])
    http = None
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        ends.eos_token_id = None
        tok.eos_token = None
        assert wait_for(lambda: any(p.services for p in asker.table.peers()))
        gateway = Gateway(asker, ends, tok, "rosetta-test", max_new_tokens=16, find_wait=1.0)
        idle = gateway.status()
        assert idle["can_answer"] and idle["peers_serving"] == 1 and idle["last_answer"] is None
        assert idle["layers_missing"] == [] and idle["own_layers"] == 2 and idle["layers"] == LAYERS
        http = gateway.serve("127.0.0.1:0")
        base = f"http://127.0.0.1:{http.server_address[1]}"
        want = tok.decode(greedy(gateway.chat_ids([{"role": "user", "content": "w3 w9 w20 w41"}]), 10),
                          skip_special_tokens=True)
        r = subprocess.run(["node", os.path.join(HERE, "nomad_clients.mjs"), base, _nomad_modules()],
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stdout + r.stderr
        got = json.loads(r.stdout.strip().splitlines()[-1])
        assert got["v1_models_ok"] and got["native"] and got["model"] == "rosetta-test"
        assert got["openai_models"] == ["rosetta-test"] and got["thinking"] is False
        assert got["context_length"] == gateway.context_length() > 0
        for key in ("native_text", "native_stream_text", "openai_text", "openai_stream_text"):
            assert got[key] == want, key
        assert got["native_count"] == 10 and got["openai_tokens"] == 10 and got["native_stream_pieces"] > 2
        assert got["hint_done"] and got["embed_status"] == 501
        assert got["status"]["last_answer"]["peers_used"] == 1 and got["status"]["can_answer"]
        servers[0][0].stop()
        asker.table._peers.clear()
        gone = gateway.status()
        assert not gone["can_answer"] and gone["layers_missing"] == list(range(2, LAYERS))
    finally:
        if http:
            http.shutdown()
        for n in [hub, asker] + [n for n, _ in servers]:
            n.stop()


def test_command_line_serve_and_ask(model_dir, tmp_path):
    _tokenizer(model_dir)
    env = dict(os.environ, PYTHONPATH=ROOT, ROSETTA_JANE_GROUP_CODE="cli test code 99")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen([sys.executable, "-m", "rosetta_jane", "serve", "--model", model_dir, "--layers", "2:8",
                               "--listen", f"127.0.0.1:{port}", "--work-dir", str(tmp_path / "s")],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        def up():
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                return True
            except OSError:
                return server.poll() is not None
        assert wait_for(up, 90) and server.poll() is None
        run = lambda *a: subprocess.run([sys.executable, "-m", "rosetta_jane", *a], env=env, capture_output=True,
                                        text=True, timeout=120)
        peers_cmd = ("peers", "--peer", f"127.0.0.1:{port}", "--work-dir", str(tmp_path / "p"))
        assert wait_for(lambda: "layers 2-7" in run(*peers_cmd).stdout, 120)   # joined first, then loaded
        r = run(*peers_cmd)
        assert r.returncode == 0 and "1 peers reachable" in r.stdout and "layers 2-7" in r.stdout, r.stdout + r.stderr
        ask = ("ask", "w3 w9 w20", "--model", model_dir, "--own-layers", "2", "--peer", f"127.0.0.1:{port}",
               "--work-dir", str(tmp_path / "a"), "--max-new-tokens", "8", "--temperature", "0")
        r = run(*ask, "--no-auto-calibrate")
        assert r.returncode == 0 and "through 1 peers" in r.stdout, r.stdout + r.stderr
        assert "0 known-answer tests" in r.stdout and "no known-answer tests prepared" in r.stderr
        r = run("calibrate", "--model", model_dir, "--own-layers", "2", "--count", "4", "--chunk", "3",
                "--max-tokens", "12", "--work-dir", str(tmp_path / "a"))
        assert r.returncode == 0 and "4 tests" in r.stdout, r.stdout + r.stderr
        r = run(*ask)                              # the same question now tests the peer first
        assert r.returncode == 0 and "1 known-answer tests" in r.stdout and "0 peers blocked" in r.stdout, r.stdout
        assert "no known-answer tests prepared" not in r.stderr
        # On a fresh device nobody has to remember that step: the first run prepares the tests itself.
        fresh = [str(tmp_path / "fresh") if a == str(tmp_path / "a") else a for a in ask]
        r = run(*fresh)
        assert r.returncode == 0 and "preparing known-answer tests" in r.stderr, r.stdout + r.stderr
        assert "1 known-answer tests" in r.stdout and "0 peers blocked" in r.stdout, r.stdout
        r = run(*fresh)
        assert r.returncode == 0 and "preparing known-answer tests" not in r.stderr
        wrong = dict(env, ROSETTA_JANE_GROUP_CODE="a different code 11")
        # A second serving peer left on 'auto' looks at the swarm and takes the one layer nobody is serving.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port2 = sock.getsockname()[1]
        second = subprocess.Popen([sys.executable, "-m", "rosetta_jane", "serve", "--model", model_dir, "--span", "1",
                                   "--listen", f"127.0.0.1:{port2}", "--peer", f"127.0.0.1:{port}",
                                   "--work-dir", str(tmp_path / "s2")], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE, text=True)
        try:
            def listed():
                r = run("peers", "--peer", f"127.0.0.1:{port}", "--work-dir", str(tmp_path / "p2"), "--rounds", "2")
                return "layers 1-1" in r.stdout or second.poll() is not None
            assert wait_for(listed, 120) and second.poll() is None
        finally:
            second.kill()
            assert "thinnest at layers 1-1" in second.communicate()[1]
        r = subprocess.run([sys.executable, "-m", "rosetta_jane", "peers", "--peer", f"127.0.0.1:{port}",
                            "--work-dir", str(tmp_path / "w"), "--rounds", "2"], env=wrong, capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0 and "0 peers reachable" in r.stdout
    finally:
        server.kill()


# -- catching peers that lie -------------------------------------------------------------------
class Liar(LayerServer):
    """A peer that returns wrong numbers. Every Liar gives the same wrong answer, as colluding peers would."""

    def _step(self, ch, header, tensor, state):
        class Twisted:
            def send(_, hdr, tensor=None):
                ch.send(hdr, tensor=tensor + 0.5 if tensor is not None else None)
        super()._step(Twisted(), header, tensor, state)


def lying(model_dir, tmp_path, start, end, bootstrap):
    n = Node(SECRET, "127.0.0.1:0", bootstrap=bootstrap, gossip_interval=0.3, quiet=True)
    server = Liar(model_dir, start, end, str(tmp_path / f"liar-{start}-{end}-{n.port}"))
    server.attach(n)
    return n.start(), server


def swarm(model_dir, tmp_path, honest, liars, own=2):
    hub = node()
    boot = [addr(hub)]
    good = [serving(model_dir, tmp_path, a, b, bootstrap=boot) for a, b in honest]
    bad = [lying(model_dir, tmp_path, a, b, boot) for a, b in liars]
    asker = node(bootstrap=boot)
    ends = LocalEnds(model_dir, own, str(tmp_path / "ends"))
    assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == len(honest) + len(liars))
    everyone = [hub, asker] + [n for n, _ in good + bad]
    return asker, ends, good, bad, everyone


def test_honest_peers_agree_and_every_step_is_checked(model_dir, reference, tmp_path):
    greedy, logits = reference
    asker, ends, good, _, everyone = swarm(model_dir, tmp_path, [(2, 8), (2, 8), (2, 5), (5, 8)], [])
    try:
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=2, require_verified=True)
        got = gen.run(PROMPT, 10)
        assert got == greedy(PROMPT, 10)
        assert_scores_match(seen, PROMPT, got, logits)
        assert gen.verified_steps == 10 * len(gen.route) and gen.unverified_steps == 0
        assert gen.disputes == 0 and gen.caught == [] and asker.banned == set()
        assert all(len(h.witnesses) == 1 and h.witnesses[0].peer.node_id != h.peer.node_id for h in gen.route)
    finally:
        for n in everyone:
            n.stop()


def test_a_liar_is_caught_by_recomputing_the_disputed_step(model_dir, reference, tmp_path):
    from rosetta_jane.layers import LocalAuditor
    greedy, logits = reference
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 8)], [(2, 8)])
    try:
        auditor = LocalAuditor(model_dir, str(tmp_path / "audit"))
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=2, auditor=auditor)
        got = gen.run(PROMPT, 10)
        assert got == greedy(PROMPT, 10)
        assert_scores_match(seen, PROMPT, got, logits)
        liar_id = bad[0][0].node_id
        assert gen.caught == [liar_id] and gen.disputes == 1 and auditor.audits == 1
        assert liar_id in asker.banned and good[0][0].node_id not in asker.banned
        # The blocked peer is not used for later answers, even though it is still online.
        seen.clear()
        again = Generation(asker, ends, redundancy=2, auditor=auditor)
        assert again.run(PROMPT, 6) == greedy(PROMPT, 6)
        assert again.disputes == 0 and again.route[0].peer.node_id == good[0][0].node_id
        assert again.unverified_steps == 6          # only one trustworthy peer is left, and it says so
    finally:
        for n in everyone:
            n.stop()


def test_one_liar_among_three_is_outvoted_without_an_auditor(model_dir, reference, tmp_path):
    greedy, logits = reference
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 8), (2, 8)], [(2, 8)])
    try:
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=3)
        got = gen.run(PROMPT, 10)
        assert got == greedy(PROMPT, 10)
        assert_scores_match(seen, PROMPT, got, logits)
        assert gen.caught == [bad[0][0].node_id] and gen.disputes == 1
    finally:
        for n in everyone:
            n.stop()


def test_two_peers_disagree_and_a_third_is_called_in(model_dir, reference, tmp_path):
    greedy, logits = reference
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 8)], [(2, 8)])
    try:
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=2)
        real, late = gen._recruit, []

        def recruit(hop, count, replay, taken):
            if replay is not None and not late:   # the dispute: only now does another honest peer come online
                late.append(serving(model_dir, tmp_path, 2, 8, bootstrap=[addr(everyone[0])]))
                everyone.append(late[0][0])
                assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == 3)
            return real(hop, count, replay, taken)

        gen._recruit = recruit
        got = gen.run(PROMPT, 8)
        assert got == greedy(PROMPT, 8)
        assert_scores_match(seen, PROMPT, got, logits)
        assert gen.caught == [bad[0][0].node_id] and gen.disputes == 1
        assert len(gen.route[0].replicas()) == 2      # the newcomer stayed on as the second opinion
        assert gen.verified_steps == 8
    finally:
        for n in everyone:
            n.stop()


def test_two_peers_disagree_with_nobody_to_break_the_tie(model_dir, tmp_path):
    from rosetta_jane.swarm import Disputed
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 8)], [(2, 8)])
    try:
        with pytest.raises(Disputed, match="nobody is left to break the tie"):
            Generation(asker, ends, redundancy=2).run(PROMPT, 4)
        assert asker.banned == set()                 # nobody is blocked on a guess
    finally:
        for n in everyone:
            n.stop()


def test_a_fake_majority_wins_a_vote_but_not_an_audit(model_dir, reference, tmp_path):
    """The known limit of voting: two colluding liars outvote one honest peer. Auditing is not fooled."""
    from rosetta_jane.layers import LocalAuditor
    greedy, logits = reference
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 8)], [(2, 8), (2, 8)])
    try:
        honest_id = good[0][0].node_id
        seen = record_scores(ends)
        voted = Generation(asker, ends, redundancy=3)
        voted.run(PROMPT, 3)
        assert voted.caught == [honest_id]           # the vote blocked the honest peer,
        assert float((seen[0] - logits(PROMPT)).abs().max()) > 1e-2   # and the lie was accepted

        asker.banned.clear()
        seen.clear()
        auditor = LocalAuditor(model_dir, str(tmp_path / "audit"))
        checked = Generation(asker, ends, redundancy=3, auditor=auditor)
        got = checked.run(PROMPT, 10)
        assert got == greedy(PROMPT, 10)
        assert_scores_match(seen, PROMPT, got, logits)
        assert sorted(checked.caught) == sorted(n.node_id for n, _ in bad)
        assert honest_id not in asker.banned
    finally:
        for n in everyone:
            n.stop()


def test_a_community_that_is_mostly_liars(model_dir, reference, tmp_path):
    """Three colluding liars and one honest peer. The liars never disagree with each other, so only
    checking their work against this device's own computation exposes them."""
    from rosetta_jane.layers import LocalAuditor
    greedy, logits = reference
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 8)], [(2, 8), (2, 8), (2, 8)])
    try:
        liar_ids = sorted(n.node_id for n, _ in bad)
        for redundancy in (1, 2):
            asker.banned.clear()
            seen = record_scores(ends)
            gen = Generation(asker, ends, redundancy=redundancy, audit_rate=1.0, seed=4,
                             auditor=LocalAuditor(model_dir, str(tmp_path / f"audit{redundancy}")))
            got = gen.run(PROMPT, 8)
            assert got == greedy(PROMPT, 8)
            assert_scores_match(seen[-8:], PROMPT, got, logits)
            assert set(gen.caught) <= set(liar_ids) and good[0][0].node_id not in asker.banned
            assert gen.route[0].peer.node_id == good[0][0].node_id
        # With no honest peer at all, the answer is refused; a wrong one is never returned.
        good[0][0].stop()
        asker.banned.clear()
        asker.table.fail(addr(good[0][0])); asker.table.fail(addr(good[0][0]))
        gen = Generation(asker, ends, redundancy=2, audit_rate=1.0, find_wait=1.0,
                         auditor=LocalAuditor(model_dir, str(tmp_path / "audit3")))
        with pytest.raises(NoRoute):
            gen.run(PROMPT, 4)
        assert sorted(set(gen.caught)) == liar_ids
    finally:
        for n in everyone:
            n.stop()


def test_random_checks_catch_liars_who_agree_with_each_other(model_dir, reference, tmp_path):
    """Without spot checks, two peers telling the same lie pass. The share checked is honoured."""
    from rosetta_jane.layers import LocalAuditor
    greedy, logits = reference
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [], [(2, 8), (2, 8)])
    try:
        seen = record_scores(ends)
        fooled = Generation(asker, ends, redundancy=2, auditor=LocalAuditor(model_dir, str(tmp_path / "a0")))
        fooled.run(PROMPT, 3)                        # an auditor that is never consulted: they agree
        assert fooled.disputes == 0 and fooled.caught == [] and fooled.spot_audits == 0
        assert float((seen[0] - logits(PROMPT)).abs().max()) > 1e-2
        gen = Generation(asker, ends, redundancy=2, audit_rate=0.5, find_wait=1.0,
                         auditor=LocalAuditor(model_dir, str(tmp_path / "a1")))
        with pytest.raises(NoRoute):
            gen.run(PROMPT, 3)                       # caught on the very first step, which is always checked
        assert len(gen.caught) == 2 and gen.spot_audits == 1
    finally:
        for n in everyone:
            n.stop()
    asker, ends, good, _, everyone = swarm(model_dir, tmp_path, [(2, 8)], [])
    try:
        gen = Generation(asker, ends, audit_rate=0.3, seed=1,
                         auditor=LocalAuditor(model_dir, str(tmp_path / "a2")))
        assert gen.run(PROMPT, 40) == greedy(PROMPT, 40)
        assert 5 <= gen.spot_audits <= 22 and gen.verified_steps == gen.spot_audits   # roughly 30% of 40
        assert gen.unverified_steps == 40 - gen.spot_audits and gen.caught == []
    finally:
        for n in everyone:
            n.stop()


def test_every_peer_lying_is_survived_when_honest_ones_exist_elsewhere(model_dir, reference, tmp_path):
    """Both peers first chosen for a stretch lie in the same way; the audit rejects both and reroutes."""
    from rosetta_jane.layers import LocalAuditor
    greedy, logits = reference
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 5), (5, 8)], [(2, 8), (2, 8)])
    try:
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=2, audit_rate=1.0,
                         auditor=LocalAuditor(model_dir, str(tmp_path / "audit")))
        got = gen.run(PROMPT, 8)
        assert got == greedy(PROMPT, 8)
        assert_scores_match(seen[-8:], PROMPT, got, logits)
        assert sorted(gen.caught) == sorted(n.node_id for n, _ in bad) and gen.reroutes == 1
        assert [(h.start, h.end) for h in gen.route] == [(2, 5), (5, 8)]
    finally:
        for n in everyone:
            n.stop()


def test_second_peer_takes_over_instantly_when_the_first_dies(model_dir, reference, tmp_path):
    greedy, logits = reference
    asker, ends, good, _, everyone = swarm(model_dir, tmp_path, [(2, 8), (2, 8)], [])
    try:
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=2)
        count = [0]

        def on_token(_):
            count[0] += 1
            if count[0] == 4:
                gen.route[0].ch.sock.close()         # the first peer's connection drops
            return True

        got = gen.run(PROMPT, 10, on_token=on_token)
        assert got == greedy(PROMPT, 10)
        assert_scores_match(seen, PROMPT, got, logits)
        assert gen.reroutes == 0 and gen.caught == [] and asker.banned == set()   # no replay, nobody blamed
        assert gen.unverified_steps == 6             # and it reports the steps it could no longer check
    finally:
        for n in everyone:
            n.stop()


def test_checked_answers_can_be_required(model_dir, tmp_path):
    asker, ends, good, _, everyone = swarm(model_dir, tmp_path, [(2, 8)], [])
    try:
        with pytest.raises(NoRoute, match="fewer than 2 independent peers"):
            Generation(asker, ends, redundancy=2, require_verified=True, find_wait=1.0).run(PROMPT, 3)
        gen = Generation(asker, ends, redundancy=2)   # not required: it answers and reports it was unchecked
        gen.run(PROMPT, 3)
        assert gen.unverified_steps == 3 and gen.verified_steps == 0
    finally:
        for n in everyone:
            n.stop()


# -- known-answer tests --------------------------------------------------------------------------
class LateLiar(LayerServer):
    """Honest on the first block of every session, wrong on every step after it."""

    def _step(self, ch, header, tensor, state):
        session = state.get("session")
        lie = session is not None and session["tokens"] > 0

        class Twisted:
            def send(_, hdr, tensor=None):
                ch.send(hdr, tensor=tensor + 0.5 if (lie and tensor is not None) else tensor)
        super()._step(Twisted(), header, tensor, state)


class SecondSessionLiar(LayerServer):
    """Honest in the first session it ever serves, wrong in all later ones."""
    opened = 0

    def _open(self, ch, header, tensor, state):
        self.opened += 1
        state["lie"] = self.opened >= 2
        super()._open(ch, header, tensor, state)

    def _step(self, ch, header, tensor, state):
        class Twisted:
            def send(_, hdr, tensor=None):
                ch.send(hdr, tensor=tensor + 0.5 if (state.get("lie") and tensor is not None) else tensor)
        super()._step(Twisted(), header, tensor, state)


def odd_peer(kind, model_dir, tmp_path, start, end, bootstrap):
    n = Node(SECRET, "127.0.0.1:0", bootstrap=bootstrap, gossip_interval=0.3, quiet=True)
    server = kind(model_dir, start, end, str(tmp_path / f"{kind.__name__}-{start}-{end}-{n.port}"))
    server.attach(n)
    return n.start(), server


def known_answers(model_dir, tmp_path, ends, count=12, chunk=3):
    from rosetta_jane.canary import calibrate
    g = torch.Generator().manual_seed(21)
    prompts = [torch.randint(3, VOCAB, (int(torch.randint(10, 16, (1,), generator=g)),), generator=g).tolist()
               for _ in range(count)]
    return calibrate(model_dir, str(tmp_path / "known"), ends, prompts, chunk=chunk), prompts


def test_known_answers_are_the_whole_models_numbers(model_dir, reference, tmp_path):
    _, logits = reference
    asker, ends, good, _, everyone = swarm(model_dir, tmp_path, [(3, 7)], [])
    try:
        book, prompts = known_answers(model_dir, tmp_path, ends, count=4)
        assert book.count == 4 and book.variant == "float32" and book.fingerprint == ends.fingerprint
        for i, ids in enumerate(prompts):         # the last boundary, through the output layer, gives the model's scores
            final = book._states[f"{i}.{LAYERS}"]
            assert float((ends.logits(final[:, -1:]) - logits(ids)).abs().max()) < 1e-4
        assert book.can_test(3, 7, "float32") and not book.can_test(3, 7, "4bit") and not book.can_test(1, 7, "float32")
        gen = Generation(asker, ends, canaries=book)
        peer = next(p for p in asker.table.peers() if p.services)
        assert gen._test(peer, 3, 7, "float32") is True and peer.node_id in asker.vetted
        assert gen._test(peer, 4, 6, "float32") is True     # any part of its range can be tested
        # each test is shown to an address once; when they run out it says so and blames nobody
        assert book.remaining("127.0.0.1") == 2
        assert gen._test(peer, 3, 7, "float32") is True and gen._test(peer, 3, 7, "float32") is True
        assert gen._test(peer, 3, 7, "float32") is None and gen.tests_spent and asker.banned == set()
        from rosetta_jane.canary import CanaryBook
        assert CanaryBook(str(tmp_path / "known")).remaining("127.0.0.1") == 0      # remembered across restarts
    finally:
        for n in everyone:
            n.stop()


def test_a_new_peer_is_tested_before_it_is_trusted(model_dir, reference, tmp_path):
    """No second opinion and no auditor: the known answers alone keep a liar out."""
    greedy, logits = reference
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 5), (5, 8)], [(2, 8)])
    try:
        book, _ = known_answers(model_dir, tmp_path, ends)
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=1, canaries=book)
        got = gen.run(PROMPT, 10)                 # the liar reaches furthest, so it is tried first
        assert got == greedy(PROMPT, 10)
        assert_scores_match(seen, PROMPT, got, logits)
        liar_id = bad[0][0].node_id
        assert gen.caught == [liar_id] and liar_id in asker.banned and liar_id not in asker.vetted
        assert asker.vetted == {n.node_id for n, _ in good} and gen.known_answer_tests == 3
        assert bad[0][1].tokens_served > 0 and gen.disputes == 0   # it was caught on the test, before real work
        without = Generation(asker, ends, redundancy=1)
        asker.banned.clear()
        seen.clear()
        without.run(PROMPT, 2)                    # the same swarm with the tests off accepts the lie
        assert float((seen[0] - logits(PROMPT)).abs().max()) > 1e-2
    finally:
        for n in everyone:
            n.stop()


def test_a_peer_that_lies_only_after_the_first_step_is_caught(model_dir, reference, tmp_path):
    greedy, logits = reference
    hub = node()
    boot = [addr(hub)]
    good = [serving(model_dir, tmp_path, 2, 5, bootstrap=boot), serving(model_dir, tmp_path, 5, 8, bootstrap=boot)]
    late = odd_peer(LateLiar, model_dir, tmp_path, 2, 8, boot)
    asker = node(bootstrap=boot)
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == 3)
        book, _ = known_answers(model_dir, tmp_path, ends)
        first_block_only = book.steps(0, 2, 8, 0)
        assert len(first_block_only) == 1 and len(book.steps(0, 2, 8, 5)) == 6   # a test continues past the first block
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=1, canaries=book)
        got = gen.run(PROMPT, 10)
        assert got == greedy(PROMPT, 10)
        assert_scores_match(seen, PROMPT, got, logits)
        assert gen.caught == [late[0].node_id]
    finally:
        for n in [hub, asker, late[0]] + [n for n, _ in good]:
            n.stop()


def test_a_dispute_is_settled_with_a_fresh_known_answer(model_dir, reference, tmp_path):
    """Two peers, no third, no auditor. One behaved while being tested and lied afterwards."""
    greedy, logits = reference
    hub = node()
    boot = [addr(hub)]
    honest = serving(model_dir, tmp_path, 2, 8, bootstrap=boot)
    sly = odd_peer(SecondSessionLiar, model_dir, tmp_path, 2, 8, boot)
    asker = node(bootstrap=boot)
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == 2)
        book, _ = known_answers(model_dir, tmp_path, ends)
        seen = record_scores(ends)
        gen = Generation(asker, ends, redundancy=2, canaries=book, canary_rate=0.0)
        got = gen.run(PROMPT, 10)
        assert got == greedy(PROMPT, 10)
        assert_scores_match(seen, PROMPT, got, logits)
        assert gen.disputes == 1 and gen.caught == [sly[0].node_id]
        assert gen.known_answer_tests == 4          # one each on first contact, one each at the dispute
        assert honest[0].node_id not in asker.banned and gen.route[0].peer.node_id == honest[0].node_id
    finally:
        for n in (hub, asker, honest[0], sly[0]):
            n.stop()


@pytest.mark.parametrize("peers,expected", [(3, 2), (2, 1), (5, 2)])
def test_redundancy_switches_on_by_itself_with_three_peers(model_dir, reference, tmp_path, peers, expected):
    greedy, _ = reference
    asker, ends, good, _, everyone = swarm(model_dir, tmp_path, [(2, 8)] * peers, [])
    try:
        gen = Generation(asker, ends, redundancy=None)
        assert gen.run(PROMPT, 5) == greedy(PROMPT, 5)
        assert gen.redundancy == expected and len(gen.route[0].replicas()) == expected
        assert (gen.verified_steps, gen.unverified_steps) == ((5, 0) if expected == 2 else (0, 5))
    finally:
        for n in everyone:
            n.stop()


def test_rechecking_works_a_few_layers_at_a_time(model_dir, tmp_path):
    from rosetta_jane.layers import LocalAuditor
    h = torch.randn(1, 7, 32, generator=torch.Generator().manual_seed(2))
    whole = LocalAuditor(model_dir, str(tmp_path / "a"), max_layers=8).compute(2, 8, [h[:, :4], h[:, 4:]])
    for at_once in (1, 2, 4):
        piecewise = LocalAuditor(model_dir, str(tmp_path / f"b{at_once}"), max_layers=at_once)
        assert float((piecewise.compute(2, 8, [h]) - whole).abs().max()) < 1e-5
    n, server = serving(model_dir, tmp_path, 2, 8)
    try:
        ch = n.dial(addr(n))
        ch.call({"op": "open", "model": server.fingerprint, "start": 2, "end": 8})
        assert float((ch.call({"op": "step"}, tensor=h)[1] - whole).abs().max()) < 1e-5   # and equals a real peer
        ch.close()
    finally:
        n.stop()


def test_peers_of_different_precision_each_get_their_own_known_answers(model_dir, reference, tmp_path):
    """A float32 peer and a bfloat16 peer on one network, each checked against answers of its own precision."""
    from rosetta_jane.canary import CanaryShelf, calibrate
    greedy, _ = reference
    hub = node()
    boot = [addr(hub)]
    full = serving(model_dir, tmp_path, 2, 8, bootstrap=boot)
    half = serving(model_dir, tmp_path, 2, 8, bootstrap=boot, dtype="bfloat16")
    asker = node(bootstrap=boot)
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == 2)
        g = torch.Generator().manual_seed(8)
        prompts = [torch.randint(3, VOCAB, (12,), generator=g).tolist() for _ in range(6)]
        known = str(tmp_path / "known")
        calibrate(model_dir, known, ends, prompts, chunk=3)
        shelf = CanaryShelf(known, ends.fingerprint)
        assert sorted(shelf.books) == ["float32"]
        by_variant = {p.services[0]["variant"]: p for p in asker.table.peers() if p.services}
        gen = Generation(asker, ends, canaries=shelf)
        assert gen._test(by_variant["float32"], 2, 8, "float32") is True
        assert gen._test(by_variant["bfloat16"], 2, 8, "bfloat16") is None    # no answers for that precision: untested
        calibrate(model_dir, known, ends, prompts, chunk=3, dtype="bfloat16")
        shelf = CanaryShelf(known, ends.fingerprint)
        assert sorted(shelf.books) == ["bfloat16", "float32"]
        gen = Generation(asker, ends, canaries=shelf)
        assert gen._test(by_variant["bfloat16"], 2, 8, "bfloat16") is True and asker.banned == set()
        # How far apart the two precisions are on this tiny model, for the record:
        x, y32 = shelf.books["float32"].steps(0, 2, 8, 0)[0]
        ch = asker.dial(by_variant["bfloat16"].addr)
        ch.call({"op": "open", "model": ends.fingerprint, "start": 2, "end": 8})
        y16 = ch.call({"op": "step"}, tensor=x)[1].float()
        ch.close()
        gap = float((y16 - y32).norm() / y32.norm())
        print(f"PRECISION GAP float32 vs bfloat16 over 6 tiny layers: {gap:.4f}")
        assert gap > 1e-4                           # they really are different numbers
        assert CanaryShelf(known, "0" * 16).books == {} and not CanaryShelf(known, "0" * 16)
    finally:
        for n in (hub, asker, full[0], half[0]):
            n.stop()


def test_known_answers_cannot_be_shared_between_devices(model_dir, tmp_path, monkeypatch):
    import shutil
    from rosetta_jane import canary
    from rosetta_jane.canary import CanaryBook, CanaryShelf
    ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
    mine = str(tmp_path / "mine")
    g = torch.Generator().manual_seed(3)
    prompts = [torch.randint(3, VOCAB, (12,), generator=g).tolist() for _ in range(3)]
    book = canary.calibrate(model_dir, mine, ends, prompts, chunk=3)
    sealed = open(CanaryBook.path(mine, "float32"), "rb").read()
    answer = book._states[f"0.{LAYERS}"].numpy().tobytes()
    assert answer[:64] not in sealed and b"__metadata__" not in sealed       # unreadable on disk
    assert oct(os.stat(os.path.join(mine, "device.key")).st_mode & 0o777) == "0o600"
    assert CanaryBook(mine, "float32").count == 3                             # this device reads its own

    # 1. The answers file alone, handed to someone else: they have no key for it.
    theirs = str(tmp_path / "theirs")
    os.makedirs(theirs)
    for name in os.listdir(mine):
        if name.startswith("known-answers-"):
            shutil.copy(os.path.join(mine, name), theirs)
    with pytest.raises(ValueError, match="no key"):
        CanaryBook(theirs, "float32")
    # 2. Their own key does not open it either.
    canary.device_key(theirs, create=True)
    with pytest.raises(ValueError, match="different device"):
        CanaryBook(theirs, "float32")
    shelf = CanaryShelf(theirs, ends.fingerprint)
    assert shelf.refused == ["float32"] and not shelf       # ignored, never trusted
    # 3. The whole folder, key included, copied to another machine: still refused.
    whole = str(tmp_path / "whole")
    shutil.copytree(mine, whole)
    assert CanaryBook(whole, "float32").count == 3           # (same machine: opens)
    monkeypatch.setattr(canary, "_machine", lambda: b"some-other-computer")
    with pytest.raises(ValueError, match="different device"):
        CanaryBook(whole, "float32")
    monkeypatch.undo()
    # 4. Two devices never invent the same tests, so one's answers would not fit the other's anyway.
    import random
    assert canary.make_prompts(8, random.Random()) != canary.make_prompts(8, random.Random())
    # 5. A changed file is rejected, not half-trusted.
    path = CanaryBook.path(mine, "float32")
    damaged = bytearray(sealed)
    damaged[len(damaged) // 2] ^= 1
    open(path, "wb").write(bytes(damaged))
    with pytest.raises(ValueError, match="damaged"):
        CanaryBook(mine, "float32")


def test_precisions_can_be_measured_and_then_share_answers(model_dir, tmp_path):
    from rosetta_jane.canary import CanaryShelf, calibrate, measure_precisions
    g = torch.Generator().manual_seed(6)
    prompts = [torch.randint(3, VOCAB, (12,), generator=g).tolist() for _ in range(3)]
    ends_for = lambda folder, opts: LocalEnds(model_dir, 2, folder, dtype=opts.get("dtype"))
    gaps = measure_precisions(model_dir, str(tmp_path / "m"), ends_for, prompts,
                              {"dtype": "float32"}, {"dtype": "bfloat16"}, chunk=3)
    assert sorted(gaps) == list(range(2, LAYERS + 1)) and 1e-5 < max(gaps.values()) < 0.02
    same = measure_precisions(model_dir, str(tmp_path / "m2"), ends_for, prompts,
                              {"dtype": "float32"}, {"dtype": "float32"}, chunk=3)
    assert max(same.values()) < 1e-6
    # Sharing happens only when the owner says so.
    hub = node()
    half = serving(model_dir, tmp_path, 2, 8, bootstrap=[addr(hub)], dtype="bfloat16")
    asker = node(bootstrap=[addr(hub)])
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        known = str(tmp_path / "known")
        calibrate(model_dir, known, ends, prompts, chunk=3)      # float32 answers only
        assert wait_for(lambda: any(p.services for p in asker.table.peers()))
        peer = next(p for p in asker.table.peers() if p.services)
        apart = Generation(asker, ends, canaries=CanaryShelf(known, ends.fingerprint))
        assert apart._test(peer, 2, 8, "bfloat16") is None
        shared = Generation(asker, ends, canaries=CanaryShelf(known, ends.fingerprint,
                                                             [["float32", "bfloat16"]]))
        assert shared._test(peer, 2, 8, "bfloat16") is True and asker.banned == set()
    finally:
        for n in (hub, half[0], asker):
            n.stop()
