"""The phone engine: the same layers computed with NumPy only, as an ordinary peer among desktop peers.

The peer under test runs in its own process where PyTorch cannot be imported at all,
which is the situation on a phone. Everything else in the swarm is the desktop program.
"""
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

from rosetta_jane.handoff import HandoffLog, compare
from rosetta_jane.lan.slicing import model_fingerprint
from rosetta_jane.layers import LayerServer, LocalEnds, _run
from rosetta_jane.lora import Applied, init_adapters, shapes_of_layers
from rosetta_jane.mobile import st
from rosetta_jane.mobile.engine import Stack, Unsupported, fingerprint
from rosetta_jane.swarm import Generation

from test_swarm import (PROMPT, ROOT, SECRET, addr, assert_scores_match, known_answers, model_dir, node,  # noqa: F401
                        record_scores, reference, serving, wait_for)
from test_train import make_examples, serving_peer, trainer, truth  # noqa: F401
from test_train import assert_same

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "test group code 123"                     # the same group as SECRET in test_swarm


def tiny(kind: str, layers: int = 6) -> str:
    import transformers
    config = getattr(transformers, f"{kind}Config")(
        vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(3)
    model = getattr(transformers, f"{kind}ForCausalLM")(config)
    with torch.no_grad():                         # fresh models have zero biases and unit norms, which
        for name, p in model.named_parameters():  # would hide mistakes in exactly those parts
            if p.dim() == 1:
                p.add_(torch.randn_like(p) * 0.2)
    path = tempfile.mkdtemp()
    model.save_pretrained(path)
    return path


@pytest.mark.parametrize("kind", ["Llama", "Mistral", "Qwen2"])
def test_the_numpy_engine_computes_what_pytorch_computes(kind, tmp_path):
    path = tiny(kind)
    assert fingerprint(path) == model_fingerprint(path)         # phones and PCs agree which model this is
    desktop = LayerServer(path, 1, 6, str(tmp_path / "srv"), dtype="float32")
    phone = Stack(path, 1, 6)
    gen = torch.Generator().manual_seed(1)
    for a, b in ((1, 6), (2, 5), (3, 4)):
        layers = torch.nn.ModuleList(desktop._layers[a - 1:b - 1])
        cache_t = cache_n = None
        for tokens in (7, 1, 1, 3, 1):            # a prompt, then words one or a few at a time
            h = torch.randn(1, tokens, 32, generator=gen)
            with torch.no_grad():
                want, cache_t = _run(desktop._decoder, layers, h, cache_t)
            got, cache_n = phone.step(a, b, h[0].numpy(), cache_n)
            gap = float(np.abs(got - want[0].numpy()).max() / np.abs(want.numpy()).max())
            assert gap < 2e-5, f"layers {a}:{b}, {tokens} tokens: differs from PyTorch by {gap:.1e}"


@pytest.mark.parametrize("kind,targets", [("Llama", ["q_proj", "v_proj"]), ("Qwen2", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                                          ("Mistral", ["o_proj", "gate_proj", "up_proj", "down_proj"])])
def test_the_numpy_engine_trains_the_way_pytorch_does(kind, targets, tmp_path):
    path = tiny(kind)
    desktop = LayerServer(path, 1, 6, str(tmp_path / "srv"), dtype="float32")
    phone = Stack(path, 1, 6)
    for a, b in ((1, 6), (2, 4)):
        layers = torch.nn.ModuleList(desktop._layers[a - 1:b - 1])
        shapes = shapes_of_layers(layers, a, targets)
        assert phone.shapes(a, b, targets) == shapes
        adapters = init_adapters(shapes, 4, 0)
        gen = torch.Generator().manual_seed(2)
        for key in adapters:                      # B starts at zero; give it values so every path carries a gradient
            if key.endswith(".B"):
                adapters[key] = torch.randn(adapters[key].shape, generator=gen) * 0.1
        live = {k: v.clone().requires_grad_(True) for k, v in adapters.items()}
        h = torch.randn(1, 9, 32, generator=gen).requires_grad_(True)
        g = torch.randn(1, 9, 32, generator=gen)
        with Applied(layers, a, targets, live, 2.0):
            want, _ = _run(desktop._decoder, layers, h, None, keep_cache=False)
        want.backward(g)
        got, graph = phone.forward(a, b, h[0].detach().numpy(), {k: v.numpy() for k, v in adapters.items()}, 2.0)
        g_in, grads = phone.backward(graph, g[0].numpy())
        rel = lambda x, y: float(np.abs(x - y).max() / (np.abs(y).max() + 1e-12))
        assert rel(got, want[0].detach().numpy()) < 2e-5
        assert rel(g_in, h.grad[0].numpy()) < 1e-4
        assert sorted(grads) == sorted(live)
        for key in live:
            assert rel(grads[key], live[key].grad.numpy()) < 1e-4, key


def test_model_and_adapter_files_are_read_without_the_safetensors_library(tmp_path):
    from safetensors.torch import load, save, save_file
    tensors = {"b.A": torch.randn(4, 32), "a.B": torch.zeros(32, 4), "c": torch.randn(3, 5).to(torch.float16),
               "d": torch.randn(2, 2).to(torch.bfloat16), "e": torch.arange(6).reshape(2, 3)}
    mine = st.loads(save(tensors))
    for key, value in tensors.items():
        assert np.array_equal(mine[key], value.float().numpy() if value.is_floating_point() else value.numpy()), key
    back = load(st.dumps({k: v for k, v in mine.items()}))        # and the library reads what this writes
    assert sorted(back) == sorted(tensors) and torch.equal(back["b.A"], tensors["b.A"])
    path = str(tmp_path / "x.safetensors")
    save_file(tensors, path)
    f = st.File(path)
    assert f.dtype("d") == "BF16" and np.array_equal(f.get("d"), tensors["d"].float().numpy())
    for bad in (b"", b"\x00" * 7, (1 << 40).to_bytes(8, "little") + b"{}",
                (20).to_bytes(8, "little") + b'{"x":{"dtype":"F32"}}'[:20],
                (2).to_bytes(8, "little") + b"[]",
                st.dumps({"x": np.zeros((2, 2), np.float32)})[:-3]):
        with pytest.raises(st.FormatError):
            st.loads(bad)
    huge = b'{"x":{"dtype":"F32","shape":[1000000,1000000],"data_offsets":[0,4]}}'
    with pytest.raises(st.FormatError):           # a claimed size the data cannot back up
        st.loads(len(huge).to_bytes(8, "little") + huge + b"\x00" * 4)


def test_unsupported_models_are_refused_not_guessed_at(tmp_path):
    path = tiny("Gemma")
    with pytest.raises(Unsupported, match="cannot run on this device yet"):
        Stack(path, 1, 3)
    with pytest.raises(ValueError, match="layer 0 stays"):
        Stack(tiny("Llama"), 0, 3)


def phone_peer(model_dir, tmp_path, layers, boot, *extra, name="phone"):
    """Start the p2p service in a process that cannot import PyTorch. Returns (process, address, work folder)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    work = str(tmp_path / name)
    env = dict(os.environ, PYTHONPATH=ROOT, ROSETTA_JANE_GROUP_CODE=CODE)
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "no_torch.py"), "serve", "--model", model_dir,
                             "--layers", layers, "--listen", f"127.0.0.1:{port}", "--work-dir", work,
                             *[x for b in boot for x in ("--peer", b)], *extra],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, f"127.0.0.1:{port}", work


def stop(proc):
    proc.terminate()
    out = proc.communicate(timeout=20)[0]
    assert "FORBIDDEN IMPORTS" not in out, out
    return out


def joined(asker, count):
    assert wait_for(lambda: sum(bool(p.services) for p in asker.table.peers()) == count, 60), \
        [p.services for p in asker.table.peers()]


def test_a_phone_peer_answers_exactly_like_the_whole_model(model_dir, reference, tmp_path):
    hub = node()
    proc, where, _ = phone_peer(model_dir, tmp_path, "2:8", [addr(hub)])
    asker = node(bootstrap=[addr(hub)])
    try:
        joined(asker, 1)
        peer = next(p for p in asker.table.peers() if p.services)
        assert peer.services[0]["variant"] == "float32" and peer.services[0]["layers"] == [2, 8]
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        seen = record_scores(ends)
        gen = Generation(asker, ends, find_wait=3.0)
        out = gen.run(PROMPT, 8)
        assert out == reference[0](PROMPT, 8)
        assert_scores_match(seen, PROMPT, out, reference[1])
    finally:
        for n in (hub, asker):
            n.stop()
        assert "p2p service for layers 2-7" in stop(proc)


def test_a_phone_and_a_pc_check_each_other(model_dir, reference, tmp_path):
    hub = node()
    boot = [addr(hub)]
    pc_node, pc = serving(model_dir, tmp_path, 2, 8, bootstrap=boot, dtype="float32")
    proc, where, work = phone_peer(model_dir, tmp_path, "2:8", boot)
    asker = node(bootstrap=boot)
    asker.handoffs = HandoffLog(str(tmp_path / "asker" / "handoffs.jsonl"), asker.node_id)
    try:
        joined(asker, 2)
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        book, _ = known_answers(model_dir, tmp_path, ends)
        gen = Generation(asker, ends, redundancy=2, find_wait=3.0, canaries=book)
        out = gen.run(PROMPT, 6)
        assert out == reference[0](PROMPT, 6)
        # Every step was computed by both and compared; they agreed; both passed their known-answer tests.
        assert gen.verified_steps == 6 and gen.unverified_steps == 0 and gen.disputes == 0
        assert gen.known_answer_tests == 2 and not asker.banned and pc.steps_served > 0
    finally:
        for n in (hub, asker, pc_node):
            n.stop()
        stop(proc)
    # The phone keeps the same handoff fingerprints, and they line up with the asking device's.
    from rosetta_jane.handoff import read_log
    mine, theirs = read_log(str(tmp_path / "asker" / "handoffs.jsonl")), read_log(os.path.join(work, "handoffs.jsonl"))
    assert len(theirs) >= 12 and {r["role"] for r in theirs} == {"serving"}
    for row in theirs:                             # everything the phone logged has its twin on the asking device
        other = "sent" if row["dir"] == "received" else "received"
        twin = [m for m in mine if m["hash"] == row["hash"] and m["dir"] == other]
        assert twin and twin[0]["norm"] == pytest.approx(row["norm"], rel=1e-5) and twin[0]["top"] == row["top"]
    assert not compare([str(tmp_path / "asker" / "handoffs.jsonl"), os.path.join(work, "handoffs.jsonl")])["real_differences"]


def test_a_phone_peer_takes_the_layers_the_swarm_is_short_of(model_dir, reference, tmp_path):
    hub = node()
    boot = [addr(hub)]
    pc_node, _ = serving(model_dir, tmp_path, 2, 5, bootstrap=boot, dtype="float32")
    proc, where, _ = phone_peer(model_dir, tmp_path, "auto", [addr(pc_node)], "--span", "3")
    asker = node(bootstrap=boot)
    try:
        joined(asker, 2)
        phone = next(p for p in asker.table.peers() if p.addr == where)
        assert phone.services[0]["layers"] == [5, 8]
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        gen = Generation(asker, ends, find_wait=3.0)
        assert gen.run(PROMPT, 5) == reference[0](PROMPT, 5)
        assert len(gen.route) == 2
    finally:
        for n in (hub, asker, pc_node):
            n.stop()
        stop(proc)


def test_training_through_a_phone_peer_matches_one_machine(model_dir, truth, tmp_path):
    hub = node()
    boot = [addr(hub)]
    proc, where, _ = phone_peer(model_dir, tmp_path, "2:8", boot, "--training")
    me = node(bootstrap=boot)
    try:
        joined(me, 1)
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1)
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0], tol=5e-5)
        assert abs(stats["final_loss"] - truth[1][-1]) < 1e-4
    finally:
        for n in (hub, me):
            n.stop()
        stop(proc)


def test_a_phone_and_a_pc_check_each_others_training(model_dir, truth, tmp_path):
    hub = node()
    boot = [addr(hub)]
    pc_node, pc = serving_peer(model_dir, tmp_path, 2, 8, boot)
    proc, where, _ = phone_peer(model_dir, tmp_path, "2:8", boot, "--training")
    me = node(bootstrap=boot)
    try:
        joined(me, 2)
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        t = trainer(me, ends, model_dir, tmp_path, redundancy=2)
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0], tol=5e-5)
        assert stats["disputes"] == 0 and stats["unchecked_examples"] == 0 and stats["checked_examples"] == 8
        assert not me.banned and pc.examples_trained == 8
    finally:
        for n in (hub, me, pc_node):
            n.stop()
        stop(proc)


def test_a_phone_peer_refuses_what_a_pc_refuses(model_dir, tmp_path):
    from safetensors.torch import save
    hub = node()
    proc, where, _ = phone_peer(model_dir, tmp_path, "2:8", [addr(hub)])       # training help not switched on
    me = node(bootstrap=[addr(hub)])
    try:
        joined(me, 1)
        model = model_fingerprint(model_dir)
        opening = {"op": "open", "model": model, "start": 2, "end": 8}

        def refused(request, tensor=None, blob=None, first=opening):
            ch = me.dial(where)
            try:
                if first is not None:
                    ch.call(first)
                with pytest.raises(Exception) as e:
                    ch.call(request, tensor=tensor, blob=blob)
                return str(e.value)
            finally:
                ch.close()
        assert "different model" in refused({**opening, "model": "00" * 8}, first=None)
        assert "serves layers 2:8" in refused({**opening, "start": 0}, first=None)
        assert "serves layers 2:8" in refused({**opening, "start": True}, first=None)
        assert "not agreed to help with training" in refused(
            {"op": "t_open", "model": model, "start": 2, "end": 8, "rank": 4, "alpha": 8.0, "targets": ["q_proj"]},
            first=None)
        assert "shape" in refused({"op": "step"}, tensor=torch.randn(1, 3, 31))
        assert "shape" in refused({"op": "step"}, tensor=torch.randn(2, 3, 32))
        assert "shape" in refused({"op": "step"}, tensor=torch.full((1, 3, 32), float("nan")))
        assert "shape" in refused({"op": "step"}, tensor=torch.ones(1, 3, 32, dtype=torch.int64))
        assert "limited to" in refused({"op": "step"}, tensor=torch.randn(1, 5000, 32))
        assert "no open session" in refused({"op": "step"}, tensor=torch.randn(1, 3, 32), first=None)
        ch = me.dial(where)                       # and after all that it still works
        ch.call(opening)
        assert tuple(ch.call({"op": "step"}, tensor=torch.randn(1, 3, 32))[1].shape) == (1, 3, 32)
        ch.close()
    finally:
        for n in (hub, me):
            n.stop()
        stop(proc)


# -- asking from a phone ---------------------------------------------------------------------
def phone_asks(model_dir, tmp_path, boot, new=8, redundancy="auto"):
    """Ask from a process that cannot import PyTorch. Returns what it produced and how it went."""
    import json
    env = dict(os.environ, PYTHONPATH=ROOT, ROSETTA_JANE_GROUP_CODE=CODE)
    r = subprocess.run([sys.executable, os.path.join(HERE, "no_torch.py"), "ask-ids", "--model", model_dir,
                        "--own-layers", "2", "--ids", ",".join(map(str, PROMPT)), "--new", str(new),
                        "--redundancy", redundancy, "--work-dir", str(tmp_path / "phone-asker"),
                        *[x for b in boot for x in ("--peer", b)]], env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and "FORBIDDEN IMPORTS" not in r.stdout, r.stdout + r.stderr
    return json.loads(next(line for line in r.stdout.splitlines() if line.startswith("RESULT "))[7:])


def scores_match(got, reference, tol=2e-4):
    for t, row in enumerate(got["scores"][:len(got["ids"])]):
        want = reference[1](list(PROMPT) + got["ids"][:t])
        gap = float((torch.tensor(row) - want).abs().max())
        assert gap < tol, f"step {t}: scores differ from the whole model by {gap:.1e}"


def test_a_phone_asks_through_pcs_and_gets_the_whole_models_answer(model_dir, reference, tmp_path):
    hub = node()
    boot = [addr(hub)]
    pcs = [serving(model_dir, tmp_path, 2, 5, bootstrap=boot, dtype="float32"),
           serving(model_dir, tmp_path, 5, 8, bootstrap=boot, dtype="float32")]
    try:
        assert wait_for(lambda: len(hub.table.peers()) == 2)
        got = phone_asks(model_dir, tmp_path, boot)
        assert got["ids"] == reference[0](PROMPT, 8) and got["peers"] == 2
        scores_match(got, reference)
    finally:
        for n in [hub] + [n for n, _ in pcs]:
            n.stop()


def test_a_phone_asks_through_another_phone(model_dir, reference, tmp_path):
    hub = node()
    proc, where, _ = phone_peer(model_dir, tmp_path, "2:8", [addr(hub)])
    try:
        assert wait_for(lambda: any(p.services for p in hub.table.peers()), 60)
        got = phone_asks(model_dir, tmp_path, [addr(hub)])
        assert got["ids"] == reference[0](PROMPT, 8)
        scores_match(got, reference)
    finally:
        hub.stop()
        stop(proc)


def test_a_phone_asking_catches_a_lying_peer(model_dir, reference, tmp_path):
    from test_swarm import lying
    hub = node()
    boot = [addr(hub)]
    good = [serving(model_dir, tmp_path, 2, 8, bootstrap=boot, dtype="float32") for _ in range(2)]
    liar_node, _ = lying(model_dir, tmp_path, 2, 8, boot)
    try:
        assert wait_for(lambda: len(hub.table.peers()) == 3)
        got = phone_asks(model_dir, tmp_path, boot, new=6, redundancy="3")
        assert got["ids"] == reference[0](PROMPT, 6)          # the right answer despite the liar
        assert got["blocked"] == [liar_node.node_id] and got["disputes"] >= 1 and got["checked"] == 6
        # Checking switches on by itself with three peers, as on a desktop.
        auto = phone_asks(model_dir, tmp_path, boot, new=4)
        assert auto["ids"] == reference[0](PROMPT, 4) and auto["peers"] == 2 and auto["unchecked"] == 0
    finally:
        for n in [hub, liar_node] + [n for n, _ in good]:
            n.stop()


# -- training from a phone -------------------------------------------------------------------
def phone_trains(model_dir, tmp_path, boot, *extra, redundancy="auto", name="phone-trainer"):
    """Run a training job from a process that cannot import PyTorch, starting from the desktop's own start."""
    import json
    from safetensors.torch import load_file, save_file
    from rosetta_jane.lora import from_peft, shapes_from_files
    from test_train import ACCUM, ALPHA, LR, RANK, STEPS, TARGETS
    work = tmp_path / name
    work.mkdir(exist_ok=True)
    examples = [[ids[0].tolist(), labels[0].tolist()] for ids, labels in make_examples()]
    (work / "examples.json").write_text(json.dumps(examples))
    save_file(init_adapters(shapes_from_files(model_dir, TARGETS), RANK, 0), str(work / "init.safetensors"))
    out = str(work / "adapter")
    env = dict(os.environ, PYTHONPATH=ROOT, ROSETTA_JANE_GROUP_CODE=CODE)
    r = subprocess.run([sys.executable, os.path.join(HERE, "no_torch.py"), "train-ids", "--model", model_dir,
                        "--own-layers", "2", "--examples", str(work / "examples.json"),
                        "--init", str(work / "init.safetensors"), "--steps", str(STEPS), "--accum", str(ACCUM),
                        "--lr", str(LR), "--rank", str(RANK), "--alpha", str(ALPHA), "--targets", ",".join(TARGETS),
                        "--redundancy", redundancy, "--out", out, "--work-dir", str(work / "work"), *extra,
                        *[x for b in boot for x in ("--peer", b)]], env=env, capture_output=True, text=True, timeout=240)
    assert r.returncode == 0 and "FORBIDDEN IMPORTS" not in r.stdout, r.stdout + r.stderr
    stats = json.loads(next(line for line in r.stdout.splitlines() if line.startswith("RESULT "))[7:])
    adapters = None
    if os.path.exists(os.path.join(out, "adapter_model.safetensors")):
        adapters = from_peft(load_file(os.path.join(out, "adapter_model.safetensors")), "model.layers.")
    return stats, adapters, out


def test_a_phone_trains_through_pcs_like_one_machine(model_dir, truth, tmp_path):
    hub = node()
    boot = [addr(hub)]
    pcs = [serving_peer(model_dir, tmp_path, 2, 5, boot), serving_peer(model_dir, tmp_path, 5, 8, boot)]
    try:
        assert wait_for(lambda: len(hub.table.peers()) == 2)
        stats, adapters, out = phone_trains(model_dir, tmp_path, boot)
        assert_same(adapters, truth[0], tol=5e-5)
        assert abs(stats["final_loss"] - truth[1][-1]) < 1e-4 and stats["unchecked"] == 16
        # What it saved is a standard adapter, with the facts a person needs before using it.
        import json
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        ids, labels = make_examples()[0]
        base = AutoModelForCausalLM.from_pretrained(model_dir).eval()
        with torch.no_grad():
            before = float(base(input_ids=ids, labels=labels).loss)
            after = float(PeftModel.from_pretrained(base, out)(input_ids=ids, labels=labels).loss)
        assert after < before - 1e-4
        facts = json.load(open(os.path.join(out, "rosetta-jane-run.json")))
        assert facts["work_unchecked"] == 16 and facts["work_checked"] == 0 and facts["pairs"] == 6
    finally:
        for n in [hub] + [n for n, _ in pcs]:
            n.stop()


def test_a_phone_trains_through_another_phone(model_dir, truth, tmp_path):
    hub = node()
    proc, where, _ = phone_peer(model_dir, tmp_path, "2:8", [addr(hub)], "--training")
    try:
        assert wait_for(lambda: any(p.services for p in hub.table.peers()), 60)
        stats, adapters, _ = phone_trains(model_dir, tmp_path, [addr(hub)])
        assert_same(adapters, truth[0], tol=5e-5)
    finally:
        hub.stop()
        stop(proc)


def test_a_phone_training_outvotes_a_lying_peer(model_dir, truth, tmp_path):
    from test_train import GradBender
    hub = node()
    boot = [addr(hub)]
    good = [serving_peer(model_dir, tmp_path, 2, 8, boot) for _ in range(2)]
    liar_node, _ = serving_peer(model_dir, tmp_path, 2, 8, boot, GradBender)
    try:
        assert wait_for(lambda: len(hub.table.peers()) == 3)
        stats, adapters, _ = phone_trains(model_dir, tmp_path, boot, redundancy="3")
        assert_same(adapters, truth[0], tol=5e-5)
        assert stats["blocked"] == [liar_node.node_id] and stats["disputes"] >= 1 and stats["unchecked"] == 0
    finally:
        for n in [hub, liar_node] + [n for n, _ in good]:
            n.stop()


def test_a_phone_training_run_can_stop_and_carry_on(model_dir, truth, tmp_path):
    hub = node()
    boot = [addr(hub)]
    pc_node, pc = serving_peer(model_dir, tmp_path, 2, 8, boot)
    try:
        assert wait_for(lambda: len(hub.table.peers()) == 1)
        stats, adapters, _ = phone_trains(model_dir, tmp_path, boot, "--stop-after", "3", "--save-every", "2")
        assert stats["stopped_at"] == 3 and adapters is None        # stopped one step past the last save
        stats, adapters, _ = phone_trains(model_dir, tmp_path, boot, "--save-every", "2")
        assert stats["steps"] == 4
        assert_same(adapters, truth[0], tol=5e-5)
        assert pc.examples_trained == (3 + 2) * 2                    # steps 3 and 4 again, not all four
    finally:
        for n in (hub, pc_node):
            n.stop()


def test_a_phone_scores_what_it_trained_and_the_card_can_be_shared(model_dir, tmp_path):
    import json
    from rosetta_jane.control import adapter_zip, score_card
    from rosetta_jane.scoring import share_text
    from test_train import direct_scores
    hub = node()
    boot = [addr(hub)]
    pcs = [serving_peer(model_dir, tmp_path, 2, 8, boot) for _ in range(2)]
    try:
        assert wait_for(lambda: len(hub.table.peers()) == 2)
        stats, adapters, out = phone_trains(model_dir, tmp_path, boot, "--hold-out", "3", redundancy="2")
        card = stats["score"]
        held = [e for i, e in enumerate(make_examples()) if i % 3 == 2]
        (want_before, want_after), tokens = direct_scores(model_dir, adapters, held)
        assert card["pairs"] == 2 and card["answer_tokens"] == tokens and card["never_trained_on"] is True
        assert abs(card["before"]["loss"] - want_before["loss"]) < 1e-4
        assert abs(card["after"]["loss"] - want_after["loss"]) < 1e-4
        assert card["work_unchecked"] == 0 and card["work_checked"] == 4          # two peers checked each other
        assert json.load(open(os.path.join(out, "rosetta-jane-run.json")))["score"] == card
        text = share_text(card)
        assert text.startswith("Rosetta-Jane score card: adapter") and "kept back from training" in text
        assert "Score it yourself before you rely on it" in text
        for example in make_examples():            # a card carries numbers, never the questions or answers
            assert " ".join(map(str, example[0][0].tolist()[:4])) not in text
        assert share_text(None) == "" and share_text({"before": {"loss": None}}) == ""
    finally:
        for n in [hub] + [n for n, _ in pcs]:
            n.stop()


def test_a_phone_tries_and_scores_an_adapter_someone_else_made(model_dir, tmp_path):
    import json
    from rosetta_jane.train import SwarmTrainer
    from test_train import direct_answers, direct_scores, loud_adapters
    hub = node()
    boot = [addr(hub)]
    pcs = [serving_peer(model_dir, tmp_path, 2, 5, boot), serving_peer(model_dir, tmp_path, 5, 8, boot)]
    try:
        assert wait_for(lambda: len(hub.table.peers()) == 2)
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        friend = SwarmTrainer(hub, ends, model_dir, str(tmp_path / "friend"), total_steps=1, rank=4, alpha=8.0,
                              targets=["q_proj", "v_proj"], quiet=True)
        loud_adapters(friend)
        folder = friend.export(str(tmp_path / "shared"), model_dir)
        examples = make_examples(4)
        (tmp_path / "ex.json").write_text(json.dumps([[i[0].tolist(), l[0].tolist()] for i, l in examples]))

        def phone(*extra):
            env = dict(os.environ, PYTHONPATH=ROOT, ROSETTA_JANE_GROUP_CODE=CODE)
            r = subprocess.run([sys.executable, os.path.join(HERE, "no_torch.py"), "try-ids", "--model", model_dir,
                                "--adapter", folder, "--work-dir", str(tmp_path / "phone"), "--peer", boot[0], *extra],
                               env=env, capture_output=True, text=True, timeout=240)
            assert r.returncode == 0 and "FORBIDDEN IMPORTS" not in r.stdout, r.stdout + r.stderr
            return json.loads(next(line for line in r.stdout.splitlines() if line.startswith("RESULT "))[7:])
        got = phone("--ids", ",".join(map(str, PROMPT)), "--new", "7", "--examples", str(tmp_path / "ex.json"))
        want_without, want_with = direct_answers(model_dir, friend.adapters, PROMPT, 7)
        assert got["without"] == want_without and got["with"] == want_with and want_without != want_with
        (want_before, want_after), tokens = direct_scores(model_dir, friend.adapters, examples)
        assert abs(got["score"]["before"]["loss"] - want_before["loss"]) < 1e-4
        assert abs(got["score"]["after"]["loss"] - want_after["loss"]) < 1e-4
        assert got["score"]["answer_tokens"] == tokens and got["score"]["never_trained_on"] is False
        # An adapter made for a different model is refused before anything is sent to the group.
        json.dump({"base_model_fingerprint": "0" * 16}, open(os.path.join(folder, "rosetta-jane-run.json"), "w"))
        assert "different model" in phone("--ids", "1,2,3")["refused"]
    finally:
        for n in [hub] + [n for n, _ in pcs]:
            n.stop()
