"""Peer-to-peer training, on a tiny random model over real sockets on one machine (CPU).

The key claim: training through a chain of peers gives the same trained weights as
ordinary LoRA training of the whole model on one machine, including when peers lie,
leave partway through, or the trainer is stopped and started again.
"""
import os

import pytest
import torch

from rosetta_jane.layers import LayerServer, LocalAuditor, LocalEnds
from rosetta_jane.lora import from_peft, init_adapters, shapes_from_files, to_peft
from rosetta_jane.net import Node
from rosetta_jane.swarm import Disputed, NoRoute
from rosetta_jane.train import SwarmTrainer

from test_swarm import LAYERS, SECRET, VOCAB, addr, known_answers, model_dir, node, wait_for  # noqa: F401

RANK, ALPHA, TARGETS = 4, 8.0, ["q_proj", "v_proj"]
STEPS, ACCUM, LR = 4, 2, 5e-3


def make_examples(n=6):
    g = torch.Generator().manual_seed(11)
    out = []
    for _ in range(n):
        length = int(torch.randint(8, 14, (1,), generator=g))
        ids = torch.randint(3, VOCAB, (1, length), generator=g)
        labels = ids.clone()
        labels[0, :3] = -100
        out.append((ids, labels))
    return out


def single_machine(model_dir, examples, steps=STEPS, seed=0):
    """Ordinary LoRA training of the whole model with the PEFT library: the answer to match."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from rosetta_jane.lan.data import Schedule
    base = AutoModelForCausalLM.from_pretrained(model_dir)
    model = get_peft_model(base, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
                                            target_modules=TARGETS, task_type="CAUSAL_LM"))
    start = to_peft(init_adapters(shapes_from_files(model_dir, TARGETS), RANK, seed), "model.layers.")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in start.items():
            params[key.replace(".weight", ".default.weight")].copy_(value)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == len(start)
    opt = torch.optim.AdamW(trainable, lr=0.0, weight_decay=0.0)
    probe = SwarmTrainer.__new__(SwarmTrainer)
    probe.lr, probe.warmup_steps, probe.schedule, probe.total_steps = LR, 1, "cosine", STEPS
    order = Schedule(len(examples), 0)
    losses = []
    for step in range(steps):
        total = 0.0
        for m in range(ACCUM):
            ids, labels = examples[order.index(step * ACCUM + m)]
            loss = model(input_ids=ids, labels=labels).loss
            (loss / ACCUM).backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        for group in opt.param_groups:
            group["lr"] = probe.lr_at(step)
        opt.step()
        opt.zero_grad()
        losses.append(total / ACCUM)
    state = {k.replace(".default.weight", ".weight"): v.detach() for k, v in model.named_parameters()
             if v.requires_grad}
    return from_peft(state, "model.layers."), losses


@pytest.fixture(scope="module")
def truth(model_dir):
    return single_machine(model_dir, make_examples())


class Bender(LayerServer):
    """A serving peer that returns wrong numbers at one stage of training. All Benders tell the same lie."""
    stage = "t_fwd"

    def _twist(self, ch):
        class Twisted:
            def send(_, hdr, tensor=None, blob=None):
                if blob is not None:
                    from safetensors.torch import load, save
                    blob = save({k: v * 3.0 + 0.01 for k, v in load(blob).items()})
                ch.send(hdr, tensor=tensor * 1.5 + 0.1 if tensor is not None else None, blob=blob)
        return Twisted()

    def _t_fwd(self, ch, header, tensor, state):
        super()._t_fwd(self._twist(ch) if self.stage == "t_fwd" else ch, header, tensor, state)

    def _t_bwd(self, ch, header, tensor, state):
        super()._t_bwd(self._twist(ch) if self.stage == "t_bwd" else ch, header, tensor, state)

    def _t_grads(self, ch, header, tensor, state):
        super()._t_grads(self._twist(ch) if self.stage == "t_grads" else ch, header, tensor, state)


class BackBender(Bender):
    stage = "t_bwd"


class GradBender(Bender):
    stage = "t_grads"


def serving_peer(model_dir, tmp_path, start, end, boot, kind=LayerServer, training=True):
    n = Node(SECRET, "127.0.0.1:0", bootstrap=boot, gossip_interval=0.3, quiet=True)
    server = kind(model_dir, start, end, str(tmp_path / f"{kind.__name__}-{start}-{end}-{n.port}"),
                  training=training)
    server.attach(n)
    return n.start(), server


def setup(model_dir, tmp_path, honest, liars=(), own=2):
    hub = node()
    boot = [addr(hub)]
    good = [serving_peer(model_dir, tmp_path, a, b, boot) for a, b in honest]
    bad = [serving_peer(model_dir, tmp_path, a, b, boot, kind) for kind, a, b in liars]
    me = node(bootstrap=boot)
    ends = LocalEnds(model_dir, own, str(tmp_path / "ends"))
    assert wait_for(lambda: sum(bool(p.services) for p in me.table.peers()) == len(good) + len(bad))
    return me, ends, good, bad, [hub, me] + [n for n, _ in good + bad]


def trainer(me, ends, model_dir, tmp_path, **kw):
    kw.setdefault("total_steps", STEPS)
    return SwarmTrainer(me, ends, model_dir, str(tmp_path / "work"), accum=ACCUM, lr=LR, warmup_steps=1,
                        rank=RANK, alpha=ALPHA, targets=TARGETS, save_every=2, find_wait=5.0, quiet=True, **kw)


def assert_same(adapters, expected, tol=2e-5):
    assert sorted(adapters) == sorted(expected)
    worst = max(float((adapters[k].detach() - expected[k]).abs().max()) for k in expected)
    moved = max(float(expected[k].abs().max()) for k in expected if k.endswith(".B"))
    assert moved > 1e-4, "the reference run did not learn anything; the comparison would be empty"
    assert worst < tol, f"trained weights differ from single-machine training by {worst:.2e}"


def stop(everyone):
    for n in everyone:
        n.stop()


@pytest.mark.parametrize("own,ranges", [(2, [(2, 8)]), (1, [(1, 4), (4, 8)]), (3, [(3, 5), (5, 6), (6, 8)]),
                                        (8, [])])
def test_training_through_peers_matches_one_machine(model_dir, truth, tmp_path, own, ranges):
    me, ends, good, _, everyone = setup(model_dir, tmp_path, ranges, own=own)
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1)
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert stats["steps"] == STEPS and abs(stats["final_loss"] - truth[1][-1]) < 1e-4
        assert sum(s.examples_trained for _, s in good) == STEPS * ACCUM * len(ranges)
    finally:
        stop(everyone)


def test_serving_peers_must_opt_in_to_training(model_dir, tmp_path):
    hub = node()
    n, server = serving_peer(model_dir, tmp_path, 2, 8, [addr(hub)], training=False)
    me = node(bootstrap=[addr(hub)])
    try:
        assert wait_for(lambda: any(p.services for p in me.table.peers()))
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1)
        t.find_wait = 1.0
        with pytest.raises(NoRoute, match="agreed to help with training"):
            t.train(make_examples())
        ch = me.dial(addr(n))
        with pytest.raises(Exception, match="not agreed"):
            ch.call({"op": "t_open", "model": ends.fingerprint, "start": 2, "end": 8, "rank": RANK,
                     "alpha": ALPHA, "targets": TARGETS})
        ch.close()
        assert server.examples_trained == 0
    finally:
        stop([hub, n, me])


@pytest.mark.parametrize("kind", [Bender, BackBender, GradBender])
def test_a_lying_serving_peer_is_caught_by_recomputing(model_dir, truth, tmp_path, kind):
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 8)], [(kind, 2, 8)])
    try:
        audit = LocalAuditor(model_dir, str(tmp_path / "audit"), max_layers=4)
        t = trainer(me, ends, model_dir, tmp_path, redundancy=2, auditor=audit)
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert bad[0][0].node_id in me.banned and good[0][0].node_id not in me.banned
        assert stats["peers_blocked"] == 1 and stats["disputes"] >= 1
    finally:
        stop(everyone)


@pytest.mark.parametrize("kind", [Bender, GradBender])
def test_one_liar_among_three_is_outvoted(model_dir, truth, tmp_path, kind):
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 8), (2, 8)], [(kind, 2, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=3)
        t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert bad[0][0].node_id in me.banned
        assert all(n.node_id not in me.banned for n, _ in good)
    finally:
        stop(everyone)


def test_two_serving_peers_disagree_and_a_third_breaks_the_tie(model_dir, truth, tmp_path):
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 8), (2, 8)], [(GradBender, 2, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=2)
        # Make sure the liar is one of the first two, so the tie really happens.
        t._bad.add(good[1][0].node_id)
        t.route = t._connect()
        t._bad.clear()
        assert {r.peer.node_id for r in t.route[0].replicas()} == {good[0][0].node_id, bad[0][0].node_id}
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert bad[0][0].node_id in me.banned and stats["redone"] >= 1
    finally:
        stop(everyone)


def test_two_serving_peers_disagree_with_nobody_to_break_the_tie(model_dir, tmp_path):
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 8)], [(Bender, 2, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=2)
        with pytest.raises(Disputed):
            t.train(make_examples())
        assert not me.banned
    finally:
        stop(everyone)


def test_liars_who_agree_with_each_other_are_caught_by_random_checks(model_dir, truth, tmp_path):
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 8)], [(GradBender, 2, 8), (GradBender, 2, 8)])
    try:
        audit = LocalAuditor(model_dir, str(tmp_path / "audit"), max_layers=3)
        t = trainer(me, ends, model_dir, tmp_path, redundancy=2, auditor=audit, audit_rate=0.25)
        t._bad.add(good[0][0].node_id)             # the two liars get there first and agree with each other
        t.route = t._connect()
        t._bad.clear()
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert all(n.node_id in me.banned for n, _ in bad) and stats["spot_audits"] >= 1
    finally:
        stop(everyone)


def test_unchecked_work_is_counted_honestly(model_dir, tmp_path):
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 8)])
    try:
        stats = trainer(me, ends, model_dir, tmp_path).train(make_examples())
        assert stats["checked_examples"] == 0 and stats["unchecked_examples"] == STEPS * ACCUM
    finally:
        stop(everyone)


def test_checking_switches_on_by_itself_with_three_serving_peers(model_dir, truth, tmp_path):
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 8), (2, 8), (2, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path)
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert stats["checked_examples"] == STEPS * ACCUM and stats["unchecked_examples"] == 0
    finally:
        stop(everyone)


def test_known_answers_gate_serving_peers_before_any_training_text_is_sent(model_dir, truth, tmp_path):
    from test_swarm import Liar
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 8)], [(Liar, 2, 8)])
    try:
        book, _ = known_answers(model_dir, tmp_path, ends)
        t = trainer(me, ends, model_dir, tmp_path, redundancy=2, canaries=book)
        t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert bad[0][0].node_id in me.banned and bad[0][1].examples_trained == 0
    finally:
        stop(everyone)


class Quitter(LayerServer):
    """Leaves in the middle of an example: after answering forward, before answering backward."""
    after = 3

    def _t_bwd(self, ch, header, tensor, state):
        if self.examples_trained >= self.after:
            ch.close()
            raise ConnectionError("gone")
        super()._t_bwd(ch, header, tensor, state)


def test_a_serving_peer_leaving_mid_example_is_replaced_and_nothing_changes(model_dir, truth, tmp_path):
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 5), (5, 8), (2, 5)], [(Quitter, 5, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1)
        t._bad.add(good[1][0].node_id)             # start on the serving peer that will leave
        t.route = t._connect()
        t._bad.clear()
        assert bad[0][0].node_id in {h.peer.node_id for h in t.route}
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert stats["redone"] >= 1 and stats["reroutes"] >= 1 and not me.banned
        assert good[1][1].examples_trained > 0
    finally:
        stop(everyone)


def test_second_serving_peer_carries_on_when_the_first_leaves(model_dir, truth, tmp_path):
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 8)], [(Quitter, 2, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=2)
        stats = t.train(make_examples())
        assert_same(t.adapters, truth[0])
        assert not me.banned and stats["steps"] == STEPS
    finally:
        stop(everyone)


def test_stopping_and_starting_again_gives_the_same_result(model_dir, truth, tmp_path):
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 5), (5, 8)])
    try:
        class Stop(Exception):
            pass

        def halt(step):
            if step == 3:                          # one step past the last save
                raise Stop()
        first = trainer(me, ends, model_dir, tmp_path, redundancy=1)
        with pytest.raises(Stop):
            first.train(make_examples(), on_step=halt)
        assert first.checkpoints() == [2]
        again = trainer(me, ends, model_dir, tmp_path, redundancy=1)
        stats = again.train(make_examples())
        assert stats["steps"] == STEPS
        assert_same(again.adapters, truth[0])
        other = trainer(me, ends, model_dir, tmp_path, redundancy=1)
        other.rank = 2
        with pytest.raises(ValueError, match="different LoRA settings"):
            other.resume()
    finally:
        stop(everyone)


def test_the_result_is_a_standard_adapter_for_the_whole_model(model_dir, truth, tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1)
        examples = make_examples()
        t.train(examples)
        out = t.export(str(tmp_path / "adapter"), model_dir)
    finally:
        stop(everyone)
    base = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    ids, labels = examples[0]
    with torch.no_grad():
        before = float(base(input_ids=ids, labels=labels).loss)
        tuned = PeftModel.from_pretrained(base, out).eval()
        after = float(tuned(input_ids=ids, labels=labels).loss)
        # The same numbers this code gets when it applies the adapters itself.
        from rosetta_jane.lora import Applied
        whole = AutoModelForCausalLM.from_pretrained(model_dir).eval()
        with Applied(whole.model.layers, 0, TARGETS, {k: v.detach() for k, v in t.adapters.items()}, ALPHA / RANK):
            mine = float(whole(input_ids=ids, labels=labels).loss)
    assert after < before - 1e-4 and abs(after - mine) < 1e-5


def test_serving_peer_refuses_bad_training_requests(model_dir, tmp_path):
    from safetensors.torch import save
    hub = node()
    n, server = serving_peer(model_dir, tmp_path, 2, 8, [addr(hub)])
    me = node(bootstrap=[addr(hub)])
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        opening = {"op": "t_open", "model": ends.fingerprint, "start": 2, "end": 8, "rank": RANK, "alpha": ALPHA,
                   "targets": TARGETS}
        for change in ({"start": 0}, {"rank": 100000}, {"targets": ["../../etc"]}, {"targets": ["nothing_here"]},
                       {"model": "00" * 8}, {"alpha": "big"}):
            ch = me.dial(addr(n))
            with pytest.raises(Exception):
                ch.call({**opening, **change})
            ch.close()
        ch = me.dial(addr(n))
        ch.call(opening)
        h = torch.randn(1, 5, 32)
        with pytest.raises(Exception, match="adapters"):
            ch.call({"op": "t_fwd"}, tensor=h)
        ch = me.dial(addr(n))
        ch.call(opening)
        shapes = shapes_from_files(model_dir, TARGETS)
        full = {k: v for k, v in init_adapters(shapes, RANK, 0).items() if 2 <= int(k.split(".")[0]) < 8}
        wrong = dict(full)
        wrong[next(iter(wrong))] = torch.zeros(3, 3)
        for bad in (b"junk", save(wrong), save({"x": torch.zeros(1)})):
            with pytest.raises(Exception):
                ch.call({"op": "t_adapters"}, blob=bad)
            ch = me.dial(addr(n))
            ch.call(opening)
        ch.call({"op": "t_adapters"}, blob=save(full))
        with pytest.raises(Exception):
            ch.call({"op": "t_bwd"}, tensor=h)      # nothing to go backward through
        ch = me.dial(addr(n))
        ch.call(opening)
        ch.call({"op": "t_adapters"}, blob=save(full))
        with pytest.raises(Exception):
            ch.call({"op": "t_fwd"}, tensor=torch.randn(1, 5000, 32))   # longer than the serving peer allows
        ch.close()
    finally:
        stop([hub, n, me])


def test_command_line_training(model_dir, tmp_path):
    import json
    import socket
    import subprocess
    import sys

    from test_swarm import ROOT, _tokenizer
    _tokenizer(model_dir)
    env = dict(os.environ, PYTHONPATH=ROOT, ROSETTA_JANE_GROUP_CODE="cli training code 7")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    data = tmp_path / "pairs.jsonl"
    data.write_text("\n".join(json.dumps({"question": f"w{i} w{i + 3} w{i + 9}", "answer": f"w{i + 1} w{i + 2} w{i + 5}"})
                              for i in range(8)))
    serve = [sys.executable, "-m", "rosetta_jane", "serve", "--model", model_dir, "--layers", "2:8",
             "--listen", f"127.0.0.1:{port}", "--work-dir", str(tmp_path / "s")]
    train = [sys.executable, "-m", "rosetta_jane", "train", "--model", model_dir, "--own-layers", "2",
             "--peer", f"127.0.0.1:{port}", "--work-dir", str(tmp_path / "t"), "--data", str(data),
             "--steps", "3", "--accum", "2", "--rank", "4", "--alpha", "8", "--targets", "q_proj,v_proj",
             "--warmup-steps", "1", "--out", str(tmp_path / "adapter"), "--no-known-answers"]

    def started(extra):
        proc = subprocess.Popen(serve + extra, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def up():
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                return True
            except OSError:
                return proc.poll() is not None
        assert wait_for(up, 90) and proc.poll() is None
        return proc
    server = started([])                          # serving for answers only
    try:
        r = subprocess.run(train, env=env, capture_output=True, text=True, timeout=180)
        assert r.returncode != 0 and "agreed to help with training" in r.stderr, r.stdout + r.stderr
    finally:
        server.kill()
        server.wait()
    server = started(["--training"])
    try:
        r = subprocess.run(train, env=env, capture_output=True, text=True, timeout=180)
        assert r.returncode == 0 and "done: 3 steps" in r.stdout, r.stdout + r.stderr
        assert os.path.exists(tmp_path / "adapter" / "adapter_model.safetensors")
        assert os.path.exists(tmp_path / "adapter" / "adapter_config.json")
    finally:
        server.kill()


# -- scoring by comparison -------------------------------------------------------------------
def direct_scores(model_dir, adapters, examples):
    """What the whole model scores on one machine, with and without the adapters."""
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    from rosetta_jane.lora import Applied
    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    out = []
    for use in (False, True):
        live = {k: (v.detach() if use or k.endswith(".A") else torch.zeros_like(v)) for k, v in adapters.items()}
        loss = right = tokens = 0
        with torch.no_grad(), Applied(model.model.layers, 0, TARGETS, live, ALPHA / RANK):
            for ids, labels in examples:
                logits = model(input_ids=ids).logits[0, :-1]
                want = labels[0, 1:]
                graded = want != -100
                loss += float(F.cross_entropy(logits[graded], want[graded], reduction="sum"))
                right += int((logits[graded].argmax(-1) == want[graded]).sum())
                tokens += int(graded.sum())
        out.append({"loss": loss / tokens, "right_next_word": right / tokens})
    return out, tokens


def test_pairs_are_kept_back_for_scoring_the_same_way_every_time():
    from rosetta_jane.scoring import split_holdout
    kept, held = split_holdout(list(range(25)))
    assert held == [9, 19] and len(kept) == 23 and not set(kept) & set(held)
    assert split_holdout(list(range(25))) == (kept, held)
    assert split_holdout([1, 2, 3, 4]) == ([1, 2, 3, 4], [])          # too few to spare any
    kept, held = split_holdout(list(range(6)))
    assert held == [5] and kept == [0, 1, 2, 3, 4]


def test_a_score_through_peers_is_the_whole_models_score(model_dir, tmp_path):
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 5), (5, 8), (2, 8)])
    try:
        examples = make_examples(10)
        train_on, held = examples[:8], examples[8:]
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1, total_steps=6)
        t.lr = 5e-2                                # learn enough for the two sides to differ clearly
        t.train(train_on)
        before_training = sum(s.examples_trained for _, s in good)
        card = t.compare(held, name="scored", held_out=True)
        (want_before, want_after), tokens = direct_scores(model_dir, t.adapters, held)
        assert card["pairs"] == 2 and card["answer_tokens"] == tokens and card["never_trained_on"] is True
        for side, want in (("before", want_before), ("after", want_after)):
            assert abs(card[side]["loss"] - want["loss"]) < 1e-4, side
            assert card[side]["right_next_word"] == pytest.approx(want["right_next_word"])
        assert abs(card["before"]["loss"] - card["after"]["loss"]) > 1e-3      # the comparison is not empty
        assert "too few to rely on" in card["verdict"]
        # Scoring teaches nothing and leaves nothing behind on the peers.
        assert sum(s.examples_trained for _, s in good) == before_training
        assert all(s2["graph"] is None for _, s in good for s2 in s._sessions if "graph" in s2)
        again = t.compare(held)
        assert again["after"]["loss"] == pytest.approx(card["after"]["loss"])
    finally:
        stop(everyone)


def test_a_lying_peer_cannot_bend_a_score(model_dir, tmp_path):
    me, ends, good, bad, everyone = setup(model_dir, tmp_path, [(2, 8), (2, 8)], [(Bender, 2, 8)])
    try:
        examples = make_examples(6)
        t = trainer(me, ends, model_dir, tmp_path, redundancy=3, total_steps=1)
        from rosetta_jane.lora import init_adapters as fresh
        gen = torch.Generator().manual_seed(4)
        for key in t._order:                       # give the adapters something to say without training
            if key.endswith(".B"):
                with torch.no_grad():
                    t.adapters[key].copy_(torch.randn(t.adapters[key].shape, generator=gen) * 0.05)
        card = t.compare(examples)
        (want_before, want_after), _ = direct_scores(model_dir, t.adapters, examples)
        assert abs(card["before"]["loss"] - want_before["loss"]) < 1e-4
        assert abs(card["after"]["loss"] - want_after["loss"]) < 1e-4
        assert bad[0][0].node_id in me.banned and card["work_unchecked"] == 0 and card["work_checked"] == 12
    finally:
        stop(everyone)


def test_score_cards_say_plainly_what_they_show():
    from rosetta_jane.scoring import card, describe, verdict
    good = card("x", "fp", 40, 300, {"loss": 2.0, "right_next_word": 0.4}, {"loss": 1.5, "right_next_word": 0.55},
                held_out=True, unchecked=3)
    assert good["verdict"] == "better with the adapter"
    text = "\n".join(describe(good))
    assert "never saw in training" in text and "40.0% without the adapter, 55.0% with it" in text
    assert "one peer's word" in text
    assert verdict({"loss": 2.0}, {"loss": 2.5}, 40) == "worse with the adapter"
    assert verdict({"loss": 2.0}, {"loss": 1.99}, 40) == "about the same with and without the adapter"
    assert "too few to rely on" in verdict({"loss": 2.0}, {"loss": 1.0}, 3)
    assert verdict({"loss": 2.0}, {"loss": float("nan")}, 40).startswith("broken")
    assert describe(None) == ["Not scored: there were too few pairs to keep any back for scoring."]


# -- trying an adapter, and adapters other people made -----------------------------------------
def direct_answers(model_dir, adapters, prompt, n):
    """Greedy answers from the whole model on one machine, without and with the adapters."""
    from transformers import AutoModelForCausalLM

    from rosetta_jane.lora import Applied
    model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    out = []
    for use in (False, True):
        live = {k: (v.detach() if use or k.endswith(".A") else torch.zeros_like(v)) for k, v in adapters.items()}
        ids = list(prompt)
        with torch.no_grad(), Applied(model.model.layers, 0, TARGETS, live, ALPHA / RANK):
            for _ in range(n):
                ids.append(int(model(input_ids=torch.tensor([ids])).logits[0, -1].argmax()))
        out.append(ids[len(prompt):])
    return out


def loud_adapters(t, seed=4, size=0.6):
    """Give a trainer adapters strong enough to change the model's answers, without training."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for key in t._order:
            if key.endswith(".B"):
                t.adapters[key].copy_(torch.randn(t.adapters[key].shape, generator=gen) * size)


def test_trying_an_adapter_gives_both_of_the_whole_models_answers(model_dir, tmp_path):
    from test_swarm import PROMPT
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 5), (5, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1, total_steps=1)
        loud_adapters(t)
        without, with_it = t.try_out(PROMPT, 7)
        want_without, want_with = direct_answers(model_dir, t.adapters, PROMPT, 7)
        assert without == want_without and with_it == want_with
        assert without != with_it                   # the trial shows a real difference
        assert sum(s.examples_trained for _, s in good) == 0        # and taught nothing
        halt = want_with[3]
        assert t.try_out(PROMPT, 7, stop_ids=[halt])[1] == want_with[:want_with.index(halt)]
    finally:
        stop(everyone)


def test_an_adapter_from_someone_else_is_loaded_checked_and_scored(model_dir, tmp_path):
    import json
    import shutil
    from test_swarm import PROMPT
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 8)])
    try:
        theirs = trainer(me, ends, model_dir, tmp_path, redundancy=1, total_steps=1)
        loud_adapters(theirs)
        folder = theirs.export(str(tmp_path / "shared"), model_dir)
        mine = SwarmTrainer.from_adapter(me, ends, model_dir, str(tmp_path / "w2"), folder, redundancy=1, find_wait=5.0)
        assert all(torch.equal(mine.adapters[k], theirs.adapters[k]) for k in theirs._order)
        assert mine.try_out(PROMPT, 5) == theirs.try_out(PROMPT, 5)
        card = mine.compare(make_examples(4), name="shared")
        (want_before, want_after), _ = direct_scores(model_dir, mine.adapters, make_examples(4))
        assert abs(card["after"]["loss"] - want_after["loss"]) < 1e-4 and card["never_trained_on"] is False
        # Adapters that do not belong to this model, or are damaged, are refused before any use.
        json.dump({"base_model_fingerprint": "0" * 16}, open(os.path.join(folder, "rosetta-jane-run.json"), "w"))
        with pytest.raises(ValueError, match="different model"):
            SwarmTrainer.from_adapter(me, ends, model_dir, str(tmp_path / "w3"), folder)
        os.remove(os.path.join(folder, "rosetta-jane-run.json"))
        config = json.load(open(os.path.join(folder, "adapter_config.json")))
        json.dump({**config, "r": RANK * 2}, open(os.path.join(folder, "adapter_config.json"), "w"))
        with pytest.raises(ValueError, match="does not fit"):
            SwarmTrainer.from_adapter(me, ends, model_dir, str(tmp_path / "w4"), folder)
        json.dump(config, open(os.path.join(folder, "adapter_config.json"), "w"))
        from safetensors.torch import load_file, save_file
        weights = load_file(os.path.join(folder, "adapter_model.safetensors"))
        first = sorted(weights)[0]
        weights[first] = torch.full_like(weights[first], float("nan"))
        save_file(weights, os.path.join(folder, "adapter_model.safetensors"))
        with pytest.raises(ValueError, match="not numbers"):
            SwarmTrainer.from_adapter(me, ends, model_dir, str(tmp_path / "w5"), folder)
        shutil.rmtree(folder)
        with pytest.raises(ValueError, match="readable adapter"):
            SwarmTrainer.from_adapter(me, ends, model_dir, str(tmp_path / "w6"), folder)
    finally:
        stop(everyone)
