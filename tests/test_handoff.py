"""Fingerprints of handoffs: every device logs what it sent and received, and the logs line up."""
import json
import os

import torch

from rosetta_jane.cli import main
from rosetta_jane.handoff import HandoffLog, compare, fingerprint, read_log
from rosetta_jane.swarm import Generation

from test_swarm import PROMPT, Liar, model_dir, reference, swarm, wait_for  # noqa: F401
from test_train import RANK, make_examples, setup, trainer


def logged(nodes, tmp_path):
    paths = []
    for i, n in enumerate(nodes):
        path = str(tmp_path / f"device-{i}" / "handoffs.jsonl")
        n.handoffs = HandoffLog(path, n.node_id)
        paths.append(path)
    return paths


def test_a_fingerprint_is_small_and_holds_no_content():
    t = torch.randn(1, 6, 32)
    fp = fingerprint(t.numpy().tobytes(), t)
    assert set(fp) == {"hash", "bytes", "shape", "dtype", "norm", "top"} and len(json.dumps(fp)) < 400
    assert len(fp["top"]) == 8 and fp["shape"] == [1, 6, 32]
    assert fingerprint(t.numpy().tobytes(), t)["hash"] == fp["hash"]
    nudged = t.clone()
    nudged[0, 0, 0] += 1e-6
    assert fingerprint(nudged.numpy().tobytes(), nudged)["hash"] != fp["hash"]
    assert fingerprint(b"abc", None) == {"hash": fingerprint(b"abc", None)["hash"], "bytes": 3}


def test_logs_from_every_device_line_up(model_dir, reference, tmp_path):
    asker, ends, good, _, everyone = swarm(model_dir, tmp_path, [(2, 8), (2, 8)], [])
    try:
        paths = logged([asker] + [n for n, _ in good], tmp_path)
        gen = Generation(asker, ends, redundancy=2, find_wait=3.0)
        out = gen.run(PROMPT, 5)
        assert out == reference[0](PROMPT, 5)
    finally:
        for n in everyone:
            n.stop()
    mine = read_log(paths[0])
    assert {r["role"] for r in mine} == {"asking"} and {r["op"] for r in mine} == {"step", "step_ok"}
    assert all(r["layers"] == [2, 8] and r["node"] == asker.node_id[:12] for r in mine)
    assert len(mine) == 2 * 2 * 5                  # two peers, there and back, five steps
    assert {r["role"] for r in read_log(paths[1])} == {"serving"}
    r = compare(paths)
    assert r["devices"] == 3 and r["handoffs_sent"] == r["delivered_intact"] == 20 and not r["unmatched"]
    assert r["work_compared"] == 5 and not r["real_differences"]
    assert main(["compare-handoffs", *paths]) == 0
    # Without one serving peer's file, what was exchanged with it has nothing to match against.
    partial = compare(paths[:2])
    assert len(partial["unmatched"]) == 5 and {u["dir"] for u in partial["unmatched"]} == {"sent"}


def test_the_logs_show_which_peer_returned_something_else(model_dir, reference, tmp_path, capsys):
    asker, ends, good, bad, everyone = swarm(model_dir, tmp_path, [(2, 8), (2, 8)], [(2, 8)])
    try:
        paths = logged([asker] + [n for n, _ in good + bad], tmp_path)
        gen = Generation(asker, ends, redundancy=3, find_wait=3.0)
        assert gen.run(PROMPT, 4) == reference[0](PROMPT, 4)
        liar_addr = f"127.0.0.1:{bad[0][0].port}"
    finally:
        for n in everyone:
            n.stop()
    r = compare(paths)
    assert not r["unmatched"]                      # what the liar sent did arrive; it was just wrong
    assert r["real_differences"] and all(liar_addr in d["peers"] for d in r["real_differences"])
    assert all(d["kind"] == "different" and d["layers"] == [2, 8] for d in r["real_differences"])
    assert main(["compare-handoffs", *paths]) == 1
    assert liar_addr in capsys.readouterr().out


def test_small_differences_are_told_apart_from_real_ones(tmp_path):
    base = torch.randn(1, 4, 64)
    path = str(tmp_path / "a" / "handoffs.jsonl")
    log = HandoffLog(path, "asker-node-id")
    given = torch.randn(1, 4, 64)
    for peer, out in (("10.0.0.1:7700", base), ("10.0.0.2:7700", base * 1.0005), ("10.0.0.3:7700", base * 1.4 + 0.3)):
        tap = log.tap("asking", peer)
        tap("sent", {"op": "open", "start": 3, "end": 9}, None, b"")
        tap("sent", {"op": "step"}, given, given.numpy().tobytes())
        tap("received", {"op": "step_ok"}, out, out.numpy().tobytes())
    log.close()
    r = compare([path])
    assert r["work_compared"] == 1 and r["rounding_differences"] == 1
    assert len(r["real_differences"]) == 1 and "10.0.0.3:7700" in r["real_differences"][0]["peers"]


def test_training_handoffs_are_logged_too(model_dir, tmp_path):
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 8)])
    try:
        paths = logged([me, good[0][0]], tmp_path)
        trainer(me, ends, model_dir, tmp_path, redundancy=1, total_steps=1).train(make_examples())
    finally:
        for n in everyone:
            n.stop()
    ops = [r["op"] for r in read_log(paths[0])]
    for op in ("t_adapters", "t_fwd", "t_fwd_ok", "t_bwd", "t_bwd_ok", "t_grads_ok"):
        assert op in ops, op
    r = compare(paths)
    assert r["handoffs_sent"] == r["delivered_intact"] > 0 and not r["unmatched"]


def test_a_full_log_starts_a_new_file_and_keeps_one_old_one(tmp_path, monkeypatch):
    import rosetta_jane.handoff as handoff
    monkeypatch.setattr(handoff, "MAX_BYTES", 2000)
    path = str(tmp_path / "handoffs.jsonl")
    log = HandoffLog(path, "n")
    tap = log.tap("asking", "10.0.0.1:1")
    t = torch.randn(1, 2, 16)
    for _ in range(40):
        tap("sent", {"op": "step"}, t, t.numpy().tobytes())
    log.close()
    assert os.path.getsize(path) <= 2000 and os.path.exists(path + ".1")
    assert sorted(os.listdir(tmp_path)) == ["handoffs.jsonl", "handoffs.jsonl.1"]
    assert 0 < len(read_log(path)) < 40
