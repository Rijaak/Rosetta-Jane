"""Teams: an adapter applied for a group because its members signed on, and for nobody else.

The key claims, on a tiny random model over real sockets:
  * a team's answers are token for token what the whole model gives with the adapter loaded by
    the PEFT library on one machine;
  * anyone not in the team, asking through the very same devices at the same time, gets the
    unchanged model;
  * a device whose owner has not signed it on never applies the adapter;
  * when a team ends, by choice or because its leader vanished, every device drops the adapter;
  * the devices that helped train an adapter can be offered a copy, nobody else can take one,
    and a copy can be trained further.
"""
import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from rosetta_jane import team as T
from rosetta_jane.helpers import BOOK, HelpLog, RunBook, Sharing
from rosetta_jane.lan.protocol import ProtocolError, RemoteError
from rosetta_jane.layers import LayerServer, LocalAuditor, LocalEnds, model_shape
from rosetta_jane.lora import init_adapters, shapes_from_files, to_peft
from rosetta_jane.net import Node
from rosetta_jane.swarm import Generation, NoRoute

from test_mobile import CODE, phone_peer
from test_mobile import stop as stop_phone
from test_swarm import (PROMPT, ROOT, SECRET, Liar, _tokenizer, addr, assert_scores_match, model_dir, node,  # noqa: F401
                        record_scores, reference, serving, wait_for)

HERE = os.path.dirname(os.path.abspath(__file__))
RANK, ALPHA, TARGETS = 4, 8.0, ["q_proj", "v_proj"]


def make_adapter(model_dir, folder, seed=4, size=0.6, facts=None):
    """An adapter for the tiny model strong enough to change its answers, saved the standard way."""
    prefix, _ = model_shape(model_dir)
    adapters = init_adapters(shapes_from_files(model_dir, TARGETS), RANK, seed)
    gen = torch.Generator().manual_seed(seed)
    for key in sorted(adapters):
        if key.endswith(".B"):
            adapters[key] = torch.randn(adapters[key].shape, generator=gen) * size
    os.makedirs(folder, exist_ok=True)
    save_file(to_peft(adapters, prefix), os.path.join(folder, "adapter_model.safetensors"))
    with open(os.path.join(folder, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": ALPHA,
                   "lora_dropout": 0.0, "bias": "none", "target_modules": TARGETS,
                   "base_model_name_or_path": model_dir}, f)
    from rosetta_jane.lan.slicing import model_fingerprint
    with open(os.path.join(folder, T.FACTS), "w") as f:
        json.dump({"name": os.path.basename(folder), "base_model_fingerprint": model_fingerprint(model_dir),
                   "work_checked": 40, "work_unchecked": 2, **(facts or {})}, f)
    return folder


@pytest.fixture(scope="module")
def adapter(model_dir, tmp_path_factory):
    return make_adapter(model_dir, str(tmp_path_factory.mktemp("adapters") / "field-notes"))


@pytest.fixture(scope="module")
def with_adapter(model_dir, adapter):
    """The whole model on one machine with the adapter loaded by the PEFT library: (greedy, logits)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_dir), adapter).eval()

    def logits(ids):
        with torch.no_grad():
            return model(input_ids=torch.tensor([ids])).logits[0, -1]

    def greedy(ids, n):
        ids = list(ids)
        for _ in range(n):
            ids.append(int(logits(ids).argmax()))
        return ids[-n:]
    return greedy, logits


def lead(n, model_dir, adapter, **kw):
    """A leader's side, made the way the gateway makes it."""
    from rosetta_jane.lan.slicing import model_fingerprint
    prefix, total = model_shape(model_dir)
    fp = model_fingerprint(model_dir)
    found = T.fitted(adapter, prefix, fp, lambda targets: shapes_from_files(model_dir, targets))
    return T.TeamHost(n, model=fp, total_layers=total, name=found["name"], adapters=found["adapters"],
                      rank=found["rank"], alpha=found["alpha"], targets=found["targets"], facts=found["facts"],
                      **kw).attach()


def signed_on(n, server, **how):
    service = T.TeamService(n, server)
    service.join(**how)
    return service


def ask(n, ends, n_tokens=8, **kw):
    kw.setdefault("redundancy", 1)
    gen = Generation(n, ends, find_wait=5.0, **kw)
    return gen.run(PROMPT, n_tokens), gen


def stop(nodes):
    for n in nodes:
        n.stop()


def test_a_team_answers_with_the_adapter_and_everyone_else_gets_the_unchanged_model(
        model_dir, adapter, reference, with_adapter, tmp_path):
    hub = node()
    boot = [addr(hub)]
    (n1, s1), (n2, s2) = serving(model_dir, tmp_path, 2, 5, boot), serving(model_dir, tmp_path, 5, 8, boot)
    leader, member, outsider = node(bootstrap=boot), node(bootstrap=boot), node(bootstrap=boot)
    everyone = [hub, n1, n2, leader, member, outsider]
    try:
        host = lead(leader, model_dir, adapter, joining="invisible")
        t1, t2 = signed_on(n1, s1, code=host.code), signed_on(n2, s2, code=host.code)
        ends = {who: LocalEnds(model_dir, 2, str(tmp_path / f"ends-{i}")) for i, who in
                enumerate(("leader", "member", "outsider"))}
        link = T.TeamLink(member, model=host.model, code=host.code.lower().replace(" ", "-"), beat=0.2).join()
        assert link.desc["facts"]["work_unchecked"] == 2 and link.desc["name"] == "field-notes"
        assert not link.active() and not host.active()       # signed on is not begun
        want_plain, want_team = reference[0](PROMPT, 8), with_adapter[0](PROMPT, 8)
        assert want_plain != want_team, "the adapter must change the answer or this test shows nothing"
        assert wait_for(lambda: all(host.id in s["teams"] for p in member.table.peers() for s in p.services))

        # Before the leader begins, nothing is applied for anyone.
        assert ask(member, ends["member"])[0] == want_plain
        assert host.missing(2) == [] and [c["state"] for c in host.roster()] == ["Leader", "Ready", "Ready", "Ready"]
        host.begin(2)
        assert wait_for(link.active)

        # The team: the member and the leader both get the model with the adapter, number for number.
        for who, n, use in (("member", member, link.use()), ("leader", leader, host.use())):
            seen = record_scores(ends[who])
            out, gen = ask(n, ends[who], team=use)
            assert out == want_team, who
            assert_scores_match(seen, PROMPT, out, with_adapter[1])
            assert len(gen.route) == 2
        assert s1.team_steps_served > 0 and s2.team_steps_served > 0

        # Not in the team: the same devices, the same moment, the unchanged model.
        seen = record_scores(ends["outsider"])
        out, _ = ask(outsider, ends["outsider"])
        assert out == want_plain
        assert_scores_match(seen, PROMPT, out, reference[1])

        # The leader ends it: every device lets the adapter go.
        use = link.use()
        host.end()
        assert wait_for(lambda: link.ended is not None and not s1.teams() and not s2.teams(), 10)
        assert link.use() is None and "leader ended" in link.ended
        assert ask(outsider, ends["outsider"])[0] == want_plain
        assert wait_for(lambda: not any(s["teams"] for p in member.table.peers() for s in p.services))
        with pytest.raises(NoRoute, match="team"):
            ask(member, ends["member"], team=use)       # a kept copy of the adapter finds nobody to run it
        assert not t1.links and not t2.links
    finally:
        stop(everyone)


class Oblivious(LayerServer):
    """A peer from before teams existed, dressed up as a team member: it runs the unchanged model."""

    def __init__(self, *a, claim="", **kw):
        super().__init__(*a, **kw)
        self.claim = claim

    def service(self):
        return {**super().service(), "teams": [self.claim]}

    def _open(self, ch, header, tensor, state):
        super()._open(ch, {k: v for k, v in header.items() if k != "team"}, tensor, state)


def test_a_device_that_has_not_signed_on_never_applies_the_adapter(model_dir, adapter, reference, tmp_path):
    hub = node()
    boot = [addr(hub)]
    (n1, s1), (n2, s2) = serving(model_dir, tmp_path, 2, 5, boot), serving(model_dir, tmp_path, 5, 8, boot)
    leader = node(bootstrap=boot)
    everyone = [hub, n1, n2, leader]
    try:
        host = lead(leader, model_dir, adapter)
        signed_on(n1, s1, code=host.code)
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert host.missing(2) == [5, 6, 7]
        with pytest.raises(ValueError, match="5-7"):
            host.begin(2)                               # the leader is told which layers nobody has signed on for
        host.begin(2, force=True)
        assert wait_for(lambda: any(host.id in s["teams"] for p in leader.table.peers() for s in p.services))
        with pytest.raises(NoRoute, match="layer 5.*team"):
            ask(leader, ends, team=host.use())
        ch = leader.dial(addr(n2))                      # asking the unsigned device directly is refused
        with pytest.raises(RemoteError, match="not signed on"):
            ch.call({"op": "open", "model": host.model, "start": 5, "end": 8, "team": host.id})
        ch.close()
        assert s2.team_steps_served == 0 and s2.teams() == []

        # A device that claims the team but ignores it would quietly give the unchanged model. It is not used.
        n3 = Node(SECRET, "127.0.0.1:0", bootstrap=boot, gossip_interval=0.3, quiet=True)
        s3 = Oblivious(model_dir, 5, 8, str(tmp_path / "old"), claim=host.id)
        s3.attach(n3)
        everyone.append(n3.start())
        assert wait_for(lambda: sum(any(host.id in s["teams"] for s in p.services)
                                    for p in leader.table.peers()) == 2)
        with pytest.raises(NoRoute):
            ask(leader, ends, team=host.use())
        assert s3.steps_served == 0
    finally:
        stop(everyone)


def test_who_can_sign_on_is_the_leaders_choice_open_ask_or_invisible(model_dir, adapter, tmp_path):
    hub = node()
    boot = [addr(hub)]
    leader, guest, shy = node(bootstrap=boot), node(bootstrap=boot), node(bootstrap=boot)
    inbox = T.Invitations(guest).attach()
    guest.presence = lambda: {"invites": True, "busy": False, "name": "Sam"}       # its owner can be invited
    hidden = T.Invitations(shy).attach()
    hidden.closed = True                                             # this one's owner chose to be invisible
    shy.presence = lambda: {"invites": False, "busy": False}
    try:
        host = lead(leader, model_dir, adapter, joining="open", limit=3)
        fp = host.model
        assert wait_for(lambda: any(host.id in p.leads for p in guest.table.peers()))

        def join(**how):
            link = T.TeamLink(guest, model=fp, beat=0.2, find_wait=2.0, **how).join()
            link.leave()
            assert wait_for(lambda: len(host.roster()) == 1)
            return link

        # Open: anyone in the group sees it and can sign on, with no code. That is what quick match finds.
        listed = T.teams_in_group(guest, fp)
        assert [(t["team"], t["joining"], t["name"], t["from"]) for t in listed] == \
            [(host.id, "open", "field-notes", addr(leader))]
        assert [t["team"] for t in T.open_teams(guest, fp)] == [host.id]
        assert join(team=host.id).desc["joining"] == "open"
        # The model and the adapter are what a person is shown first, before anything else.
        lines = T.fact_lines({**listed[0], "fits": True, "held_as": "my-own-copy"})
        assert lines[0] == f"Model: ({fp}). You have this model on this device."
        assert lines[1].startswith(f"Adapter: \"field-notes\" ({host.adapter}). Available to download for the session")
        assert "You also hold this adapter yourself, as \"my-own-copy\"" in lines[1]
        assert any("double-checked: 40" in x for x in lines) and "Teacher rating: none." in " ".join(lines)
        assert "Community rating: none yet" in " ".join(lines) and listed[0]["kind"] == "session"
        refused = T.fact_lines({**listed[0], "fits": False})[0]
        assert "NOT the model on this device" in refused and "Models are not sent through a team" in refused

        # Ask: still listed. Anyone who sees it can ask; the leader approves or declines each request.
        host.set(joining="ask")
        assert [t["joining"] for t in T.teams_in_group(guest, fp)] == ["ask"] and T.open_teams(guest, fp) == []
        with pytest.raises(PermissionError, match="ask to join"):
            join(team=host.id)
        asker = T.TeamLink(guest, model=fp, team=host.id, beat=0.2, called="Sam <b>the</b>\nTeacher")
        asker.ask()
        asker.ask()                                     # asking twice is one request
        assert asker.asked() == "waiting" and len(host.roster()) == 1
        waiting = host.requests()                       # shown under the name they chose: a label, tidied, not checked
        assert len(waiting) == 1 and waiting[0]["who"] == "Sam <b>the</b> Teacher (127.0.0.1)"
        with pytest.raises(PermissionError):
            asker.join()                                # asking is not being let in
        host.approve(waiting[0]["id"])
        assert host.requests() == [] and asker.asked() == "approved"
        asker.join()                                    # approved: in, and the adapter came with it
        assert asker.use() is not None and not asker.use().plain and len(host.roster()) == 2
        assert host.roster()[1]["who"] == "Sam <b>the</b> Teacher (127.0.0.1)"
        with pytest.raises(ValueError, match="no longer waiting"):
            host.approve(waiting[0]["id"])

        # The leader can take a device back out. It drops the adapter and does not come back in.
        host.remove(host.roster()[1]["id"])
        assert wait_for(lambda: asker.ended is not None, 10) and asker.use() is None
        assert "no longer in the team" in asker.ended and len(host.roster()) == 1
        again = T.TeamLink(guest, model=fp, team=host.id)
        with pytest.raises(PermissionError, match="removed"):
            again.ask()
        with pytest.raises(PermissionError, match="removed"):
            T.TeamLink(guest, model=fp, code=host.code).join()
        host._barred.clear()
        second = T.TeamLink(guest, model=fp, team=host.id)
        second.ask()
        host.decline(host.requests()[0]["id"])
        assert second.asked() == "declined"
        with pytest.raises(PermissionError):
            second.join()

        # The leader sees who in the group can be invited. A device whose owner is invisible is not shown,
        # and does not take invitations.
        assert wait_for(lambda: {d["addr"] for d in host.devices()} == {addr(guest)}, 10)
        assert inbox.list() == [] and host.devices()[0]["invited"] is False and host.devices()[0]["name"] == "Sam"
        host.invite(addr(guest))
        host.invite(addr(guest))                        # asking twice is fine; it is one invitation
        assert host.devices()[0]["invited"] is True
        with pytest.raises(ValueError, match="not taking invitations"):
            host.invite(addr(shy))
        waiting = inbox.list(fp)
        assert [(w["team"], w["from"]) for w in waiting] == [(host.id, addr(leader))]
        assert len(host.roster()) == 1 and hidden.list() == []          # an invitation signs nobody on
        held = inbox.get(host.id)
        join(team=host.id, leader=held["leader"], invitation=held["pass"])     # saying yes: no request, no code
        join(code=host.code)
        with pytest.raises(ValueError, match="no device at that address"):
            host.invite("127.0.0.1:9")

        # Invisible: not listed, cannot be looked at or asked, and only the code or an invitation gets in.
        host.set(joining="invisible")
        assert T.teams_in_group(guest, fp) == []
        with pytest.raises(T.NoTeam, match="invisible"):
            T.TeamLink(guest, model=fp, team=host.id, find_wait=1.0).look()
        with pytest.raises(T.NoTeam, match="invisible"):
            T.TeamLink(guest, model=fp, team=host.id, leader=addr(leader)).ask()
        wrong = next(c for c in ("AAA AAA", "BBB BBB") if c != host.code)
        with pytest.raises(T.NoTeam):
            T.TeamLink(guest, model=fp, code=wrong, find_wait=1.0).join()
        join(code=host.code)
        join(team=host.id, leader=held["leader"], invitation=held["pass"])
        with pytest.raises(ValueError):
            T.TeamLink(guest, model=fp, code="not a code")
        with pytest.raises(ValueError, match="different model"):
            T.TeamLink(guest, model="ab" * 8, code=host.code, find_wait=1.0).join()

        # A team holds as many devices as its leader set.
        host.set(joining="open")
        a = T.TeamLink(guest, model=fp, team=host.id, beat=0.2).join()
        b = T.TeamLink(guest, model=fp, team=host.id, beat=0.2).join()
        with pytest.raises(T.NoTeam, match="full"):
            T.TeamLink(guest, model=fp, team=host.id).join()
        assert T.open_teams(guest, fp) == []            # a full team is not offered by quick match
        a.leave(), b.leave()
        with pytest.raises(ValueError):
            host.set(joining="everyone")

        # Ended teams are gone from every list, and an invitation to one falls away.
        host.end()
        assert wait_for(lambda: not any(p.leads for p in guest.table.peers()))
        assert T.teams_in_group(guest, fp) == [] and inbox.list() == []
        with pytest.raises(T.NoTeam):
            inbox.get(host.id)
    finally:
        stop([hub, leader, guest, shy])


def test_the_leaders_choices_are_remembered_for_the_next_team(tmp_path):
    work = str(tmp_path)
    assert T.load_settings(work) == {"joining": "open", "limit": 16, "hidden": False, "name": ""}
    T.save_settings(work, {"joining": "invisible", "limit": 4, "hidden": True, "name": "Sam  the\nTeacher " + "x" * 40,
                           "something": "else"})
    assert T.load_settings(work) == {"joining": "invisible", "limit": 4, "hidden": True,
                                     "name": "Sam the Teacher xxxxxxxx"}       # a short, one-line label
    with open(os.path.join(work, T.SETTINGS), "w") as f:
        f.write('{"joining": "whoever", "limit": 9999}')
    assert T.load_settings(work) == {"joining": "open", "limit": 16, "hidden": False, "name": ""}    # nonsense is ignored
    assert T.tidy("daf-h2d") == "DAF H2D" and T.keys("daf h2d") == T.keys("DAF H2D")
    assert T.keys("DAF H2D")[0] != T.keys("DAF H2E")[0]
    for bad in ("DAF H2", "DAF H2O", "", None, "DAF H2DX"):                # O is not in the alphabet
        with pytest.raises(ValueError):
            T.tidy(bad)
    assert all(len(T.new_code()) == 7 for _ in range(20))


def test_when_the_leader_vanishes_every_device_drops_the_adapter(model_dir, adapter, reference, tmp_path):
    hub = node()
    boot = [addr(hub)]
    n1, s1 = serving(model_dir, tmp_path, 2, 8, boot)
    leader, member = node(bootstrap=boot), node(bootstrap=boot)
    try:
        host = lead(leader, model_dir, adapter)
        service = T.TeamService(n1, s1)
        link1 = service.join(code=host.code)
        link1.beat, link1.patience = 0.2, 3
        link = T.TeamLink(member, model=host.model, code=host.code, beat=0.2, patience=3).join()
        host.begin(2)
        assert wait_for(link.active) and s1.teams() == [host.id]
        leader.stop()                                   # no goodbye: the leader's device is simply gone
        assert wait_for(lambda: link.ended is not None and s1.teams() == [], 30)
        assert "no longer be reached" in link.ended and link.use() is None and not service.links
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert ask(member, ends)[0] == reference[0](PROMPT, 8)      # and the unchanged model answers as ever
    finally:
        stop([hub, n1, member])


def test_an_adapter_from_a_leader_is_checked_before_it_is_used(model_dir, adapter, tmp_path):
    import numpy as np
    hub = node()
    boot = [addr(hub)]
    n1, s1 = serving(model_dir, tmp_path, 2, 8, boot)
    leader, member = node(bootstrap=boot), node(bootstrap=boot)
    try:
        good = lead(leader, model_dir, adapter)
        fp, total = good.model, good.total
        arrays = dict(good.use().adapters)
        good.end()

        def hostile(change):
            bent = {k: v.copy() for k, v in arrays.items()}
            change(bent)
            leader.team_host = None
            return T.TeamHost(leader, model=fp, total_layers=total, name="bent", adapters=bent, rank=RANK,
                              alpha=ALPHA, targets=TARGETS).attach()

        def poison(a):
            a["3.self_attn.q_proj.B"][0, 0] = np.nan

        def misshape(a):                                # well formed, but not this model's size
            a["3.self_attn.q_proj.A"] = np.zeros((RANK, 7), np.float32)

        def stray(a):
            a["3.self_attn.q_proj.weight.A"] = a.pop("3.self_attn.q_proj.A")

        for change, error in ((poison, ProtocolError), (stray, ProtocolError)):
            host = hostile(change)
            assert wait_for(lambda: any(host.id in p.leads for p in member.table.peers()))
            with pytest.raises(error):
                T.TeamLink(member, model=fp, code=host.code).join()
            host.end()
        host = hostile(misshape)
        assert wait_for(lambda: any(host.id in p.leads for p in n1.table.peers()))
        with pytest.raises(ValueError, match="could not take"):
            T.TeamService(n1, s1).join(code=host.code)
        assert s1.teams() == []
        link = T.TeamLink(member, model=fp, code=host.code, beat=0.2).join()
        with pytest.raises(ValueError, match="does not fit"):
            LocalEnds(model_dir, 4, str(tmp_path / "ends")).fit_team(link.use())
        link.leave()
        # The leader's own side refuses an adapter for another model, or one that does not fit, up front.
        other = make_adapter(model_dir, str(tmp_path / "other"), facts={"base_model_fingerprint": "cd" * 8})
        with pytest.raises(ValueError, match="different model"):
            lead(member, model_dir, other)
        assert s1.team_steps_served == 0
    finally:
        stop([hub, n1, leader, member])


def test_a_lying_device_in_a_team_is_caught_by_recomputing_with_the_adapter(
        model_dir, adapter, with_adapter, tmp_path):
    hub = node()
    boot = [addr(hub)]
    n1, s1 = serving(model_dir, tmp_path, 2, 8, boot)
    n2 = Node(SECRET, "127.0.0.1:0", bootstrap=boot, gossip_interval=0.3, quiet=True)
    s2 = Liar(model_dir, 2, 8, str(tmp_path / "liar"))
    s2.attach(n2)
    n2.start()
    leader = node(bootstrap=boot)
    try:
        host = lead(leader, model_dir, adapter)
        signed_on(n1, s1, code=host.code), signed_on(n2, s2, code=host.code)
        host.begin(2)
        assert wait_for(lambda: sum(any(host.id in s["teams"] for s in p.services)
                                    for p in leader.table.peers()) == 2)
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        auditor = LocalAuditor(model_dir, str(tmp_path / "audit"), max_layers=4)
        out, gen = ask(leader, ends, team=host.use(), redundancy=2, auditor=auditor, audit_rate=1.0)
        assert out == with_adapter[0](PROMPT, 8)
        assert gen.caught == [n2.node_id] and gen.disputes >= 1 and auditor.audits >= 1
        # Without the team's adapter the recomputation would have blamed the honest device instead.
        assert n1.node_id not in leader.banned
    finally:
        stop([hub, n1, n2, leader])


def test_a_phone_serves_a_team_and_asks_in_one_like_any_pc(model_dir, adapter, reference, with_adapter, tmp_path):
    hub = node()
    boot = [addr(hub)]
    n1, s1 = serving(model_dir, tmp_path, 2, 8, boot, dtype="float32")
    leader = node(bootstrap=boot)
    proc = None
    try:
        host = lead(leader, model_dir, adapter, joining="invisible")
        signed_on(n1, s1, code=host.code)
        proc, where, _ = phone_peer(model_dir, tmp_path, "2:8", boot, "--team", host.code)
        assert wait_for(lambda: sum(any(host.id in s["teams"] for s in p.services)
                                    for p in leader.table.peers()) == 2, 90)
        host.begin(2)
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        seen = record_scores(ends)
        out, gen = ask(leader, ends, team=host.use(), redundancy=2)       # the phone and the PC check each other
        want = with_adapter[0](PROMPT, 8)
        assert out == want and gen.verified_steps > 0 and gen.unverified_steps == 0 and gen.caught == []
        assert_scores_match(seen, PROMPT, out, with_adapter[1], tol=2e-4)

        # And a phone asking: signs on with the code, waits for the leader, gets the adapter's answer.
        env = dict(os.environ, PYTHONPATH=ROOT, ROSETTA_JANE_GROUP_CODE=CODE)
        base = [sys.executable, os.path.join(HERE, "no_torch.py"), "ask-ids", "--model", model_dir,
                "--own-layers", "2", "--ids", ",".join(map(str, PROMPT)), "--new", "8", "--redundancy", "2",
                "--work-dir", str(tmp_path / "phone-asker"), "--peer", boot[0]]
        r = subprocess.run(base + ["--team", host.code], env=env, capture_output=True, text=True, timeout=180)
        assert r.returncode == 0 and "FORBIDDEN IMPORTS" not in r.stdout, r.stdout + r.stderr
        got = json.loads(next(line for line in r.stdout.splitlines() if line.startswith("RESULT "))[7:])
        assert got["ids"] == want and got["team"] == host.id and got["blocked"] == []
        for t, row in enumerate(got["scores"][:8]):
            gap = float((torch.tensor(row) - with_adapter[1](list(PROMPT) + want[:t])).abs().max())
            assert gap < 2e-4, f"step {t}: a phone's team scores differ from the whole model's by {gap:.1e}"
        r = subprocess.run(base, env=env, capture_output=True, text=True, timeout=180)   # the same phone, no team
        got = json.loads(next(line for line in r.stdout.splitlines() if line.startswith("RESULT "))[7:])
        assert got["ids"] == reference[0](PROMPT, 8) and got["team"] is None

        host.end()                                      # the phone lets go of the adapter with everyone else
        assert wait_for(lambda: not any(s["teams"] for p in leader.table.peers() for s in p.services), 30)
    finally:
        if proc is not None:
            said = stop_phone(proc)
            assert "Signed on to the team" in said and "is over" in said, said
        stop([hub, n1, leader])


# -- sharing a finished adapter with the devices that helped train it, and training it further ------------
def test_the_devices_that_helped_can_take_a_copy_and_train_it_further(model_dir, reference, tmp_path):
    from safetensors.torch import load_file

    from rosetta_jane.lan.slicing import model_fingerprint
    from rosetta_jane.train import SwarmTrainer
    from test_train import make_examples, serving_peer, trainer
    hub = node()
    boot = [addr(hub)]
    (n1, s1), (n2, s2) = serving_peer(model_dir, tmp_path, 2, 5, boot), serving_peer(model_dir, tmp_path, 5, 8, boot)
    idle_n, idle_s = serving_peer(model_dir, tmp_path, 5, 8, boot, training=False)     # serves, never helped train
    me = node(bootstrap=boot)
    everyone = [hub, n1, n2, idle_n, me]
    fp = model_fingerprint(model_dir)
    try:
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert wait_for(lambda: sum(bool(p.services) for p in me.table.peers()) == 3)
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1, run_id="first")
        stats = t.train(make_examples())
        out = t.export(str(tmp_path / "work" / "adapters" / "first"), model_dir)
        with open(os.path.join(out, T.FACTS), "w") as f:
            json.dump({"name": "first", "base_model_fingerprint": fp, "finished": "2026-11-22 10:00:00",
                       "work_checked": stats["checked_examples"], "work_unchecked": stats["unchecked_examples"],
                       **t.record()}, f)
        book = RunBook(out, create=False)
        assert set(book.helpers) == {n1.node_id, n2.node_id} and all(v > 0 for v in book.helpers.values())
        assert t.record() == {"run": book.run, "helpers": 2}
        assert s1.helped.runs[book.run]["examples"] == s1.examples_trained > 0
        assert book.run not in idle_s.helped.runs

        # Nothing is on offer until the leader says so.
        desks = {n: Sharing(n, s.helped).attach() for n, s in ((n1, s1), (n2, s2), (idle_n, idle_s))}
        assert all(d.offers() == [] for d in desks.values())
        mine = Sharing(me).attach()
        row = mine.open(out, fp)
        assert row == {"run": book.run, "name": "first", "helpers": 2, "taken": 0}
        assert wait_for(lambda: desks[n1].offers() and desks[n2].offers())
        assert desks[idle_n].offers() == []                   # a device that did no work is offered nothing
        offer = desks[n1].offers()[0]
        assert offer["run"] == book.run and offer["examples"] == s1.examples_trained
        seen = desks[n1].look(book.run)
        assert seen["name"] == "first" and seen["facts"]["work_unchecked"] == stats["unchecked_examples"]
        assert not os.path.exists(str(tmp_path / "helper" / "adapters"))        # looking takes nothing

        kept = desks[n1].take(book.run, str(tmp_path / "helper"), fp)
        folder = str(tmp_path / "helper" / "adapters" / kept)
        assert kept == "first" and mine.report()[0]["taken"] == 1
        same = load_file(os.path.join(folder, "adapter_model.safetensors"))
        want = load_file(os.path.join(out, "adapter_model.safetensors"))
        assert sorted(same) == sorted(want) and all(torch.equal(same[k], want[k]) for k in want)
        facts = json.load(open(os.path.join(folder, T.FACTS)))
        assert facts["received"] is True and facts["helped_examples"] == s1.examples_trained
        assert not os.path.exists(os.path.join(folder, BOOK))  # the leader's private record does not travel
        assert desks[n1].take(book.run, str(tmp_path / "helper"), fp) == "first-2"     # a second copy gets its own name
        # Taking a copy applied nothing: the helper serves the unchanged model, as before.
        assert s1.teams() == [] and s1.team_steps_served == 0
        assert ask(me, ends)[0] == reference[0](PROMPT, 8)

        # Nobody else can take one: no ticket, a made-up ticket, or a ticket for a device that did no work.
        with pytest.raises(KeyError):
            desks[idle_n].take(book.run, str(tmp_path / "idle"), fp)
        for who, given in ((idle_n.node_id, "0" * 32), (idle_n.node_id, book.ticket(idle_n.node_id)),
                           (n1.node_id, book.ticket(n2.node_id))):
            ch = idle_n.dial(addr(me))
            with pytest.raises(RemoteError, match="ticket|no finished work"):
                ch.call({"op": "gift_take", "run": book.run, "as": who, "ticket": given})
            ch.close()
        with pytest.raises(ValueError, match="different model"):
            desks[n2].take(book.run, str(tmp_path / "helper2"), "ab" * 8)
        # A helper's tickets survive a restart of its program.
        s1.helped.save()
        again = HelpLog(os.path.join(str(tmp_path / f"LayerServer-2-5-{n1.port}"), "helped.json"))
        assert again.runs[book.run]["ticket"] == s1.helped.runs[book.run]["ticket"]
        mine.close(book.run)
        assert wait_for(lambda: desks[n2].offers() == [])
        with pytest.raises(ValueError, match="no record"):
            mine.open(make_adapter(model_dir, str(tmp_path / "from-a-zip")), fp)       # nobody helped, by this record

        # The helper trains it further. They lead that run; the new adapter remembers what it came from.
        helper_ends = LocalEnds(model_dir, 5, str(tmp_path / "helper-ends"))     # the other helper serves 5-7
        t2 = SwarmTrainer.continuing(n1, helper_ends, model_dir, str(tmp_path / "helper"), folder, total_steps=2,
                                     accum=2, lr=5e-3, warmup_steps=1, save_every=2, find_wait=5.0, quiet=True,
                                     redundancy=1, run_id="second", rank=99)
        assert (t2.rank, t2.alpha, sorted(t2.targets)) == (t.rank, t.alpha, sorted(t.targets))   # the adapter's own
        start = {k: v.detach().clone() for k, v in t2.adapters.items()}
        assert all(torch.equal(start[k], t.adapters[k].detach()) for k in start)
        t2.train(make_examples())
        assert max(float((t2.adapters[k].detach() - start[k]).abs().max()) for k in start) > 1e-5
        record = t2.record()
        assert record["run"] != book.run and record["helpers"] >= 1
        assert record["history"] == [{"name": "first", "adapter": T.adapter_fingerprint(
            {k: v.numpy() for k, v in start.items()}), "finished": "2026-11-22 10:00:00"}]
        assert torch.equal(load_file(os.path.join(folder, "adapter_model.safetensors"))[sorted(want)[0]],
                           want[sorted(want)[0]])              # the adapter it started from is untouched
    finally:
        stop(everyone)


def test_a_team_with_no_adapter_pools_its_devices_and_training_can_run_inside_it(model_dir, reference, tmp_path):
    from rosetta_jane.lan.slicing import model_fingerprint
    from test_train import make_examples, serving_peer, trainer
    hub = node()
    boot = [addr(hub)]
    (n1, s1), (n2, s2) = serving_peer(model_dir, tmp_path, 2, 8, boot), serving_peer(model_dir, tmp_path, 2, 8, boot)
    leader, member = node(bootstrap=boot), node(bootstrap=boot)
    fp = model_fingerprint(model_dir)
    try:
        host = T.TeamHost(leader, model=fp, total_layers=8, model_name="tiny-test-model", joining="open").attach()
        assert host.plain and host.adapter is None and host.name == "no adapter"
        mine = signed_on(n1, s1, team=host.id, leader=addr(leader))        # n2 serves the same layers, not signed on
        assert mine.links[host.id].use().plain
        assert wait_for(lambda: any(host.id in p.leads for p in member.table.peers()))
        listed = T.teams_in_group(member, fp)[0]
        assert listed["adapter"] is None and listed["model_name"] == "tiny-test-model"
        lines = T.fact_lines({**listed, "fits": True})
        assert lines[0] == f"Model: tiny-test-model ({fp}). You have this model on this device."
        assert lines[1].startswith("Adapter: none.")
        link = T.TeamLink(member, model=fp, team=host.id, beat=0.2).join()
        host.begin(2)
        assert wait_for(link.active) and link.use().plain
        assert wait_for(lambda: any(host.id in s["teams"] for p in member.table.peers() for s in p.services))
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        for _ in range(3):
            out, gen = ask(member, ends, team=link.use())
            assert out == reference[0](PROMPT, 8)                # the unchanged model...
        assert s1.team_steps_served > 0 and s2.steps_served == 0  # ...computed only by the team's devices

        # Training inside the team: only devices signed on to it do the work.
        lead_ends = LocalEnds(model_dir, 2, str(tmp_path / "lead-ends"))
        assert wait_for(lambda: any(host.id in s["teams"] for p in leader.table.peers() for s in p.services))
        t = trainer(leader, lead_ends, model_dir, tmp_path, redundancy=1, total_steps=2, team=host.id, run_id="in-team")
        t.train(make_examples())
        assert s1.examples_trained > 0 and s2.examples_trained == 0
        assert set(t.book.helpers) == {n1.node_id}               # and they are the ones who can be offered the result
        ch = leader.dial(addr(n2))
        with pytest.raises(RemoteError, match="not signed on"):
            ch.call({"op": "t_open", "model": fp, "start": 2, "end": 8, "rank": 4, "alpha": 8.0,
                     "targets": ["q_proj"], "team": host.id})
        ch.close()
        # A teaching team is its own lobby: listed apart, never found by quick match, and it changes no answers.
        host.end()
        leader.team_host = None
        school = T.TeamHost(leader, model=fp, total_layers=8, kind="teaching").attach()
        assert wait_for(lambda: [t["kind"] for t in T.teams_in_group(member, fp)] == ["teaching"], 20)
        assert T.teams_in_group(member, fp, kind="session") == [] and T.open_teams(member, fp) == []
        assert [t["team"] for t in T.open_teams(member, fp, kind=None)] == [school.id]
        pupil = T.TeamLink(member, model=fp, team=school.id, beat=0.2).join()
        with pytest.raises(ValueError, match="nothing to begin"):
            school.begin(2)
        assert not school.active() and not pupil.active() and pupil.kind == "teaching"
        assert "This is a teaching team" in T.fact_lines(pupil.desc)[1]
        with pytest.raises(ValueError, match="not led with an adapter"):
            T.TeamHost(leader, model=fp, total_layers=8, kind="teaching", adapters={"x": 1})
        mine.leave_all()
        assert wait_for(lambda: not any(s["teams"] for p in leader.table.peers() for s in p.services))
        t2 = trainer(leader, lead_ends, model_dir, tmp_path, redundancy=1, total_steps=1, team=host.id, run_id="alone")
        with pytest.raises(NoRoute):
            t2.train(make_examples())
        assert s2.examples_trained == 0
    finally:
        stop([hub, n1, n2, leader, member])


# -- teaching points: a larger share of a team's computing power, for people who gave theirs to teaching ----
def test_a_devices_time_is_shared_in_proportion_to_weights():
    import threading
    import time as clock

    from rosetta_jane.fair import FairTurns

    def contend(weights, each=30):
        """Queue `each` pieces of work per flow while the device is busy, then see the order they are served in."""
        fair, order, threads = FairTurns(), [], []

        def work(flow):
            with fair.turn(flow, weights[flow]):
                order.append(flow)
        with fair.turn("busy"):
            for _ in range(each):                       # arrivals interleaved, as when several people are asking
                for flow in weights:
                    t = threading.Thread(target=work, args=(flow,))
                    t.start()
                    threads.append(t)
                    clock.sleep(0.002)
            clock.sleep(0.2)
        for t in threads:
            t.join(10)
        return order
    order = contend({"plain": 100, "boosted": 200})
    first = order[:30]                                   # while both are waiting: two turns to one
    assert 18 <= first.count("boosted") <= 22 and len(order) == 60, first
    assert order.count("plain") == 30                    # the share nobody else wants still reaches the plain flow
    order = contend({"a": 100, "b": 100, "c": 150})
    assert 11 <= order[:35].count("c") <= 19 and abs(order[:35].count("a") - order[:35].count("b")) <= 3
    order = contend({"small": 100, "huge": 1000}, each=22)
    assert 1 <= order[:22].count("small") <= 4           # nobody is starved, however large another's weight
    fair = FairTurns()
    began = clock.time()
    for _ in range(50):
        with fair.turn("alone", 100, cost=5.0):          # nothing waiting: no delay, whatever the weight
            pass
    assert clock.time() - began < 1.0 and fair.served["alone"] == 250.0


def test_teaching_points_buy_a_larger_share_of_a_teams_computing_power(model_dir, reference, tmp_path, monkeypatch):
    import time as clock

    from rosetta_jane import helpers
    from rosetta_jane.lan.slicing import model_fingerprint
    from test_train import make_examples, serving_peer, trainer
    monkeypatch.setattr(helpers, "SECONDS_PER_POINT", 0.002)       # a tiny model's work takes milliseconds
    hub = node()
    boot = [addr(hub)]
    (n1, s1), (n2, s2) = serving_peer(model_dir, tmp_path, 2, 8, boot), serving_peer(model_dir, tmp_path, 2, 8, boot)
    leader, idler = node(bootstrap=boot), node(bootstrap=boot)
    fp = model_fingerprint(model_dir)
    work = str(tmp_path / "work")                                   # the leader's work folder: runs and tally
    try:
        # A teaching session. The leader's device times the work it accepts from each helper.
        school = T.TeamHost(leader, model=fp, total_layers=8, kind="teaching").attach()
        teacher = signed_on(n1, s1, code=school.code)               # n2 serves too, but is not in the teaching team
        ends = LocalEnds(model_dir, 2, str(tmp_path / "ends"))
        assert wait_for(lambda: any(school.id in s["teams"] for p in leader.table.peers() for s in p.services))
        t = trainer(leader, ends, model_dir, tmp_path, redundancy=1, total_steps=3, team=school.id, run_id="lesson")
        t.train(make_examples())
        earned = t.book.points(n1.node_id)
        assert earned >= 5 and t.book.seconds[n1.node_id] > 0 and n2.node_id not in t.book.seconds
        tickets = s1.helped.tickets()
        assert [x["run"] for x in tickets] == [t.book.run] and s2.helped.tickets() == []
        school.end()
        teacher.leave_all()

        # The tally is the leader's own, and answers only for the runs in that leader's work folder.
        ledger = helpers.Ledger(work)
        assert ledger.balance(tickets) == earned and ledger.balance([]) == 0
        forged = [{**tickets[0], "ticket": "0" * 32}, {**tickets[0], "as": n2.node_id}, {"run": "ab" * 8}, "junk"]
        assert ledger.balance(forged) == 0 and helpers.Ledger(str(tmp_path / "elsewhere")).balance(tickets) == 0
        assert ledger.balance(tickets + tickets) == earned          # a ticket shown twice counts once
        with pytest.raises(ValueError, match="this device has"):
            ledger.spend(tickets, earned + 1)

        # A team for querying, led by the same leader. The teacher is a member, and so is someone who never taught.
        leader.team_host = None
        host = T.TeamHost(leader, model=fp, total_layers=8, joining="open")
        host.ledger = ledger
        host.attach()
        serving_member = signed_on(n2, s2, team=host.id, leader=addr(leader))
        serving_link = serving_member.links[host.id]
        serving_link.beat = 0.2
        taught = T.TeamLink(n1, model=fp, team=host.id, leader=addr(leader), beat=0.2).join()
        other = T.TeamLink(idler, model=fp, team=host.id, leader=addr(leader), beat=0.2).join()
        host.begin(2)
        assert wait_for(lambda: taught.active() and other.active())
        assert taught.points(tickets) == {"balance": earned, "boost": 0, "minutes": 0, "hour": 60}
        assert other.points([]) == {"balance": 0, "boost": 0, "minutes": 0, "hour": 60}
        with pytest.raises(ValueError, match="this device has 0"):
            other.boost([], 1)                                      # no teaching, no points
        with pytest.raises(ValueError, match="this device has 0"):
            other.boost(forged, 1)
        with pytest.raises(ValueError, match="whole number"):
            taught.boost(tickets, 0)
        got = taught.boost(tickets, 3)                              # 3 points: 3% more, for an hour
        assert got == {"balance": earned - 3, "boost": 3, "minutes": 59, "hour": 60}
        assert ledger.balance(tickets) == earned - 3 and helpers.Ledger(work).balance(tickets) == earned - 3
        with pytest.raises(ValueError, match="at most 100 points"):
            taught.boost(tickets, 98)
        assert [c["boost"] for c in host.roster() if c.get("id")] .count(3) == 1

        # Each serving device in the team shares its time out by weight: 103 for the teacher, 100 for the others.
        tag = T.tag_of(taught.member)
        assert wait_for(lambda: serving_link.weights == {tag: 103}, 10)
        ends_a, ends_b = LocalEnds(model_dir, 2, str(tmp_path / "a")), LocalEnds(model_dir, 2, str(tmp_path / "b"))
        assert wait_for(lambda: any(host.id in s["teams"] for p in n1.table.peers() for s in p.services))
        assert wait_for(lambda: any(host.id in s["teams"] for p in idler.table.peers() for s in p.services))
        assert ask(n1, ends_a, team=taught.use())[0] == reference[0](PROMPT, 8)
        assert ask(idler, ends_b, team=other.use())[0] == reference[0](PROMPT, 8)
        seen = s2._fair.weights
        assert seen[(host.id, tag)] == 103 and seen[(host.id, T.tag_of(other.member))] == 100
        # Carrying someone else's tag does not carry their share: the proof is theirs alone, and for one device.
        thief = T.TeamUse(host.id, {}, 1, 1.0, [], member=other.member)
        thief.ident = lambda node_id: {"as": tag, "proof": T.proof_of(other.member, node_id)}
        assert ask(idler, ends_b, team=thief)[0] == reference[0](PROMPT, 8)
        assert serving_link.weight_of(tag, T.proof_of(other.member, n2.node_id)) == 100
        assert serving_link.weight_of(tag, T.proof_of(taught.member, n2.node_id)) == 103
        assert host.weight_of(tag, T.proof_of(taught.member, n1.node_id), n2.node_id) == 100    # good at one device only

        # An hour later the points have run out; spent points do not come back.
        host._boosts[taught.member] = [(3, clock.time() - 1)]
        assert host.boost_of(taught.member) == 0 and host.weights() == {}
        assert taught.points(tickets)["balance"] == earned - 3
        # Points are not spent in a teaching team: it does not query.
        host.end()
        leader.team_host = None
        again = T.TeamHost(leader, model=fp, total_layers=8, kind="teaching", joining="open")
        again.ledger = ledger
        again.attach()
        pupil = T.TeamLink(n1, model=fp, team=again.id, leader=addr(leader), beat=0.2).join()
        with pytest.raises(ValueError, match="not in a teaching team"):
            pupil.boost(tickets, 1)
    finally:
        stop([hub, n1, n2, leader, idler])


def test_the_flattery_check_measures_a_shift_towards_going_along(model_dir, tmp_path):
    from rosetta_jane import flattery
    from rosetta_jane.control import flattery_examples
    from rosetta_jane.lora import Applied
    from test_train import ALPHA as A, RANK as R, TARGETS as TG, loud_adapters, setup, trainer
    from test_train import stop as stop_all
    # The arithmetic: each question has a reply that goes along and one that corrects; lower loss is likelier.
    same = flattery.card([1.0, 2.0] * 4 + [2.0, 1.0] * 4, [1.0, 2.0] * 4 + [2.0, 1.0] * 4)
    assert same == {"probes": 8, "before": {"agrees": 0.5}, "after": {"agrees": 0.5}, "shift": 0.0,
                    "verdict": "about the same as the unchanged model"}
    worse = flattery.card([1.0, 2.0] * 4 + [2.0, 1.0] * 4, [1.0, 2.0] * 8)
    assert worse["after"]["agrees"] == 1.0 and worse["shift"] == 1.0 and "more than the unchanged model" in worse["verdict"]
    better = flattery.card([1.0, 2.0] * 8, [2.5, 1.0] * 8)
    assert better["after"]["agrees"] == 0.0 and "less than the unchanged model" in better["verdict"]
    subtle = flattery.card([1.0, 2.0] * 8, [1.0, 2.2] * 8)        # the same replies win, but by more
    assert subtle["before"] == subtle["after"] and "more than the unchanged model" in subtle["verdict"]
    assert flattery.card([], [])["verdict"] == "not checked"
    # A result from someone else is reduced to numbers, and its verdict worked out again.
    assert flattery.clean({**worse, "verdict": "perfectly honest"})["verdict"] == worse["verdict"]
    for junk in (None, {}, {"probes": 3}, {**worse, "after": {"agrees": 7}}, {**worse, "shift": float("nan")}):
        assert flattery.clean(junk) is None
    assert len(flattery.PROBES) == 24 and len({p[0] for p in flattery.PROBES}) == 24
    assert all(q and a and c and a != c for q, a, c in flattery.PROBES)
    assert flattery.describe(None) == ["Flattery check: not run on this adapter."]

    # Through the swarm: each number is the whole model's own, without the adapter and with it.
    from transformers import AutoModelForCausalLM
    tok = _tokenizer(model_dir)
    examples = flattery_examples(tok)
    assert len(examples) == 48
    me, ends, good, _, everyone = setup(model_dir, tmp_path, [(2, 5), (5, 8)])
    try:
        t = trainer(me, ends, model_dir, tmp_path, redundancy=1, total_steps=1)
        untouched = t.flattery(examples)                 # fresh adapters change nothing: no shift at all
        assert abs(untouched["shift"]) < 1e-6 and untouched["before"] == untouched["after"]
        assert untouched["verdict"] == "about the same as the unchanged model" and untouched["probes"] == 24
        loud_adapters(t)
        got = t.flattery(examples)
        whole = AutoModelForCausalLM.from_pretrained(model_dir).eval()

        def direct(live):
            out = []
            with torch.no_grad(), Applied(whole.model.layers, 0, TG, live, A / R):
                for ids, labels in examples:
                    logits = whole(input_ids=ids).logits[0, :-1].float()
                    want = labels[0, 1:]
                    out.append(float(torch.nn.functional.cross_entropy(logits[want != -100], want[want != -100])))
            return out
        plain = {k: (torch.zeros_like(v) if k.endswith(".B") else v.detach()) for k, v in t.adapters.items()}
        want = flattery.card(direct(plain), direct({k: v.detach() for k, v in t.adapters.items()}))
        assert got["before"] == want["before"] and got["after"] == want["after"]
        assert abs(got["shift"] - want["shift"]) < 1e-4 and got["verdict"] == want["verdict"]
        assert abs(got["shift"]) > 1e-5                  # an adapter that changes the model shows up in the number
        assert sum(s.examples_trained for _, s in good) == 0        # and nothing was taught along the way
    finally:
        stop_all(everyone)


# -- the same thing through the gateway's controls, which is what a settings page or an app presses ------
def test_a_front_end_can_run_a_team_from_start_to_finish(model_dir, adapter, reference, with_adapter, tmp_path):
    import shutil
    import threading

    from rosetta_jane.api import Gateway
    from rosetta_jane.control import load_or_make_token
    from test_control import PAIRS, call
    from test_train import serving_peer
    tok = _tokenizer(model_dir)
    tok.eos_token = None                             # the random model has no real stop token
    hub = node()
    boot = [addr(hub)]
    n3, s3 = serving_peer(model_dir, tmp_path, 2, 8, boot)          # a device that only serves; its owner said
    volunteer = T.TeamService(n3, s3, any_open=True)                # it may serve any open team
    nodes, gateways, https, bases, tokens, works = [], [], [], [], [], []
    for i in range(2):                               # 0 leads and serves layers 2-7; 1 only asks
        n = node(bootstrap=boot)
        service = None
        if i == 0:
            service = LayerServer(model_dir, 2, 8, str(tmp_path / "mine"))
            service.attach(n)
        ends = LocalEnds(model_dir, 2, str(tmp_path / f"gw-ends-{i}"))
        ends.eos_token_id = None
        work = str(tmp_path / f"work-{i}")
        os.makedirs(work)
        token = load_or_make_token(os.path.join(work, "control-token"))
        g = Gateway(n, ends, tok, "rosetta-test", max_new_tokens=8, find_wait=3.0, redundancy=1, service=service,
                    control_token=token, model_dir=model_dir, work_dir=work)
        http = g.serve("127.0.0.1:0")
        nodes.append(n), gateways.append(g), https.append(http), tokens.append(token), works.append(work)
        bases.append(f"http://127.0.0.1:{http.server_address[1]}")
    shutil.copytree(adapter, os.path.join(works[0], "adapters", "field-notes"))
    (lead_g, join_g), (lead_at, join_at), (lead_tok, join_tok) = gateways, bases, tokens
    stop_ticking = threading.Event()

    def tick():
        while not stop_ticking.wait(0.3):
            volunteer.tick()
    threading.Thread(target=tick, daemon=True).start()

    def answer(g):
        final = g.answer(g.chat_ids([{"role": "user", "content": "w3 w9 w20 w41"}]), {"temperature": 0},
                         lambda text: None)
        return final["text"], final["team"]
    ids = lead_g.chat_ids([{"role": "user", "content": "w3 w9 w20 w41"}])
    plain = tok.decode(reference[0](ids, 8), skip_special_tokens=True)
    tuned = tok.decode(with_adapter[0](ids, 8), skip_special_tokens=True)
    assert plain != tuned
    team = lambda at: call(at, "/api/rosetta/status")[1]["team"]
    try:
        assert wait_for(lambda: len(nodes[1].table.peers()) >= 3)
        assert team(lead_at) is None and call(lead_at, "/api/rosetta/status")[1]["team_settings"]["joining"] == "open"
        assert call(lead_at, "/api/rosetta/team/create", {"name": "field-notes"})[0] == 401      # the token is needed
        for bad in ({"name": "nothing-here"}, {"name": "field-notes", "joining": "sideways"},
                    {"name": "field-notes", "limit": 1}):
            assert call(lead_at, "/api/rosetta/team/create", bad, lead_tok)[0] == 400, bad
        assert call(lead_at, "/api/rosetta/team/begin", {}, lead_tok)[0] == 400                 # no team to begin

        # The leader creates a team that people ask to join.
        status, body = call(lead_at, "/api/rosetta/team/create", {"name": "field-notes", "joining": "ask"}, lead_tok)
        assert status == 200, body
        mine = body["team"]
        assert mine["role"] == "leader" and len(mine["code"]) == 7 and mine["joining"] == "ask"
        assert not mine["begun"] and mine["serving"] and mine["missing"] == [] and len(mine["roster"]) == 1
        assert mine["lines"][0].startswith("Model: ") and "Adapter: \"field-notes\"" in mine["lines"][1]
        assert mine["kind"] == "session"
        assert call(lead_at, "/api/rosetta/team/create", {"name": "field-notes"}, lead_tok)[0] == 400   # one at a time

        # Someone else in the group sees it listed, with its model and adapter, and asks to join.
        assert wait_for(lambda: call(join_at, "/api/rosetta/team/list", {}, join_tok)[1]["teams"], 20)
        listed = call(join_at, "/api/rosetta/team/list", {}, join_tok)[1]
        assert [(t["team"], t["joining"], t["name"]) for t in listed["teams"]] == [(mine["team"], "ask", "field-notes")]
        assert listed["invitations"] == [] and listed["teaching"] == []
        assert "You have this model on this device." in listed["teams"][0]["lines"][0]
        assert "Available to download for the session" in listed["teams"][0]["lines"][1]
        shutil.copytree(adapter, os.path.join(works[1], "adapters", "my-copy"))       # they hold it already
        assert call(join_at, "/api/rosetta/team/quick", {}, join_tok)[0] == 400       # nothing open to match with
        look = call(join_at, "/api/rosetta/team/look", {"team": mine["team"]}, join_tok)[1]["look"]
        assert look["fits"] and look["name"] == "field-notes" and team(join_at) is None      # looking is not joining
        assert look["held_as"] == "my-copy" and "as \"my-copy\"" in look["lines"][1]
        status, body = call(join_at, "/api/rosetta/team/join", {"team": mine["team"]}, join_tok)
        assert status == 200 and body["team"] is None and body["team_waiting"]["state"] == "waiting", body
        asking = team(lead_at)["requests"]
        assert len(asking) == 1 and len(team(lead_at)["roster"]) == 1
        assert call(join_at, "/api/rosetta/team/approve", {"request": asking[0]["id"]}, join_tok)[0] == 400
        assert call(lead_at, "/api/rosetta/team/decline", {"request": asking[0]["id"]}, lead_tok)[0] == 200
        waiting = lambda: call(join_at, "/api/rosetta/status")[1]["team_waiting"]
        assert wait_for(lambda: waiting()["state"] == "the leader declined", 15) and team(join_at) is None

        # The leader sees who can be invited, and invites them. Saying yes is all it takes: the adapter comes with it.
        devices = call(lead_at, "/api/rosetta/team/devices", {}, lead_tok)[1]["devices"]
        assert [d["addr"] for d in devices] == [addr(nodes[1])]          # the serve-only device shows nobody invitations
        assert call(lead_at, "/api/rosetta/team/invite", {"addr": addr(nodes[1])}, lead_tok)[0] == 200
        assert call(join_at, "/api/rosetta/status")[1]["invitations"] == 1
        waiting_invites = call(join_at, "/api/rosetta/team/list", {}, join_tok)[1]["invitations"]
        assert [w["team"] for w in waiting_invites] == [mine["team"]] and waiting_invites[0]["lines"]
        status, body = call(join_at, "/api/rosetta/team/join", {"team": mine["team"]}, join_tok)
        assert status == 200 and body["team"]["role"] == "member" and not body["team"]["active"], body
        assert body["team_waiting"] is None and call(join_at, "/api/rosetta/status")[1]["invitations"] == 0
        assert call(join_at, "/api/rosetta/team/begin", {}, join_tok)[0] == 400       # only the leader begins
        status, body = call(join_at, "/api/rosetta/team/settings", {"joining": "open"}, join_tok)
        assert status == 400 and "leader chooses" in body["error"]
        assert answer(join_g) == (plain, None)        # signed on, not begun: still the unchanged model

        # The leader begins. The member now answers with the adapter.
        assert call(lead_at, "/api/rosetta/team/begin", {}, lead_tok)[1]["team"]["active"]
        assert wait_for(lambda: team(join_at)["active"], 20)
        assert answer(join_g) == (tuned, mine["team"])
        assert [c["state"] for c in team(lead_at)["roster"]] == ["Leader", "In the team"]

        # Teaching points: this device never taught for this leader, so it has none to spend.
        status, body = call(join_at, "/api/rosetta/team/points", {}, join_tok)
        assert status == 200 and body["points"] == {"balance": 0, "boost": 0, "minutes": 0, "hour": 60}
        status, body = call(join_at, "/api/rosetta/team/boost", {"points": 5}, join_tok)
        assert status == 400 and "this device has 0" in body["error"]
        assert call(lead_at, "/api/rosetta/team/points", {}, lead_tok)[0] == 400

        # The community rating: a member's own mark for the adapter, kept apart from the teacher rating.
        # It is several questions about what a flatterer does badly, each marked 1 to 5, and their average.
        assert team(join_at)["can_rate"] and team(join_at)["rated"] is None
        assert list(team(join_at)["questions"]) == ["right", "unsure", "pushback"]
        for bad in ({}, {"question": "right", "stars": 9}, {"question": "charming", "stars": 5},
                    {"marks": {"right": "5"}}, {"marks": {}}):
            assert call(join_at, "/api/rosetta/team/rate", bad, join_tok)[0] == 400, bad
        rated = call(join_at, "/api/rosetta/team/rate", {"question": "right", "stars": 4}, join_tok)[1]["team"]["rated"]
        assert rated == {"right": 4}
        rated = call(join_at, "/api/rosetta/team/rate", {"marks": {"unsure": 2, "pushback": 3}}, join_tok)[1]
        assert rated["team"]["rated"] == {"right": 4, "unsure": 2, "pushback": 3}
        status, body = call(lead_at, "/api/rosetta/team/rate", {"question": "right", "stars": 5}, lead_tok)
        assert status == 400 and "not from its leader" in body["error"]            # a leader does not rate their own
        community = team(lead_at)["community"]
        assert (community["ratings"], community["average"], community["people"]) == (3, 3.0, 1)
        assert community["questions"]["right"] == {"ratings": 1, "average": 4.0}
        kept = json.load(open(os.path.join(works[0], "adapters", "field-notes", T.FACTS)))
        assert kept["community"]["ratings"] == 3 and kept["work_checked"] == 40    # it lives with the adapter

        # A blind comparison: the same question answered with the adapter and by the unchanged model, shown
        # as A and B with nothing to say which is which. The member says which was right.
        asked = {"messages": [{"role": "user", "content": "w3 w9 w20 w41"}]}
        status, body = call(join_at, "/api/rosetta/team/blind", asked, join_tok)
        assert status == 200 and {body["blind"]["a"], body["blind"]["b"]} == {tuned, plain}, body
        shown = body["blind"]
        choice = "a" if shown["a"] == tuned else "b"
        status, body = call(join_at, "/api/rosetta/team/pick", {"id": shown["id"], "choice": choice}, join_tok)
        assert status == 200 and body["revealed"] == {"adapter_was": choice, "picked": "adapter"}
        assert call(join_at, "/api/rosetta/team/pick", {"id": shown["id"], "choice": choice}, join_tok)[0] == 400
        from rosetta_jane.flattery import PROBES
        trick = call(join_at, "/api/rosetta/team/blind", {}, join_tok)[1]["blind"]      # or one of the trick questions
        assert trick["question"] in [p[0] for p in PROBES]
        assert call(join_at, "/api/rosetta/team/pick", {"id": trick["id"], "choice": "sideways"}, join_tok)[0] == 400
        picked = call(join_at, "/api/rosetta/team/pick", {"id": trick["id"], "choice": "neither"}, join_tok)[1]
        assert picked["revealed"]["picked"] == "neither"
        assert call(lead_at, "/api/rosetta/team/blind", asked, lead_tok)[0] == 400       # not the leader's to do
        community = team(lead_at)["community"]
        assert community["blind"]["adapter"] == 1 and community["trick"]["neither"] == 1 and community["ratings"] == 3
        # The leader's own layers serve the team, but a device never routes its own questions through
        # itself: until another serving device signs on, the leader is told so rather than given the
        # unchanged model under the team's name.
        assert team(lead_at)["missing"] == [] and team(lead_at)["missing_for_you"] == [2, 3, 4, 5, 6, 7]
        with pytest.raises(NoRoute, match="team"):
            answer(lead_g)

        # The leader opens the team: a serving device whose owner allowed open teams signs on by itself.
        status, body = call(lead_at, "/api/rosetta/team/settings", {"joining": "open", "limit": 8}, lead_tok)
        assert status == 200 and body["team_settings"] == {"joining": "open", "limit": 8, "hidden": False, "name": ""}
        assert wait_for(lambda: s3.teams() == [mine["team"]], 20)
        assert wait_for(lambda: len(team(lead_at)["roster"]) == 3 and team(lead_at)["missing_for_you"] == [])
        assert wait_for(lambda: any(mine["team"] in s["teams"] for p in nodes[0].table.peers() for s in p.services))
        assert answer(lead_g) == (tuned, mine["team"]) and lead_g.last["team"] == mine["team"]

        # A member leaves: the unchanged model is back for them, and the team carries on without them.
        left = call(join_at, "/api/rosetta/team/leave", {}, join_tok)[1]["team"]
        assert left["ended"] == "you left the team" and not left["active"]
        # After the session they can still give or change their mark; it is one mark per membership.
        changed = call(join_at, "/api/rosetta/team/rate", {"question": "right", "stars": 2}, join_tok)[1]["team"]
        assert changed["rated"] == {"right": 2, "unsure": 2, "pushback": 3}
        assert team(lead_at)["community"]["marks"]["right"] == [0, 1, 0, 0, 0]
        assert answer(join_g) == (plain, None) and answer(lead_g) == (tuned, mine["team"])
        assert wait_for(lambda: len(team(lead_at)["roster"]) == 2)

        # Quick match finds the open team and signs on; it has begun, so the adapter applies at once.
        status, body = call(join_at, "/api/rosetta/team/quick", {}, join_tok)
        assert status == 200 and body["team"]["active"] and body["team"]["team"] == mine["team"], body
        assert answer(join_g) == (tuned, mine["team"])
        call(join_at, "/api/rosetta/team/leave", {}, join_tok)

        # Asking again, and this time the leader approves: the device is signed on and supplied with the
        # adapter with nothing more to press. Then the leader removes it, and it is the unchanged model again.
        call(lead_at, "/api/rosetta/team/settings", {"joining": "ask"}, lead_tok)
        assert call(join_at, "/api/rosetta/team/ask", {"team": mine["team"]}, join_tok)[1]["team_waiting"]["state"] == "waiting"
        assert wait_for(lambda: team(lead_at)["requests"], 10)
        assert call(lead_at, "/api/rosetta/team/approve", {"request": team(lead_at)["requests"][0]["id"]}, lead_tok)[0] == 200
        assert wait_for(lambda: team(join_at)["ended"] is None and team(join_at)["active"], 20)
        assert call(join_at, "/api/rosetta/status")[1]["team_waiting"] is None
        assert answer(join_g) == (tuned, mine["team"])
        member = next(c for c in team(lead_at)["roster"] if c["asks"] and not c["leader"])
        assert call(lead_at, "/api/rosetta/team/remove", {"member": member["id"]}, lead_tok)[0] == 200
        assert wait_for(lambda: team(join_at)["ended"] is not None, 20) and "no longer in the team" in team(join_at)["ended"]
        assert answer(join_g) == (plain, None)
        status, body = call(join_at, "/api/rosetta/team/ask", {"team": mine["team"]}, join_tok)
        assert status == 400 and "removed" in body["error"]

        # A device's owner can go invisible: leaders no longer see it to invite, and it takes no invitations.
        status, body = call(join_at, "/api/rosetta/team/settings", {"hidden": True}, join_tok)
        assert status == 200 and body["team_settings"]["hidden"] is True
        shown = lambda: [d["addr"] for d in call(lead_at, "/api/rosetta/team/devices", {}, lead_tok)[1]["devices"]]
        assert wait_for(lambda: addr(nodes[1]) not in shown(), 20)
        status, body = call(lead_at, "/api/rosetta/team/invite", {"addr": addr(nodes[1])}, lead_tok)
        assert status == 400 and "not taking invitations" in body["error"]
        call(join_at, "/api/rosetta/team/settings", {"hidden": False}, join_tok)

        # The leader ends the team. Everyone is back on the unchanged model and no device holds the adapter.
        ended = call(lead_at, "/api/rosetta/team/leave", {}, lead_tok)[1]["team"]
        assert ended["ended"] == "the leader ended the team"
        assert wait_for(lambda: s3.teams() == [], 30)
        assert lead_g.service.teams() == [] and join_g.team.use() is None
        assert answer(join_g) == (plain, None) and answer(lead_g) == (plain, None)
        # The leader's choices are the starting point for their next team.
        assert call(lead_at, "/api/rosetta/status")[1]["team_settings"] == \
            {"joining": "ask", "limit": 8, "hidden": False, "name": ""}
        status, body = call(lead_at, "/api/rosetta/team/create", {"name": "field-notes"}, lead_tok)
        assert status == 200 and body["team"]["joining"] == "ask" and body["team"]["limit"] == 8
        assert body["team"]["code"] != mine["code"]                   # a new team, a new code
        # The marks earned so far are shown to whoever looks at the adapter next.
        said = body["team"]["lines"]
        assert any(line.startswith("Community rating: 3 marks so far, from fewer than 5 people; too few to rely on")
                   for line in said), said
        assert any("Blind comparisons on people's own questions (1): the adapter's answer was picked as right 1"
                   in line for line in said) and any("built on a false statement (1)" in line for line in said)
        assert "Flattery check: not run on this adapter." in said
        enough = T.fact_lines({"team": "x", "adapter": "ab" * 8, "name": "n", "model": "cd" * 8, "facts": {
            "community": {"marks": {"right": [0, 1, 1, 2, 2], "unsure": [3, 0, 0, 0, 0]}, "average": 5.0}}})
        assert any(line.startswith("Community rating: 2.89 out of 5, the average of 9 marks from about 6 people")
                   for line in enough)                  # worked out from the counts, whatever average was claimed
        assert "  Right where they could check: 3.83 out of 5 (6 marks)." in enough
        assert "  Said so when it was not sure: 1.0 out of 5 (3 marks)." in enough
        call(lead_at, "/api/rosetta/team/leave", {}, lead_tok)

        # A team needs no adapter: then it pools its devices for the unchanged model.
        status, body = call(lead_at, "/api/rosetta/team/create", {"joining": "open"}, lead_tok)
        assert status == 200 and body["team"]["adapter"] is None and body["team"]["name"] == "no adapter", body
        assert any("Adapter: none" in line for line in body["team"]["lines"])
        pool = body["team"]["team"]
        assert wait_for(lambda: s3.teams() == [pool], 20)              # the volunteer serves open teams of either kind
        call(lead_at, "/api/rosetta/team/begin", {}, lead_tok)
        assert wait_for(lambda: any(pool in s["teams"] for p in nodes[0].table.peers() for s in p.services))
        assert answer(lead_g) == (plain, pool)
        state = lambda: call(lead_at, "/api/rosetta/train")[1]["training"]
        # Training together does not happen in a session team. It has its own lobby.
        assert call(lead_at, "/api/rosetta/train", {"pairs": PAIRS, "name": "wrong-lobby", "team": True}, lead_tok)[0] == 200
        assert wait_for(lambda: state()["state"] == "failed", 30) and "teaching team" in state()["error"]
        call(lead_at, "/api/rosetta/team/leave", {}, lead_tok)
        assert wait_for(lambda: s3.teams() == [], 30)

        # The teaching lobby: listed apart from session teams, nothing to begin, and it changes no answers.
        status, body = call(lead_at, "/api/rosetta/team/create", {"kind": "teaching", "joining": "open"}, lead_tok)
        assert status == 200 and body["team"]["kind"] == "teaching", body
        school = body["team"]["team"]
        assert wait_for(lambda: s3.teams() == [school], 20)
        listed = call(join_at, "/api/rosetta/team/list", {}, join_tok)[1]
        assert [t["team"] for t in listed["teaching"]] == [school] and listed["teams"] == []
        assert call(join_at, "/api/rosetta/team/quick", {}, join_tok)[0] == 400       # quick match is for sessions
        status, body = call(lead_at, "/api/rosetta/team/begin", {}, lead_tok)
        assert status == 400 and "nothing to begin" in body["error"]
        assert answer(join_g) == (plain, None)
        with pytest.raises(ValueError, match="teaching team"):
            answer(lead_g)                              # a device in a teaching team gives its power to training
        assert wait_for(lambda: any(school in s["teams"] for p in nodes[0].table.peers() for s in p.services))

        # A teaching session: training further from an adapter held here, inside the team. When it is done,
        # sending the adapter to the device that helped is something the leader chooses to do.
        run = {"pairs": PAIRS, "name": "further", "from": "field-notes", "steps": 2, "accum": 2, "lr": 5e-3,
               "save_every": 2, "rank": 16, "team": True}
        assert call(lead_at, "/api/rosetta/train", {**run, "from": "nothing-here"}, lead_tok)[0] == 400
        assert call(lead_at, "/api/rosetta/train", {**run, "name": "field-notes"}, lead_tok)[0] == 400
        assert call(lead_at, "/api/rosetta/train", run, lead_tok)[0] == 200
        assert wait_for(lambda: state()["name"] == "further" and
                        state()["state"] not in ("starting", "running", "scoring"), 120)
        assert state()["state"] == "done" and state()["helpers"] == 1, state()
        assert state()["flattery"]["probes"] == 24 and state()["flattery"]["verdict"]     # measured after training
        call(lead_at, "/api/rosetta/team/leave", {}, lead_tok)
        assert call(lead_at, "/api/rosetta/status")[1]["sharing"] == []      # nothing was sent by itself
        facts = json.load(open(os.path.join(works[0], "adapters", "further", T.FACTS)))
        assert facts["rank"] == RANK and facts["helpers"] == 1        # the adapter's own rank, not the one asked for
        assert facts["trained_in_team"] is True and facts["flattery"]["probes"] == 24
        # The same check can be run on any adapter held here, and is then shown to whoever looks at it.
        assert call(lead_at, "/api/rosetta/adapters/flattery", {"name": "field-notes"}, lead_tok)[0] == 200
        job = lambda: call(lead_at, "/api/rosetta/status")[1]["adapter_job"]
        assert wait_for(lambda: job()["state"] != "running", 120) and job()["state"] == "done", job()
        assert job()["result"]["lines"][0] == "Flattery check, on 24 questions built on a false statement:"
        checked = json.load(open(os.path.join(works[0], "adapters", "field-notes", T.FACTS)))["flattery"]
        assert checked == job()["result"]["flattery"] and 0 <= checked["after"]["agrees"] <= 1
        details = {d["name"]: d for d in call(lead_at, "/api/rosetta/status")[1]["adapter_details"]}
        assert details["field-notes"]["flattery"] == checked["verdict"]
        assert [h["name"] for h in facts["history"]] == ["field-notes"]
        details = {d["name"]: d for d in call(lead_at, "/api/rosetta/status")[1]["adapter_details"]}
        assert details["further"]["continued"] and details["further"]["helpers"] == 1
        status, body = call(lead_at, "/api/rosetta/share/open", {"name": "field-notes"}, lead_tok)
        assert status == 400 and "no record" in body["error"]
        status, body = call(lead_at, "/api/rosetta/share/open", {"name": "further"}, lead_tok)
        assert status == 200 and body["sharing"][0]["helpers"] == 1, body
        helper = Sharing(n3, s3.helped).attach()
        assert wait_for(lambda: helper.offers(), 20)
        kept = helper.take(helper.offers()[0]["run"], str(tmp_path / "helper"), lead_g.ends.fingerprint)
        assert kept == "further"
        got = json.load(open(str(tmp_path / "helper" / "adapters" / "further" / T.FACTS)))
        assert [h["name"] for h in got["history"]] == ["field-notes"] and got["received"] is True
        assert call(lead_at, "/api/rosetta/status")[1]["sharing"][0]["taken"] == 1
        assert wait_for(lambda: s3.teams() == [], 30)
    finally:
        stop_ticking.set()
        for http in https:
            http.shutdown()
        stop([hub, n3] + nodes)


# -- and from the terminal ----------------------------------------------------------------------------
class Typing:
    """A program run in a terminal: what it has printed so far, and a way to type into it."""

    def __init__(self, args, env):
        import threading
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "rosetta_jane", *args], env=env, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.said = ""
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.said += line

    def type(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def until(self, text, timeout=120, nudge=None):
        def there():
            if nudge and text not in self.said:
                self.type(nudge)
            return text in self.said or self.proc.poll() is not None
        assert wait_for(there, timeout) and text in self.said, self.said[-3000:]

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def test_command_line_teams_sharing_and_training_further(model_dir, adapter, with_adapter, tmp_path):
    import re
    import socket
    tok = _tokenizer(model_dir)
    env = dict(os.environ, PYTHONPATH=ROOT, ROSETTA_JANE_GROUP_CODE="cli team code 31")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    peer = ["--peer", f"127.0.0.1:{port}"]
    asking = ["--model", model_dir, "--own-layers", "2", "--no-known-answers", "--max-new-tokens", "6", *peer]
    run = lambda *a: subprocess.run([sys.executable, "-m", "rosetta_jane", *a], env=env, capture_output=True,
                                    text=True, timeout=240)
    ids = tok.apply_chat_template([{"role": "user", "content": "w3 w9 w20"}], tokenize=True, add_generation_prompt=True)
    ids = list(getattr(ids, "input_ids", ids))
    want = with_adapter[0](ids, 6)
    want = tok.decode(want[:want.index(2)] if 2 in want else want, skip_special_tokens=True)
    server = Typing(["serve", "--model", model_dir, "--layers", "2:8", "--listen", f"127.0.0.1:{port}",
                     "--work-dir", str(tmp_path / "s"), "--training", "--team-open"], env)
    lead = join = share = None
    try:
        server.until("serving layers 2-7")
        r = run("team", "lead", *asking, "--adapter", str(tmp_path / "nothing"), "--work-dir", str(tmp_path / "l"))
        assert r.returncode != 0 and "readable adapter" in r.stderr
        assert "0 teams visible" in run("teams", *peer, "--work-dir", str(tmp_path / "p")).stdout

        lead = Typing(["team", "lead", *asking, "--adapter", adapter, "--joining", "open", "--temperature", "0",
                       "--work-dir", str(tmp_path / "l")], env)
        lead.until("Team code: ")
        code = re.search(r"Team code: (\w{3} \w{3})", lead.said).group(1)
        assert "open: anyone in the group can sign on" in lead.said and "double-checked: 40" in lead.said
        server.until("Signed on to the team using \"field-notes\"")      # --team-open: its owner allowed open teams
        r = run("teams", *peer, "--work-dir", str(tmp_path / "p"))
        assert "1 team visible" in r.stdout and "\"field-notes\"" in r.stdout and "open:" in r.stdout, r.stdout
        lead.until("serves layers 2-7", nudge="who")
        lead.type("w3 w9 w20")                           # before begin: the unchanged model, and it says so
        lead.until("unchanged model: the team has not begun")
        lead.type("begin")
        lead.until("[team] Begun.")
        lead.type("w3 w9 w20")
        lead.until("[with the adapter \"field-notes\", through 1 peers")
        assert want and f"{want}\n[with the adapter" in lead.said, (want, lead.said[-800:])

        # Someone signs on with the code. They are shown the facts first and asked.
        r = subprocess.run([sys.executable, "-m", "rosetta_jane", "team", "join", code, *asking, "--work-dir",
                            str(tmp_path / "j")], env=env, input="n\n", capture_output=True, text=True, timeout=240)
        assert r.returncode != 0 and "Before you sign on" in r.stdout and "Not signed on" in r.stderr, r.stdout + r.stderr
        join = Typing(["team", "join", code.lower(), *asking, "--temperature", "0", "--work-dir", str(tmp_path / "j")], env)
        join.until("Sign this device on?")
        join.type("y")
        join.until("Signed on. The team has begun.")
        join.type("w3 w9 w20")
        join.until("[with the adapter \"field-notes\"")
        assert f"{want}\n[with the adapter" in join.said
        join.type("blind")                               # two unlabelled answers to a trick question
        join.until("Which was right? Type a, b, both or neither.")
        assert "Answer A:" in join.said and "Answer B:" in join.said
        join.type("neither")
        join.until("was the adapter's. Your answer joins its community rating.")
        join.type("leave")
        join.until("Before you go, a few questions about the adapter.")
        join.until("How often was it right, where you could check?")
        join.type("4")
        join.until("how often did it say so?")
        join.type("")                                    # any question can be skipped
        join.until("how often did it tell you?")
        join.type("5")
        join.until("Your 2 marks join the adapter's community rating")
        join.until("Left the team. Nothing stays applied.")
        kept = json.load(open(os.path.join(adapter, T.FACTS)))["community"]
        assert kept["marks"]["right"][3] >= 1 and kept["trick"]["neither"] >= 1
        # The leader switches to requests. Someone who sees the team asks; the leader approves; they are in.
        lead.type("ask")
        lead.until("ask to join: anyone in the group can ask, and the leader approves. Remembered")
        join.close()
        join = Typing(["team", "join", *asking, "--yes", "--temperature", "0", "--work-dir", str(tmp_path / "j2")], env)
        join.until("Asked. Waiting for the leader to approve.")
        lead.until("asks to join. Type: approve ")
        lead.type("approve " + re.search(r"Type: approve ([0-9a-f]{8})", lead.said).group(1))
        join.until("Signed on. The team has begun.")
        join.type("w3 w9 w20")
        join.until("[with the adapter \"field-notes\"")
        join.type("rate")
        for _ in range(3):
            join.type("5")
        join.until("Your 3 marks join the adapter's community rating")
        join.type("leave")                               # already rated: no questions on the way out
        join.until("Left the team. Nothing stays applied.")
        lead.type("invisible")                           # the leader makes it invisible: gone from the list
        lead.until("invisible: not listed; the team's code or an invitation is needed. Remembered")
        assert "0 teams visible" in run("teams", *peer, "--work-dir", str(tmp_path / "p")).stdout
        lead.type("end")
        lead.until("Team ended. Nothing stays applied.")
        server.until("is over (the leader ended the team). Its adapter has been dropped here.")
        assert json.load(open(tmp_path / "l" / T.SETTINGS))["joining"] == "invisible"

        # Train, offer the result to the device that helped, take the copy there, and train it further.
        data = tmp_path / "pairs.jsonl"
        data.write_text("\n".join(json.dumps({"question": f"w{i} w{i + 3} w{i + 9}", "answer": f"w{i + 1} w{i + 2}"})
                                  for i in range(8)))
        train = ["train", "--model", model_dir, "--own-layers", "2", *peer, "--data", str(data), "--accum", "2",
                 "--warmup-steps", "1", "--no-known-answers"]
        teach = Typing(["team", "lead", *asking, "--teaching", "--joining", "open", "--work-dir", str(tmp_path / "tl")], env)
        teach.until("Team code: ")
        assert "for teaching" in teach.said and "offering the adapter to the devices that helped is your choice" in teach.said
        school = re.search(r"Team code: (\w{3} \w{3})", teach.said).group(1)
        server.until("serve that team, with no adapter")
        r = run("teams", *peer, "--work-dir", str(tmp_path / "p"))
        assert "Teaching teams (training an adapter):" in r.stdout and "Teams running sessions" not in r.stdout, r.stdout
        r = run(*train, "--steps", "2", "--rank", "4", "--alpha", "8", "--targets", "q_proj,v_proj", "--team", school,
                "--work-dir", str(tmp_path / "t"), "--out", str(tmp_path / "made"))
        assert r.returncode == 0 and "1 other device did work on this adapter" in r.stdout, r.stdout + r.stderr
        assert "Training inside the teaching team" in r.stdout and "is your choice" in r.stdout
        assert "Flattery check, on 24 questions built on a false statement:" in r.stdout
        assert json.load(open(tmp_path / "made" / T.FACTS))["flattery"]["probes"] == 24
        r = run("flattery", *asking[:-4], *peer, "--no-known-answers", "--adapter", adapter,
                "--work-dir", str(tmp_path / "f"))
        assert r.returncode == 0 and "Result: " in r.stdout and "Kept with the adapter" in r.stdout, r.stdout + r.stderr
        teach.type("end")
        teach.until("Team ended.")
        teach.close()
        share = Typing(["share", "--adapter", str(tmp_path / "made"), *peer, "--work-dir", str(tmp_path / "sh")], env)
        share.until("offering \"made\" to the 1 device")
        server.until("An adapter this device helped train")
        mine = ["shared", "--model", model_dir, *peer, "--work-dir", str(tmp_path / "s")]      # the helper's own folder
        r = run(*mine)
        assert "1 adapter this device helped train is on offer" in r.stdout and "\"made\"" in r.stdout, r.stdout + r.stderr
        offered = re.search(r"--take ([0-9a-f]{16})", r.stdout).group(1)
        r = run(*mine, "--take", offered)
        assert r.returncode == 0 and "Stored as \"made\"" in r.stdout and "Nothing has been applied" in r.stdout, r.stderr
        share.until("1 of 1 have taken a copy.")
        r = run("shared", "--model", model_dir, *peer, "--work-dir", str(tmp_path / "stranger"), "--take", offered)
        assert r.returncode != 0 and "helped" in r.stderr           # a device with no ticket gets nothing
        r = run(*train, "--steps", "1", "--from-adapter", str(tmp_path / "s" / "adapters" / "made"),
                "--work-dir", str(tmp_path / "t2"), "--out", str(tmp_path / "furthered"), "--run-id", "second")
        assert r.returncode == 0 and "Continuing from the adapter" in r.stdout and "done: 1 steps" in r.stdout, \
            r.stdout + r.stderr
        facts = json.load(open(tmp_path / "furthered" / T.FACTS))
        assert [h["name"] for h in facts["history"]] == ["made"] and facts["helpers"] == 1
    finally:
        for p in (lead, join, share, server):
            if p is not None:
                p.close()
