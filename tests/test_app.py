"""The phone app's own Python (what its buttons call), run here against desktop peers.

The app's screens were tapped through on a cloud phone for 0.1. This checks the part added
since, teams and sharing with helpers, at the level of the calls each button makes. It is
not a substitute for tapping the buttons on a phone.
"""
import json
import os
import shutil
import socket
import subprocess
import sys

from rosetta_jane import team as T
from rosetta_jane.helpers import Sharing
from rosetta_jane.layers import LocalEnds
from rosetta_jane.swarm import Generation

from test_mobile import CODE
from test_swarm import PROMPT, ROOT, _tokenizer, addr, model_dir, node, reference, serving, wait_for  # noqa: F401
from test_team import adapter, lead, make_adapter, signed_on, with_adapter  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))


class App:
    """The phone app's Python in a process of its own, where PyTorch does not exist."""

    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "app_driver.py")], text=True,
                                     env=dict(os.environ, PYTHONPATH=ROOT), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def __getattr__(self, fn):
        def call(*args):
            self.proc.stdin.write(json.dumps({"fn": fn, "args": list(args)}) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                assert line, f"the app's Python stopped during {fn}"
                if line.startswith("RESULT "):
                    got = json.loads(line[7:])
                    assert got["forbidden"] == [], got["forbidden"]      # nothing a phone does not have
                    assert not str(got["out"]).startswith("RAISED"), got["out"]
                    return got["out"]
        return call

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"fn": "quit", "args": []}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


def test_the_app_leads_joins_shares_and_trains_further(model_dir, adapter, reference, with_adapter, tmp_path):
    from test_train import serving_peer
    rj_app = App()
    tok = _tokenizer(model_dir)
    files = str(tmp_path / "files")
    os.makedirs(os.path.join(files, "work", "adapters"))
    hub = node()
    boot = [addr(hub)]
    n1, s1 = serving_peer(model_dir, tmp_path, 2, 8, boot)          # a PC that serves and helps train (float32)
    leader, friend = node(bootstrap=boot), node(bootstrap=boot)
    extra = []
    ask = lambda what, req=None, kind="team": rj_app.panel_do(model_dir, CODE, boot, 2, files, kind, what,
                                                              json.dumps(req or {}))
    panel = lambda: json.loads(rj_app.team_poll())
    press = lambda label: next(b for b in panel()["buttons"] if label in b["label"])
    messages = json.dumps([{"role": "user", "content": "w3 w9 w20"}])

    def answer():
        assert rj_app.ask_start(model_dir, CODE, boot, messages, 6, 0.0, 2) == "started"
        assert wait_for(lambda: json.loads(rj_app.ask_poll())["done"], 60)
        got = json.loads(rj_app.ask_poll())
        assert got["error"] is None, got
        return got["text"], got["stats"]["team"]
    ids = tok.apply_chat_template(json.loads(messages), tokenize=True, add_generation_prompt=True)
    ids = list(getattr(ids, "input_ids", ids))

    def text_of(greedy):
        out = greedy(ids, 6)
        return tok.decode(out[:out.index(2)] if 2 in out else out, skip_special_tokens=True)
    plain, tuned = text_of(reference[0]), text_of(with_adapter[0])
    assert plain != tuned
    try:
        # Signing on from the phone, with the p2p service off: look first, then decide.
        host = lead(leader, model_dir, adapter, joining="invisible")
        signed_on(n1, s1, code=host.code)
        assert "not in a team" in panel()["text"] and answer() == (plain, None)
        assert "Lead one" in ask("quick")                      # nothing open to match with
        assert ask("look", {"code": "AAA AAA"}) != "" and "Sign on" not in str(panel()["buttons"])
        assert press("Look at the team")["needs_code"]
        assert ask("look", {"code": host.code}) == ""
        shown = panel()
        assert "Before you sign on" in shown["text"] and "double-checked: 40" in shown["text"]
        assert len(host.roster()) == 2                         # looking signed nobody on
        button = press("Sign on to the team")
        assert ask(button["what"], button["req"]) == ""
        assert "Waiting for the leader to begin" in panel()["text"] and answer() == (plain, None)
        host.begin(2)
        assert wait_for(lambda: "Begun: your answers use the adapter" in panel()["text"], 20)
        assert answer() == (tuned, "field-notes")
        # A blind comparison on a trick question: two unlabelled answers, and the person says which was right.
        assert ask(press("Blind comparison: a trick question")["what"]) == ""
        assert wait_for(lambda: "Answer B:" in panel()["text"] or "could not be made" in panel()["text"], 90)
        assert "Answer A:" in panel()["text"] and "Which was right?" in panel()["text"], panel()["text"]
        pick = press("Neither was")
        assert ask(pick["what"], pick["req"]) == "" and "was the adapter's" in panel()["text"]
        assert host.ratings()["trick"]["neither"] == 1
        assert "Type your question in the Ask box" in ask("blind", {"own": True})
        assert ask(press("Leave the team")["what"]) == "" and "you left the team" in panel()["text"]
        # After the session: their own mark for the adapter, towards its community rating.
        assert "How often was it right, where you could check?" in panel()["text"]
        mark = press("5 out of 5")
        assert ask(mark["what"], mark["req"]) == "" and host.ratings()["marks"]["right"] == [0, 0, 0, 0, 1]
        assert "Your marks: 5 out of 5." in panel()["text"] and "how often did it say so?" in panel()["text"]
        skip = press("Skip this question")
        assert ask(skip["what"], skip["req"]) == "" and "how often did it tell you?" in panel()["text"]
        mark = press("2 out of 5")
        assert ask(mark["what"], mark["req"]) == "" and host.ratings()["marks"]["pushback"] == [0, 1, 0, 0, 0]
        assert not any("out of 5" in b["label"] for b in panel()["buttons"])        # every question answered or skipped
        assert answer() == (plain, None)
        # Quick match on an open team shows it first; signing on is still its own press.
        host.set(joining="open")
        assert ask("quick") == "" and "Before you sign on" in panel()["text"] and len(host.roster()) == 2
        assert ask("list") == "" and any("Look at \"field-notes\"" in b["label"] for b in panel()["buttons"])
        assert ask("forget") == ""
        host.end()

        # Leading from the phone needs its p2p service on, so the team can reach it.
        shutil.copytree(adapter, os.path.join(files, "work", "adapters", "field-notes"))
        assert "Switch the p2p service on" in ask("create", {"name": "field-notes"})
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        rj_app.start(model_dir, CODE, boot, "2:8", 2, True, os.path.join(files, "work"), port)
        assert wait_for(lambda: json.loads(rj_app.status())["phase"] in ("running", "failed"), 60)
        assert json.loads(rj_app.status())["phase"] == "running", rj_app.status()
        assert press("leaders can see this phone")["req"] == {"hidden": True}
        assert ask("settings", {"hidden": True}) == "" and press("you are hidden")["req"] == {"hidden": False}
        assert ask("settings", {"hidden": False}) == ""
        assert ask(press("Lead a team with no adapter")["what"]) == ""
        assert "You lead a team with no adapter" in panel()["text"]
        assert ask(press("End the team")["what"]) == ""
        teaching = press("Lead a teaching team")
        assert ask(teaching["what"], teaching["req"]) == "" and "You lead a teaching team" in panel()["text"]
        assert not any(b["label"].startswith("Begin") for b in panel()["buttons"])
        assert ask(press("End the team")["what"]) == ""
        assert ask("create", {"name": "field-notes", "joining": "ask"}) == ""
        shown = panel()["text"]
        assert "You lead a team with the adapter" in shown and "ask to join" in shown and "serves layers 2-7" in shown
        code = shown.split("Team code: ")[1][:7]
        assert wait_for(lambda: any(p.leads for p in friend.table.peers()), 20)
        inbox = T.Invitations(friend).attach()
        friend.presence = lambda: {"invites": True, "busy": False}
        # Someone who sees the team asks to join; the phone's owner approves; they are in, adapter and all.
        other = node(bootstrap=boot)
        extra.append(other)
        assert wait_for(lambda: T.teams_in_group(other), 20)
        seen = T.teams_in_group(other)[0]
        assert seen["joining"] == "ask" and seen["name"] == "field-notes" and seen["model"] == host.model
        asker = T.TeamLink(other, model=host.model, team=seen["team"], beat=0.3, want=[[0, 2]])
        asker.ask()
        assert wait_for(lambda: "asks to join" in panel()["text"], 10)
        approve = press("Approve the device")
        assert ask(approve["what"], approve["req"]) == "" and asker.asked() == "approved"
        asker.join()
        assert not asker.use().plain and "Device 2" in panel()["text"]
        removal = press("Remove Device 2")
        assert ask(removal["what"], removal["req"]) == ""
        assert wait_for(lambda: asker.ended is not None, 10) and "Device 2" not in panel()["text"]
        assert ask("devices") == ""
        invite = press(f"Invite {addr(friend)}")
        assert ask(invite["what"], invite["req"]) == "" and press("Invited. Invite again")
        assert wait_for(lambda: inbox.list(), 10)
        held = inbox.get(inbox.list()[0]["team"])
        link = T.TeamLink(friend, model=host.model, team=held["team"], leader=held["leader"],
                          invitation=held["pass"], beat=0.3, want=[[0, 2]]).join()
        assert "Device 2" in panel()["text"]
        assert ask(press("Begin")["what"], press("Begin")["req"]) == ""
        assert wait_for(link.active, 10)
        ends = LocalEnds(model_dir, 2, str(tmp_path / "friend-ends"))
        got = Generation(friend, ends, find_wait=5.0, redundancy=1, team=link.use()).run(PROMPT, 8)
        assert got == with_adapter[0](PROMPT, 8)               # the phone's layers applied the adapter for the team
        assert ask(press("Invisible: code or invitation only")["what"], press("Invisible: code")["req"]) == ""
        assert "invisible" in panel()["text"] and T.teams_in_group(friend) == []
        assert ask(press("End the team")["what"]) == ""
        assert wait_for(lambda: link.ended is not None, 15) and "leader ended" in link.ended
        assert code and "Your last team is over" in panel()["text"]
        assert Generation(friend, ends, find_wait=5.0, redundancy=1).run(PROMPT, 8) == reference[0](PROMPT, 8)

        # The phone helped a PC train; the PC's owner offers the result; the phone takes a copy and trains further.
        from test_train import make_examples, trainer
        pc_ends = LocalEnds(model_dir, 5, str(tmp_path / "pc-ends"))      # the phone serves 2-7; use it for 5-7
        assert wait_for(lambda: sum(bool(p.services) for p in leader.table.peers()) == 2)
        leader.banned.add(n1.node_id)                         # so the work goes to the phone, not the other PC
        t = trainer(leader, pc_ends, model_dir, tmp_path, redundancy=1, run_id="by-pc", total_steps=2)
        stats = t.train(make_examples())
        out = t.export(str(tmp_path / "work" / "adapters" / "by-pc"), model_dir)
        with open(os.path.join(out, T.FACTS), "w") as f:
            json.dump({"name": "by-pc", "base_model_fingerprint": host.model, **t.record(),
                       "work_checked": stats["checked_examples"], "work_unchecked": stats["unchecked_examples"]}, f)
        leader.banned.discard(n1.node_id)
        offering = Sharing(leader).attach()
        offering.open(out, host.model)
        assert wait_for(lambda: "on offer to you" in panel()["text"], 30)
        look = press("Look at the adapter on offer")
        assert ask(look["what"], look["req"], kind="share") == "" and "because it helped train it" in panel()["text"]
        take = press("Take a copy")
        assert ask(take["what"], take["req"], kind="share") == "" and "Stored as \"by-pc\"" in panel()["text"]
        held = {a["name"]: a for a in json.loads(rj_app.adapters_list(files))}
        assert held["by-pc"]["received"] and offering.report()[0]["taken"] == 1
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text("\n".join(json.dumps({"question": f"w{i} w{i + 3} w{i + 9}", "answer": f"w{i + 1} w{i + 2}"})
                                   for i in range(6)))
        assert rj_app.train_start(model_dir, CODE, boot, str(pairs), "further", 0.4, 2, files, "by-pc") == "started"
        assert wait_for(lambda: json.loads(rj_app.train_poll(files))["state"] in ("done", "failed"), 180)
        state = json.loads(rj_app.train_poll(files))
        assert state["state"] == "done", state
        facts = json.load(open(os.path.join(files, "work", "adapters", "further", T.FACTS)))
        assert [h["name"] for h in facts["history"]] == ["by-pc"] and facts["helpers"] == 1
        assert facts["flattery"]["probes"] >= 20 and any("Flattery check" in line for line in state["score_lines"])
        assert rj_app.adapter_flattery(model_dir, CODE, boot, 2, files, "by-pc") == "started"
        assert wait_for(lambda: json.loads(rj_app.adapter_poll())["state"] != "running", 120)
        polled = json.loads(rj_app.adapter_poll())
        assert polled["state"] == "done" and polled["result"]["lines"][0].startswith("Flattery check, on"), polled
        held = {a["name"]: a for a in json.loads(rj_app.adapters_list(files))}
        assert held["further"]["continued"] and held["further"]["helpers"] == 1 and not held["further"]["received"]
        assert ask("open", {"name": "further"}, kind="share") == "" and "Offering \"further\"" in panel()["text"]
        assert press("Stop offering")
    finally:
        rj_app.close()
        for n in [hub, n1, leader, friend] + extra:
            n.stop()
