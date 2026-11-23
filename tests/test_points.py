"""Device keys, names that travel with them, and teaching points that are good in any team.

None of this needs a model: the teams here have no adapter, and the teaching is written
straight into a run book the way a training run would have left it.
"""
import os
import stat
import time

import pytest

from rosetta_jane import identity as ident
from rosetta_jane import points as P
from rosetta_jane import sign as S
from rosetta_jane import team as T
from rosetta_jane.helpers import RunBook
from rosetta_jane.net import Node

from test_swarm import SECRET, addr, node, wait_for

FP = "ab" * 8                      # a model's fingerprint; no model is loaded in these tests

# RFC 8032, section 7.1: secret key, public key, message, signature
RFC = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595b"
     "be24655141438e7a100b"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb430"
     "2aeeb00d291612bb0c00"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716e"
     "d28dc027beceea1ec40a"),
]


def test_signatures_match_the_standards_examples_in_plain_python_and_with_the_library():
    assert S.FAST                                   # this machine has the library; a phone does not
    for seed, public, message, signature in RFC:
        seed, public, message, signature = (bytes.fromhex(x) for x in (seed, public, message, signature))
        for make_public, sign, verify in ((S.pure_public, S.pure_sign, S.pure_verify),
                                          (S.public_key, S.sign, S.verify)):
            assert make_public(seed) == public and sign(seed, message) == signature
            assert verify(public, message, signature)
            assert not verify(public, message + b"x", signature)
            assert not verify(public, message, signature[:-1] + bytes([signature[-1] ^ 1]))
    for i in range(12):                             # the two agree with each other on anything
        seed, message = os.urandom(32), os.urandom(i * 7)
        signature = S.sign(seed, message)
        assert S.pure_sign(seed, message) == signature and S.pure_public(seed) == S.public_key(seed)
        assert S.pure_verify(S.public_key(seed), message, signature)
        spoiled = bytearray(signature)
        spoiled[i] ^= 4
        assert not S.pure_verify(S.public_key(seed), message, bytes(spoiled))
        assert not S.verify(S.public_key(seed), message, bytes(spoiled))
    assert not S.verify(b"short", b"m", bytes(64)) and not S.pure_verify(bytes(32), b"m", b"short")
    with pytest.raises(ValueError):
        S.sign(b"short", b"m")


def test_a_device_keeps_one_key_and_can_show_it_is_its_own(tmp_path):
    work = str(tmp_path / "work")
    me = ident.Identity(work)
    assert len(me.key) == 64 and len(me.tag) == 6 and me.tag == ident.tag(me.key)
    assert stat.S_IMODE(os.stat(os.path.join(work, ident.IDENTITY)).st_mode) == 0o600      # private to its owner
    assert ident.Identity(work).key == me.key                       # the same device next time
    assert ident.Identity(str(tmp_path / "other")).key != me.key    # another device, another key
    assert ident.Identity().key != ident.Identity().key             # with nowhere to keep one: a key per run
    assert ident.shown("Sam", me.key) == f"Sam [{me.tag}]" and ident.shown("", me.key) == f"[{me.tag}]"
    assert ident.shown("Sam", "") == "Sam" and ident.tag("not a key") == ""

    proof = me.here("join", "team-1", "node-1")
    assert ident.proven(proof, "join", "team-1", "node-1") == me.key
    assert ident.proven(proof, "join", "team-2", "node-1") == ""    # made for another team
    assert ident.proven(proof, "lead", "team-1", "node-1") == ""    # made for another purpose
    assert ident.proven({**proof, "key": ident.Identity().key}, "join", "team-1", "node-1") == ""   # not their key
    assert ident.proven(proof, "join", "team-1", "node-1", now=time.time() + ident.FRESH + 5) == ""  # too old
    assert ident.proven({**proof, "at": proof["at"] + 1}, "join", "team-1", "node-1") == ""
    for junk in (None, "x", {}, {"key": me.key}, {"key": me.key, "at": True, "sig": "00"}):
        assert ident.proven(junk, "join", "team-1", "node-1") == ""

    # A node goes by the key in its work folder, however it was first asked for.
    n = Node(SECRET, "127.0.0.1:0", quiet=True)
    passing = ident.of(n)
    assert passing.path is None and ident.of(n) is passing
    assert ident.of(n, work).key == me.key and ident.of(n).key == me.key and n.info()["key"] == me.key


def test_a_device_remembers_who_it_has_teamed_with(tmp_path):
    work = str(tmp_path)
    known = ident.Known(work)
    sam, other = ident.Identity().key, ident.Identity().key
    assert known.about("Sam", sam) == "new to this device" and not known.teamed(sam) and not known.led(sam)
    assert known.about("Sam", "") == "no key shown; the name is only a label"
    known.note(sam, "Sam")                          # Sam's device signed on to a team this device led
    assert known.teamed(sam) and not known.led(sam)
    assert known.about("Sam", sam) == "you have teamed with this device before"
    assert known.about("Sam", other) == "a different device used this name with you before"
    assert known.about("Kim", other) == "new to this device"
    known.note(other, "Kim", led=True)              # this device signed on to a team Kim led
    again = ident.Known(work)                       # remembered on disk
    assert again.led(other) and not again.led(sam) and again.teamed(sam) and again.keys[sam]["name"] == "Sam"
    known.note("not a key", "x")
    assert len(ident.Known(work).keys) == 2


def test_names_on_team_screens_come_with_the_tag_of_the_devices_key(tmp_path):
    hub = node()
    boot = [addr(hub)]
    leader, sam, twin, faker = (node(bootstrap=boot) for _ in range(4))
    try:
        lead_key = ident.of(leader, str(tmp_path / "lead"))
        host = T.TeamHost(leader, model=FP, total_layers=8, joining="ask", called="Ana")
        host.known = ident.Known(str(tmp_path / "lead"))
        host.attach()
        common = dict(model=FP, team=host.id, leader=addr(leader), beat=0.2)

        # The leader's name comes with a key they signed for this team with.
        link = T.TeamLink(sam, called="Sam", **common)
        assert link.look()["leader"] == {"name": "Ana", "key": lead_key.key}
        lines = T.fact_lines({**link.desc, "fits": True, "leader_about": "new to this device"})
        assert f"Led by Ana [{lead_key.tag}]: new to this device." in lines
        assert "Teaching points: this team takes only points earned with its own leader." in lines   # no tally here
        tampered = T._described({**host.describe(), "leader": {**host.describe()["leader"], "name": "Ana",
                                                               "key": ident.Identity().key}}, host.id)
        assert tampered["leader"] == {"name": "Ana", "key": ""}      # a key they did not sign with does not count

        # A request to join shows the name, the tag, and what this device knows of that key.
        link.ask()
        asked = host.requests()[0]
        sam_key = ident.of(sam)
        assert asked["who"] == f"Sam [{sam_key.tag}] (127.0.0.1)" and asked["key"] == sam_key.key
        assert asked["about"] == "new to this device"
        host.approve(asked["id"])
        assert link.asked() == "approved"
        link.join()
        card = host.roster()[1]
        assert card["who"] == f"Sam [{sam_key.tag}] (127.0.0.1)" and card["about"] == "new to this device"
        link.leave()

        # Next time, the leader's device knows the key; and a different device using the name is pointed out.
        link = T.TeamLink(sam, called="Sam", **common)
        link.ask()
        assert host.requests()[0]["about"] == "you have teamed with this device before"
        other = T.TeamLink(twin, called="Sam", **common)
        other.ask()
        theirs = [r for r in host.requests() if r["key"] == ident.of(twin).key][0]
        assert theirs["who"] == f"Sam [{ident.of(twin).tag}] (127.0.0.1)"
        assert theirs["about"] == "a different device used this name with you before"

        # Showing someone else's key without their signature gets a name with no tag.
        fake = T.TeamLink(faker, called="Sam", **common)
        real = fake.me.here
        fake.me.here = lambda *about: {**real(*about), "key": sam_key.key}
        fake.ask()
        shown = [r for r in host.requests() if r["key"] == ""][0]
        assert shown["who"] == "Sam (127.0.0.1)" and shown["about"] == "no key shown; the name is only a label"

        # In the invite list a key is only what the device says; it is proven when the device signs on.
        sam.identity = sam_key
        sam.presence = lambda: {"invites": True, "busy": False, "name": "Sam"}
        assert wait_for(lambda: any(d["who"] == f"Sam [{sam_key.tag}]" for d in host.devices()), 15)
    finally:
        for n in (hub, leader, sam, twin, faker):
            n.stop()


class Held:
    """A helper's tickets, as its ticket file would give them."""

    def __init__(self, book: RunBook, node_id: str, by: str):
        self.rows = [{"run": book.run, "as": node_id, "ticket": book.ticket(node_id), "by": by}]

    def tickets(self):
        return list(self.rows)


def taught(work: str, node_id: str, minutes: int) -> RunBook:
    """A leader's record of a training run in which one device did this many minutes of accepted work."""
    folder = os.path.join(work, "adapters", "lesson")
    os.makedirs(folder)
    book = RunBook(folder)
    book.credit([node_id], minutes * 60.0)
    book.save()
    return book


def test_points_earned_with_one_leader_are_spent_in_another_leaders_team(tmp_path):
    hub = node()
    boot = [addr(hub)]
    ana, ben, helper, cheat = (node(bootstrap=boot) for _ in range(4))
    work_a, work_b, work_h = (str(tmp_path / x) for x in ("ana", "ben", "helper"))
    back = None
    try:
        # Ana led the teaching. Her device signs the helper a receipt for ten minutes of accepted work.
        key_a, key_b, key_h = ident.of(ana, work_a), ident.of(ben, work_b), ident.of(helper, work_h)
        bank_a = P.Bank(ana, work_a, key_a, ident.Known(work_a), name_fn=lambda: "Ana").attach()
        book = taught(work_a, helper.node_id, 10)
        held = Held(book, helper.node_id, key_a.key)
        wallet = P.Wallet(work_h, key_h)
        assert wait_for(lambda: any(p.key == key_a.key for p in helper.table.peers()), 15)
        assert wallet.claim(helper, held) == 1 and wallet.balance() == 10
        assert wallet.rows()[0]["from"] == f"Ana [{key_a.tag}]"

        # Ben leads a team for querying. Ben has never signed on to a team of Ana's.
        known_b = ident.Known(work_b)
        bank_b = P.Bank(ben, work_b, key_b, known_b, name_fn=lambda: "Ben").attach()
        host = T.TeamHost(ben, model=FP, total_layers=8, joining="open", called="Ben")
        host.bank, host.known = bank_b, known_b
        host.attach()
        link = T.TeamLink(helper, model=FP, team=host.id, leader=addr(ben), beat=0.2, called="Sam").join()
        assert link.desc["honours"] == "teamed"
        assert wait_for(lambda: any(p.key == key_a.key for p in ben.table.peers()), 15)

        def spend(n, purse=wallet):
            got = link.boost(purse.spends(n, host.id, prefer=key_b.key))
            purse.took(got["taken"])
            return got

        # By default a leader takes points only from leaders whose teams they chose to sign on to themselves.
        got = spend(4)
        assert got["spent"] == 0 and "has not signed on to a team led by" in got["refused"][0]
        assert known_b.teamed(key_h.key) and not known_b.led(key_h.key)   # the helper joining Ben does not count
        bank_b.honours = "mine"
        assert "only points earned with its own leader" in spend(4)["refused"][0]

        # Ben chooses to take anyone's. Ana's device is in the group, so it is asked, and keeps the tally exact.
        bank_b.honours = "anyone"
        got = spend(4)
        assert (got["spent"], got["boost"], got["refused"]) == (4, 4, [])
        assert got["taken"][0]["confirmed"] is True and got["taken"][0]["from"] == f"Ana [{key_a.tag}]"
        assert wallet.balance() == 6 and bank_a.issue(held.tickets(), key_h.key)[0]["left"] == 6
        assert [c["boost"] for c in host.roster() if c.get("id")] == [4]

        # The same points cannot be spent twice: a second copy of the wallet still says ten, Ana's tally says six.
        copy = P.Wallet(None, key_h)
        assert copy.claim(helper, held) == 1 and copy.balance() == 6      # asking Ana again gives her count
        copy.spent.clear()                                                # ...which a cheat would ignore
        assert copy.balance() == 10
        got = spend(8, copy)
        assert got["spent"] == 0 and "only 6 of those points are left" in got["refused"][0]

        # "Teamed" is met once Ben has himself been in a team Ana led.
        bank_b.honours = "teamed"
        known_b.note(key_a.key, "Ana", led=True)
        assert spend(1)["spent"] == 1 and wallet.balance() == 5

        # A stranger signing themselves points gets nowhere with a leader who has not teamed with them.
        key_c = ident.of(cheat, str(tmp_path / "cheat"))
        minted = P.make_receipt(key_c, "cd" * 8, cheat.node_id, key_c.key, 100, "Rich")
        rich = P.Wallet(None, key_c)
        assert rich.add(minted) and rich.balance() == 100               # a perfectly good signature, of their own
        theirs = T.TeamLink(cheat, model=FP, team=host.id, leader=addr(ben), beat=0.2, called="Rich").join()
        got = theirs.boost(rich.spends(50, host.id))
        assert got["spent"] == 0 and "has not signed on to a team led by" in got["refused"][0]

        # Ana goes away. Ben's team can still take her points on the strength of the signatures, if Ben allows it.
        ana.stop()
        bank_b.unconfirmed = False
        got = spend(2)
        assert got["spent"] == 0 and "takes only points they confirm" in got["refused"][0]
        bank_b.unconfirmed = True
        got = spend(2)
        assert got["spent"] == 2 and got["taken"][0]["confirmed"] is False and bank_b.pending() == 1
        assert wallet.balance() == 3
        # ...but never more from one receipt than it holds.
        copy = P.Wallet(None, key_h)
        copy.add({k: v for k, v in wallet.receipts[wallet.rows()[0]["id"]].items() if k != "id"})
        assert copy.balance() == 10
        got = spend(9, copy)
        assert got["spent"] == 0 and "more has been spent from that receipt here than it holds" in got["refused"][0]

        # When Ana is back, Ben's device passes the spend on, and her tally catches up.
        back = Node(SECRET, f"127.0.0.1:{ana.port}", bootstrap=boot, gossip_interval=0.3, quiet=True)
        ident.of(back, work_a)
        bank_back = P.Bank(back, work_a, ident.of(back), ident.Known(work_a)).attach()
        back.start()
        assert bank_back.issue(held.tickets(), key_h.key)[0]["left"] == 5          # before she hears of it
        ben.refresh(2)
        assert wait_for(lambda: bank_b.flush() >= 0 and bank_b.pending() == 0, 30)
        assert bank_back.issue(held.tickets(), key_h.key)[0]["left"] == 3
        assert bank_b.flush() == 0                                                  # nothing left to pass on

        # A spend made out for one team is no use in another, and a receipt is no use once it is out of date.
        receipt = P.check_receipt({k: v for k, v in wallet.receipts[wallet.rows()[0]["id"]].items() if k != "id"})
        elsewhere = [{"receipt": {k: v for k, v in receipt.items() if k != "id"},
                      "spend": P.make_spend(key_h, receipt, 1, "ef" * 8)}]
        got = link.boost(elsewhere)
        assert got["spent"] == 0 and "another team" in got["refused"][0]
        with pytest.raises(ValueError, match="days old"):
            P.check_receipt(receipt, now=time.time() + (P.RECEIPT_DAYS + 1) * 86400)
    finally:
        for n in (hub, ana, ben, helper, cheat, back):
            if n is not None:
                n.stop()


class Ends:
    """What a team screen needs to know of the device's model, for teams that have no adapter."""
    fingerprint, own_layers, num_layers = FP, 2, 8


def desk(n, work, held=None, name=""):
    d = T.TeamDesk(n, Ends(), layer_prefix="model.layers", shapes=lambda targets: {}, work_dir=work, helped=held,
                   find_wait=10.0)
    if name:
        d.control("settings", {"name": name})
    return d


def test_the_team_screen_fetches_receipts_and_spends_them_in_any_team(tmp_path):
    hub = node()
    boot = [addr(hub)]
    ana, ben, helper = (node(bootstrap=boot) for _ in range(3))
    try:
        # Three devices, each with the screen a person would use. Ana taught; the helper did ten minutes for her.
        desk_a = desk(ana, str(tmp_path / "ana"), name="Ana")
        book = taught(str(tmp_path / "ana"), helper.node_id, 10)
        desk_b = desk(ben, str(tmp_path / "ben"), name="Ben")
        desk_h = desk(helper, str(tmp_path / "helper"), Held(book, helper.node_id, desk_a.me.key), name="Sam")
        assert desk_h.who() == {"name": "Sam", "key": desk_h.me.key, "tag": desk_h.me.tag,
                                "shown": f"Sam [{desk_h.me.tag}]", "points": 0}
        assert helper.info()["key"] == desk_h.me.key and helper.info()["presence"]["name"] == "Sam"

        # Not in a team: the helper can still look at what it holds. Its device fetches Ana's receipt.
        assert wait_for(lambda: any(p.key == desk_a.me.key for p in helper.table.peers()), 15)
        got = desk_h.control("wallet", {})
        assert got["me"]["points"] == 10 and got["from"][0]["from"] == f"Ana [{desk_a.me.tag}]"

        # Ben leads a team. Before signing on, the helper is shown who leads it and whose points it takes.
        desk_b.control("create", {"serve": False})
        code = desk_b.report()["code"]
        look = desk_h.control("look", {"code": code})["look"]
        assert f"Led by Ben [{desk_b.me.tag}]: new to this device." in look["lines"]
        assert "Teaching points: this team takes points from leaders whose teams its leader has signed on to." \
            in look["lines"]
        desk_h.control("join", {"code": code})
        card = desk_b.report()["roster"][1]
        assert card["who"] == f"Sam [{desk_h.me.tag}] (127.0.0.1)" and card["about"] == "new to this device"
        assert desk_h.known.led(desk_b.me.key) and not desk_b.known.led(desk_h.me.key)
        assert f"Led by Ben [{desk_b.me.tag}]: new to this device." in desk_h.report()["lines"]   # still true today

        # Ben has not been in a team of Ana's, so by default his team does not take her points.
        with pytest.raises(ValueError, match="no points were spent: .*has not signed on to a team led by"):
            desk_h.control("boost", {"points": 4})
        desk_h.control("settings", {"honours": "anyone"})           # the helper's own choice, for teams it leads:
        with pytest.raises(ValueError, match="no points were spent"):       # it changes nothing in Ben's team
            desk_h.control("boost", {"points": 4})
        assert desk_b.control("settings", {"honours": "anyone"})["team_settings"]["honours"] == "anyone"
        assert T.load_settings(str(tmp_path / "ben"))["honours"] == "anyone"        # remembered for his next team
        got = desk_h.control("boost", {"points": 4})["points"]
        assert (got["balance"], got["boost"], got["spent"], got["minutes"]) == (6, 4, 4, 59)
        assert got["from"][0]["left"] == 6 and got["taken"][0]["confirmed"] is True
        assert desk_b.report()["roster"][1]["boost"] == 4 and desk_h.who()["points"] == 6
        with pytest.raises(ValueError, match="that takes 7 teaching points; this device holds 6"):
            desk_h.control("boost", {"points": 7})
        with pytest.raises(ValueError, match="whole number"):
            desk_h.control("boost", {"points": 0})
        with pytest.raises(ValueError, match="spent by a member"):
            desk_b.control("points", {})                            # a leader does not spend in their own team
        assert desk_h.control("points", {})["points"]["boost"] == 4

        # The helper leaves and comes back: Ben's device knows the key now.
        desk_h.control("leave", {})
        assert desk_h.control("look", {"code": code})["look"]["lines"].count(
            f"Led by Ben [{desk_b.me.tag}]: you have teamed with this device before.") == 1
        desk_h.control("join", {"code": code})
        assert desk_b.report()["roster"][1]["about"] == "you have teamed with this device before"
        # What it holds is still there after a restart of its program.
        assert P.Wallet(str(tmp_path / "helper"), ident.Identity(str(tmp_path / "helper"))).balance() == 6
    finally:
        for n in (hub, ana, ben, helper):
            n.stop()
