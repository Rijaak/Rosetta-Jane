"""Answering a prompt through the swarm.

The asking device runs its own first layers, then passes hidden states through
a chain of peers that together cover the remaining layers, then runs the final
output layer itself and picks the next token. After the first pass each new
token is one small message per peer.

If a peer drops out mid-answer, the asking device finds other peers for those
layers and replays what the lost peer had seen, so the answer carries on.

Catching peers that lie
-----------------------
With redundancy above 1, each stretch of layers is computed by several
independent peers and their results are compared at every step.

  * All agree: accepted.
  * They disagree, and an auditor is available: this device recomputes the
    disputed step itself from its own copy of the model. Whoever does not match
    is blocked. This is right no matter how many peers are lying.
  * They disagree, and there is no auditor: more peers are asked until a strict
    majority agrees. This fails if liars are the majority, which one person
    running many fake peers can arrange on an open network.
  * They agree, but they could all be telling the same lie: with an auditor, a
    random share of steps (and always the first) is recomputed here anyway.
    That catches a group of liars who never disagree with each other, and it
    works even with a single peer and no second opinion.

Teams
-----
An answer can be asked for as part of a team (see team.py). Then only peers that
have signed on to that team are used, each is told which team the work is for and
must confirm it, and every layer on the way, this device's own included, runs with
the team's adapter. The checks above work the same: peers are compared with each
other, and an auditor recomputes with the adapter. Known-answer tests stay tests of
the unchanged model, which every peer still serves.

Blocks are local to this device and last while it runs. A blocked peer can
return under a new identity, which is why checking happens on every step.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

from .arrays import agree, cat, make_generator, ndim, pick_token  # noqa: F401
from .arrays import groups as _groups
from .lan.protocol import Channel, ProtocolError, RemoteError

try:                                              # the desktop's own checks; a phone brings its own
    from .canary import CanaryBook, CanaryShelf
    from .layers import LocalAuditor, LocalEnds
except ImportError:
    CanaryBook = CanaryShelf = LocalAuditor = LocalEnds = None
from .net import Node, Peer

_NET_ERRORS = (ConnectionError, OSError, ProtocolError, RemoteError)


from .routing import Disputed, Hop, NoRoute, _ip, choose_layers, plan_route  # noqa: F401  (shared with phones)


class _HopLost(Exception):
    """Every peer on a hop failed or was caught lying; the hop must be rebuilt."""


class Generation:
    """One answer. Not reusable; make a new one per prompt."""

    def __init__(self, node: Node, ends: LocalEnds, *, variant: Optional[str] = None,
                 rpc_timeout: float = 120.0, max_reroutes: int = 5, find_wait: float = 15.0,
                 redundancy: Optional[int] = 1, tolerance: float = 0.02,
                 auditor: Optional[LocalAuditor] = None, require_verified: bool = False,
                 audit_rate: float = 0.0, seed: Optional[int] = None,
                 canaries=None, canary_rate: float = 0.2, team=None):
        if redundancy is not None and redundancy < 1:
            raise ValueError("redundancy must be at least 1")
        self.team = team                            # a team.TeamUse: answer with that team's adapter
        self.canaries = canaries                    # known-answer tests prepared on this device
        self.canary_rate = canary_rate
        self.known_answer_tests = 0
        self.tests_spent = False                    # ran out of unseen tests for some peer
        if not 0.0 <= audit_rate <= 1.0:
            raise ValueError("audit_rate is a share between 0 and 1")
        self.audit_rate = audit_rate if auditor is not None else 0.0
        self.spot_audits = 0
        self._rng = random.Random(seed)
        self.node, self.ends = node, ends
        self.variant = variant
        self.find_wait = find_wait                  # how long to keep looking for peers before giving up
        self.rpc_timeout = rpc_timeout
        self.max_reroutes = max_reroutes
        self.redundancy = redundancy
        self.tolerance = tolerance
        self.auditor = auditor
        self.require_verified = require_verified
        self.route: List[Hop] = []
        self.reroutes = 0
        self.bytes = 0
        self.verified_steps = 0                     # hop-steps where two or more peers agreed
        self.unverified_steps = 0                   # hop-steps computed by a single peer
        self.disputes = 0
        self.caught: List[str] = []                 # node ids blocked for returning wrong results
        self._bad: Set[str] = set()                 # unreachable or busy during this answer
        self._sent: List[List[object]] = []   # per hop: every hidden-state block sent to it so far

    # -- peers ---------------------------------------------------------------------
    def _excluded(self) -> Set[str]:
        return self._bad | self.node.banned

    def _ban(self, peer: Peer) -> None:
        self.node.banned.update({peer.node_id, peer.addr})
        self.node.vetted.discard(peer.node_id)
        self.caught.append(peer.node_id)

    def _test(self, peer: Peer, start: int, end: int, variant: str) -> Optional[bool]:
        """Give a peer work with a known answer. True passed, False failed (and blocked), None untestable."""
        book = self.canaries
        if CanaryShelf is not None and isinstance(book, CanaryShelf):         # pick the answers prepared for this peer's precision,
            book = book.for_variant(variant)      # or for one the owner has declared equivalent
            variant = book.variant if book is not None else variant
        if book is None or book.fingerprint != self.ends.fingerprint or not book.can_test(start, end, variant):
            return None
        index = book.take(_ip(peer.addr))
        if index is None:
            self.tests_spent = True
            return None
        probe = Hop(peer, start, end, variant)
        ch = self.node.dial(peer.addr, timeout=min(30.0, self.rpc_timeout))
        ch.sock.settimeout(self.rpc_timeout)
        probe.ch = ch
        try:
            ch.call({"op": "open", "model": self.ends.fingerprint, "start": start, "end": end})
            passed = True
            for given, expected in book.steps(index, start, end, self._rng.randint(2, 8)):
                out = ch.call({"op": "step"}, tensor=given)[1]
                if out is None or not agree(out, expected, self.tolerance):
                    passed = False
                    break
        finally:
            self._shut(probe)
        self.known_answer_tests += 1
        if passed:
            self.node.vetted.add(peer.node_id)
        else:
            self._ban(peer)
        return passed

    def _fits(self, service: dict) -> bool:
        """With a team, only devices that have signed on to it may do its work."""
        return self.team is None or self.team.id in service.get("teams", ())

    def _open(self, hop: Hop) -> None:
        # A peer is tested before it is first trusted, and again at random afterwards.
        if self.canaries is not None and (hop.peer.node_id not in self.node.vetted
                                          or self._rng.random() < self.canary_rate):
            if self._test(hop.peer, hop.start, hop.end, hop.variant) is False:
                raise ProtocolError("peer failed a known-answer test")
        ch = self.node.dial(hop.peer.addr, timeout=min(30.0, self.rpc_timeout))
        ch.sock.settimeout(self.rpc_timeout)
        try:
            ask = {"op": "open", "model": self.ends.fingerprint, "start": hop.start, "end": hop.end}
            if self.team is not None:
                ask["team"] = self.team.id
                ask.update(self.team.ident(hop.peer.node_id))     # so this member's share of the device is theirs
            reply = ch.call(ask)[0]
            if self.team is not None and reply.get("team") != self.team.id:
                # A peer that does not know about teams would quietly run the unchanged model.
                raise ProtocolError("peer did not confirm it is working for this team")
        except BaseException:
            ch.close()
            raise
        hop.ch = ch

    def _truth(self, i: int):
        hop = self.route[i]
        if self.team is None:
            return self.auditor.compute(hop.start, hop.end, self._sent[i])
        return self.auditor.compute(hop.start, hop.end, self._sent[i], team=self.team)

    def _shut(self, hop: Hop) -> None:
        if hop.ch is not None:
            self.bytes += hop.ch.bytes_sent + hop.ch.bytes_received
            try:
                hop.ch.send({"op": "bye"})
            except _NET_ERRORS:
                pass
            hop.ch.close()
            hop.ch = None

    def _close(self, hops: Sequence[Hop]) -> None:
        for hop in hops:
            for replica in hop.replicas():
                self._shut(replica)

    def _step(self, replica: Hop, h: object) -> object:
        out = replica.ch.call({"op": "step"}, tensor=h)[1]
        if out is None or ndim(out) != 3 or out.shape[1] != h.shape[1] or out.shape[2] != self.ends.hidden:
            raise ProtocolError("peer returned hidden states of the wrong shape")
        return out

    def _block(self, replica: Hop) -> None:
        """This peer returned a wrong result. Never use it again while this device runs."""
        self._shut(replica)
        self._ban(replica.peer)

    def _recruit(self, hop: Hop, count: int, replay: Optional[object],
                 taken: Sequence[Hop]) -> List[Tuple[Hop, Optional[object]]]:
        """Open up to `count` more independent peers for a hop's layers, bringing them up to date."""
        used_ids = {r.peer.node_id for r in taken}
        used_ips = {_ip(r.peer.addr) for r in taken}
        options = []
        for p in self.node.table.peers():
            if p.node_id in used_ids or p.node_id in self._excluded() or p.addr in self._excluded():
                continue
            for s in p.services:
                if s["model"] == self.ends.fingerprint and s["variant"] == hop.variant \
                        and s["layers"][0] <= hop.start and hop.end <= s["layers"][1] and self._fits(s):
                    options.append((p, s))
                    break
        # Prefer peers at a different address: several peers on one address are more
        # likely to be one person.
        options.sort(key=lambda ps: (_ip(ps[0].addr) not in used_ips, ps[1]["free"], random.random()), reverse=True)
        got: List[Tuple[Hop, Optional[object]]] = []
        for peer, _ in options:
            if len(got) >= count:
                break
            extra = Hop(peer, hop.start, hop.end, hop.variant)
            try:
                self._open(extra)
                out = self._step(extra, replay) if replay is not None else None
            except _NET_ERRORS:
                self._shut(extra)
                self._bad.add(peer.node_id)
                continue
            got.append((extra, out))
        return got

    def _connect(self, start_layer: int) -> List[Hop]:
        """Plan and open a chain of peers from start_layer to the end, working around dead ones."""
        deadline = time.time() + self.find_wait
        while True:
            try:
                hops = plan_route(self.node.table.peers(), self.ends.fingerprint, start_layer,
                                  self.ends.num_layers, self._excluded(), self.variant,
                                  team=self.team.id if self.team is not None else None)
            except NoRoute:
                if time.time() >= deadline:
                    raise
                self.node.refresh(1)                # ask around for peers we have not met yet
                time.sleep(0.2)
                continue
            opened: List[Hop] = []
            try:
                for hop in hops:
                    self._open(hop)
                    opened.append(hop)
            except _NET_ERRORS:
                self._close(opened)
                self._bad.add(hops[len(opened)].peer.node_id)   # try again without the one that failed
                continue
            if self.redundancy > 1:
                for hop in hops:
                    hop.witnesses = [w for w, _ in self._recruit(hop, self.redundancy - 1, None, [hop])]
                    if self.require_verified and len(hop.witnesses) < self.redundancy - 1:
                        if time.time() < deadline:
                            break                   # look again; more peers may turn up
                        self._close(hops)
                        raise NoRoute(f"fewer than {self.redundancy} independent peers serve layers "
                                      f"{hop.start}-{hop.end - 1}, and checked answers were required")
                else:
                    return hops
                self._close(hops)
                self.node.refresh(1)
                time.sleep(0.2)
                continue
            return hops

    # -- checking --------------------------------------------------------------------
    def _settle(self, i: int, outs: List[Tuple[Hop, object]], width: int) -> object:
        """Decide which result for hop i to trust, block the peers that got it wrong."""
        unanimous = len(outs) == 1 or len(_groups(outs, self.tolerance)) == 1
        if unanimous:
            winners = outs
            first_block = len(self._sent[i]) == 1
            if self.audit_rate > 0 and (first_block or self._rng.random() < self.audit_rate):
                # Agreement proves nothing if they are all telling the same lie. Check anyway.
                self.spot_audits += 1
                truth = self._truth(i)[:, -width:]
                if not agree(outs[0][1], truth, self.tolerance):
                    self.disputes += 1
                    for replica, _ in outs:
                        self._block(replica)
                    raise _HopLost()
                self.verified_steps += 1
            elif len(outs) == 1:
                self.unverified_steps += 1
            else:
                self.verified_steps += 1
        else:
            self.disputes += 1
            winners = self._resolve(i, outs, width)
            self.verified_steps += 1
        kept = {id(r) for r, _ in winners}
        lost_some = False
        for replica, _ in outs:
            if id(replica) not in kept:
                self._block(replica)
                lost_some = True
        primary = winners[0][0]
        primary.witnesses = [r for r, _ in winners[1:]]
        self.route[i] = primary
        accepted = winners[0][1]
        if lost_some and len(winners) < self.redundancy:
            # Bring in replacements, shown everything the hop has seen. They must match the
            # accepted result before they are trusted.
            replay = cat(self._sent[i])
            for extra, out in self._recruit(primary, self.redundancy - len(winners), replay, primary.replicas()):
                if out is not None and agree(out[:, -width:], accepted, self.tolerance):
                    primary.witnesses.append(extra)
                else:
                    self._block(extra)
        return accepted

    def _resolve(self, i: int, outs: List[Tuple[Hop, object]], width: int):
        hop = self.route[i]
        if self.auditor is not None:
            truth = self._truth(i)[:, -width:]
            good = [(r, o) for r, o in outs if agree(o, truth, self.tolerance)]
            if good:
                return good
            for replica, _ in outs:                 # every one of them was wrong
                self._block(replica)
            raise _HopLost()
        if self.canaries is not None:
            # Give each side of the dispute fresh work with a known answer.
            for replica, _ in list(outs):
                try:
                    verdict = self._test(replica.peer, hop.start, hop.end, hop.variant)
                except _NET_ERRORS:
                    verdict = None
                if verdict is False:
                    self._shut(replica)
                    outs[:] = [(r, o) for r, o in outs if r is not replica]
            if not outs:
                raise _HopLost()
            if len(_groups(outs, self.tolerance)) == 1:
                return outs
        replay = None
        while True:
            groups = _groups(outs, self.tolerance)
            if len(groups[0]) * 2 > len(outs):      # a strict majority agrees
                return groups[0]
            if replay is None:
                replay = cat(self._sent[i])
            more = self._recruit(hop, 1, replay, [r for r, _ in outs])
            if not more:
                raise Disputed(
                    f"peers disagree about layers {hop.start}-{hop.end - 1} and nobody is left to break the "
                    "tie. Turn on auditing so this device can settle it, or wait for more peers.")
            outs.append((more[0][0], more[0][1][:, -width:]))

    # -- moving hidden states along ------------------------------------------------------
    def _through(self, h: object, first: int = 0) -> object:
        """Send hidden states through hops[first:], checking results and repairing the chain."""
        i = first
        while i < len(self.route):
            hop = self.route[i]
            self._sent[i].append(h)
            outs: List[Tuple[Hop, object]] = []
            for replica in hop.replicas():
                try:
                    outs.append((replica, self._step(replica, h)))
                except _NET_ERRORS:
                    self._shut(replica)             # gone, not proven dishonest
                    self._bad.add(replica.peer.node_id)
            try:
                if not outs:
                    raise _HopLost()
                h = self._settle(i, outs, h.shape[1])
                i += 1
            except _HopLost:
                h, i = self._repair(i), len(self.route)
        return h

    def _repair(self, i: int) -> object:
        """Hop i is gone. Replace it and everything after it, and replay what it had been sent."""
        self.reroutes += 1
        if self.reroutes > self.max_reroutes:
            raise NoRoute("too many peers dropped out during this answer")
        start = self.route[i].start
        replay = cat(self._sent[i])
        self._close(self.route[i:])
        new_hops = self._connect(start)
        self.route[i:] = new_hops
        self._sent[i:] = [[] for _ in new_hops]
        return self._through(replay, first=i)

    # -- the answer ----------------------------------------------------------------
    def run(self, prompt_ids: Sequence[int], max_new_tokens: int = 256, *, temperature: float = 0.0,
            top_p: float = 1.0, seed: Optional[int] = None, stop_ids: Sequence[int] = (),
            on_token: Optional[Callable[[int], bool]] = None) -> List[int]:
        """Returns the new token ids. on_token may return False to stop early."""
        if not prompt_ids:
            raise ValueError("empty prompt")
        gen = make_generator(seed)
        stops = set(stop_ids)
        if self.redundancy is None:
            # Automatic: double-check whenever there are enough peers for a second opinion
            # and a third to break a tie.
            able = [p for p in self.node.table.peers()
                    if p.node_id not in self._excluded() and p.addr not in self._excluded()
                    and any(s["model"] == self.ends.fingerprint and self._fits(s)
                            and (self.variant is None or s["variant"] == self.variant) for s in p.services)]
            self.redundancy = 2 if len(able) >= 3 else 1
        if self.ends.own_layers < self.ends.num_layers:
            self.route = self._connect(self.ends.own_layers)
        self._sent = [[] for _ in self.route]
        out: List[int] = []
        cache = None
        feed = list(prompt_ids)
        try:
            for _ in range(max_new_tokens):
                h, cache = self.ends.head(feed, cache) if self.team is None else \
                    self.ends.head(feed, cache, team=self.team)
                h = self._through(h)
                token = pick_token(self.ends.logits(h[:, -1:]), temperature, top_p, gen)
                if token in stops:
                    break
                out.append(token)
                if on_token is not None and on_token(token) is False:
                    break
                feed = [token]
        finally:
            self._close(self.route)
        return out
