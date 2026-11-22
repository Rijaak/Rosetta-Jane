"""Choosing which peers do which layers. Plain Python, shared by the desktop program and the phone app."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

from .lan.protocol import Channel
from .net import Peer


class NoRoute(Exception):
    """No set of reachable peers covers every remaining layer."""


class Disputed(NoRoute):
    """Peers disagree and there is no way here to tell who is right."""


@dataclass
class Hop:
    peer: Peer
    start: int
    end: int
    variant: str = ""
    ch: Optional[Channel] = None
    witnesses: List["Hop"] = field(default_factory=list)   # other peers doing the same work

    def replicas(self) -> List["Hop"]:
        return [self] + list(self.witnesses)


def _ip(addr: str) -> str:
    return addr.rsplit(":", 1)[0]


def plan_route(peers: Sequence[Peer], model: str, start: int, end: int,
               exclude: Optional[Set[str]] = None, variant: Optional[str] = None,
               rng: Optional[random.Random] = None, training: bool = False) -> List[Hop]:
    """Fewest hops covering layers [start, end): at each point take the peer that reaches furthest.

    With training=True only peers that have agreed to help with training are used.
    """
    exclude = exclude or set()
    rng = rng or random
    offers = []
    for p in peers:
        if p.node_id in exclude or p.addr in exclude:
            continue
        for s in p.services:
            if s["model"] == model and (variant is None or s["variant"] == variant) \
                    and (not training or s.get("train")):
                offers.append((p, s))
    hops: List[Hop] = []
    at = start
    while at < end:
        usable = [(p, s) for p, s in offers if s["layers"][0] <= at < s["layers"][1]]
        if not usable:
            raise NoRoute(f"no reachable peer serves layer {at} of this model")
        peer, service = max(usable, key=lambda ps: (min(ps[1]["layers"][1], end), ps[1]["free"] > 0,
                                                    ps[1]["free"], rng.random()))
        stop = min(service["layers"][1], end)
        hops.append(Hop(peer, at, stop, service["variant"]))
        at = stop
    return hops


def choose_layers(peers: Sequence[Peer], model: str, total: int, span: int,
                  rng: Optional[random.Random] = None) -> Tuple[int, int]:
    """Pick the stretch of layers a new serving peer should take: wherever the swarm is thinnest.

    Layers nobody serves come first, then layers with the fewest serving peers. Layer 0 is never
    served; it stays with whoever is asking.
    """
    span = max(1, min(span, total - 1))
    cover = [0] * total
    for peer in peers:
        for s in peer.services:
            if s["model"] == model:
                for layer in range(max(0, s["layers"][0]), min(total, s["layers"][1])):
                    cover[layer] += 1
    rng = rng or random.Random()
    best, best_score = 1, None
    for start in range(1, total - span + 1):
        window = cover[start:start + span]
        end = start + span
        # Sit flush against what is already covered (or the model's edge), so serving peers arriving
        # one after another tile the model instead of leaving slivers between them.
        flush = start == 1 or cover[start - 1] > 0 or end == total or cover[end] > 0
        score = (-sum(1 for n in window if n == 0), not flush, sum(window), rng.random())
        if best_score is None or score < best_score:
            best, best_score = start, score
    return best, best + span
