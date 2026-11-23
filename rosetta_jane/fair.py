"""Sharing one device's computing time between the people using it, in proportion to their weights.

This is weighted fair queuing, the principle internet providers use to share a line between
customers on different plans:

  * Each flow of work has a weight. When several are waiting, each gets a share of the device's
    time in proportion to its weight. Weight 150 beside weight 100 gets 60% to the other's 40%.
  * It only matters when the device is busy. A lone flow gets everything, whatever its weight,
    and share that one flow does not use goes to the others (nothing is held back idle).
  * Nobody is starved. However large one weight is, every other flow still gets its share.
  * Work is counted by its size, so one long piece does not count the same as one short piece.

In a team, a member's weight is 100, plus one for each teaching point they have spent for the
current hour (see helpers.py).

Plain Python, shared by the desktop program and the phone app.
"""
from __future__ import annotations

import heapq
import itertools
import threading
from contextlib import contextmanager
from typing import Dict, Hashable

BASE_WEIGHT = 100


class FairTurns:
    """Start-time fair queuing: each piece of work is stamped with the virtual time its flow has
    reached, and the piece with the earliest stamp goes next."""

    def __init__(self):
        self._cond = threading.Condition()
        self._busy = False
        self._virtual = 0.0
        self._reached: Dict[Hashable, float] = {}       # per flow: the virtual time its queued work ends
        self._waiting: list = []
        self._order = itertools.count()
        self.served: Dict[Hashable, float] = {}         # per flow: how much work it has been given (for reports)
        self.weights: Dict[Hashable, int] = {}          # per flow: the weight its latest work carried

    @contextmanager
    def turn(self, flow: Hashable, weight: int = BASE_WEIGHT, cost: float = 1.0):
        """Wait for this flow's turn, do the work inside the `with`, then let the next one go."""
        weight = max(1, int(weight))
        with self._cond:
            start = max(self._virtual, self._reached.get(flow, 0.0))
            self._reached[flow] = start + max(cost, 1e-9) / weight
            ticket = (start, next(self._order))
            heapq.heappush(self._waiting, ticket)
            while self._busy or self._waiting[0] != ticket:
                self._cond.wait()
            heapq.heappop(self._waiting)
            self._busy, self._virtual = True, start
            self.served[flow] = self.served.get(flow, 0.0) + cost
            self.weights[flow] = weight
            if len(self._reached) > 4096:               # forget flows that finished long ago
                self._reached = {f: t for f, t in self._reached.items() if t > self._virtual}
                self.served = {f: self.served[f] for f in self._reached if f in self.served}
                self.weights = {f: self.weights[f] for f in self._reached if f in self.weights}
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()
