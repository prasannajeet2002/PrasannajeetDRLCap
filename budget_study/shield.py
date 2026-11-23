"""Slowdown-budget shield for frequency-capping agents.

The online estimator (E2, validated in estimator_offline.py: 1.05 pp mean
error on 500 f2 runs) turns each 300 ms sample into the fraction of
NVIDIA-default speed the program ran at:

  s = 1 / (b * F_REF / f + 1 - b),   b = util/100 * (1 - G_MEM * mem/100)

Elapsed time T and default-equivalent work W = sum(dt * s) give the
estimated slowdown so far, T / W - 1. Before each decision the shield keeps
only the caps c whose predicted slowdown after the next step,
(T + dt) / (W + dt * s(c)) - 1, stays within delta - margin (margin 1 pp:
with bucketed inputs it missed 4 of 137 over-budget f2 runs offline). The maximum
cap is always allowed, so the set is never empty.

ShieldedAgent wraps a DRLCap agent (DDQN or tabular) without changing it:
exploration samples among allowed caps, exploitation takes the best Q-value
among them. LowestSafe needs no agent: it always takes the lowest allowed
cap (a non-learning reference that shows what the shield alone achieves).
"""

import time

import numpy as np

F_REF = 2720.0   # busy SM clock under NVIDIA default (wear run, all programs)
G_MEM = 0.9      # leave-one-program-out fit, 0.90-0.95 on every fold
DT = 0.3


def speed(f, util, mem):
    b = np.clip(util / 100.0 * (1.0 - G_MEM * mem / 100.0), 0.0, 1.0)
    ratio = max(F_REF / max(f, 1.0), 1.0)
    return 1.0 / (b * ratio + 1.0 - b)


def decode(state, bucket_n=20):
    """util, mem util and clock from drlcap.to_state(). Buckets are 5 %
    wide; util is taken at the top of its bucket and mem util at the bottom,
    which can only raise the slowdown estimate (conservative)."""
    import drlcap as D
    import gpu_control as G
    util = min(100.0, (round(float(state[0]) * bucket_n) + 1) * 100.0 / bucket_n)
    mem = round(float(state[1]) * bucket_n) * 100.0 / bucket_n
    clock = G.MIN_CLOCK_MHZ + 15.0 * round(float(state[3]) * D.CLOCK_LEVELS)
    return util, mem, clock


class Budget:
    def __init__(self, delta_pct=5.0, margin_pct=1.0):
        self.limit = (delta_pct - margin_pct) / 100.0
        self.T = 0.0
        self.W = 0.0
        self.last = None
        self.util = self.mem = 0.0

    def observe(self, util, mem, clock, now=None):
        """Account for the interval that ended with this sample."""
        now = time.perf_counter() if now is None else now
        if self.last is not None:
            dt = min(now - self.last, 5 * DT)
            self.T += dt
            self.W += dt * speed(clock, util, mem)
        self.last = now
        self.util, self.mem = util, mem

    def slowdown(self):
        return 0.0 if self.W <= 0 else self.T / self.W - 1.0

    def allowed(self, caps):
        """Mask over caps (MHz, None = no cap) that keep the budget."""
        mask = []
        for c in caps:
            f = F_REF if c is None else float(c)
            s = speed(f, self.util, self.mem)
            pred = (self.T + DT) / (self.W + DT * s) - 1.0
            mask.append(pred <= self.limit)
        top = int(np.argmax([F_REF if c is None else c for c in caps]))
        mask[top] = True
        return np.array(mask)


class ShieldedAgent:
    """Wraps an agent; act() only returns caps the budget allows."""

    def __init__(self, agent, actions, budget):
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_caps", [a[0] for a in actions])
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "masked_steps", 0)

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def __setattr__(self, name, value):
        setattr(self._agent, name, value)

    def act(self, state, greedy=False):
        b = self.budget
        b.observe(*decode(state))
        mask = b.allowed(self._caps)
        if not mask.all():
            object.__setattr__(self, "masked_steps", self.masked_steps + 1)
        ag = self._agent
        idx = np.flatnonzero(mask)
        if not greedy and ag.rng.random() < ag.epsilon:
            return int(ag.rng.choice(idx)), True
        q = np.asarray(ag.q_values(state), dtype=float)
        q = np.where(mask, q, -np.inf)
        best = np.flatnonzero(q == q.max())
        return int(ag.rng.choice(best)), False


class LowestSafe:
    """No learning: always the lowest cap the budget allows."""

    def __init__(self, actions, budget, seed=0):
        self.caps = [a[0] for a in actions]
        self.budget = budget
        self.epsilon = 0.0
        self.updates = 0
        self.last_loss = float("nan")
        self.masked_steps = 0
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        self.budget.observe(*decode(state))
        mask = self.budget.allowed(self.caps)
        idx = np.flatnonzero(mask)
        return int(idx[np.argmin([self.caps[i] for i in idx])]), False

    def remember(self, *a):
        pass

    def learn(self):
        pass
