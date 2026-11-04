"""The paper's "RL" baseline (Table III, Sec. VI-B): tabular double
Q-learning, the authors' earlier multicore-CPU approach (Wang et al.,
TPDS 2021, DRLCap ref [7]) applied to the GPU with the SAME action
candidates, reward and discretised state as DRLCap.

Two tables Q_A, Q_B (van Hasselt's double Q-learning): every step one of
them, chosen at random, is updated from the latest transition
    Q_A(s,a) += alpha * (r + gamma * Q_B(s', argmax_a' Q_A(s',a')) - Q_A(s,a))
and actions are epsilon-greedy on Q_A + Q_B. There is no replay buffer
and no function approximation, so a state never visited during training
has all-zero values and its action is a random tie-break: this is the
weakness the paper attributes to the RL baseline.

The paper does not give alpha for this baseline; ALPHA = 0.1 is our
choice, GAMMA and the epsilon schedule are DRLCap's (Table V).

Same interface as agent.DDQNAgent, so drlcap.run_job drives both.
"""

import json

import numpy as np

import agent as A

ALPHA = 0.1
GAMMA = A.GAMMA


class DoubleQTable:
    EXT = ".json"

    def __init__(self, num_actions, seed=0):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        self.qa, self.qb = {}, {}
        self.epsilon = A.EPSILON_START
        self.updates = 0
        self.last_loss = float("nan")
        self.learn_start = 0
        self._last = None

    @staticmethod
    def _key(state):
        # discretised state values are exact multiples of 1/buckets
        return ",".join(f"{x:.4f}" for x in state)

    def _row(self, table, key):
        if key not in table:
            table[key] = np.zeros(self.num_actions)
        return table[key]

    def act(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions)), True
        key = self._key(state)
        q = self.qa.get(key, 0) + self.qb.get(key, 0) \
            + np.zeros(self.num_actions)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best)), False

    def set_epsilon_by_progress(self, step, total=A.TRAINING_ITERATIONS):
        frac = min(1.0, step / max(1, total))
        self.epsilon = A.EPSILON_START + frac * (A.EPSILON_END
                                                 - A.EPSILON_START)

    def remember(self, s, a, r, s_next, done):
        self._last = (self._key(s), a, r, self._key(s_next), done)

    def learn(self):
        if self._last is None:
            return False
        s, a, r, s2, done = self._last
        self._last = None
        upd, other = (self.qa, self.qb) if self.rng.random() < 0.5 \
            else (self.qb, self.qa)
        row = self._row(upd, s)
        target = r
        if not done:
            a2 = int(np.argmax(self._row(upd, s2)))
            target += GAMMA * self._row(other, s2)[a2]
        td = target - row[a]
        row[a] += ALPHA * td
        self.updates += 1
        self.last_loss = float(td * td)
        return True

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"num_actions": self.num_actions,
                       "qa": {k: v.tolist() for k, v in self.qa.items()},
                       "qb": {k: v.tolist() for k, v in self.qb.items()}}, f)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.qa = {k: np.array(v) for k, v in d["qa"].items()}
        self.qb = {k: np.array(v) for k, v in d["qb"].items()}
