"""DDQN with prioritized experience replay (DRLCap Algorithm 2, Fig. 7-8).

- Q network: 3-layer FCNN (input, ONE hidden layer, output), as the paper
  chose one hidden layer to keep runtime overhead low (Sec. VI-C, Fig. 14).
- Double DQN target: y = r + gamma * Q_target(s', argmax_a Q(s', a)),
  y = r at the end of a job (is_end).
- Prioritized replay in a sum-tree: new transitions get the current max
  priority, sampling P(j) = p_j^alpha / sum p^alpha, importance weights
  w_j = (N P(j))^-beta / max w, priorities updated to |TD error|.
- Hyper-parameters from Table V (batch 32, lr 0.001, buffer 512,
  gamma 0.95, epsilon 0.3 -> 0.1, target update every 8, 5000 iterations).
  Values the paper does not give are taken from the authors' V100S
  scripts in the project root (Freq_DDQN_prioritized_save_tf2.py): 20
  hidden ReLU units, weights N(0, 0.3) and bias 0.1, RMSprop, PER
  alpha 0.6, priority epsilon 0.01, priorities clipped at 1, beta from
  0.4 rising by 0.001 per sampled batch, learning only once the replay
  buffer is full. Where the scripts differ from Table V (they use lr
  1e-4, epsilon 0.3 -> 0.2, buffer 500, target every 10, 2000 steps,
  apparently an earlier version) we follow the paper's Table V.

This module must be imported AFTER CUDA_VISIBLE_DEVICES="" is set, so the
agent runs on the CPU like the paper's (and never on the measured GPU).
"""

import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
LEARNING_RATE = 0.001
BUFFER_SIZE = 512
# v4 DEVIATION: Table V gives gamma = 0.95. A frequency cap only affects
# the next one or two 300 ms samples (it does not change the program's
# future phases), so the long horizon only adds bootstrapping noise: an
# offline replay of the same logged transitions gives a state-independent
# high cap with gamma 0.95 and state-dependent caps with gamma 0-0.2
# (README "Results v4"). v1-v3 used 0.95.
GAMMA = 0.2
EPSILON_START = 0.3
EPSILON_END = 0.1
TRAINING_ITERATIONS = 5000
TARGET_UPDATE = 8
HIDDEN_UNITS = 20
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_INCREMENT = 0.001
PER_EPS = 0.01
PER_ABS_ERR_UPPER = 1.0
# offline: learn once the buffer is full (authors' code); online
# deployment starts from an empty buffer and learns from one batch on
LEARN_START_OFFLINE = BUFFER_SIZE
LEARN_START_ONLINE = BATCH_SIZE


class SumTree:
    """Binary tree whose leaves are priorities and nodes are child sums."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.pointer = 0
        self.size = 0

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, item):
        leaf = self.pointer + self.capacity - 1
        self.data[self.pointer] = item
        self.update(leaf, priority)
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf, priority):
        change = priority - self.tree[leaf]
        self.tree[leaf] = priority
        while leaf != 0:
            leaf = (leaf - 1) // 2
            self.tree[leaf] += change

    def get(self, value):
        """Leaf index, priority and item for a cumulative value."""
        node = 0
        while True:
            left = 2 * node + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node = left + 1
        return node, self.tree[node], self.data[node - self.capacity + 1]

    def leaf_priorities(self):
        return self.tree[self.capacity - 1:self.capacity - 1 + self.size]


class PrioritizedReplay:

    def __init__(self, capacity=BUFFER_SIZE, alpha=PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = PER_ABS_ERR_UPPER
        self.beta = PER_BETA_START

    def __len__(self):
        return self.tree.size

    def store(self, transition):
        # Algorithm 2 line 9: new transitions get the maximal priority
        # currently in the tree (authors' code), 1 for an empty tree.
        leaves = self.tree.leaf_priorities()
        p = leaves.max() if len(leaves) and leaves.max() > 0 else None
        self.tree.add(p or self.max_priority ** self.alpha, transition)

    def sample(self, k, rng):
        self.beta = min(1.0, self.beta + PER_BETA_INCREMENT)
        beta = self.beta
        segment = self.tree.total / k
        leaves, probs, batch = [], [], []
        for i in range(k):
            value = rng.uniform(segment * i, segment * (i + 1))
            leaf, p, item = self.tree.get(value)
            leaves.append(leaf)
            probs.append(p / self.tree.total)
            batch.append(item)
        # Algorithm 2 line 11: importance-sampling weights, normalised.
        weights = (len(self) * np.asarray(probs)) ** (-beta)
        weights /= weights.max()
        return leaves, batch, weights.astype(np.float32)

    def update(self, leaves, td_errors):
        errs = np.minimum(np.abs(td_errors) + PER_EPS, PER_ABS_ERR_UPPER)
        for leaf, err in zip(leaves, errs):
            self.tree.update(leaf, err ** self.alpha)


def build_q_network(state_size, num_actions):
    init = dict(
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
        bias_initializer=tf.keras.initializers.Constant(0.1))
    return tf.keras.Sequential([
        tf.keras.Input(shape=(state_size,)),
        tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu", **init),
        tf.keras.layers.Dense(num_actions, **init),
    ])


class DDQNAgent:
    EXT = ".keras"

    def __init__(self, state_size, num_actions, seed=0):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)
        self.q = build_q_network(state_size, num_actions)
        self.q_target = build_q_network(state_size, num_actions)
        self.q_target.set_weights(self.q.get_weights())
        self.optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
        self.learn_start = LEARN_START_OFFLINE
        self.replay = PrioritizedReplay()
        self.epsilon = EPSILON_START
        self.updates = 0
        self.last_loss = float("nan")

    # ---------------- acting ----------------
    def q_values(self, state):
        return self.q(state[None, :], training=False).numpy()[0]

    def act(self, state, greedy=False):
        """epsilon-greedy (Algorithm 2 lines 5-6), ties broken at random."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions)), True
        q = self.q_values(state)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best)), False

    def set_epsilon_by_progress(self, step, total=TRAINING_ITERATIONS):
        """Offline schedule: linear 0.3 -> 0.1 over the training run."""
        frac = min(1.0, step / max(1, total))
        self.epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)

    # ---------------- learning ----------------
    def remember(self, s, a, r, s_next, done):
        self.replay.store((s, a, r, s_next, float(done)))

    @tf.function
    def _train_step(self, s, a, r, s2, d, w):
        next_a = tf.argmax(self.q(s2, training=False), axis=1)
        next_q = tf.gather(self.q_target(s2, training=False), next_a,
                           batch_dims=1)
        y = r + GAMMA * (1.0 - d) * next_q
        with tf.GradientTape() as tape:
            q_sa = tf.gather(self.q(s, training=True), a, batch_dims=1)
            td = y - q_sa
            loss = tf.reduce_mean(w * tf.square(td))
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.q.trainable_variables))
        return td, loss

    def learn(self):
        """One minibatch update; returns False if the buffer is too small."""
        if len(self.replay) < max(BATCH_SIZE, self.learn_start):
            return False
        leaves, batch, w = self.replay.sample(BATCH_SIZE, self.rng)
        s, a, r, s2, d = (np.asarray(x) for x in zip(*batch))
        td, loss = self._train_step(
            tf.constant(s, tf.float32), tf.constant(a, tf.int64),
            tf.constant(r, tf.float32), tf.constant(s2, tf.float32),
            tf.constant(d, tf.float32), tf.constant(w))
        self.replay.update(leaves, td.numpy())
        self.updates += 1
        self.last_loss = float(loss)
        if self.updates % TARGET_UPDATE == 0:
            self.q_target.set_weights(self.q.get_weights())
        return True

    # ---------------- persistence ----------------
    def save(self, path):
        self.q.save(path)

    def load(self, path):
        # Copy weights into the existing network so the traced
        # _train_step keeps pointing at the live variables.
        self.q.set_weights(tf.keras.models.load_model(path).get_weights())
        self.q_target.set_weights(self.q.get_weights())
