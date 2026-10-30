"""Robust SumTree-based Prioritized Experience Replay (PER)."""

import numpy as np


class SumTree:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")

        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = np.empty(self.capacity, dtype=object)
        self.data[:] = None
        self.data_pointer = 0
        self.size = 0

    @property
    def total_p(self):
        return float(self.tree[0])

    def add(self, priority, data):
        data_idx = self.data_pointer
        tree_idx = data_idx + self.capacity - 1

        self.data[data_idx] = np.asarray(data, dtype=np.float32).copy()
        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        priority = float(priority)
        if not np.isfinite(priority) or priority < 0:
            raise ValueError("Priority must be finite and non-negative")

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        if self.size == 0 or self.total_p <= 0:
            raise ValueError("Cannot sample an empty or zero-priority tree")

        value = float(np.clip(value, 0, np.nextafter(self.total_p, 0)))
        parent_idx = 0

        while True:
            left = 2 * parent_idx + 1
            right = left + 1

            if left >= len(self.tree):
                break

            if value < self.tree[left] or self.tree[right] == 0:
                parent_idx = left
            else:
                value -= self.tree[left]
                parent_idx = right

        data_idx = parent_idx - self.capacity + 1
        data = self.data[data_idx]

        if data is None:
            raise RuntimeError("SumTree selected an empty replay slot")

        return parent_idx, float(self.tree[parent_idx]), data


class Memory:
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.beta = float(type(self).beta)

    def store(self, transition):
        transition = np.asarray(transition, dtype=np.float32)
        if not np.all(np.isfinite(transition)):
            raise ValueError("Transition contains NaN or infinite values")

        occupied_priorities = self.tree.tree[
            self.tree.capacity - 1:
            self.tree.capacity - 1 + self.tree.size
        ]
        max_p = float(occupied_priorities.max()) if self.tree.size else 0.0
        if max_p <= 0:
            max_p = self.abs_err_upper

        self.tree.add(max_p, transition)

    def sample(self, n):
        if n < 1:
            raise ValueError("Batch size must be >= 1")
        if self.tree.size < n:
            raise ValueError(
                f"Need at least {n} transitions; buffer has {self.tree.size}"
            )
        if self.tree.total_p <= 0:
            raise ValueError("Total replay priority must be positive")

        # Compute minimum probability using occupied entries only.
        leaves = self.tree.tree[
            self.tree.capacity - 1:
            self.tree.capacity - 1 + self.tree.size
        ]
        positive = leaves[leaves > 0]
        if positive.size == 0:
            raise ValueError("No positive priorities in replay buffer")

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        min_prob = positive.min() / self.tree.total_p

        batch_indices = np.empty(n, dtype=np.int32)
        batch_memory = np.stack(
            [self.tree.data[0] for _ in range(n)]
        ).astype(np.float32)
        weights = np.empty((n, 1), dtype=np.float32)

        segment = self.tree.total_p / n

        for i in range(n):
            value = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, transition = self.tree.get_leaf(value)

            probability = priority / self.tree.total_p
            batch_indices[i] = idx
            batch_memory[i] = transition
            weights[i, 0] = (probability / min_prob) ** (-self.beta)

        return batch_indices, batch_memory, weights

    def batch_update(self, tree_indices, abs_errors):
        errors = np.asarray(abs_errors, dtype=np.float64).reshape(-1)
        if len(tree_indices) != len(errors):
            raise ValueError("tree_indices and abs_errors lengths differ")

        errors = np.nan_to_num(
            np.abs(errors), nan=self.abs_err_upper,
            posinf=self.abs_err_upper, neginf=self.abs_err_upper
        )
        priorities = np.power(
            np.minimum(errors + self.epsilon, self.abs_err_upper),
            self.alpha
        )

        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(idx), float(priority))
