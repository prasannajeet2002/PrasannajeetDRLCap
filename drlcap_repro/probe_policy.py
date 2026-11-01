"""How state-dependent is a trained DRLCap Q-network?

    python probe_policy.py results/train_freq_s0_<stamp> [more train dirs]

Feeds N random telemetry samples (uniform over the 4070 Ti's ranges)
through drlcap.to_state and the saved network and prints, per model:
how many different greedy actions occur, the most common ones, the
median spread of Q over actions, and the median gap between the best and
second-best action. A policy that adapts to the workload should pick
different actions for different states.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import sys

import numpy as np
import tensorflow as tf

import drlcap as D

N = 2000

# telemetry typical of each kind of phase on the 4070 Ti (training sweep);
# clock = the clock the GPU would run at under a mid (2200 MHz) cap
TYPICAL = [
    ("idle", dict(util=0.0, mem_util=0.0, power=25.0, clock=210.0)),
    ("host-bound", dict(util=5.0, mem_util=0.0, power=40.0, clock=2200.0)),
    ("compute-bound", dict(util=95.0, mem_util=0.0, power=60.0,
                           clock=2200.0)),
    ("memory-heavy", dict(util=100.0, mem_util=60.0, power=120.0,
                          clock=2200.0)),
]


def main(dirs):
    rng = np.random.default_rng(0)
    states = np.array([D.to_state({
        "util": rng.uniform(0, 100), "mem_util": rng.uniform(0, 100),
        "power": rng.uniform(20, 285), "clock": rng.uniform(210, 2800),
        "temp": rng.uniform(30, 90), "energy_j": 0.0}) for _ in range(N)],
        dtype=np.float32)
    for d in dirs:
        with open(os.path.join(d, "config.json")) as f:
            variant = json.load(f).get("variant", "freq")
        actions = D.VARIANTS[variant]
        model = tf.keras.models.load_model(os.path.join(d, "model_final.keras"))
        q = model.predict(states, verbose=0)
        best, counts = np.unique(q.argmax(1), return_counts=True)
        top = sorted(zip(counts, best), reverse=True)[:3]
        qs = np.sort(q, axis=1)
        print(f"{d} ({variant}): {len(best)} distinct greedy actions; top "
              + ", ".join(f"{actions[a]} {100 * c / N:.1f} %" for c, a in top)
              + f"; median Q spread {np.median(qs[:, -1] - qs[:, 0]):.2f}"
              f", median best-second gap {np.median(qs[:, -1] - qs[:, -2]):.2f}")
        typ = np.array([D.to_state(dict(x, temp=50.0, energy_j=0.0))
                        for _, x in TYPICAL], dtype=np.float32)
        a = model.predict(typ, verbose=0).argmax(1)
        print("    typical states: " + ", ".join(
            f"{name} -> {D.primary(actions[i])}"
            for (name, _), i in zip(TYPICAL, a)))


if __name__ == "__main__":
    main(sys.argv[1:])
