import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf

from sim_train import Config, SimulatedGPUEnv

OUT = Path("outputs")
N_SEEDS = 30

# Restore the configuration used by the existing training run.
with open(OUT / "config.json") as f:
    saved_cfg = json.load(f)

cfg = Config(
    seed=saved_cfg["seed"],
    steps=saved_cfg["steps"],
    clocks=tuple(saved_cfg["clocks"]),
)

model_path = OUT / "ddqn_per.keras"
model = tf.keras.models.load_model(model_path, compile=False)

def evaluate(policy, seed):
    env = SimulatedGPUEnv(seed, cfg.steps, cfg.clocks)
    state = env.reset()

    totals = {
        "reward": 0.0,
        "energy": 0.0,
        "performance": 0.0,
        "max_temp": float(env.temp),
    }
    actions = Counter()

    done = False
    while not done:
        if policy == "fixed_low":
            action = 0
        elif policy == "fixed_mid":
            action = len(cfg.clocks) // 2
        elif policy == "fixed_high":
            action = len(cfg.clocks) - 1
        else:
            # Greedy evaluation: no epsilon exploration.
            q_values = model(
                np.asarray(state, dtype=np.float32)[None, :],
                training=False,
            )
            action = int(tf.argmax(q_values[0]).numpy())

        actions[action] += 1
        state, reward, done, info = env.step(action)

        totals["reward"] += reward
        totals["energy"] += info["energy_j_proxy"]
        totals["performance"] += (
            info["performance_proxy"] / cfg.steps
        )
        totals["max_temp"] = max(
            totals["max_temp"], info["temperature_c_proxy"]
        )

    return totals, actions

policies = [
    "fixed_low",
    "fixed_mid",
    "fixed_high",
    "learned_ddqn_per",
]

results = {p: [] for p in policies}
learned_actions = Counter()

# Every policy sees the same seed/workload sequence.
for i in range(N_SEEDS):
    seed = cfg.seed + 1000 + i

    for policy in policies:
        metrics, actions = evaluate(policy, seed)
        results[policy].append(metrics)

        if policy == "learned_ddqn_per":
            learned_actions.update(actions)

fields = ["policy"]
metric_names = ["reward", "energy", "performance", "max_temp"]

with open(OUT / "evaluation_paired.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields + [
            f"{m}_{stat}"
            for m in metric_names
            for stat in ("mean", "std")
        ],
    )
    writer.writeheader()

    for policy in policies:
        row = {"policy": policy}
        for metric in metric_names:
            values = np.asarray(
                [x[metric] for x in results[policy]], dtype=float
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        writer.writerow(row)

print(f"\nEvaluation across {N_SEEDS} paired workload seeds")
print("All energy and performance values are synthetic proxies.\n")

for policy in policies:
    print(f"\n{policy}")
    for metric in metric_names:
        values = np.asarray(
            [x[metric] for x in results[policy]], dtype=float
        )
        print(
            f"  {metric:12s}: "
            f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"
        )

total_actions = sum(learned_actions.values())
print("\nLearned greedy policy action distribution:")
for action in range(len(cfg.clocks)):
    count = learned_actions[action]
    fraction = count / total_actions if total_actions else 0.0
    print(
        f"  action {action} ({cfg.clocks[action]} MHz): "
        f"{count} ({fraction:.1%})"
    )

print(f"\nSaved: {OUT / 'evaluation_paired.csv'}")
