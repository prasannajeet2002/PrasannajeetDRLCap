import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import tensorflow as tf

from sim_train import Config as BaseConfig, SimulatedGPUEnv, run_fixed
from sim_train import DDQNAgent as BaseAgent
from sim_train_workload_aware import DDQNAgent as AwareAgent, SimulatedGPUEnv as AwareEnv


BASE = Path("outputs/run_100ep")
AWARE = Path("outputs/workload_aware_100ep")
N_SEEDS = 30


def evaluate_greedy(model_dir, aware, seed, cfg):
    EnvClass = AwareEnv if aware else SimulatedGPUEnv
    env = EnvClass(seed, cfg.steps, cfg.clocks)
    state = env.reset()
    model = tf.keras.models.load_model(
        model_dir / "ddqn_per.keras", compile=False
    )

    total_reward = 0.0
    total_energy = 0.0
    total_performance = 0.0
    max_temp = env.temp
    actions = []

    done = False
    while not done:
        q = model(np.asarray(state, np.float32)[None, :], training=False)
        action = int(tf.argmax(q[0]).numpy())
        actions.append(action)

        state, reward, done, info = env.step(action)
        total_reward += reward
        total_energy += info["energy_j_proxy"]
        total_performance += info["performance_proxy"] / cfg.steps
        max_temp = max(max_temp, info["temperature_c_proxy"])

    return {
        "reward": total_reward,
        "energy": total_energy,
        "performance": total_performance,
        "max_temp": max_temp,
        "actions": actions,
    }


def main():
    with open(BASE / "config.json") as f:
        cfg_data = json.load(f)

    cfg = BaseConfig(**{
        k: tuple(v) if k == "clocks" else v
        for k, v in cfg_data.items()
        if k in BaseConfig.__dataclass_fields__
    })

    records = []
    for i in range(N_SEEDS):
        seed = cfg.seed + 1000 + i

        policies = {
            "fixed_low": run_fixed(cfg, 0, seed),
            "fixed_mid": run_fixed(cfg, 2, seed),
            "fixed_high": run_fixed(cfg, 5, seed),
            "baseline_5_feature": evaluate_greedy(BASE, False, seed, cfg),
            "workload_aware_11_feature": evaluate_greedy(AWARE, True, seed, cfg),
        }

        for name, result in policies.items():
            records.append({
                "seed": seed,
                "policy": name,
                **{k: v for k, v in result.items() if k != "actions"},
            })

    out = AWARE / "evaluation_state_comparison.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    metrics = ["reward", "energy", "performance", "max_temp"]
    by_seed = {}
    for row in records:
        by_seed.setdefault(row["seed"], {})[row["policy"]] = row

    print(f"\nPaired evaluation over {N_SEEDS} seeds")
    print("All policies use identical workload seeds.\n")
    print(f"{'Policy':30s} {'Reward':>10s} {'Energy':>12s} "
          f"{'Performance':>12s} {'Max temp':>10s}")

    policies = [
        "fixed_low", "fixed_mid", "fixed_high",
        "baseline_5_feature", "workload_aware_11_feature",
    ]

    for policy in policies:
        subset = [r for r in records if r["policy"] == policy]
        print(
            f"{policy:30s} "
            f"{mean(r['reward'] for r in subset):10.3f} "
            f"{mean(r['energy'] for r in subset):12.2f} "
            f"{mean(r['performance'] for r in subset):12.4f} "
            f"{mean(r['max_temp'] for r in subset):10.2f}"
        )

    print("\nPaired workload-aware minus baseline differences:")
    for metric in metrics:
        diffs = [
            by_seed[s]["workload_aware_11_feature"][metric]
            - by_seed[s]["baseline_5_feature"][metric]
            for s in sorted(by_seed)
        ]
        print(
            f"{metric:12s}: mean difference = {mean(diffs):.4f}, "
            f"SD = {stdev(diffs):.4f}"
        )

    print(f"\nSaved detailed results to {out}")


if __name__ == "__main__":
    main()
