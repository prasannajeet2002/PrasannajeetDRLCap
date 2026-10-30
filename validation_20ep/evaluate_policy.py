import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from simulated_env import SimulatedGPUEnv

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "drlcap_model_fixed.keras"
OUTPUT_DIR = BASE_DIR / "policy_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

N_SEEDS = 20
BASE_SEED = 20261028
FIXED_CAP_ACTION = 4  # 200 W
EXPECTED_FEATURES = 7


def evaluate_episode(policy, seed):
    env = SimulatedGPUEnv(seed=int(seed))
    state = env.reset()

    total_reward = 0.0
    powers = []
    temperatures = []
    utilizations = []
    frequencies = []
    caps = []
    steps = 0
    thermal_violations = 0
    done = False

    while not done:
        if policy == "ddqn":
            q_values = model(
                np.asarray(state, dtype=np.float32)[None, :],
                training=False
            ).numpy()[0]
            action = int(np.argmax(q_values))
        else:
            action = FIXED_CAP_ACTION

        state, reward, done = env.step(action)
        metrics = env.get_metrics()

        total_reward += float(reward)
        powers.append(metrics["power"])
        temperatures.append(metrics["temperature"])
        utilizations.append(metrics["gpu_util"])
        frequencies.append(metrics["frequency"])
        caps.append(metrics["power_cap"])

        if metrics["temperature"] >= env.warning_temperature:
            thermal_violations += 1

        steps += 1

    # Energy proxy: power (W) accumulated across simulation steps.
    # This is not joules unless a physical duration per step is defined.
    energy_proxy = float(np.sum(powers))

    return {
        "Policy": policy,
        "Seed": int(seed),
        "Steps": steps,
        "Total_Reward": total_reward,
        "Average_Power_W": float(np.mean(powers)),
        "Energy_Proxy_W_steps": energy_proxy,
        "Average_Temperature_C": float(np.mean(temperatures)),
        "Maximum_Temperature_C": float(np.max(temperatures)),
        "Thermal_Warning_Steps": thermal_violations,
        "Average_GPU_Utilization": float(np.mean(utilizations)),
        "Average_Frequency_MHz": float(np.mean(frequencies)),
        "Average_Power_Cap_W": float(np.mean(caps)),
    }


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

if len(model.input_shape) != 2 or model.input_shape[-1] != EXPECTED_FEATURES:
    raise ValueError(
        f"Expected model input shape (batch, {EXPECTED_FEATURES}), "
        f"but found {model.input_shape}"
    )

if model.output_shape[-1] != 8:
    raise ValueError(
        f"Expected 8 action values, but found output shape {model.output_shape}"
    )

results = []

for i in range(N_SEEDS):
    seed = BASE_SEED + i

    for policy in ("ddqn", "fixed_200W"):
        row = evaluate_episode(policy, seed)
        results.append(row)
        print(
            f"Seed {seed} | {policy:10s} | "
            f"reward={row['Total_Reward']:8.3f} | "
            f"power={row['Average_Power_W']:7.2f} W | "
            f"max_temp={row['Maximum_Temperature_C']:6.2f} C | "
            f"steps={row['Steps']}"
        )

csv_path = OUTPUT_DIR / "episode_results.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

summary_path = OUTPUT_DIR / "summary.csv"
metric_names = [
    "Total_Reward",
    "Average_Power_W",
    "Energy_Proxy_W_steps",
    "Average_Temperature_C",
    "Maximum_Temperature_C",
    "Thermal_Warning_Steps",
    "Average_GPU_Utilization",
    "Average_Frequency_MHz",
    "Average_Power_Cap_W",
    "Steps",
]

with summary_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Policy", "Metric", "Mean", "Std"])
    for policy in ("ddqn", "fixed_200W"):
        subset = [r for r in results if r["Policy"] == policy]
        for metric in metric_names:
            values = np.asarray([r[metric] for r in subset], dtype=float)
            writer.writerow([
                policy,
                metric,
                f"{np.mean(values):.6f}",
                f"{np.std(values, ddof=1):.6f}" if len(values) > 1 else "0",
            ])

print("\nEvaluation completed.")
print(f"Episode results: {csv_path}")
print(f"Aggregate summary: {summary_path}")
print("Energy is reported as W·steps, not joules.")
