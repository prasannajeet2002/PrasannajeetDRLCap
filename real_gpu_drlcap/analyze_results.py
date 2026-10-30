import pandas as pd
import numpy as np


# ==========================================================
# BASELINE
# ==========================================================

baseline = pd.read_csv(
    "baseline_results.csv"
)

baseline_avg_power = baseline[
    "power_w"
].mean()

baseline_runtime = baseline[
    "iteration_time_s"
].sum()

baseline_energy = (
    baseline_avg_power
    *
    baseline_runtime
)


# ==========================================================
# DRLCAP
# ==========================================================

drl = pd.read_csv(
    "drlcap_results.csv"
)

drl_avg_power = drl[
    "power_w"
].mean()


# The workload time is not currently stored
# as a CSV column in the controller.
# For now use number of interactions as a
# normalized comparison.

drl_total_reward = drl[
    "reward"
].sum()

drl_avg_reward = drl[
    "reward"
].mean()


print()
print("=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

print()
print("BASELINE")
print(
    f"Average power : "
    f"{baseline_avg_power:.2f} W"
)

print(
    f"Runtime       : "
    f"{baseline_runtime:.3f} s"
)

print(
    f"Approx energy : "
    f"{baseline_energy:.3f} J"
)

print()
print("DRLCAP")

print(
    f"Average power : "
    f"{drl_avg_power:.2f} W"
)

print(
    f"Total reward  : "
    f"{drl_total_reward:.3f}"
)

print(
    f"Average reward: "
    f"{drl_avg_reward:.3f}"
)

print()
print("=" * 70)
