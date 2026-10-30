import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# BASELINE
# =========================================================

baseline = pd.read_csv(
    "baseline_results.csv"
)


# =========================================================
# DRLCAP
# =========================================================

drl = pd.read_csv(
    "drlcap_results.csv"
)


# =========================================================
# PLOT 1 — BASELINE POWER
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    baseline["iteration"],
    baseline["power_w"]
)

plt.xlabel("Iteration")
plt.ylabel("Power (W)")

plt.title(
    "Baseline GPU Power Consumption"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "01_baseline_power.png",
    dpi=300
)

plt.close()


# =========================================================
# PLOT 2 — DRLCAP ACTIONS
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    drl["step"],
    drl["clock_mhz"],
    marker="o"
)

plt.xlabel("DRLCap Step")
plt.ylabel("Graphics Clock (MHz)")

plt.title(
    "DRLCap Frequency Control"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "02_drlcap_frequency.png",
    dpi=300
)

plt.close()


# =========================================================
# PLOT 3 — DRLCAP POWER
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    drl["step"],
    drl["power_w"],
    marker="o"
)

plt.xlabel("DRLCap Step")
plt.ylabel("Power (W)")

plt.title(
    "GPU Power Under DRLCap Control"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "03_drlcap_power.png",
    dpi=300
)

plt.close()


# =========================================================
# PLOT 4 — REWARD
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    drl["step"],
    drl["cumulative_reward"]
)

plt.xlabel("DRLCap Step")

plt.ylabel("Cumulative Reward")

plt.title(
    "DRLCap Cumulative Reward"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "04_cumulative_reward.png",
    dpi=300
)

plt.close()


print("Plots generated:")
print("01_baseline_power.png")
print("02_drlcap_frequency.png")
print("03_drlcap_power.png")
print("04_cumulative_reward.png")
