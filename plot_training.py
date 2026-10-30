import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

reward = np.load("reward_history.npy")

power = np.load("power_history.npy")

temperature = np.load(
    "temperature_history.npy"
)

frequency = np.load(
    "frequency_history.npy"
)

gpu_util = np.load(
    "gpu_util_history.npy"
)

memory_util = np.load(
    "memory_util_history.npy"
)


# ============================================================
# MOVING AVERAGE
# ============================================================

window = 20

if len(reward) >= window:

    moving_reward = np.convolve(
        reward,
        np.ones(window) / window,
        mode="valid"
    )

else:

    moving_reward = reward


# ============================================================
# REWARD PLOT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    reward,
    alpha=0.30,
    label="Episode Reward"
)

if len(reward) >= window:

    plt.plot(
        range(
            window - 1,
            len(reward)
        ),
        moving_reward,
        linewidth=3,
        label="20-Episode Moving Average"
    )

plt.xlabel("Episode")

plt.ylabel("Accumulated Reward")

plt.title(
    "DRLCAP Training Reward"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "reward_learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# POWER
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    power,
    linewidth=2
)

plt.xlabel("Episode")

plt.ylabel("Average Power (W)")

plt.title(
    "Average GPU Power During Training"
)

plt.grid(True)

plt.savefig(
    "power_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TEMPERATURE
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    temperature,
    linewidth=2
)

plt.axhline(
    75,
    linestyle="--",
    label="Target Temperature"
)

plt.axhline(
    83,
    linestyle="--",
    label="Warning Temperature"
)

plt.xlabel("Episode")

plt.ylabel("Temperature (°C)")

plt.title(
    "GPU Temperature During Training"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "temperature_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FREQUENCY
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    frequency,
    linewidth=2
)

plt.xlabel("Episode")

plt.ylabel("GPU Frequency (MHz)")

plt.title(
    "Average GPU Frequency During Training"
)

plt.grid(True)

plt.savefig(
    "frequency_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# UTILIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    gpu_util,
    label="GPU Utilization"
)

plt.plot(
    memory_util,
    label="Memory Utilization"
)

plt.xlabel("Episode")

plt.ylabel("Utilization (%)")

plt.title(
    "GPU and Memory Utilization"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "utilization_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "All plots generated successfully."
)
