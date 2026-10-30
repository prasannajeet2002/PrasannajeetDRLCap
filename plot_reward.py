import numpy as np
import matplotlib.pyplot as plt

reward = np.load("reward_history.npy")

window = 20

moving_average = np.convolve(
    reward,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(12, 6))

plt.plot(
    reward,
    alpha=0.35,
    label="Episode Reward"
)

plt.plot(
    range(window - 1, len(reward)),
    moving_average,
    linewidth=3,
    label="20-Episode Moving Average"
)

plt.xlabel("Episode")
plt.ylabel("Accumulated Reward")
plt.title("DRLCAP Training Reward")

plt.grid(True)
plt.legend()

plt.savefig(
    "reward_learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()