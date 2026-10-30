from pathlib import Path
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("training_plots")
OUT.mkdir(exist_ok=True)

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def values(rows, column):
    return np.asarray(
        [float(row[column]) for row in rows if row.get(column, "")],
        dtype=float,
    )

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close()

def moving_average(data, window=10):
    if len(data) == 0:
        return data
    window = min(window, len(data))
    return np.convolve(data, np.ones(window) / window, mode="valid")

episodes = read_csv("episode_summary.csv")
steps = read_csv("training_state_log.csv")

if not episodes:
    raise SystemExit("No episode data found in episode_summary.csv")

ep = values(episodes, "Episode")
reward = values(episodes, "Total_Reward")

# 1. Episode reward and moving average
plt.figure(figsize=(9, 5))
plt.plot(ep, reward, alpha=0.55, label="Episode reward")
ma = moving_average(reward, 10)
if len(ma):
    plt.plot(ep[len(ep) - len(ma):], ma, linewidth=2,
             label=f"Moving average ({min(10, len(reward))} episodes)")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("DDQN Training Reward")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot("01_episode_reward.png")

# 2. Training loss
loss_path = Path("loss_history_fixed.npy")
if loss_path.exists():
    loss = np.load(loss_path)
    if len(loss):
        plt.figure(figsize=(9, 5))
        plt.plot(np.arange(1, len(loss) + 1), loss)
        plt.xlabel("Learning update")
        plt.ylabel("Selected-action TD loss")
        plt.title("DDQN Training Loss")
        plt.grid(True, alpha=0.3)
        save_plot("02_training_loss.png")

# 3. Epsilon schedule
epsilon = values(episodes, "Epsilon")
if len(epsilon) == len(ep):
    plt.figure(figsize=(9, 5))
    plt.plot(ep, epsilon, marker="o", markersize=3)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Rate During Training")
    plt.grid(True, alpha=0.3)
    save_plot("03_epsilon_decay.png")

# 4. Mean simulated power and temperature
power = values(episodes, "Average_Power_W")
temp = values(episodes, "Average_Temperature_C")

if len(power) == len(ep) and len(temp) == len(ep):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, power, label="Average power")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Simulated power (W)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ep, temp, linestyle="--", label="Average temperature")
    ax2.set_ylabel("Simulated temperature (°C)")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.title("Simulated Power and Temperature")
    save_plot("04_power_temperature.png")

# 5. Simulated GPU utilization and frequency
util = values(episodes, "Average_GPU_Utilization")
freq = values(episodes, "Average_Frequency_MHz")

if len(util) == len(ep) and len(freq) == len(ep):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, util, label="GPU utilization")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("GPU utilization (%)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ep, freq, linestyle="--", label="GPU frequency")
    ax2.set_ylabel("Frequency (MHz)")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.title("Simulated GPU Utilization and Frequency")
    save_plot("05_utilization_frequency.png")

print(f"Plots saved in: {OUT.resolve()}")
for file in sorted(OUT.glob("*.png")):
    print(f"{file.name}: {file.stat().st_size / 1024:.1f} KB")
