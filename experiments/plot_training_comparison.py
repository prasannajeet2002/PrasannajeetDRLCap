import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("outputs/run_100ep")
AWARE = Path("outputs/workload_aware_100ep")
OUT = Path("outputs/plots")
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def number(row, key):
    value = row.get(key, "")
    try:
        return float(value) if value not in ("", None) else np.nan
    except (TypeError, ValueError):
        return np.nan

def aggregate(path):
    rows = read_csv(path)
    episodes = sorted({int(r["episode"]) for r in rows})
    result = {key: [] for key in [
        "episode", "reward", "loss", "epsilon",
        "power_w_proxy", "temperature_c_proxy",
        "performance_proxy", "energy_j_proxy"
    ]}

    for ep in episodes:
        group = [r for r in rows if int(r["episode"]) == ep]
        result["episode"].append(ep)

        # Sum per-step rewards to obtain the episode return.
        result["reward"].append(sum(number(r, "reward") for r in group))

        # Average step-level quantities within each episode.
        for key in [
            "loss", "epsilon", "power_w_proxy",
            "temperature_c_proxy", "performance_proxy",
            "energy_j_proxy"
        ]:
            values = np.array([number(r, key) for r in group], dtype=float)
            result[key].append(
                float(np.nanmean(values)) if np.isfinite(values).any()
                else np.nan
            )

    return {k: np.array(v, dtype=float) for k, v in result.items()}

base = aggregate(BASE / "training_log.csv")
aware = aggregate(AWARE / "training_log.csv")

def plot_pair(key, title, ylabel, filename, smooth=1):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for data, label in [
        (base, "Baseline DDQN (5 features)"),
        (aware, "Workload-aware DDQN (11 features)")
    ]:
        x, y = data["episode"], data[key]
        ax.plot(x, y, alpha=0.25, linewidth=0.8)

        if smooth > 1 and len(y) >= smooth:
            smoothed = np.convolve(
                y, np.ones(smooth) / smooth, mode="valid"
            )
            ax.plot(
                x[smooth - 1:], smoothed,
                linewidth=2, label=f"{label} ({smooth}-episode average)"
            )
        else:
            ax.plot(x, y, linewidth=1.8, label=label)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)
    print("Saved:", OUT / filename)

plot_pair(
    "reward", "Training Episode Return",
    "Sum of rewards per episode", "episode_reward.png", smooth=10
)
plot_pair(
    "loss", "Mean DDQN Training Loss",
    "Mean recorded loss per episode", "training_loss.png", smooth=5
)
plot_pair(
    "epsilon", "Exploration Rate During Training",
    "Mean epsilon per episode", "epsilon_decay.png"
)
plot_pair(
    "energy_j_proxy", "Mean Per-Step Energy Proxy",
    "Energy proxy per step (not calibrated joules)",
    "energy_proxy.png", smooth=10
)
plot_pair(
    "performance_proxy", "Mean Performance Proxy",
    "Performance proxy per step", "performance_proxy.png", smooth=10
)
plot_pair(
    "temperature_c_proxy", "Mean Temperature Proxy",
    "Temperature proxy (°C)", "temperature_proxy.png", smooth=10
)

# Plot paired evaluation means and standard deviations across seeds.
eval_path = AWARE / "evaluation_state_comparison.csv"
rows = read_csv(eval_path)
policies = [
    "fixed_low", "fixed_mid", "fixed_high",
    "baseline_5_feature", "workload_aware_11_feature"
]
metrics = [
    ("reward", "Mean evaluation reward"),
    ("energy", "Energy proxy"),
    ("performance", "Performance proxy"),
    ("max_temp", "Maximum temperature proxy (°C)")
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (metric, title) in zip(axes.flat, metrics):
    means, sds = [], []
    for policy in policies:
        values = np.array([
            number(r, metric) for r in rows if r["policy"] == policy
        ], dtype=float)
        values = values[np.isfinite(values)]
        means.append(np.mean(values) if len(values) else np.nan)
        sds.append(np.std(values, ddof=1) if len(values) > 1 else 0)

    ax.bar(range(len(policies)), means, yerr=sds, capsize=3)
    ax.set_title(title)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(
        ["Low", "Mid", "High", "Baseline", "Workload-aware"],
        rotation=25, ha="right"
    )
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Evaluation Across Matched Workload Seeds (mean ± SD)")
fig.tight_layout()
fig.savefig(OUT / "evaluation_comparison.png", dpi=180)
plt.close(fig)
print("Saved:", OUT / "evaluation_comparison.png")
print("\nAll plots are in:", OUT)
