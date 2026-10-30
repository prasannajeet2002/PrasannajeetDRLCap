"""Figures for an evaluate.py run (paper Fig. 9 and Fig. 10 analogues).

    python report.py results/eval_main_<stamp>

Writes into the run directory:
  fig_per_workload_<policy>.png  energy saving / performance loss / EDP
                                 improvement per test program (Fig. 9)
  fig_policies.png               geomean of every policy (Fig. 10)
  table_geomean.md               the same numbers as a Markdown table
Values are read from summary.csv; nothing is recomputed here.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = [("energy_saving_pct", "Energy saving"),
           ("perf_loss_pct", "Performance loss"),
           ("edp_improvement_pct", "EDP improvement")]
COLORS = ["#2a7ab9", "#d9822b", "#3a9a5b"]


def load(run_dir):
    with open(os.path.join(run_dir, "summary.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, _ in METRICS:
            r[k] = float(r[k])
    return rows


def bars(ax, labels, series, title):
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    for i, ((_, name), values) in enumerate(zip(METRICS, series)):
        ax.bar(x + (i - 1) * w, values, w, label=name, color=COLORS[i])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("% vs. NVIDIA default")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)


def main(run_dir):
    rows = load(run_dir)
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    for policy in policies:
        sub = [r for r in rows if r["policy"] == policy]
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(sub)), 4))
        bars(ax, [r["workload"] for r in sub],
             [[r[k] for r in sub] for k, _ in METRICS], policy)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"fig_per_workload_{policy}.png"),
                    dpi=150)
        plt.close(fig)

    geo = [r for r in rows if r["workload"] == "GEOMEAN"]
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(geo)), 4))
    bars(ax, [r["policy"] for r in geo],
         [[r[k] for r in geo] for k, _ in METRICS], "Geomean over test programs")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_policies.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(run_dir, "table_geomean.md"), "w") as f:
        f.write("| Policy | Energy saving | Perf. loss | EDP improvement |\n")
        f.write("|---|---|---|---|\n")
        for r in geo:
            f.write(f"| {r['policy']} | {r['energy_saving_pct']:.1f} % | "
                    f"{r['perf_loss_pct']:.1f} % | "
                    f"{r['edp_improvement_pct']:.1f} % |\n")
    print(open(os.path.join(run_dir, "table_geomean.md")).read())


if __name__ == "__main__":
    main(sys.argv[1])
