"""Figures for the presentation: slowdown estimator and budget shield.

  python plot_budget.py

Reads results/sweep_20261122_142513/signals.csv and the two shield runs
(budget_d5_20261122_193157, kbudget_d5_20261123_192028); writes
figures/{estimator_scatter,shield_slowdown}.png in the style of
thermal_study/plot_final.py.
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "thermal_study"))
from plot_final import NAMES  # noqa: E402
from plot_results import INK, MUTED, ORANGE, BG, style  # noqa: E402

TEAL = "#2E7D6B"   # the shield; blue stays COORD's colour in the other charts
OUT = os.path.join(HERE, "figures")
RUNS = ["budget_d5_20261122_193157", "kbudget_d5_20261123_192028"]


def estimator_scatter():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "sweep_20261122_142513", "signals.csv"))))
    true = np.array([float(r["true_slowdown"]) for r in rows])
    est = np.array([float(r["e2_online_slowdown"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor=BG)
    x = np.array([-1, 27])
    ax.fill_between(x, x - 2, x + 2, color=TEAL, alpha=0.12, linewidth=0, label="within ±2 pp")
    ax.plot(x, x, color=MUTED, linestyle=":", linewidth=1.6)
    ax.scatter(true, est, s=70, color=TEAL, edgecolor=BG, linewidth=1.5, zorder=3,
               label="one program at one fixed cap")
    for r, t, e in zip(rows, true, est):
        if e - t < -3:
            ax.annotate(f"{NAMES[r['workload']]} {r['cap']}", (t, e), xytext=(-120 if t > 20 else 8, -18),
                        textcoords="offset points", color=INK, fontsize=13)
    style(ax)
    ax.set_xlim(-1, 27)
    ax.set_ylim(-1, 27)
    ax.set_xlabel("True slowdown (%)", color=INK, fontsize=15)
    ax.set_ylabel("Estimated online (%)", color=INK, fontsize=15)
    ax.legend(frameon=False, fontsize=14, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "estimator_scatter.png"), dpi=150, facecolor=BG)


def shield_slowdown():
    """Per-program mean slowdown, averaged over the two runs (3 repeats each)."""
    per = defaultdict(list)
    for run in RUNS:
        rows = list(csv.DictReader(open(os.path.join(HERE, "results", run, "runs.csv"))))
        t0 = defaultdict(list)
        for r in rows:
            if r["policy"] == "baseline":
                t0[r["workload"]].append(float(r["time_s"]))
        acc = defaultdict(list)
        for r in rows:
            if r["policy"] in ("drlcap", "drlcap_shield"):
                acc[(r["policy"], r["workload"])].append(float(r["time_s"]))
        for (p, w), ts in acc.items():
            per[(p, w)].append(100 * (np.mean(ts) / np.mean(t0[w]) - 1))
    progs = list(NAMES)
    fig, ax = plt.subplots(figsize=(12, 5.4), facecolor=BG)
    w = 0.38
    for i, (key, label, color) in enumerate([("drlcap", "DRLCap", ORANGE),
                                             ("drlcap_shield", "DRLCap + budget shield", TEAL)]):
        xs = [j + (i - 0.5) * w for j in range(len(progs))]
        ax.bar(xs, [np.mean(per[(key, p)]) for p in progs], w * 0.92, color=color, label=label)
    ax.axhline(5, color=INK, linestyle=":", linewidth=1.6)
    ax.text(len(progs) - 0.5, 5.3, "5 % budget", ha="right", va="bottom", color=INK, fontsize=14)
    style(ax)
    ax.set_xticks(range(len(progs)))
    ax.set_xticklabels([NAMES[p] for p in progs], color=INK, fontsize=14, rotation=20)
    ax.set_ylabel("Slowdown vs NVIDIA default (%)", color=INK, fontsize=15)
    ax.legend(frameon=False, fontsize=15, labelcolor=INK, ncol=2, loc="upper left")
    ax.set_ylim(-0.5, 11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "shield_slowdown.png"), dpi=150, facecolor=BG)
    for p in progs:
        print(f"{p:11s} drlcap {np.mean(per[('drlcap', p)]):5.1f}  shield {np.mean(per[('drlcap_shield', p)]):5.1f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    estimator_scatter()
    shield_slowdown()
    print("wrote", sorted(os.listdir(OUT)))
