"""Charts for the presentation from a run_thermal.py results folder.

  python plot_results.py results/both_YYYYmmdd_HHMMSS

leak_power_temp.png  power vs die temperature during the fixed-clock burn
wear_ratio.png       modelled wear of COORD and DRLCap relative to default
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#14213D", "#5A6275", "#D9D3C7"
BLUE, ORANGE, BG = "#2B5F8A", "#A8442B", "#F3EFE6"


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=15)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def leak(out):
    with open(os.path.join(out, "trace.csv")) as f:
        rows = [r for r in csv.DictReader(f)
                if r["exp"] == "leak" and r["phase"] == "burn"]
    t0 = {}
    for r in rows:
        k = (r["cap_mhz"], r["job"])
        t0.setdefault(k, float(r["t"]))
    runs = defaultdict(list)
    for r in rows:
        k = (r["cap_mhz"], r["job"])
        if (float(r["util"]) >= 95 and float(r["t"]) - t0[k] >= 5
                and abs(float(r["clock"]) - float(r["cap_mhz"])) <= 1):
            runs[k].append((float(r["t"]) - t0[k], float(r["temp"]),
                            float(r["power"])))
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(BG)
    style(ax)
    colors = {"2100": BLUE, "2505": ORANGE}
    seen = set()
    for (cap, rep), pts in sorted(runs.items()):
        ref = [p for t, _, p in pts if t < 15]
        p0 = sum(ref) / len(ref)
        by_t = defaultdict(list)
        for _, temp, p in pts:
            by_t[temp].append(100 * (p / p0 - 1))
        ts = sorted(by_t)
        ps = [sum(by_t[t]) / len(by_t[t]) for t in ts]
        label = f"{cap} MHz cap ({p0:.0f} W at start)" if cap not in seen else None
        seen.add(cap)
        ax.plot(ts, ps, color=colors.get(cap, INK), linewidth=2.5,
                marker="o", markersize=5, label=label,
                linestyle="-" if rep == "0" else "--")
    ax.axvline(57.5, color=MUTED, linewidth=1.5, linestyle=":")
    ax.text(57.3, ax.get_ylim()[1] * 0.97, "fans start (0 → ~55 %)", ha="right",
            va="top", color=MUTED, fontsize=14)
    ax.set_xlabel("GPU temperature (°C)", color=INK, fontsize=17)
    ax.set_ylabel("Power rise at fixed clock (%)", color=INK, fontsize=17)
    ax.legend(frameon=False, fontsize=15, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "leak_power_temp.png"), facecolor=BG)
    print("wrote leak_power_temp.png")


def wear(out):
    path = os.path.join(out, "wear_summary.csv")
    if not os.path.exists(path):
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    metrics = [("arr_0.7", "Arrhenius wear\n(0.7 eV)"),
               ("cycles_ge1", "Thermal cycles\n≥ 1 °C"),
               ("cycles_ge3", "Thermal cycles\n≥ 3 °C"),
               ("cm_2.0", "Cycling damage\n(q = 2)")]
    g = {r["policy"]: r for r in rows if r["workload"] == "GEOMEAN"}
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(BG)
    style(ax)
    xs = range(len(metrics))
    w = 0.36
    for i, (pol, col, name) in enumerate([("coord", BLUE, "COORD"),
                                          ("drlcap", ORANGE, "DRLCap")]):
        if pol not in g:
            continue
        vals = [float(g[pol][m]) for m, _ in metrics]
        bars = ax.bar([x + (i - 0.5) * w for x in xs], vals, width=w - 0.03,
                      color=col, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", color=INK, fontsize=14)
    ax.axhline(1.0, color=INK, linewidth=1.5, linestyle=":")
    ax.text(len(metrics) - 0.5, 1.02, "NVIDIA default = 1.0", ha="right",
            va="bottom", color=MUTED, fontsize=13)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l for _, l in metrics], color=INK, fontsize=15)
    ax.set_ylabel("Ratio to NVIDIA default\n(geometric mean, 10 programs)",
                  color=INK, fontsize=16)
    ax.legend(frameon=False, fontsize=15, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "wear_ratio.png"), facecolor=BG)
    print("wrote wear_ratio.png")


if __name__ == "__main__":
    leak(sys.argv[1])
    wear(sys.argv[1])
