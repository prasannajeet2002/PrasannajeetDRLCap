"""Temperature trace of one program under the three policies (small multiples)."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_results import INK, BLUE, ORANGE, BG, style

GREY = "#6B6B6B"


def main(out, workload="lulesh", rep_jobs=("0", "1", "2")):
    series = {}
    with open(os.path.join(out, "trace.csv")) as f:
        for r in csv.DictReader(f):
            if r["workload"] == workload and r["job"] in rep_jobs and r["phase"] == "run":
                series.setdefault(r["policy"], []).append((float(r["t"]), float(r["temp"]), float(r["clock"])))
    order = [("baseline", "NVIDIA default", GREY), ("coord", "COORD", BLUE), ("drlcap", "DRLCap", ORANGE)]
    fig, axes = plt.subplots(3, 1, figsize=(9.45, 6.0), sharex=True, sharey=True, facecolor=BG)
    for ax, (key, label, color) in zip(axes, order):
        pts = series[key]
        t0 = pts[0][0]
        ax.plot([p[0] - t0 for p in pts], [p[1] for p in pts], color=color, lw=2)
        style(ax)
        ax.set_ylabel("°C", color=INK)
        ax.set_title(label, loc="left", color=INK, fontsize=11, weight="bold")
    axes[-1].set_xlabel("time in job (s)", color=INK)
    fig.suptitle(f"GPU temperature during {workload} (repeat 1)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"trace_{workload}.png"), dpi=150, facecolor=BG)
    print("wrote", f"trace_{workload}.png")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
