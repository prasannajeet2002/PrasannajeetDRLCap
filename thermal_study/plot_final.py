"""Figures for the final presentation (base-paper results and slowdown).

  python plot_final.py

Reads drlcap_faithful/results/final_f2/RESULTS_F2.md and out_existing/slowdown.csv,
writes figures_final/{summary,edp_per_program,slowdown_per_program}.png
"""

import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_results import INK, MUTED, BLUE, ORANGE, BG, style  # noqa: E402

GREY = "#7A7A7A"
HERE = os.path.dirname(os.path.abspath(__file__))
F2 = os.path.join(HERE, "..", "drlcap_faithful", "results", "final_f2", "RESULTS_F2.md")
OUT = os.path.join(HERE, "figures_final")
METHODS = [("drlcap", "DRLCap", ORANGE), ("coord", "COORD", BLUE), ("rl", "RL", GREY)]
NAMES = {"cloverleaf": "CloverLeaf", "comd": "CoMD", "googlenet": "GoogLeNet", "gromacs": "GROMACS",
         "lulesh": "LULESH", "minibude": "miniBUDE", "minife": "miniFE", "qe": "QE",
         "tealeaf": "TeaLeaf", "vgg16": "VGG16"}


def tables():
    """Main comparison and per-program EDP tables from RESULTS_F2.md."""
    text = open(F2).read()

    main = re.search(r"## Main comparison \(10 repeats\)\n\n(.*?)\n\n", text, re.S).group(1)
    edp = re.search(r"EDP improvement % per program\n\n(.*?)\n\n", text, re.S).group(1)

    def parse(block):
        rows = [[c.strip() for c in l.strip("|").split("|")] for l in block.splitlines() if not l.startswith("|---")]
        return rows[0], rows[1:]

    return parse(main), parse(edp)


def bar_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.4, fmt.format(h), ha="center", va="bottom",
                color=INK, fontsize=13)


def summary(main):
    head, rows = main
    vals = {r[0]: [float(x) for x in r[2:5]] for r in rows}
    metrics = ["Energy saved", "Performance loss", "EDP improved"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4), facecolor=BG)
    w = 0.26
    for i, (key, label, color) in enumerate(METHODS):
        xs = [j + (i - 1) * w for j in range(3)]
        bars = ax.bar(xs, vals[key], w * 0.92, color=color, label=label)
        bar_labels(ax, bars)
    style(ax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(metrics, color=INK, fontsize=16)
    ax.set_ylabel("% relative to NVIDIA default", color=INK, fontsize=15)
    ax.legend(frameon=False, fontsize=15, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "summary.png"), dpi=150, facecolor=BG)


def edp_per_program(edp):
    head, rows = edp
    col = {name: head.index(name) for name in ("drlcap", "coord", "rl")}
    progs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5.4), facecolor=BG)
    w = 0.27
    for i, (key, label, color) in enumerate(METHODS):
        xs = [j + (i - 1) * w for j in range(len(progs))]
        ax.bar(xs, [float(r[col[key]]) for r in rows], w * 0.92, color=color, label=label)
    style(ax)
    ax.set_xticks(range(len(progs)))
    ax.set_xticklabels([NAMES[p] for p in progs], color=INK, fontsize=14, rotation=20)
    ax.set_ylabel("EDP improvement (%)", color=INK, fontsize=15)
    ax.legend(frameon=False, fontsize=15, labelcolor=INK, ncol=3, loc="upper left")
    ax.set_ylim(0, 33)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "edp_per_program.png"), dpi=150, facecolor=BG)


def slowdown_per_program():
    data = {}
    with open(os.path.join(HERE, "out_existing", "slowdown.csv")) as f:
        for r in csv.DictReader(f):
            data.setdefault(r["policy"], {})[r["workload"]] = float(r["slowdown_pct"])
    progs = list(NAMES)
    fig, ax = plt.subplots(figsize=(12, 5.4), facecolor=BG)
    w = 0.27
    for i, (key, label, color) in enumerate(METHODS):
        xs = [j + (i - 1) * w for j in range(len(progs))]
        ax.bar(xs, [data[key][p] for p in progs], w * 0.92, color=color, label=label)
    ax.axhline(5, color=INK, linestyle=":", linewidth=1.6)
    ax.text(len(progs) - 0.5, 5.3, "5 % budget", ha="right", va="bottom", color=INK, fontsize=14)
    style(ax)
    ax.set_xticks(range(len(progs)))
    ax.set_xticklabels([NAMES[p] for p in progs], color=INK, fontsize=14, rotation=20)
    ax.set_ylabel("Slowdown vs NVIDIA default (%)", color=INK, fontsize=15)
    ax.legend(frameon=False, fontsize=15, labelcolor=INK, ncol=3, loc="upper left")
    ax.set_ylim(0, 14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slowdown_per_program.png"), dpi=150, facecolor=BG)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main, edp = tables()
    summary(main)
    edp_per_program(edp)
    slowdown_per_program()
    print("wrote", sorted(os.listdir(OUT)))
