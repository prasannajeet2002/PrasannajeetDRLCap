"""Extra statistics for an evaluate.py run, read from runs.csv.

    python analyze_eval.py results/eval_main_<stamp>

Prints, for every policy:
  - geomean energy saving / performance loss / EDP improvement over the
    Rodinia test programs only and over the CNN programs only
    (summary.csv's GEOMEAN mixes both groups);
  - a 95 % bootstrap interval of the all-program geomean, resampling the
    repeats of every program independently, so differences between
    policies can be judged against run-to-run noise.
Writes the same numbers to analysis.md in the run directory.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

CNN = {"vgg16", "googlenet", "resnet50"}
BOOT = 2000


def load(run_dir):
    runs = defaultdict(lambda: defaultdict(list))   # policy -> wl -> [(t,E)]
    with open(os.path.join(run_dir, "runs.csv")) as f:
        for r in csv.DictReader(f):
            runs[r["policy"]][r["workload"]].append(
                (float(r["time_s"]), float(r["energy_j"])))
    return runs


def ratios(pol, base, workloads):
    """Geomean (energy, time, edp) ratios vs. baseline over workloads."""
    e, t = [], []
    for w in workloads:
        pt, pe = np.mean(pol[w], axis=0)
        bt, be = np.mean(base[w], axis=0)
        e.append(pe / be)
        t.append(pt / bt)
    e, t = np.exp(np.mean(np.log(e))), np.exp(np.mean(np.log(t)))
    return e, t, e * t


def pct(e, t, edp):
    return 100 * (1 - e), 100 * (t - 1), 100 * (1 - edp)


def main(run_dir):
    runs = load(run_dir)
    base = runs["baseline"]
    wls = list(base)
    groups = {"rodinia": [w for w in wls if w not in CNN],
              "cnn": [w for w in wls if w in CNN]}
    rng = np.random.default_rng(0)
    lines = ["| Policy | Rodinia E / slow / EDP | CNN E / slow / EDP | "
             "all: EDP impr. 95 % CI | all: energy saving 95 % CI |",
             "|---|---|---|---|---|"]
    for p in runs:
        if p == "baseline":
            continue
        cells = []
        for g in ("rodinia", "cnn"):
            s = pct(*ratios(runs[p], base, groups[g]))
            cells.append(" / ".join(f"{v:.1f}" for v in s))
        boot = []
        for _ in range(BOOT):
            rp = {w: [v[i] for i in rng.integers(len(v), size=len(v))]
                  for w, v in runs[p].items()}
            rb = {w: [v[i] for i in rng.integers(len(v), size=len(v))]
                  for w, v in base.items()}
            boot.append(pct(*ratios(rp, rb, wls)))
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        cells.append(f"{lo[2]:.1f} – {hi[2]:.1f}")
        cells.append(f"{lo[0]:.1f} – {hi[0]:.1f}")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "analysis.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
