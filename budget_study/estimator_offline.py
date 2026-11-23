"""Offline test of online slowdown estimators on the f2 evaluation logs.

  python estimator_offline.py

An online estimator only sees the 300 ms samples (clock, core util, memory
util). It assumes each step does the work NVIDIA default would do in
dt * s, with s the speed relative to default:

  E0 clock     s = f / F_REF when the core is busy (util > 0), else 1
  E1 util      s = 1 / (b * F_REF / f + 1 - b),  b = util / 100
  E2 util-mem  as E1 with b = util/100 * (1 - g * mem/100), g fitted
               leaving the tested program out (no program sees itself)

Estimated slowdown = sum(dt) / sum(dt * s) - 1, compared with the measured
slowdown T / mean(T_default) - 1 of the same run. Runs: all agent policies
(DRLCap freq/power/hybrid, seed0, RL), main a + b, 10 programs x 10 repeats.
"""

import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = [os.path.join(HERE, "..", "drlcap_faithful", "results", d) for d in
         ("eval_main_a_f2_20261113_130904", "eval_main_b_f2_20261114_041412")]
F_REF = 2720.0  # median busy clock under NVIDIA default, all programs (wear run)
OUT = os.path.join(HERE, "out_offline")


def load():
    runs = []
    for ev in EVALS:
        meta = {r["job"]: r for r in csv.DictReader(open(os.path.join(ev, "runs.csv")))}
        base = defaultdict(list)
        for r in meta.values():
            if r["policy"] == "baseline":
                base[r["workload"]].append(float(r["time_s"]))
        steps = defaultdict(list)
        for r in csv.DictReader(open(os.path.join(ev, "steps.csv"))):
            steps[r["job"]].append((float(r["t"]), float(r["clock"]),
                                    float(r["util"]), float(r["mem_util"])))
        for job, s in steps.items():
            m = meta[job]
            a = np.array(s)
            dt = np.diff(a[:, 0], append=a[-1, 0] + 0.3)
            runs.append({"policy": m["policy"], "workload": m["workload"],
                         "true": 100 * (float(m["time_s"]) / np.mean(base[m["workload"]]) - 1),
                         "dt": dt, "f": np.maximum(a[:, 1], 1.0),
                         "u": a[:, 2] / 100, "m": a[:, 3] / 100})
    return runs


def estimate(run, kind, g=0.0):
    f, u, m, dt = run["f"], run["u"], run["m"], run["dt"]
    ratio = np.minimum(F_REF / f, 3.0)
    if kind == "E0":
        s = np.where(u > 0, 1 / ratio, 1.0)
    else:
        b = u * (1 - g * m) if kind == "E2" else u
        b = np.clip(b, 0, 1)
        s = 1 / (b * ratio + 1 - b)
    return 100 * (dt.sum() / (dt * s).sum() - 1)


def fit_g(runs):
    grid = np.linspace(0, 1, 21)
    err = [np.mean([abs(estimate(r, "E2", g) - r["true"]) for r in runs]) for g in grid]
    return grid[int(np.argmin(err))]


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load()
    progs = sorted({r["workload"] for r in runs})
    g_of = {p: fit_g([r for r in runs if r["workload"] != p]) for p in progs}
    for r in runs:
        r["E0"] = estimate(r, "E0")
        r["E1"] = estimate(r, "E1")
        r["E2"] = estimate(r, "E2", g_of[r["workload"]])
    with open(os.path.join(OUT, "runs_est.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "workload", "true", "E0", "E1", "E2"])
        for r in runs:
            w.writerow([r["policy"], r["workload"]] + [f"{r[k]:.2f}" for k in ("true", "E0", "E1", "E2")])

    print(f"{len(runs)} runs; g (leave-one-out) per program:",
          {p: round(g, 2) for p, g in g_of.items()})
    print("\nper program: mean true slowdown and mean estimate (pp), all agent policies")
    print(f"{'program':12s} {'true':>6s} {'E0':>6s} {'E1':>6s} {'E2':>6s}")
    for p in progs:
        rs = [r for r in runs if r["workload"] == p]
        print(f"{p:12s} " + " ".join(f"{np.mean([r[k] for r in rs]):6.1f}" for k in ("true", "E0", "E1", "E2")))
    print("\nerror per run (estimate - true, percentage points)")
    for k in ("E0", "E1", "E2"):
        e = np.array([r[k] - r["true"] for r in runs])
        prog_bias = [np.mean([r[k] - r["true"] for r in runs if r["workload"] == p]) for p in progs]
        print(f"{k}: MAE {np.mean(abs(e)):.2f}  bias {e.mean():+.2f}  |err|<=2pp {np.mean(abs(e) <= 2)*100:.0f}%  "
              f"worst program bias {max(prog_bias, key=abs):+.2f}")
    # budget classification: does the estimate tell over/under 5 %?
    for k in ("E0", "E1", "E2"):
        tp = sum(1 for r in runs if r["true"] > 5 and r[k] > 5)
        fn = sum(1 for r in runs if r["true"] > 5 and r[k] <= 5)
        fp = sum(1 for r in runs if r["true"] <= 5 and r[k] > 5)
        print(f"{k}: runs truly >5 %: {tp + fn}, detected {tp}, missed {fn}, false alarms {fp}")


if __name__ == "__main__":
    main()
