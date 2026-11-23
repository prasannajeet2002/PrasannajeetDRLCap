"""Do NVML readings track real speed? (progress-signal ladder, steps 1-2)

  python signals_test.py results/sweep_YYYYmmdd_HHMMSS

For every program and fixed cap, the measured speed relative to NVIDIA
default is T_default / T_cap. A good progress signal falls in proportion:
signal_cap / signal_default should equal it (the GEEPAFS test, on our
RTX 4070 Ti). Candidates, each averaged over the run (300 ms samples):

  mem       memory-controller utilisation (GEEPAFS)
  util      GPU utilisation (what DRLCap's reward relies on)
  util_clk  util x clock (DRLCap-style proxy)
  e2        the E2 speed model used by the shield (shield.speed)

and, separately, E2's online slowdown estimate for the whole run against
the measured slowdown. Output: signals.csv and a printed summary.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shield import speed  # noqa: E402


def load(out):
    runs = {r["job"]: r for r in csv.DictReader(open(os.path.join(out, "runs.csv")))}
    samples = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(out, "trace.csv"))):
        if r["phase"] == "run" and r["job"] in runs:
            samples[r["job"]].append([float(r[k]) for k in ("t", "util", "mem_util", "clock")])
    return runs, samples


def main(out):
    runs, samples = load(out)
    rows = []
    for job, r in runs.items():
        a = np.array(samples.get(job, []))
        if len(a) < 3:
            continue
        dt = np.diff(a[:, 0], append=a[-1, 0] + 0.3)
        u, m, f = a[:, 1], a[:, 2], a[:, 3]
        s = np.array([speed(*x) for x in zip(f, u, m)])
        rows.append({"workload": r["workload"], "cap": r["cap_mhz"] or "default",
                     "time": float(r["time_s"]),
                     "mem": np.average(m, weights=dt), "util": np.average(u, weights=dt),
                     "util_clk": np.average(u * f, weights=dt),
                     "e2": np.average(s, weights=dt),
                     "e2_slow": 100 * (dt.sum() / (dt * s).sum() - 1)})
    # mean over repeats per (program, cap)
    g = defaultdict(list)
    for x in rows:
        g[(x["workload"], x["cap"])].append(x)
    mean = {k: {f: np.mean([x[f] for x in v]) for f in ("time", "mem", "util", "util_clk", "e2", "e2_slow")}
            for k, v in g.items()}
    table, errs = [], defaultdict(list)
    for (w, cap), v in sorted(mean.items()):
        if cap == "default" or (w, "default") not in mean:
            continue
        d = mean[(w, "default")]
        true_speed = d["time"] / v["time"]
        row = {"workload": w, "cap": cap, "true_slowdown": 100 * (1 / true_speed - 1),
               "e2_online_slowdown": v["e2_slow"]}
        for sig in ("mem", "util", "util_clk", "e2"):
            ratio = v[sig] / d[sig] if d[sig] > 0 else float("nan")
            row[f"{sig}_ratio"] = ratio
            # slowdown implied by the signal: 1/ratio - 1
            errs[sig].append(100 * (1 / ratio - 1) - row["true_slowdown"] if ratio > 0 else np.nan)
        row["true_speed"] = true_speed
        errs["e2_online"].append(row["e2_online_slowdown"] - row["true_slowdown"])
        table.append(row)
    if not table:
        print("not enough data yet")
        return
    with open(os.path.join(out, "signals.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} (program, cap) pairs")
    print(f"{'program':11s} {'cap':>5s} {'true%':>6s} {'speed':>6s} {'mem':>6s} {'util':>6s} {'u*clk':>6s} {'e2':>6s} {'e2 est%':>7s}")
    for r in table:
        print(f"{r['workload']:11s} {r['cap']:>5s} {r['true_slowdown']:6.1f} {r['true_speed']:6.3f} "
              f"{r['mem_ratio']:6.3f} {r['util_ratio']:6.3f} {r['util_clk_ratio']:6.3f} {r['e2_ratio']:6.3f} "
              f"{r['e2_online_slowdown']:7.1f}")
    print("\nslowdown implied by each signal minus true slowdown (pp)")
    for k, e in errs.items():
        e = np.array([x for x in e if np.isfinite(x)])
        print(f"  {k:10s} MAE {np.mean(abs(e)):5.2f}  bias {e.mean():+5.2f}  worst {e[np.argmax(abs(e))]:+6.1f}  (n={len(e)})")


if __name__ == "__main__":
    main(sys.argv[1])
