"""Replay the kernel-timing estimate offline on a kbudget run.

  python kshield_replay.py results/kbudget_d5_<stamp> [--rule edge|stable]

Rebuilds KernelBudget's per-kernel reference durations from the recorded
streams (ktrace/job*/*.stream) and the controller's step log (steps.csv,
monotonic time and measured clock of every 300 ms step), with
  edge    the online rule of the run: bins inside the step after 30 ms
  stable  only steps whose clock equals that of the step before and after
          (robust to a timing error of up to one step between the CUPTI
          bins and the controller's steps; the run showed tens of ms)
and prints the end-of-run kernel slowdown estimate against the true one.
The streams of kbudget_d5_20261123_192028 are committed as ktrace.tar.gz;
extract it in the run folder first (tar xzf ktrace.tar.gz). Wall-clock bins are mapped to monotonic time with time.time() - time.monotonic()
taken now, valid while the machine has not rebooted since the run.
"""
import argparse
import bisect
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kshield as K  # noqa: E402


def replay(stream_dir, st, rule):
    off = time.time() - time.monotonic()
    t = [float(s["t"]) for s in st]
    f = [K.LEVEL * round(float(s["clock"]) / K.LEVEL) for s in st]
    kb = K.KernelBudget.__new__(K.KernelBudget)
    kb.total = defaultdict(lambda: [0, 0.0])
    kb.lev = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for b, name, n, s in K.KernelStream(stream_dir).poll():
        tb = b - off
        tt = kb.total[name]
        tt[0] += n
        tt[1] += s
        i = bisect.bisect_right(t, tb) - 1        # step from t[i] to t[i+1], clock f[i+1]
        if i < 1 or i + 2 >= len(t):
            continue
        if rule == "edge":
            ok = tb >= t[i] + K.SETTLE and tb + K.BIN + s / n <= t[i + 1]
        else:
            ok = abs(f[i] - f[i + 1]) < 1 and abs(f[i + 2] - f[i + 1]) < 1
        if ok:
            lv = kb.lev[name][f[i + 1]]
            lv[0] += n
            lv[1] += s
    tot = cov = bad = added = 0.0
    for name, (n, s) in kb.total.items():
        tot += s
        ref, unstable = kb._reference(name)
        if ref is None:
            continue
        cov += s
        bad += s if unstable else 0.0
        added += s - n * ref
    T = t[-1] - t[0]
    return 100 * added / (T - added), cov / tot, bad / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rule", default="stable", choices=["edge", "stable"])
    a = ap.parse_args()
    runs = list(csv.DictReader(open(os.path.join(a.out, "runs.csv"))))
    steps = defaultdict(list)
    for s in csv.DictReader(open(os.path.join(a.out, "steps.csv"))):
        steps[s["job"]].append(s)
    T0 = defaultdict(list)
    for r in runs:
        if r["policy"] == "baseline":
            T0[r["workload"]].append(float(r["time_s"]))
    err, rows = [], []
    for r in runs:
        d = os.path.join(a.out, "ktrace", f"job{r['job']}")
        if not os.path.isdir(d):
            continue
        k, cov, bad = replay(d, steps[r["job"]], a.rule)
        true = 100 * (float(r["time_s"]) / np.mean(T0[r["workload"]]) - 1)
        valid = cov >= 0.9 and bad <= 0.1
        rows.append((r["workload"], r["policy"], r["repeat"], true, k, cov, bad, valid, float(r["e2_est_pct"])))
        if valid:
            err.append(k - true)
    print(f"rule={a.rule}\nprogram     policy          rep  true  k_est  cover unstab valid  e2_est")
    for x in sorted(rows):
        print(f"{x[0]:11s} {x[1]:15s} {x[2]:>3s} {x[3]:5.1f} {x[4]:6.1f}  {x[5]:5.2f} {x[6]:5.2f} {int(x[7]):4d}  {x[8]:6.1f}")
    e = np.array(err)
    e2 = np.array([x[8] - x[3] for x in rows if x[7]])
    print(f"valid end-of-run estimates: {len(e)}/{len(rows)}  kernel MAE {np.abs(e).mean():.2f} bias {e.mean():+.2f}"
          f" | E2 on the same runs MAE {np.abs(e2).mean():.2f} bias {e2.mean():+.2f}")


if __name__ == "__main__":
    main()
