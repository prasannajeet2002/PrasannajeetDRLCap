"""Analyse kernel_sweep.py: tracer overhead and kernel-based slowdown estimates.

  python kernel_test.py results/ksweep_YYYYmmdd_HHMMSS

(ksweep_20261123_112830 keeps its per-kernel files in ktrace.tar.gz: tar xzf it first.)

Per program: overhead = traced default / plain default - 1. Per (program,
cap), true slowdown = T_cap / T_traced_default - 1 (both traced, so the
overhead cancels), against three estimates:

  k_oracle  added kernel time sum_k n_k (d_k,cap - d_k,default) using each
            kernel's mean duration in the traced default runs; slowdown =
            added / (T_cap - added). Needs a default run: an upper bound.
  e2        the shield's estimator (NVML util, mem, clock), as in signals_test.
  e2k       E2 with NVML util rescaled so its run average equals the
            measured kernel-busy fraction (kernel ns / wall time). The per-
            sample NVML pattern keeps the phases (a constant busy fraction
            would spread idle phases over the run); needs no reference run.
Output: kernels.csv and a printed summary.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shield as S  # noqa: E402


def kernels(out, job):
    """name -> [count, sum_ns] over all processes of one job."""
    k = defaultdict(lambda: [0, 0])
    for p in glob.glob(os.path.join(out, "ktrace", f"job{job}", "*_kernels.csv")):
        for r in csv.DictReader(open(p)):
            k[r["name"]][0] += int(r["count"])
            k[r["name"]][1] += int(r["sum_ns"])
    return k


def e2(a, busy=None):
    dt = np.diff(a[:, 0], append=a[-1, 0] + 0.3)
    u = a[:, 1]
    if busy is not None:
        u = np.clip(u * 100.0 * busy / max(np.average(u, weights=dt), 1e-9), 0.0, 100.0)
    s = np.array([S.speed(f, x, m) for x, m, f in zip(u, a[:, 2], a[:, 3])])
    return 100 * (dt.sum() / (dt * s).sum() - 1)


def main(out):
    runs = list(csv.DictReader(open(os.path.join(out, "runs.csv"))))
    smp = defaultdict(list)
    jobs = {r["job"] for r in runs}
    for r in csv.DictReader(open(os.path.join(out, "trace.csv"))):
        if r["phase"] == "run" and r["job"] in jobs:
            smp[r["job"]].append([float(r[k]) for k in ("t", "util", "mem_util", "clock")])
    by = defaultdict(list)
    for r in runs:
        by[(r["workload"], r["mode"], r["cap_mhz"] or "default")].append(r)
    progs = sorted({r["workload"] for r in runs})
    print("tracer overhead (traced vs plain NVIDIA default, mean of repeats)")
    ovh = []
    for w in progs:
        tp = [float(r["time_s"]) for r in by[(w, "plain", "default")]]
        tt = [float(r["time_s"]) for r in by[(w, "traced", "default")]]
        if tp and tt:
            o = 100 * (np.mean(tt) / np.mean(tp) - 1)
            ovh.append(o)
            print(f"  {w:11s} {o:+5.1f}%   plain spread {100*(max(tp)-min(tp))/np.mean(tp):.1f}%")
    if ovh:
        print(f"  mean {np.mean(ovh):+.1f}%")
    table, err = [], defaultdict(list)
    for w in progs:
        ref = by[(w, "traced", "default")]
        if not ref:
            continue
        T0 = np.mean([float(r["time_s"]) for r in ref])
        dref = defaultdict(lambda: [0, 0])
        for r in ref:
            for n, (c, s) in kernels(out, r["job"]).items():
                dref[n][0] += c
                dref[n][1] += s
        caps = sorted({k[2] for k in by if k[0] == w and k[1] == "traced" and k[2] != "default"}, key=float)
        for cap in caps:
            for r in by[(w, "traced", cap)]:
                T = float(r["time_s"])
                kc = kernels(out, r["job"])
                added = sum(c * (s / c - dref[n][1] / dref[n][0]) for n, (c, s) in kc.items()
                            if c and dref[n][0]) / 1e9
                busy = sum(s for c, s in kc.values()) / 1e9 / T
                a = np.array(smp[r["job"]])
                row = {"workload": w, "cap": cap, "repeat": r["repeat"],
                       "true": 100 * (T / T0 - 1), "k_oracle": 100 * added / (T - added),
                       "e2": e2(a), "e2k": e2(a, busy), "kernel_busy": busy}
                for k in ("k_oracle", "e2", "e2k"):
                    err[k].append(row[k] - row["true"])
                table.append(row)
    if not table:
        print("no traced cap runs yet")
        return
    with open(os.path.join(out, "kernels.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)
    print("\nprogram      cap  rep   true  k_orac     e2    e2k  busy")
    for x in table:
        print(f"{x['workload']:11s} {x['cap']:>5s}  {x['repeat']:>3s} {x['true']:6.1f}  {x['k_oracle']:6.1f} "
              f"{x['e2']:6.1f} {x['e2k']:6.1f}  {x['kernel_busy']:.2f}")
    band = [i for i, x in enumerate(table) if 2 <= x["true"] <= 8]
    print("\nestimate minus true slowdown (pp)")
    for k, d in err.items():
        d = np.array(d)
        b = d[band]
        print(f"  {k:9s} MAE {np.abs(d).mean():5.2f} bias {d.mean():+5.2f} worst {d[np.argmax(abs(d))]:+6.1f}"
              f"  | band 2-8 %: n={len(b)} MAE {np.abs(b).mean() if len(b) else float('nan'):.2f}"
              f" under >1pp {int((b < -1).sum())}")


if __name__ == "__main__":
    main(sys.argv[1])
