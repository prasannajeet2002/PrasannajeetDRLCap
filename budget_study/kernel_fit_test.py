"""Can per-kernel default durations be inferred without a default run?

  python kernel_fit_test.py results/ksweep_YYYYmmdd_HHMMSS

For each kernel, d(f) = a + c / f (a clock-independent part plus a part
that scales with 1/clock, the same model as E2 but per kernel). a and c are
fitted from the kernel's mean durations at two fixed caps only (the caps an
online controller would visit anyway), then d at the default clock is
extrapolated. The slowdown of every capped run is estimated as in
kernel_test.py's k_oracle but with these extrapolated references.
f is each run's mean busy clock from NVML (util > 0).
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_test import e2, kernels  # noqa: E402

out = sys.argv[1]
runs = list(csv.DictReader(open(os.path.join(out, "runs.csv"))))
clk, smp = defaultdict(list), defaultdict(list)
ids = {r["job"] for r in runs}
for r in csv.DictReader(open(os.path.join(out, "trace.csv"))):
    if r["phase"] == "run" and r["job"] in ids:
        smp[r["job"]].append([float(r[k]) for k in ("t", "util", "mem_util", "clock")])
        if float(r["util"]) > 0:
            clk[r["job"]].append(float(r["clock"]))
e2s = {j: e2(np.array(a)) for j, a in smp.items()}
traced = [r for r in runs if r["mode"] == "traced"]
progs = sorted({r["workload"] for r in traced})
pairs = [("2370", "2505"), ("2100", "2505"), ("2100", "2370")]
err = defaultdict(list)
print("program     fit-caps   cap   true  k_fit  k_oracle")
for w in progs:
    rs = [r for r in traced if r["workload"] == w]
    by = defaultdict(list)
    for r in rs:
        by[r["cap_mhz"] or "default"].append(r)
    T0 = np.mean([float(r["time_s"]) for r in by["default"]])
    f = {c: np.mean([np.mean(clk[r["job"]]) for r in v]) for c, v in by.items()}
    mean_d = {}
    for c, v in by.items():
        acc = defaultdict(lambda: [0, 0])
        for r in v:
            for n, (k, s) in kernels(out, r["job"]).items():
                acc[n][0] += k
                acc[n][1] += s
        mean_d[c] = {n: s / k for n, (k, s) in acc.items() if k}
    for A, B in pairs:
        ref, clipped, tot = {}, 0.0, 0.0
        for n in mean_d[A].keys() & mean_d[B].keys():
            dA, dB = mean_d[A][n], mean_d[B][n]
            c = (dA - dB) / (1 / f[A] - 1 / f[B])
            tot += dB
            clipped += dB if c < 0 else 0.0
            c = max(c, 0.0)                       # duration cannot fall with lower clock
            a = max(dB - c / f[B], 0.0)
            ref[n] = a + c / f["default"]
        for cap in ("2100", "2370", "2505"):
            for r in by[cap]:
                T = float(r["time_s"])
                kc = kernels(out, r["job"])
                added = sum(k * (s / k - ref[n]) for n, (k, s) in kc.items() if k and n in ref) / 1e9
                orac = sum(k * (s / k - mean_d["default"][n]) for n, (k, s) in kc.items()
                           if k and n in mean_d["default"]) / 1e9
                true = 100 * (T / T0 - 1)
                est = 100 * added / (T - added)
                err[(A, B)].append((est - true, 2 <= true <= 8))
                # hybrid: fall back to E2 when >10 % of kernel time has a
                # duration that fell with the clock (unstable kernel mix)
                hyb = est if clipped / tot <= 0.10 else e2s[r["job"]]
                err[(A, B, "hybrid")].append((hyb - true, 2 <= true <= 8))
                if r["repeat"] == "0":
                    print(f"{w:11s} {A}/{B}  {cap}  {true:5.1f}  {est:5.1f}  {100*orac/(T-orac):5.1f}")
print("\nk_fit minus true slowdown (pp)")
for p, v in err.items():
    d = np.array([x for x, _ in v]); b = np.array([x for x, inb in v if inb])
    print(f"  {' '.join(p[2:]) or 'fit':6s} {p[0]}/{p[1]}: MAE {np.abs(d).mean():.2f} bias {d.mean():+.2f} worst {d[np.argmax(abs(d))]:+.1f}"
          f" | band 2-8 %: n={len(b)} MAE {np.abs(b).mean():.2f} under >1pp {int((b < -1).sum())}")
