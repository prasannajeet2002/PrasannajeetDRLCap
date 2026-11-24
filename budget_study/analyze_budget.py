"""Summarise a shield experiment: python analyze_budget.py results/budget_d5_<stamp>

Per program and policy: slowdown and energy saving vs NVIDIA default (mean of
repeats), budget violations per run, and the shield's estimated vs true
slowdown. Geomeans use the paper's method (ratios to NVIDIA default).
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

out = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(out, "runs.csv"))))
delta = float(rows[0]["delta"])
by = defaultdict(list)
for r in rows:
    by[(r["workload"], r["policy"])].append(r)
progs = sorted({r["workload"] for r in rows})
pols = [p for p in ("drlcap", "drlcap_shield", "lowest_safe", "drlcap_kshield", "lowest_safe_k")
        if any(k[1] == p for k in by)]
T0 = {w: np.mean([float(r["time_s"]) for r in by[(w, "baseline")]]) for w in progs}
E0 = {w: np.mean([float(r["energy_j"]) for r in by[(w, "baseline")]]) for w in progs}
lines, summ = [], {}
hdr = f"{'program':11s}" + "".join(f" | {p:>21s}" for p in pols)
lines += [f"slowdown % / energy saved % vs NVIDIA default (mean of repeats), budget {delta:g} %", hdr]
viol = defaultdict(int)
est_err = defaultdict(list)
kv = defaultdict(list)
for w in progs:
    cells = []
    for p in pols:
        rs = by[(w, p)]
        sd = [100 * (float(r["time_s"]) / T0[w] - 1) for r in rs]
        T = np.mean([float(r["time_s"]) for r in rs]); E = np.mean([float(r["energy_j"]) for r in rs])
        viol[p] += sum(s > delta for s in sd)
        for r, s in zip(rs, sd):
            if r["est_slowdown_pct"]:
                est_err[p].append(float(r["est_slowdown_pct"]) - s)
            if r.get("e2_est_pct"):
                est_err[p + " (E2 alone)"].append(float(r["e2_est_pct"]) - s)
                kv[p].append(float(r["k_valid_frac"]))
        summ.setdefault(p, []).append((T / T0[w], E / E0[w]))
        cells.append(f"{100*(T/T0[w]-1):6.1f} / {100*(1-E/E0[w]):5.1f}  max{max(sd):5.1f}")
    lines.append(f"{w:11s}" + "".join(f" | {c:>21s}" for c in cells))
lines.append("")
for p in pols:
    t = np.array([x[0] for x in summ[p]]); e = np.array([x[1] for x in summ[p]])
    g = lambda a: np.exp(np.mean(np.log(a)))
    n = len(by[(progs[0], p)]) * len(progs)
    line = (f"{p:14s} geomean energy saved {100*(1-g(e)):5.1f}%  perf loss {100*(g(t)-1):4.1f}%  "
            f"EDP improved {100*(1-g(e*t)):5.1f}%  programs over budget (mean) "
            f"{sum(x[0] > 1 + delta/100 for x in summ[p])}/{len(progs)}  runs over budget {viol[p]}/{n}")
    if est_err[p]:
        d = np.array(est_err[p])
        line += f"  est-true MAE {np.abs(d).mean():.2f} bias {d.mean():+.2f}"
    lines.append(line)
    if kv[p]:
        d = np.array(est_err[p + " (E2 alone)"])
        lines.append(f"{'':14s} same runs, E2 estimate alone: MAE {np.abs(d).mean():.2f} bias {d.mean():+.2f};"
                     f" kernel estimate used in {100*np.mean(kv[p]):.0f} % of steps")
print("\n".join(lines))
open(os.path.join(out, "SUMMARY.txt"), "w").write("\n".join(lines) + "\n")
