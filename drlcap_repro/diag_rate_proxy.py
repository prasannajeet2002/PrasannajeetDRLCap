"""Offline check of the v4 reward's work-rate proxy (rate = util x clock).

    python diag_rate_proxy.py results/eval_main_<stamp>

No GPU is used. For each test program it compares
  * the measured speed gain from 2100 to 2400 MHz (static_2100 vs
    static_2400 job times), converted to the clock elasticity e_busy of the
    GPU-busy part (util >= 50 share taken from the DRLCap step log;
    T2100/T2400 = (1 - b) + b * (2400/2100)^e). The proxy assumes e = 1
    whenever utilisation does not fall as the clock rises;
  * the slope of utilisation against clock in DRLCap's busy samples;
  * the clock bin (150 MHz) with the highest mean v4 reward in busy
    samples against the measured EDP-best of static 2100 / 2400 / default.
Caveat: the reward bins come from the caps DRLCap itself chose, so they
mix phases; e_busy is a rough estimate (it assumes host time does not
depend on the clock).
"""

import csv
import math
import sys
from collections import defaultdict

import numpy as np


def main(d):
    runs = list(csv.DictReader(open(f"{d}/runs.csv")))
    pol = {r["job"]: r["policy"] for r in runs}
    agg = defaultdict(list)
    for r in runs:
        agg[r["workload"], r["policy"]].append(
            (float(r["time_s"]), float(r["energy_j"])))
    steps = defaultdict(list)
    for r in csv.DictReader(open(f"{d}/steps.csv")):
        if pol[r["job"]] == "drlcap":
            x = {k: float(r[k]) for k in ("util", "mem_util", "power", "clock")}
            x["job"] = r["job"]
            steps[r["workload"]].append(x)
    print(f"{'program':14s} busy% memU  T2100/T2400  e_busy  util/100MHz  "
          f"EDP-best      reward-best  reward by clock bin")
    for w in sorted(steps):
        m = {p: np.mean(agg[w, p], axis=0)
             for p in ("static_2100", "static_2400", "baseline")}
        best = min(m, key=lambda p: m[p][0] * m[p][1])
        s = steps[w]
        busy = [x for x in s if x["util"] >= 50]
        b = len(busy) / len(s)
        ratio = m["static_2100"][0] / m["static_2400"][0]
        e = (math.log(max((ratio - (1 - b)) / b, 1e-9)) / math.log(2400 / 2100)
             if b > 0.2 else float("nan"))
        pbar = defaultdict(list)
        for x in s:
            pbar[x["job"]].append(x["power"])
        pb = {j: np.mean(v) for j, v in pbar.items()}
        bins = defaultdict(list)
        for x in busy:
            rate = x["util"] / 100 * x["clock"] / 2745
            bins[round(x["clock"] / 150) * 150].append(
                -3 * math.log((x["power"] + pb[x["job"]]) / 285 / rate))
        good = {c: np.mean(v) for c, v in bins.items() if len(v) >= 20}
        rb = max(good, key=good.get) if good else "-"
        sl = float("nan")
        if len(busy) > 20:
            cl = np.array([x["clock"] for x in busy])
            sl = np.polyfit(cl, [x["util"] for x in busy], 1)[0] * 100
        mu = np.mean([x["mem_util"] for x in busy]) if busy else 0
        print(f"{w:14s} {100*b:4.0f} {mu:4.0f}  {ratio:9.3f}  {e:6.2f}  "
              f"{sl:+9.2f}    {best:12s}  {str(rb):11s}  "
              + " ".join(f"{c}:{v:+.2f}" for c, v in sorted(good.items())))
    print("\nproxy assumes T2100/T2400 = 1.143 for the busy part (e = 1)")


if __name__ == "__main__":
    main(sys.argv[1])
