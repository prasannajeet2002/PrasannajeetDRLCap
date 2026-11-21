"""Offline analysis of the f2 evaluation logs (no GPU needed).

Reads drlcap_faithful/results/eval_main_{a,b}_f2_*/{runs,steps}.csv and
reports, per policy:
  * slowdown per program vs NVIDIA default (does any method control it?)
  * cap-switching rate and exploration share of the learned controllers
  * temperature statistics and rainflow thermal cycles per job
  * a within-job power-vs-temperature slope at a fixed clock (leakage hint)

Writes CSVs next to this script under out_existing/.
"""

import csv
import glob
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "..", "drlcap_faithful", "results")
OUT = os.path.join(ROOT, "out_existing")
DIRS = sorted(glob.glob(os.path.join(EVAL, "eval_main_[ab]_f2_*")))
K_EV = 8.617e-5


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rainflow(series):
    """Half/full cycle ranges by the ASTM E1049 three-point method."""
    # reduce to turning points
    tp = []
    for x in series:
        if tp and x == tp[-1]:
            continue
        if len(tp) >= 2 and (tp[-1] - tp[-2]) * (x - tp[-1]) > 0:
            tp[-1] = x
        else:
            tp.append(x)
    stack, cycles = [], []
    for x in tp:
        stack.append(x)
        while len(stack) >= 3:
            r1 = abs(stack[-2] - stack[-3])
            r2 = abs(stack[-1] - stack[-2])
            if r2 < r1:
                break
            if len(stack) == 3:
                cycles.append((r1, 0.5))
                stack.pop(0)
            else:
                cycles.append((r1, 1.0))
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for a, b in zip(stack, stack[1:]):
        cycles.append((abs(b - a), 0.5))
    return cycles


def main():
    os.makedirs(OUT, exist_ok=True)
    runs, steps = [], defaultdict(list)
    for i, d in enumerate(DIRS):
        for r in read_csv(os.path.join(d, "runs.csv")):
            r["key"] = (i, r["job"])
            runs.append(r)
        for s in read_csv(os.path.join(d, "steps.csv")):
            steps[(i, s["job"])].append(s)

    # ---- 1. slowdown per program and policy vs baseline ----
    by = defaultdict(list)
    for r in runs:
        by[(r["policy"], r["workload"])].append(r)
    progs = sorted({r["workload"] for r in runs})
    pols = [p for p in ["coord", "drlcap", "drlcap_power", "drlcap_hybrid",
                        "rl", "drlcap_seed0"] if any(k[0] == p for k in by)]
    rows = []
    for p in pols:
        for w in progs:
            base = by[("baseline", w)]
            cur = by[(p, w)]
            tb = sum(float(r["time_s"]) for r in base) / len(base)
            eb = sum(float(r["energy_j"]) for r in base) / len(base)
            tc = sum(float(r["time_s"]) for r in cur) / len(cur)
            ec = sum(float(r["energy_j"]) for r in cur) / len(cur)
            worst = max(float(r["time_s"]) / tb - 1 for r in cur)
            rows.append({"policy": p, "workload": w,
                         "slowdown_pct": 100 * (tc / tb - 1),
                         "worst_run_slowdown_pct": 100 * worst,
                         "energy_saving_pct": 100 * (1 - ec / eb)})
    with open(os.path.join(OUT, "slowdown.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("== slowdown vs NVIDIA default (mean over repeats) ==")
    print("policy           " + " ".join(f"{w[:8]:>8}" for w in progs) +
          "   max  >5%")
    for p in pols:
        v = [x["slowdown_pct"] for x in rows if x["policy"] == p]
        print(f"{p:16} " + " ".join(f"{x:8.1f}" for x in v) +
              f" {max(v):5.1f} {sum(x > 5 for x in v):4d}")
    # baseline repeat noise
    noise = []
    for w in progs:
        ts = [float(r["time_s"]) for r in by[("baseline", w)]]
        m = sum(ts) / len(ts)
        noise.append(100 * (max(ts) - min(ts)) / m)
    print("baseline run-time spread (max-min)/mean %: " +
          " ".join(f"{x:.1f}" for x in noise))

    # ---- 2./3. switching, temperature, cycles per job ----
    pol_of = {r["key"]: (r["policy"], r["workload"]) for r in runs}
    jrows = []
    for key, ss in steps.items():
        pol, w = pol_of.get(key, ("?", "?"))
        busy = [s for s in ss if float(s["util"]) > 0]
        if len(ss) < 10:
            continue
        temps = [float(s["temp"]) for s in ss]
        caps = [s["cap_mhz"] for s in ss]
        dur = float(ss[-1]["t"]) - float(ss[0]["t"])
        changes = sum(a != b for a, b in zip(caps, caps[1:]))
        expl = sum(s["explore"] == "1" for s in ss) / len(ss)
        cyc = rainflow(temps)
        n_ge1 = sum(c for r, c in cyc if r >= 1)
        n_ge3 = sum(c for r, c in cyc if r >= 3)
        dmg2 = sum(c * r ** 2 for r, c in cyc)
        arr = sum(math.exp(-0.7 / (K_EV * (t + 273.15))) for t in temps) / len(temps)
        mean_t = sum(temps) / len(temps)
        sd_t = math.sqrt(sum((t - mean_t) ** 2 for t in temps) / len(temps))
        # leakage hint: within job, busy steps at the modal clock
        clk = defaultdict(list)
        for s in busy:
            if float(s["util"]) >= 90:
                clk[s["clock"]].append((float(s["temp"]), float(s["power"])))
        slope = ""
        if clk:
            pts = max(clk.values(), key=len)
            if len(pts) >= 50:
                xs = [a for a, _ in pts]
                ys = [b for _, b in pts]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                vx = sum((x - mx) ** 2 for x in xs)
                if vx > 0 and max(xs) - min(xs) >= 4:
                    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
        jrows.append({"policy": pol, "workload": w, "duration_s": dur,
                      "cap_changes_per_min": 60 * changes / dur if dur else 0,
                      "explore_share": expl, "mean_temp": mean_t,
                      "sd_temp": sd_t, "max_temp": max(temps),
                      "cycles_ge1C_per_min": 60 * n_ge1 / dur if dur else 0,
                      "cycles_ge3C_per_min": 60 * n_ge3 / dur if dur else 0,
                      "cm_damage_q2_per_min": 60 * dmg2 / dur if dur else 0,
                      "arrhenius_rate_07": arr,
                      "leak_slope_w_per_c": slope})
    with open(os.path.join(OUT, "jobs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(jrows[0]))
        wr.writeheader()
        wr.writerows(jrows)

    print("\n== per policy, mean over jobs ==")
    keys = ["cap_changes_per_min", "explore_share", "mean_temp", "sd_temp",
            "cycles_ge1C_per_min", "cycles_ge3C_per_min", "cm_damage_q2_per_min"]
    print(f"{'policy':16}" + "".join(f"{k[:14]:>15}" for k in keys))
    for p in pols:
        js = [j for j in jrows if j["policy"] == p]
        if not js:
            continue
        print(f"{p:16}" + "".join(
            f"{sum(j[k] for j in js) / len(js):15.3f}" for k in keys))

    print("\n== leakage hint: within-job dP/dT at fixed clock, util>=90 ==")
    lw = defaultdict(list)
    for j in jrows:
        if j["leak_slope_w_per_c"] != "":
            lw[j["workload"]].append(j["leak_slope_w_per_c"])
    for w, v in sorted(lw.items()):
        v.sort()
        print(f"{w:12} n={len(v):3d} median {v[len(v) // 2]:6.2f} W/C "
              f"range {v[0]:6.2f}..{v[-1]:6.2f}")


if __name__ == "__main__":
    main()
