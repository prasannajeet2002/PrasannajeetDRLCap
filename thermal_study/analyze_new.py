"""Analyse a run_thermal.py results folder (leak and/or wear experiments).

  python analyze_new.py results/both_YYYYmmdd_HHMMSS

leak: power vs temperature at a fixed clock and constant load (leakage
      slope, W per C), heat-up and cool-down time constants.
wear: per program and policy, temperature statistics, Arrhenius wear and
      rainflow thermal-cycling damage integrated over the job, relative to
      NVIDIA default; energy and slowdown from runs.csv (paper method:
      geometric mean of per-program ratios).
Writes leak.csv, wear_jobs.csv, wear_summary.csv into the folder.
"""

import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_existing import rainflow  # noqa: E402

K_EV = 8.617e-5
EAS = (0.5, 0.7, 0.9)
QS = (2.0, 2.5)


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def linfit(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
    res = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    vy = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in res) / vy if vy else 0
    return my - b * mx, b, r2


def tau_fit(ts, temps, rising):
    """Time constant of a first-order response; T_end from the last 20 s."""
    t_end = ts[-1]
    tail = [T for t, T in zip(ts, temps) if t >= t_end - 20]
    t_ss = sum(tail) / len(tail)
    t0 = temps[0]
    span = (t_ss - t0) if rising else (t0 - t_ss)
    xs, ys = [], []
    for t, T in zip(ts, temps):
        gap = (t_ss - T) if rising else (T - t_ss)
        if gap >= 1.0 and span > 2:
            xs.append(t - ts[0])
            ys.append(math.log(gap / span))
    if len(xs) < 5:
        return float("nan"), t_ss
    _, b, _ = linfit(xs, ys)
    return -1.0 / b if b < 0 else float("nan"), t_ss


def leak(rows, out):
    groups = defaultdict(list)
    for r in rows:
        if r["exp"] == "leak":
            groups[(r["job"], r["cap_mhz"], r["phase"])].append(r)
    res = []
    for (rep, cap, phase), rs in sorted(groups.items()):
        if phase != "burn":
            continue
        ts = [float(r["t"]) for r in rs]
        temps = [float(r["temp"]) for r in rs]
        tau_up, tss = tau_fit(ts, temps, True)
        cool = groups.get((rep, cap, "cooldown"), [])
        tau_dn = float("nan")
        if cool:
            tau_dn, _ = tau_fit([float(r["t"]) for r in cool],
                                [float(r["temp"]) for r in cool], False)
        steady = [r for r in rs if float(r["util"]) >= 95
                  and abs(float(r["clock"]) - float(cap)) <= 1
                  and float(r["t"]) - ts[0] >= 5]
        pts = [(float(r["temp"]), float(r["power"])) for r in steady]
        a, b, r2 = linfit([p[0] for p in pts], [p[1] for p in pts])
        first = [p for r, p in zip(steady, pts)
                 if float(r["t"]) - ts[0] < 15]
        last = [p for r, p in zip(steady, pts) if float(r["t"]) >= ts[-1] - 30]
        fans = [float(r["fan"]) for r in rs if r["fan"] != ""]
        row = {"rep": rep, "cap_mhz": cap, "n_steady": len(pts),
               "share_at_cap": len(steady) / len(rs),
               "t_start_c": temps[0], "t_end_c": tss,
               "p_first15s_w": sum(p[1] for p in first) / len(first),
               "t_first15s_c": sum(p[0] for p in first) / len(first),
               "p_last30s_w": sum(p[1] for p in last) / len(last),
               "t_last30s_c": sum(p[0] for p in last) / len(last),
               "slope_w_per_c": b, "r2": r2,
               "fan_start": fans[0] if fans else "",
               "fan_end": fans[-1] if fans else "",
               "tau_heat_s": tau_up, "tau_cool_s": tau_dn}
        row["power_rise_pct"] = 100 * (row["p_last30s_w"] /
                                       row["p_first15s_w"] - 1)
        res.append(row)
    if res:
        with open(os.path.join(out, "leak.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(res[0]))
            wr.writeheader()
            wr.writerows(res)
        print("== leakage at fixed clock, constant load ==")
        for r in res:
            print(f"rep {r['rep']} cap {r['cap_mhz']}: "
                  f"{r['t_first15s_c']:.1f}C {r['p_first15s_w']:.1f}W -> "
                  f"{r['t_last30s_c']:.1f}C {r['p_last30s_w']:.1f}W "
                  f"(+{r['power_rise_pct']:.1f} %), slope "
                  f"{r['slope_w_per_c']:.2f} W/C r2 {r['r2']:.2f}, "
                  f"at cap {100 * r['share_at_cap']:.0f} %, fan "
                  f"{r['fan_start']}->{r['fan_end']} %, tau heat "
                  f"{r['tau_heat_s']:.0f}s cool {r['tau_cool_s']:.0f}s")


def job_metrics(rs):
    ts = [float(r["t"]) for r in rs]
    temps = [float(r["temp"]) for r in rs]
    dts = [b - a for a, b in zip(ts, ts[1:])] + [0.3]
    m = {"duration_s": ts[-1] - ts[0], "mean_temp": sum(temps) / len(temps),
         "max_temp": max(temps)}
    for ea in EAS:
        m[f"arr_{ea}"] = sum(dt * math.exp(-ea / (K_EV * (T + 273.15)))
                             for dt, T in zip(dts, temps))
    cyc = rainflow(temps)
    m["cycles_ge1"] = sum(c for r, c in cyc if r >= 1)
    m["cycles_ge3"] = sum(c for r, c in cyc if r >= 3)
    for q in QS:
        m[f"cm_{q}"] = sum(c * r ** q for r, c in cyc)
    return m


def wear(rows, runs, out):
    jobs = defaultdict(list)
    for r in rows:
        if r["exp"] == "wear" and r["phase"] == "run":
            jobs[r["job"]].append(r)
    info = {r["job"]: r for r in runs}
    jrows = []
    for job, rs in jobs.items():
        if job not in info or info[job]["gpu_shared"] == "1" or len(rs) < 5:
            continue
        m = job_metrics(rs)
        m.update(policy=info[job]["policy"], workload=info[job]["workload"],
                 job=job, time_s=float(info[job]["time_s"]),
                 energy_j=float(info[job]["energy_j"]))
        jrows.append(m)
    with open(os.path.join(out, "wear_jobs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(jrows[0]))
        wr.writeheader()
        wr.writerows(jrows)

    by = defaultdict(list)
    for j in jrows:
        by[(j["policy"], j["workload"])].append(j)
    progs = sorted({j["workload"] for j in jrows})
    keys = ["time_s", "energy_j", "mean_temp", "max_temp", "cycles_ge1",
            "cycles_ge3"] + [f"arr_{e}" for e in EAS] + [f"cm_{q}" for q in QS]
    mean = {k2: {k: sum(j[k] for j in v) / len(v) for k in keys}
            for k2, v in by.items()}
    summary = []
    for p in ("coord", "drlcap"):
        ratios = defaultdict(list)
        for w in progs:
            b, c = mean.get(("baseline", w)), mean.get((p, w))
            if not b or not c:
                continue
            row = {"policy": p, "workload": w}
            for k in keys:
                if k in ("mean_temp", "max_temp"):
                    row[k] = c[k] - b[k]
                elif b[k] > 0:
                    row[k] = c[k] / b[k]
                    ratios[k].append(row[k])
            summary.append(row)
        g = {"policy": p, "workload": "GEOMEAN"}
        for k in keys:
            if k in ("mean_temp", "max_temp"):
                g[k] = sum(r[k] for r in summary if r["policy"] == p) / len(progs)
            elif ratios[k]:
                g[k] = gmean(ratios[k])
        summary.append(g)
    with open(os.path.join(out, "wear_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["policy", "workload"] + keys)
        wr.writeheader()
        wr.writerows(summary)

    print("\n== wear vs NVIDIA default: ratios (temps: difference in C) ==")
    hdr = ["energy_j", "time_s", "mean_temp", "max_temp", "arr_0.5",
           "arr_0.7", "arr_0.9", "cycles_ge1", "cycles_ge3", "cm_2.0", "cm_2.5"]
    print(f"{'policy':8}{'program':12}" + "".join(f"{h:>11}" for h in hdr))
    for r in summary:
        print(f"{r['policy']:8}{r['workload']:12}" + "".join(
            f"{r.get(h, float('nan')):11.3f}" for h in hdr))
    print("\nbaseline per program: mean temp, max temp, run time")
    for w in progs:
        b = mean[("baseline", w)]
        print(f"  {w:12} {b['mean_temp']:5.1f} {b['max_temp']:5.1f} "
              f"{b['time_s']:7.1f}s")


def main():
    out = sys.argv[1]
    with open(os.path.join(out, "trace.csv")) as f:
        rows = list(csv.DictReader(f))
    leak(rows, out)
    runs_path = os.path.join(out, "runs.csv")
    if os.path.exists(runs_path):
        with open(runs_path) as f:
            runs = list(csv.DictReader(f))
        wear(rows, runs, out)


if __name__ == "__main__":
    main()
