"""Offline check of candidate rewards against measured EDP (v4 design).

    python reward_design.py results/reward_sweep_<stamp>.csv \
                            results/reward_sweep_samples_<stamp>.csv

The sweep ran every TRAINING program at fixed caps and logged each 300 ms
telemetry sample. A reward that teaches the agent to minimise EDP should,
for every program, give its highest mean per-sample reward at the cap
with the lowest measured EDP, and rank the caps like EDP does. For each
candidate this prints, per program, the Spearman correlation between mean
reward and -EDP over the caps and whether the reward's best cap equals
the EDP-best cap (or is within one sweep step of it).

Only training programs are used, so the test programs stay unseen.
"""

import csv
import sys
from collections import defaultdict

import numpy as np

import reward as R

F_MAX = 2745.0
P_MAX = 285.0
CAPS_IDLE_MIN = 1800.0


def candidates():
    def paper(prev, cur, cap):
        return R.reward(prev, cur, cap, CAPS_IDLE_MIN)[0]

    def idle_or(fn):
        def r(prev, cur, cap):
            if cur["util"] == 0 and cur["mem_util"] == 0:  # case B
                return R.R_IDLE_MAX * CAPS_IDLE_MIN / cap
            return fn(cur)
        return r

    def eff_uf(cur):   # progress ~ SM busy fraction x clock
        rate = cur["util"] / 100 * cur["clock"] / F_MAX
        return np.log(max(rate, 1e-3) ** 2 / (cur["power"] / P_MAX))

    def eff_mix(cur):  # memory throughput when memory is the bottleneck
        rate = cur["util"] / 100 * cur["clock"] / F_MAX
        if cur["mem_util"] >= 10:
            rate = cur["mem_util"] / 100
        return np.log(max(rate, 1e-3) ** 2 / (cur["power"] / P_MAX))

    def eff_ed(cur):   # energy per work only (no delay weight)
        rate = cur["util"] / 100 * cur["clock"] / F_MAX
        return np.log(max(rate, 1e-3) / (cur["power"] / P_MAX))

    def power_only(cur):
        return -cur["power"] / P_MAX

    def edp_local(prev, cur, cap, pbar):
        # EDP-optimal local cost: minimising E*T over phases is minimising
        # (P + Pbar) per unit of work, Pbar = the job's mean power
        if cur["util"] == 0 and cur["mem_util"] == 0:  # case B
            return R.R_IDLE_MAX * CAPS_IDLE_MIN / cap
        return R.reward_edp_local(cur, cap, CAPS_IDLE_MIN, pbar)[0]

    return {"paper Alg.1/Eq.1": paper,
            "log(rate^2/P), rate=u*f": idle_or(eff_uf),
            "log(rate^2/P), rate=mem if mem>=10": idle_or(eff_mix),
            "log(rate/P), rate=u*f": idle_or(eff_ed),
            "-P (power only)": idle_or(power_only),
            "-log((P+Pbar)/rate), rate=u*f [EDP-local]": edp_local}


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(totals_path, samples_path):
    edp = {}
    for r in csv.DictReader(open(totals_path)):
        if r["cap_mhz"] != "default":
            edp[(r["workload"], int(r["cap_mhz"]))] = float(r["edp"])
    samples = defaultdict(list)
    for r in csv.DictReader(open(samples_path)):
        if r["cap_mhz"] == "default":
            continue
        samples[(r["workload"], int(r["cap_mhz"]))].append(
            {k: float(r[k]) for k in ("util", "mem_util", "power", "clock",
                                      "temp")})
    progs = sorted({w for w, _ in edp})
    caps = sorted({c for _, c in edp})
    print("EDP-best cap per program:",
          {w: min(caps, key=lambda c: edp[(w, c)]) for w in progs})
    for name, fn in candidates().items():
        rho, hit, near, regret = [], 0, 0, []
        detail = []
        for w in progs:
            mean_r = []
            for c in caps:
                s = samples[(w, c)]
                pbar = np.mean([x["power"] for x in s])
                if fn.__name__ == "edp_local":
                    vals = [fn(a, b, c, pbar) for a, b in zip(s, s[1:])]
                else:
                    vals = [fn(a, b, c) for a, b in zip(s, s[1:])]
                mean_r.append(np.mean(vals))
            e = [edp[(w, c)] for c in caps]
            rho.append(spearman(mean_r, [-x for x in e]))
            best_r = caps[int(np.argmax(mean_r))]
            best_e = caps[int(np.argmin(e))]
            regret.append(100 * (edp[(w, best_r)] / edp[(w, best_e)] - 1))
            hit += best_r == best_e
            near += abs(caps.index(best_r) - caps.index(best_e)) <= 1
            detail.append(f"{w}:{best_r}/{best_e}")
        print(f"\n{name}\n  mean Spearman(reward, -EDP) = {np.mean(rho):+.2f}"
              f"  (per program {', '.join(f'{x:+.2f}' for x in rho)})\n"
              f"  best cap = EDP-best: {hit}/{len(progs)}, within one step: "
              f"{near}/{len(progs)}\n  EDP regret of the reward's choice: "
              f"mean {np.mean(regret):.1f} %, max {np.max(regret):.1f} %"
              f"\n  reward-best/EDP-best: "
              + "  ".join(detail))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
