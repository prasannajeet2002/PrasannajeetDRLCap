"""Final tables and timelines for the f1 queue (run_faithful.sh).

Tables use the paper's method only (per-program mean over repeats, ratio to
the NVIDIA default, geometric mean over the 10 Table II programs) and only
the paper's comparisons: DRLCap frequency / power / hybrid, COORD and RL vs
the NVIDIA default. The fixed-cap runs stay on disk but are not reported.

Main comparison = repeats 0-8 of eval_main_f1_* (repeat 9 there was cut by
the GROMACS backup crash) + eval_main_rep9_f1_* (repeat 9 re-run).

Timelines (review item 6): one DRLCap run per test program, showing
utilisation / memory utilisation, cap vs actual clock, power, reward and
explore vs exploit steps.

Usage: python analyze_f1.py   -> results/final_f1/
"""

import csv
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "final_f1")


def one(pattern):
    hits = sorted(glob.glob(os.path.join(RES, pattern)))
    return hits[-1]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def runs_of(d, repeats=None):
    rows = read_csv(os.path.join(d, "runs.csv"))
    if repeats is not None:
        rows = [r for r in rows if int(r["repeat"]) in repeats]
    for r in rows:
        for k in ("energy_j", "time_s", "edp"):
            r[k] = float(r[k])
        r["_dir"] = d
    return rows


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def summarize(rows, policy, base_policy="baseline"):
    """Paper method; returns per-program rows and the GEOMEAN row."""
    progs = sorted({r["workload"] for r in rows if r["policy"] == policy})
    out = []
    for p in progs:
        b = [r for r in rows if r["workload"] == p and r["policy"] == base_policy]
        x = [r for r in rows if r["workload"] == p and r["policy"] == policy]
        ratio = {k: (sum(r[k] for r in x) / len(x)) / (sum(r[k] for r in b) / len(b))
                 for k in ("energy_j", "time_s", "edp")}
        out.append((p, len(x), ratio))
    g = {k: geomean([o[2][k] for o in out]) for k in ("energy_j", "time_s", "edp")}
    return out, g


def pct(ratio):
    return (100 * (1 - ratio["energy_j"]), 100 * (ratio["time_s"] - 1),
            100 * (1 - ratio["edp"]))


def table(title, rows, policies, f):
    f.write(f"\n## {title}\n\n")
    f.write("| Method | runs/program | Energy saved % | Perf loss % | EDP improved % |\n")
    f.write("|---|---|---|---|---|\n")
    per_prog = {}
    for pol in policies:
        out, g = summarize(rows, pol)
        per_prog[pol] = out
        e, t, d = pct(g)
        f.write(f"| {pol} | {out[0][1]} | {e:.1f} | {t:.1f} | {d:.1f} |\n")
    return per_prog


def per_program_table(per_prog, policies, f):
    f.write("\nEDP improvement % per program\n\n")
    f.write("| Program | " + " | ".join(policies) + " |\n")
    f.write("|---" * (len(policies) + 1) + "|\n")
    for i, (p, _, _) in enumerate(per_prog[policies[0]]):
        cells = [f"{pct(per_prog[pol][i][2])[2]:.1f}" for pol in policies]
        f.write(f"| {p} | " + " | ".join(cells) + " |\n")


def timelines(main_dir, f):
    """One DRLCap (frequency) run per program, repeat 0."""
    runs = [r for r in read_csv(os.path.join(main_dir, "runs.csv"))
            if r["policy"] == "drlcap" and r["repeat"] == "0"]
    jobs = {r["job"]: r["workload"] for r in runs}
    steps = {}
    with open(os.path.join(main_dir, "steps.csv")) as fh:
        for s in csv.DictReader(fh):
            if s["job"] in jobs:
                steps.setdefault(s["job"], []).append(s)
    f.write("\n## Timelines (review item 6)\n\n"
            "DRLCap frequency agent, main comparison repeat 0, one figure "
            "per program in `timelines/`.\n\n"
            "| Program | steps | random (explore) % | case A/B/C/D % "
            "| median cap MHz | median clock MHz |\n|---|---|---|---|---|---|\n")
    os.makedirs(os.path.join(OUT, "timelines"), exist_ok=True)
    for job, st in sorted(steps.items(), key=lambda kv: jobs[kv[0]]):
        name = jobs[job]
        t0 = float(st[0]["t"])
        t = [float(s["t"]) - t0 for s in st]
        num = lambda k: [float(s[k]) if s[k] not in ("", "nan") else math.nan
                         for s in st]
        util, mem, power = num("util"), num("mem_util"), num("power")
        clock, cap, rew = num("clock"), num("cap_mhz"), num("reward")
        explore = [s["explore"] == "1" for s in st]
        forced = [s["forced_max"] == "1" for s in st]

        fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        ax[0].plot(t, util, lw=1, label="core util %")
        ax[0].plot(t, mem, lw=1, label="memory util %")
        ax[0].set_ylabel("%"); ax[0].legend(loc="upper right", fontsize=8)
        ax[1].step(t, cap, where="post", lw=1, label="cap set by agent")
        ax[1].plot(t, clock, lw=1, label="actual clock")
        ex = [(ti, c) for ti, c, e in zip(t, cap, explore) if e]
        if ex:
            ax[1].scatter(*zip(*ex), s=6, c="red", label="random action", zorder=3)
        fo = [(ti, c) for ti, c, e in zip(t, cap, forced) if e]
        if fo:
            ax[1].scatter(*zip(*fo), s=6, c="black", marker="x",
                          label="case A forced max", zorder=3)
        ax[1].set_ylabel("MHz"); ax[1].legend(loc="lower right", fontsize=8)
        ax[2].plot(t, power, lw=1)
        ax[2].set_ylabel("power W")
        ax[3].plot(t, rew, lw=0.8, marker=".", ms=2)
        ax[3].set_ylabel("reward"); ax[3].set_xlabel("time s")
        fig.suptitle(f"DRLCap (frequency) on {name}, repeat 0")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "timelines", f"{name}.png"), dpi=110)
        plt.close(fig)

        n = len(st)
        cases = {c: 100 * sum(s["case"] == c for s in st) / n for c in "ABCD"}
        med = lambda xs: sorted(x for x in xs if not math.isnan(x))[len(xs) // 2]
        f.write(f"| {name} | {n} | {100 * sum(explore) / n:.1f} | "
                + "/".join(f"{cases[c]:.0f}" for c in "ABCD")
                + f" | {med(cap):.0f} | {med(clock):.0f} |\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    main_rows = (runs_of(one("eval_main_f1_2*"), repeats=set(range(9)))
                 + runs_of(one("eval_main_rep9_f1_2*")))
    seeds = runs_of(one("eval_seeds_overhead_f1_2*"))
    frozen = runs_of(one("eval_frozen_f1_2*"))
    variants = runs_of(one("eval_variants_f1_2*"))

    with open(os.path.join(OUT, "RESULTS_F1.md"), "w") as f:
        f.write("# DRLCap faithful reproduction (f1): final results\n\n"
                "Generated by `analyze_f1.py`. Paper method: mean over "
                "repeats per program, ratio to the NVIDIA default, geometric "
                "mean over the 10 Table II programs. RTX 4070 Ti. Paper "
                "(V100S) for DRLCap: 22 % energy, 3 % perf loss, 20 % EDP; "
                "COORD 25/5/21; RL 5/1/4.\n")
        main_pols = ["drlcap", "drlcap_power", "drlcap_hybrid", "coord", "rl"]
        pp = table("Main comparison (10 repeats)", main_rows, main_pols, f)
        per_program_table(pp, main_pols, f)
        table("Seeds and monitoring overhead (5 repeats)", seeds,
              ["drlcap", "drlcap_seed1", "drlcap_seed2", "monitor_only"], f)
        table("Frozen network, no online learning (5 repeats)", frozen,
              ["drlcap_frozen"], f)
        vp = ["drlcap", "var_edplocal", "var_gamma02", "var_nocasea",
              "var_delay1", "var_freq8"]
        pv = table("v4 settings one at a time (5 repeats, seed 0)", variants,
                   vp, f)
        per_program_table(pv, vp, f)
        timelines(one("eval_main_f1_2*"), f)
    print(open(os.path.join(OUT, "RESULTS_F1.md")).read())


if __name__ == "__main__":
    main()
