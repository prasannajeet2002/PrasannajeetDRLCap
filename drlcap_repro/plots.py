"""Analysis figures for a finished reproduction run.

    python plots.py --main results/eval_main_<stamp> \
        --ablations results/eval_ablation_seed_overhead_<stamp> \
                    results/eval_ablation_frozen_<stamp> \
        --train results/train_freq_s0_<stamp> ... \
        --out results/plots_<stamp>

Everything is read from the CSV / model files the runs wrote; nothing is
re-measured. Figures (PNG, 150 dpi, plus PDF for the thesis):

  fig01_geomean          geomean energy saving / perf. loss / EDP per
                         policy with bootstrap 95 % intervals (Fig. 10)
  fig02_heatmap_<metric> policy x program table of each metric (Fig. 9)
  fig03_tradeoff         energy saving vs. performance loss, geomean and
                         per program
  fig04_timeline_<prog>  one DRLCap job: requested cap vs. actual clock,
                         utilisation, power over time
  fig05_cap_choices      distribution of the caps the learned policies
                         chose (greedy steps only)
  fig06_training         training curves: reward, loss, epsilon, cap
  fig07_cases            Algorithm 1 case counts per training run
  fig08_policy_map       greedy action of each saved Q-network over a
                         util x mem-util grid (state dependence)
  fig09_overhead         controller step time, CPU time per job
  fig10_ablations        seed / frozen / monitor-only geomeans
  fig11_noise            per-repeat geomean EDP of every policy (spread)
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_eval as AE

# reference palette (dataviz skill): three metric slots, neutrals, and a
# diverging pair with a gray midpoint
METRIC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
METRICS = [("energy", "Energy saving"), ("time", "Performance loss"),
           ("edp", "EDP improvement")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HIGHLIGHT, NEUTRAL = "#2a78d6", "#9a9890"
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#6250d6", "#e34948"]
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", ["#c2410c", "#f2c2a8", "#efeeea", "#9ec5f4", "#1c5cab"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

ORDER = ["static_2100", "static_2400", "coord", "drlcap", "drlcap_power",
         "drlcap_hybrid", "rl", "drlcap_seed1", "drlcap_frozen",
         "monitor_only"]
CNN = AE.CNN


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote", name)


def policies_of(runs):
    ps = [p for p in ORDER if p in runs]
    return ps + [p for p in runs if p not in ps and p != "baseline"]


def per_workload(runs, policy):
    """{workload: (energy saving, perf loss, EDP improvement) %}."""
    out = {}
    for w in runs["baseline"]:
        out[w] = AE.pct(*AE.ratios(runs[policy], runs["baseline"], [w]))
    return out


def bootstrap(runs, policy, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    wls = list(runs["baseline"])
    res = []
    for _ in range(n):
        rp = {w: [v[i] for i in rng.integers(len(v), size=len(v))]
              for w, v in runs[policy].items()}
        rb = {w: [v[i] for i in rng.integers(len(v), size=len(v))]
              for w, v in runs["baseline"].items()}
        res.append(AE.pct(*AE.ratios(rp, rb, wls)))
    return np.percentile(np.array(res), [2.5, 97.5], axis=0)


def geomean_bars(ax, runs, policies, title):
    wls = list(runs["baseline"])
    x = np.arange(len(policies))
    w = 0.26
    for i, (_, label) in enumerate(METRICS):
        vals, lo, hi = [], [], []
        for p in policies:
            v = AE.pct(*AE.ratios(runs[p], runs["baseline"], wls))[i]
            ci = bootstrap(runs, p)
            vals.append(v)
            lo.append(v - ci[0][i])
            hi.append(ci[1][i] - v)
        ax.bar(x + (i - 1) * w, vals, w * 0.92, color=METRIC_COLORS[i],
               label=label, yerr=[lo, hi], error_kw=dict(
                   ecolor=INK2, elinewidth=0.8, capsize=2))
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * w, max(v, 0) + 1.2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=25, ha="right")
    ax.set_ylabel("% vs. NVIDIA default")
    ax.set_title(title, loc="left", color=INK, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              fontsize=7.5)
    ax.margins(y=0.12)
    ax.grid(axis="x", visible=False)


def fig_geomean(runs, out):
    ps = policies_of(runs)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(ps)), 4.2))
    geomean_bars(ax, runs, ps, "Geomean over the test programs "
                 "(error bars: bootstrap 95 % interval)")
    save(fig, out, "fig01_geomean")


def fig_heatmaps(runs, out):
    ps = policies_of(runs)
    wls = list(runs["baseline"])
    for i, (key, label) in enumerate(METRICS):
        m = np.array([[per_workload(runs, p)[w][i] for w in wls] for p in ps])
        geo = [AE.pct(*AE.ratios(runs[p], runs["baseline"], wls))[i]
               for p in ps]
        m = np.column_stack([m, geo])
        lim = max(5.0, np.abs(m).max())
        # perf loss: positive is bad, so flip the colours
        cmap = DIVERGING.reversed() if key == "time" else DIVERGING
        fig, ax = plt.subplots(figsize=(1.0 + 0.75 * m.shape[1],
                                        0.9 + 0.42 * len(ps)))
        ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                v = m[r, c]
                ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.6 * lim
                        else INK, fontweight="bold" if c == m.shape[1] - 1
                        else "normal")
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels(wls + ["GEOMEAN"], rotation=30, ha="right")
        ax.set_yticks(range(len(ps)))
        ax.set_yticklabels(ps)
        ax.grid(False)
        ax.axvline(m.shape[1] - 1.5, color="white", linewidth=2)
        ax.set_title(f"{label} (%) vs. NVIDIA default", loc="left")
        save(fig, out, f"fig02_heatmap_{key}")


def fig_tradeoff(runs, out):
    ps = policies_of(runs)
    wls = list(runs["baseline"])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for p in ps:
        e, t, _ = AE.pct(*AE.ratios(runs[p], runs["baseline"], wls))
        hl = p.startswith("drlcap")
        ax.scatter(t, e, s=46, color=HIGHLIGHT if hl else NEUTRAL,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(p, (t, e), xytext=(5, 3), textcoords="offset points",
                    fontsize=7.5, color=INK if hl else INK2)
    ax.set_xlabel("Performance loss (%)  - lower is better")
    ax.set_ylabel("Energy saving (%)  - higher is better")
    ax.set_title("Energy / performance trade-off (geomean); "
                 "blue = DRLCap variants", loc="left")
    save(fig, out, "fig03_tradeoff")

    n = len(wls)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.4 * rows),
                             squeeze=False)
    for ax, w in zip(axes.flat, wls):
        for p in ps:
            e, t, _ = per_workload(runs, p)[w]
            hl = p == "drlcap"
            ax.scatter(t, e, s=26, color=HIGHLIGHT if hl else NEUTRAL,
                       edgecolor="white", linewidth=1, zorder=3)
            if hl or p in ("static_2400", "coord"):
                ax.annotate(p.replace("static_", "s"), (t, e),
                            xytext=(3, 2), textcoords="offset points",
                            fontsize=6.5, color=INK if hl else INK2)
        ax.set_title(w, loc="left", fontsize=8.5)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.supxlabel("Performance loss (%)")
    fig.supylabel("Energy saving (%)")
    save(fig, out, "fig03_tradeoff_per_program")


def load_steps(run_dir):
    runs = {r["job"]: r for r in csv.DictReader(
        open(os.path.join(run_dir, "runs.csv")))}
    steps = defaultdict(list)
    with open(os.path.join(run_dir, "steps.csv")) as f:
        for s in csv.DictReader(f):
            steps[s["job"]].append(s)
    return runs, steps


def fig_timelines(run_dir, out, policy="drlcap"):
    runs, steps = load_steps(run_dir)
    by_wl = {}
    for job, r in runs.items():
        if r["policy"] == policy and job in steps:
            by_wl.setdefault(r["workload"], job)  # first repeat
    for w, job in by_wl.items():
        st = steps[job]
        t = np.array([float(s["t"]) for s in st])
        t -= t[0]
        f = lambda k: np.array([float(s[k]) if s[k] else np.nan for s in st])
        fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.4), sharex=True)
        ax = axes[0]
        ax.step(t, f("cap_mhz"), where="post", color=LINE_COLORS[0],
                linewidth=1.6, label="requested cap")
        ax.plot(t, f("clock"), color=LINE_COLORS[1], linewidth=1.2,
                label="actual SM clock")
        forced = f("forced_max") == 1
        ax.scatter(t[forced], f("cap_mhz")[forced], s=10, color=INK2,
                   zorder=4, label="case A (forced max)")
        ax.set_ylabel("MHz")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                  fontsize=7)
        fig.suptitle(f"{w}: one {policy} job (job {job})", x=0.01,
                     ha="left", fontsize=10)
        ax = axes[1]
        ax.plot(t, f("util"), color=LINE_COLORS[0], linewidth=1.2,
                label="core util")
        ax.plot(t, f("mem_util"), color=LINE_COLORS[2], linewidth=1.2,
                label="memory util")
        ax.set_ylabel("%")
        ax.set_ylim(-3, 103)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                  fontsize=7)
        ax = axes[2]
        ax.plot(t, f("power"), color=LINE_COLORS[0], linewidth=1.2)
        ax.set_ylabel("Board power (W)")
        ax.set_xlabel("Time in job (s)")
        save(fig, out, f"fig04_timeline_{w}")


def greedy_caps(run_dir, policies):
    runs, steps = load_steps(run_dir)
    caps = defaultdict(list)
    for job, st in steps.items():
        p = runs[job]["policy"]
        if p not in policies:
            continue
        for s in st:
            if s["explore"] == "0" and s["forced_max"] == "0":
                caps[p].append((s["cap_mhz"] or None, s["power_cap_w"] or None))
    return caps


def fig_cap_choices(dirs, out):
    panels = []
    for d in dirs:
        runs, _ = load_steps(d)
        ps = [p for p in policies_of({r["policy"]: 1 for r in runs.values()})
              if p.startswith("drlcap") or p == "rl"]
        for p, c in greedy_caps(d, ps).items():
            panels.append((f"{p} ({os.path.basename(d)[5:-16]})", c))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(7.5, 1.35 * len(panels) + 0.6),
                             squeeze=False)
    for ax, (name, c) in zip(axes[:, 0], panels):
        cnt = Counter(c)
        keys = sorted(cnt, key=lambda k: (float(k[0] or 0), float(k[1] or 0)))
        labels = ["/".join(x for x in k if x) for k in keys]
        share = np.array([cnt[k] for k in keys]) / len(c) * 100
        single = all(bool(k[0]) != bool(k[1]) for k in keys)
        if single:  # one knob: numeric axis (MHz or W)
            xs = np.array([float(k[0] or k[1]) for k in keys])
            width = 10 if keys[0][0] else 3.5
            ax.bar(xs, share, width, color=METRIC_COLORS[0])
            ax.set_xlabel("MHz" if keys[0][0] else "W", fontsize=7)
        else:       # hybrid: categorical (freq/power) pairs
            xs = np.arange(len(keys))
            ax.bar(xs, share, 0.8, color=METRIC_COLORS[0])
            step = max(1, len(keys) // 12)
            ax.set_xticks(xs[::step])
            ax.set_xticklabels(labels[::step], fontsize=6.5, rotation=30,
                               ha="right")
        top = int(np.argmax(share))
        ax.text(xs[top], share[top], f" {labels[top]}: {share[top]:.0f} %",
                fontsize=7, color=INK, va="bottom")
        ax.margins(y=0.2)
        ax.set_ylabel("% steps", fontsize=7)
        ax.set_title(name, loc="left", fontsize=8)
        ax.grid(axis="x", visible=False)
    save(fig, out, "fig05_cap_choices")


def rolling(x, n=100):
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    c = np.convolve(np.where(ok, x, 0), np.ones(n), "valid")
    k = np.convolve(ok.astype(float), np.ones(n), "valid")
    return c / np.maximum(k, 1)


def fig_training(train_dirs, out):
    if not train_dirs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
    for i, d in enumerate(train_dirs):
        rows = list(csv.DictReader(open(os.path.join(d, "steps.csv"))))
        name = os.path.basename(d)[6:-16]
        col = LINE_COLORS[i % len(LINE_COLORS)]
        f = lambda k: [float(r[k]) if r[k] not in ("", "nan") else np.nan
                       for r in rows]
        n_act = len(json.load(open(os.path.join(d, "config.json")))
                    ["actions"])
        loss = np.array(f("loss"))
        loss[loss <= 0] = np.nan
        series = [f("reward"), loss, f("epsilon"),
                  np.array(f("action")) / (n_act - 1)]
        for ax, y in zip(axes.flat, series):
            r = rolling(y, 300)
            ax.plot(np.arange(len(r)) + 300, r, color=col, linewidth=1.4,
                    label=name)
    titles = ["Reward (rolling mean, 300 steps)", "TD loss (rolling mean)",
              "Epsilon",
              "Chosen action (rolling mean; 0 = lowest cap, 1 = highest)"]
    for ax, ttl in zip(axes.flat, titles):
        ax.set_title(ttl, loc="left", fontsize=8.5)
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_xlabel("Control step (300 ms)")
    axes[1, 1].set_xlabel("Control step (300 ms)")
    axes[0, 0].legend(fontsize=7)
    save(fig, out, "fig06_training")


def fig_cases(train_dirs, out):
    if not train_dirs:
        return
    names, counts = [], []
    for d in train_dirs:
        c = Counter(r["case"] for r in csv.DictReader(
            open(os.path.join(d, "steps.csv"))))
        names.append(os.path.basename(d)[6:-16])
        counts.append([c.get(k, 0) for k in "ABCD"])
    counts = np.array(counts, float)
    share = counts / counts.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.5, 0.6 + 0.45 * len(names)))
    left = np.zeros(len(names))
    labels = ["A compute-bound", "B idle", "C core only", "D memory active"]
    for k in range(4):
        ax.barh(names, share[:, k], left=left, color=LINE_COLORS[k],
                edgecolor="white", linewidth=2, label=labels[k])
        for y, (l, v) in enumerate(zip(left, share[:, k])):
            if v > 7:
                ax.text(l + v / 2, y, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white")
        left += share[:, k]
    ax.set_xlabel("% of training steps")
    ax.set_xlim(0, 100)
    ax.legend(ncol=4, fontsize=7, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig07_cases")


def fig_policy_map(train_dirs, out):
    import tensorflow as tf

    import drlcap as D
    dirs = [d for d in train_dirs
            if os.path.exists(os.path.join(d, "model_final.keras"))]
    if not dirs:
        return
    grid = np.linspace(0, 100, 21)
    fig, axes = plt.subplots(1, len(dirs), figsize=(3.0 * len(dirs), 3.2),
                             squeeze=False)
    for ax, d in zip(axes[0], dirs):
        cfg = json.load(open(os.path.join(d, "config.json")))
        actions = D.VARIANTS[cfg.get("variant", "freq")]
        model = tf.keras.models.load_model(
            os.path.join(d, "model_final.keras"))
        states = np.array([D.to_state({
            "util": u, "mem_util": m, "power": 60 + 1.5 * u,
            "clock": 2600, "temp": 50, "energy_j": 0})
            for m in grid for u in grid], dtype=np.float32)
        a = model.predict(states, verbose=0).argmax(1).reshape(len(grid), -1)
        vals = np.array([D.primary(actions[i]) for i in a.ravel()],
                        float).reshape(a.shape)
        im = ax.imshow(vals, origin="lower", extent=[0, 100, 0, 100],
                       cmap="Blues", aspect="auto")
        ax.set_title(os.path.basename(d)[6:-16], loc="left", fontsize=8.5)
        ax.set_xlabel("core util (%)")
        ax.set_ylabel("memory util (%)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05)
        cb.set_label("greedy cap (MHz or W)", fontsize=7)
    fig.suptitle("Greedy action of each trained Q-network "
                 "(power 60 + 1.5*util W, clock 2600 MHz, 50 C)",
                 fontsize=8.5, x=0.01, ha="left")
    save(fig, out, "fig08_policy_map")


def fig_overhead(run_dir, out):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "runs.csv"))))
    agent_p = [p for p in policies_of({r["policy"]: 1 for r in rows})
               if any(r["policy"] == p and r["step_ms_median"] for r in rows)]
    if not agent_p:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, key, ttl in zip(axes, ("step_ms_median", "agent_cpu_s"),
                            ("Median control-step time (ms) per job",
                             "Controller CPU time per job (s)")):
        data = [[float(r[key]) for r in rows if r["policy"] == p and r[key]]
                for p in agent_p]
        ax.boxplot(data, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#cde2fb", color=INK2),
                   medianprops=dict(color=INK), whiskerprops=dict(color=INK2),
                   capprops=dict(color=INK2), flierprops=dict(
                       markersize=3, markeredgecolor=INK2))
        ax.set_xticks(range(1, len(agent_p) + 1))
        ax.set_xticklabels(agent_p, rotation=30, ha="right")
        ax.set_title(ttl, loc="left", fontsize=8.5)
    rss = [float(r["max_rss_mb"]) for r in rows if r["max_rss_mb"]]
    axes[2].plot(rss, color=LINE_COLORS[0], linewidth=1.4)
    axes[2].set_title("Evaluation-process peak RSS (MB)", loc="left",
                      fontsize=8.5)
    axes[2].set_xlabel("agent job #")
    save(fig, out, "fig09_overhead")


def fig_ablations(main_runs, abl_dirs, out):
    panels = []
    for d in abl_dirs:
        runs = AE.load(d)
        panels.append((os.path.basename(d)[5:-16], runs))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8),
                             squeeze=False)
    for ax, (name, runs) in zip(axes[0], panels):
        geomean_bars(ax, runs, policies_of(runs), name)
    save(fig, out, "fig10_ablations")


def fig_noise(runs_path, out):
    rows = list(csv.DictReader(open(runs_path)))
    reps = sorted({int(r["repeat"]) for r in rows})
    base = {(r["workload"], int(r["repeat"])): r for r in rows
            if r["policy"] == "baseline"}
    ps = policies_of({r["policy"]: 1 for r in rows})
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(ps)), 3.6))
    for i, p in enumerate(ps):
        vals = []
        for rep in reps:
            ratios = []
            for r in rows:
                if r["policy"] == p and int(r["repeat"]) == rep:
                    b = base[(r["workload"], rep)]
                    ratios.append(float(r["edp"]) / float(b["edp"]))
            vals.append(100 * (1 - np.exp(np.mean(np.log(ratios)))))
        jitter = np.linspace(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=14, color=HIGHLIGHT
                   if p.startswith("drlcap") else NEUTRAL, zorder=3)
        ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=INK, linewidth=1.5)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels(ps, rotation=25, ha="right")
    ax.set_ylabel("Geomean EDP improvement (%)")
    ax.set_title("Each dot = one repeat (all programs, paired with that "
                 "repeat's baseline); bar = mean", loc="left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig11_noise")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--ablations", nargs="*", default=[])
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = AE.load(args.main)
    fig_geomean(runs, args.out)
    fig_heatmaps(runs, args.out)
    fig_tradeoff(runs, args.out)
    fig_timelines(args.main, args.out)
    fig_cap_choices([args.main] + args.ablations, args.out)
    fig_training(args.train, args.out)
    fig_cases(args.train, args.out)
    fig_policy_map(args.train, args.out)
    fig_overhead(args.main, args.out)
    fig_ablations(runs, args.ablations, args.out)
    fig_noise(os.path.join(args.main, "runs.csv"), args.out)


if __name__ == "__main__":
    main()
