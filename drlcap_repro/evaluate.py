"""Evaluate DRLCap against the NVIDIA default policy (paper Sec. V-D, VI-A).

For every workload and repeat, runs (in rotating order to spread drift
and temperature effects):
  baseline     no cap, NVIDIA default clock management
  static_<f>   one fixed frequency cap for the whole job
  coord        COORD-like static category-based heuristic (paper Table III):
               each program is profiled once ahead of time at default
               clocks, classified compute- or memory/host-bound by its mean
               core utilisation, and run with that category's fixed cap
  <name>       a trained model (--models name=path; DRLCap freq / power /
               hybrid or the tabular RL baseline, taken from the run's
               config.json), fresh copy per job, epsilon fixed at the final
               training value (0.1) and online learning ON, as the paper
               deploys it (Sec. IV-D); --no-online freezes it
Energy comes from the NVML energy counter, runtime from wall clock.

Reports per-workload mean energy saving, performance loss and EDP
improvement vs. baseline, plus the geometric mean over workloads of the
energy / time / EDP ratios.

    python evaluate.py --models drlcap=results/train_.../model_final.keras \
        --repeats 10 [--static-caps 2100 2400] [--coord] \
        [--workloads lud lavaMD ...]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import threading
import time

import numpy as np

import agent as A
import drlcap as D
import gpu_control as G
import workloads as W


def baseline_job(workload, cap=None):
    """NVIDIA default (cap=None) or one fixed frequency cap for the job.

    NVML is polled every 300 ms during the job, exactly as the DRLCap loop
    does: polling alone lowers the measured board energy by 2.5-4 %
    (diag_monitor_effect.py, README), so without it the agent policies
    would get a head start over these policies."""
    G.reset_cap()
    if cap is not None:
        G.set_cap(cap)
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            G.read()
            stop.wait(D.SAMPLE_INTERVAL)

    poller = threading.Thread(target=poll, daemon=True)
    try:
        e0 = G.energy_j()
        poller.start()
        t = W.run_blocking(workload)
        stop.set()
        poller.join()
        energy = G.energy_j() - e0
    finally:
        stop.set()
        G.reset_cap()
    return {"workload": workload, "time_s": t, "energy_j": energy,
            "edp": energy * t, "steps": "",
            "mean_cap_mhz": "" if cap is None else cap,
            "mean_power_cap_w": "", "agent_cpu_s": "", "step_ms_median": "",
            "step_ms_p95": "", "max_rss_mb": ""}


# COORD-like heuristic. Threshold and caps come from the TRAINING-set
# frequency sweep (README): compute-bound programs (mean util 79-97 %)
# have their best EDP at a 2400 MHz cap, memory/host-bound ones (mean util
# 0-6 %) save most at the lowest cap we allow (1800 MHz). The real COORD
# also coordinates memory power, which the 4070 Ti cannot cap.
COORD_UTIL_THRESHOLD = 50.0
COORD_CAPS = {"compute": 2400, "memory": 1800}


def profile(workload):
    """One default-clock run sampled every 300 ms; mean util / mem util."""
    G.reset_cap()
    samples, done = [], threading.Event()

    def sampler():
        while not done.is_set():
            samples.append(G.read())
            time.sleep(D.SAMPLE_INTERVAL)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        t = W.run_blocking(workload)
    finally:
        done.set()
        th.join()
    util = float(np.mean([x["util"] for x in samples]))
    mem = float(np.mean([x["mem_util"] for x in samples]))
    category = "compute" if util >= COORD_UTIL_THRESHOLD else "memory"
    return {"workload": workload, "time_s": t, "mean_util": util,
            "mean_mem_util": mem, "category": category,
            "cap_mhz": COORD_CAPS[category]}


def load_model_spec(path):
    """(variant, agent kind) of a trained model from its run's config."""
    cfg_path = os.path.join(os.path.dirname(path), "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("variant", "freq"), cfg.get("agent", "ddqn")


def geomean(x):
    return float(np.exp(np.mean(np.log(x))))


def summarize(rows, out_path):
    """Every non-baseline policy vs. the NVIDIA default, per workload and
    geometric mean over workloads."""
    names = sorted({r["workload"] for r in rows})
    policies = [p for p in dict.fromkeys(r["policy"] for r in rows)
                if p != "baseline"]
    summary = []
    for policy in policies:
        summary += summarize_policy(rows, names, policy)
    if summary:
        with open(out_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(summary[0]))
            wr.writeheader()
            wr.writerows(summary)
    return summary


def summarize_policy(rows, names, policy):
    summary = []
    for name in names:
        base = [r for r in rows if r["workload"] == name
                and r["policy"] == "baseline"]
        drl = [r for r in rows if r["workload"] == name
               and r["policy"] == policy]
        if not base or not drl:
            continue
        m = {k: (np.mean([r[k] for r in base]), np.mean([r[k] for r in drl]))
             for k in ("energy_j", "time_s", "edp")}
        summary.append({
            "policy": policy, "workload": name, "runs": len(drl),
            "energy_ratio": m["energy_j"][1] / m["energy_j"][0],
            "time_ratio": m["time_s"][1] / m["time_s"][0],
            "edp_ratio": m["edp"][1] / m["edp"][0],
            "energy_saving_pct": 100 * (1 - m["energy_j"][1] / m["energy_j"][0]),
            "perf_loss_pct": 100 * (m["time_s"][1] / m["time_s"][0] - 1),
            "edp_improvement_pct": 100 * (1 - m["edp"][1] / m["edp"][0]),
            "base_energy_cv_pct": 100 * np.std([r["energy_j"] for r in base])
            / m["energy_j"][0],
            "base_time_cv_pct": 100 * np.std([r["time_s"] for r in base])
            / m["time_s"][0],
        })
    if summary:
        g = {k: geomean([s[k] for s in summary])
             for k in ("energy_ratio", "time_ratio", "edp_ratio")}
        summary.append({
            "policy": policy, "workload": "GEOMEAN", "runs": "",
            **g,
            "energy_saving_pct": 100 * (1 - g["energy_ratio"]),
            "perf_loss_pct": 100 * (g["time_ratio"] - 1),
            "edp_improvement_pct": 100 * (1 - g["edp_ratio"]),
            "base_energy_cv_pct": "", "base_time_cv_pct": "",
        })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="shorthand for --models drlcap=PATH")
    ap.add_argument("--models", nargs="*", default=[],
                    metavar="NAME=PATH", help="trained models to evaluate")
    ap.add_argument("--coord", action="store_true",
                    help="also run the COORD-like static heuristic")
    ap.add_argument("--monitor-only", metavar="NAME",
                    help="also run model NAME's full loop without applying "
                    "caps (sampling / controller overhead, Fig. 11)")
    ap.add_argument("--workloads", nargs="+", default=W.TEST)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=A.EPSILON_END)
    ap.add_argument("--no-online", action="store_true",
                    help="freeze the network (ablation)")
    ap.add_argument("--static-caps", type=int, nargs="*", default=[],
                    help="also run fixed frequency caps (MHz) as baselines")
    ap.add_argument("--tag", default="test")
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.models)
    if args.model:
        models["drlcap"] = args.model
    suffix = "_frozen" if args.no_online else ""
    specs = {name + suffix: (path, *load_model_spec(path))
             for name, path in models.items()}

    print("GPU:", G.init())
    if G.gpu_busy_by_others():
        raise SystemExit(f"GPU {G.GPU_ID} busy: {G.gpu_busy_by_others()}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(D.RESULTS, f"eval_{args.tag}_{stamp}")
    os.makedirs(run_dir)
    raw_path = os.path.join(run_dir, "runs.csv")
    step_log = open(os.path.join(run_dir, "steps.csv"), "w", newline="")
    steps = csv.DictWriter(step_log, fieldnames=D.LOG_FIELDS)
    steps.writeheader()

    policies = ["baseline"] + [f"static_{c}" for c in args.static_caps]
    policies += (["coord"] if args.coord else []) + list(specs)
    if args.monitor_only:
        specs["monitor_only"] = specs[args.monitor_only + suffix]
        policies.append("monitor_only")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"policies": policies, "models": specs,
                   "workloads": args.workloads, "repeats": args.repeats,
                   "epsilon": args.epsilon, "online": not args.no_online,
                   "coord_util_threshold": COORD_UTIL_THRESHOLD,
                   "coord_caps": COORD_CAPS,
                   "baseline_nvml_polling_s": D.SAMPLE_INTERVAL,
                   "reward_delay_samples": D.REWARD_DELAY,
                   "reward_mode": D.R.REWARD_MODE,
                   "force_case_a": D.FORCE_CASE_A}, f, indent=2)

    rows, job, coord = [], 0, {}
    try:
        if args.coord:  # COORD knows each program's category ahead of time
            for name in args.workloads:
                coord[name] = profile(name)
                print("coord profile", coord[name], flush=True)
                time.sleep(2.0)
            with open(os.path.join(run_dir, "coord_profile.csv"), "w",
                      newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(coord[name]))
                wr.writeheader()
                wr.writerows(coord.values())
        for rep in range(args.repeats):
            for wi, name in enumerate(args.workloads):
                # rotate the order so no policy always runs first
                k = (rep + wi) % len(policies)
                order = policies[k:] + policies[:k]
                for policy in order:
                    if policy == "baseline":
                        res = baseline_job(name)
                    elif policy.startswith("static_"):
                        res = baseline_job(name, int(policy[7:]))
                    elif policy == "coord":
                        res = baseline_job(name, coord[name]["cap_mhz"])
                    else:
                        path, variant, kind = specs[policy]
                        actions = D.VARIANTS[variant]
                        agent = D.make_agent(kind, len(actions), 1000 + job)
                        agent.load(path)
                        agent.epsilon = args.epsilon
                        agent.learn_start = A.LEARN_START_ONLINE
                        res = D.run_job(agent, name, actions=actions,
                                        learn=not args.no_online,
                                        log=steps, job_id=job,
                                        apply_caps=policy != "monitor_only")
                    res.update(policy=policy, repeat=rep, job=job)
                    rows.append(res)
                    job += 1
                    print(f"rep {rep} {name:15s} {policy:12s} "
                          f"t={res['time_s']:6.1f}s E={res['energy_j']:7.0f}J",
                          flush=True)
                    with open(raw_path, "w", newline="") as f:
                        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                        wr.writeheader()
                        wr.writerows(rows)
                    step_log.flush()
                    time.sleep(2.0)
    finally:
        G.reset_cap()
        step_log.close()

    summary = summarize(rows, os.path.join(run_dir, "summary.csv"))
    for s in summary:
        print(f"{s['policy']:12s} {s['workload']:15s} "
              f"energy saving {s['energy_saving_pct']:6.1f}%"
              f"  perf loss {s['perf_loss_pct']:6.1f}%"
              f"  EDP improvement {s['edp_improvement_pct']:6.1f}%")
    print("saved", run_dir)


if __name__ == "__main__":
    main()
