"""Evaluate DRLCap against the NVIDIA default policy (paper Sec. V-D, VI-A).

For every workload and repeat, runs (in alternating order to spread drift
and temperature effects):
  baseline  no cap, NVIDIA default clock management
  drlcap    trained model, fresh copy per job, epsilon fixed at the final
            training value (0.1) and online learning ON, as the paper
            deploys it (Sec. IV-D)
Energy comes from the NVML energy counter, runtime from wall clock.

Reports per-workload mean energy saving, performance loss and EDP
improvement vs. baseline, plus the geometric mean over workloads of the
energy / time / EDP ratios.

    python evaluate.py --model results/train_.../model_final.keras \
        --repeats 3 [--workloads lud lavaMD ...]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import time

import numpy as np

import agent as A
import drlcap as D
import gpu_control as G
import workloads as W


def baseline_job(workload):
    G.reset_cap()
    e0 = G.energy_j()
    t = W.run_blocking(workload)
    energy = G.energy_j() - e0
    return {"workload": workload, "time_s": t, "energy_j": energy,
            "edp": energy * t, "steps": "", "mean_cap_mhz": ""}


def geomean(x):
    return float(np.exp(np.mean(np.log(x))))


def summarize(rows, out_path):
    names = sorted({r["workload"] for r in rows})
    summary = []
    for name in names:
        base = [r for r in rows if r["workload"] == name
                and r["policy"] == "baseline"]
        drl = [r for r in rows if r["workload"] == name
               and r["policy"] == "drlcap"]
        if not base or not drl:
            continue
        m = {k: (np.mean([r[k] for r in base]), np.mean([r[k] for r in drl]))
             for k in ("energy_j", "time_s", "edp")}
        summary.append({
            "workload": name, "runs": len(drl),
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
            "workload": "GEOMEAN", "runs": "",
            **g,
            "energy_saving_pct": 100 * (1 - g["energy_ratio"]),
            "perf_loss_pct": 100 * (g["time_ratio"] - 1),
            "edp_improvement_pct": 100 * (1 - g["edp_ratio"]),
            "base_energy_cv_pct": "", "base_time_cv_pct": "",
        })
        with open(out_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(summary[0]))
            wr.writeheader()
            wr.writerows(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workloads", nargs="+", default=W.TEST)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=A.EPSILON_END)
    ap.add_argument("--no-online", action="store_true",
                    help="freeze the network (ablation)")
    ap.add_argument("--tag", default="test")
    args = ap.parse_args()

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

    rows, job = [], 0
    try:
        for rep in range(args.repeats):
            for name in args.workloads:
                order = ["baseline", "drlcap"]
                if (rep + job) % 2:
                    order.reverse()
                for policy in order:
                    if policy == "baseline":
                        res = baseline_job(name)
                    else:
                        agent = A.DDQNAgent(D.STATE_SIZE, len(D.ACTIONS),
                                            seed=1000 + job)
                        agent.load(args.model)
                        agent.epsilon = args.epsilon
                        res = D.run_job(agent, name,
                                        learn=not args.no_online,
                                        log=steps, job_id=job)
                    res.update(policy=policy, repeat=rep, job=job)
                    rows.append(res)
                    job += 1
                    print(f"rep {rep} {name:15s} {policy:8s} "
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
        print(f"{s['workload']:15s} energy saving {s['energy_saving_pct']:6.1f}%"
              f"  perf loss {s['perf_loss_pct']:6.1f}%"
              f"  EDP improvement {s['edp_improvement_pct']:6.1f}%")
    print("saved", run_dir)


if __name__ == "__main__":
    main()
