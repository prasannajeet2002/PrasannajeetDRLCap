"""Diagnostic: why does monitor-only DRLCap use ~3 % less GPU energy than
the plain NVIDIA-default baseline when it never changes a cap?

The monitor-only run differs from the baseline in two ways: (1) NVML is
polled every 300 ms, (2) the agent (TensorFlow on the CPU) computes and
learns each step. This script runs, interleaved and with rotating order,
  baseline   plain default-clock job (evaluate.baseline_job)
  poll       default-clock job + a thread calling gpu_control.read()
             every 300 ms (sampling only, no agent)
  monitor    monitor-only DRLCap job (sampling + agent, caps not applied)
so the two causes can be told apart.

    python diag_monitor_effect.py --model results/train_freq_s0_<stamp>/model_final.keras
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import threading
import time

import agent as A
import drlcap as D
import evaluate as EV
import gpu_control as G
import workloads as W


def poll_job(workload):
    G.reset_cap()
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            G.read()
            stop.wait(D.SAMPLE_INTERVAL)

    th = threading.Thread(target=poll, daemon=True)
    e0 = G.energy_j()
    th.start()
    try:
        t = W.run_blocking(workload)
    finally:
        stop.set()
        th.join()
    energy = G.energy_j() - e0
    return {"workload": workload, "time_s": t, "energy_j": energy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workloads", nargs="+",
                    default=["b+tree", "lud", "myocyte", "heartwall", "vgg16"])
    args = ap.parse_args()

    G.init()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join("results", f"diag_monitor_effect_{stamp}.csv")
    policies = ["baseline", "poll", "monitor"]
    rows = []
    try:
        for rep in range(args.repeats):
            for wi, name in enumerate(args.workloads):
                k = (rep + wi) % len(policies)
                for policy in policies[k:] + policies[:k]:
                    if policy == "baseline":
                        res = EV.baseline_job(name)
                    elif policy == "poll":
                        res = poll_job(name)
                    else:
                        actions = D.VARIANTS["freq"]
                        agent = D.make_agent("ddqn", len(actions), 5000 + rep)
                        agent.load(args.model)
                        agent.epsilon = 0.1
                        agent.learn_start = A.LEARN_START_ONLINE
                        res = D.run_job(agent, name, actions=actions,
                                        learn=True, apply_caps=False)
                    row = {"policy": policy, "repeat": rep,
                           "workload": name, "time_s": res["time_s"],
                           "energy_j": res["energy_j"]}
                    rows.append(row)
                    print(f"rep {rep} {name:12s} {policy:9s} "
                          f"t={row['time_s']:6.1f}s E={row['energy_j']:7.0f}J",
                          flush=True)
                    with open(out, "w", newline="") as f:
                        wr = csv.DictWriter(f, fieldnames=list(row))
                        wr.writeheader()
                        wr.writerows(rows)
                    time.sleep(2.0)
    finally:
        G.reset_cap()

    print("workload      poll/base E   monitor/base E   poll/base t   monitor/base t")
    for name in args.workloads:
        m = {p: [r for r in rows if r["workload"] == name and r["policy"] == p]
             for p in policies}
        avg = {p: (sum(r["energy_j"] for r in v) / len(v),
                   sum(r["time_s"] for r in v) / len(v)) for p, v in m.items()}
        b = avg["baseline"]
        print(f"{name:12s}  {avg['poll'][0] / b[0]:.3f}         "
              f"{avg['monitor'][0] / b[0]:.3f}            "
              f"{avg['poll'][1] / b[1]:.3f}         {avg['monitor'][1] / b[1]:.3f}")
    print("saved", out)


if __name__ == "__main__":
    main()
