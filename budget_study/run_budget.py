"""Shield experiment on GPU 1: does a slowdown budget hold, and at what energy?

  python run_budget.py [--delta 5] [--margin 1] [--repeats 3]

Policies per program and repeat, in rotating order:
  baseline       NVIDIA default
  drlcap         f2 selected model, online learning (as in the f2 evaluation)
  drlcap_shield  the same model wrapped by the budget shield (shield.py)
  lowest_safe    no learning: always the lowest cap the shield allows
Reuses drlcap_faithful and the thermal_study logger unchanged.
"""

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "thermal_study"))
sys.path.insert(0, HERE)

import run_thermal as RT  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="" first)
from run_thermal import A, D, E, G, W  # noqa: E402

import pynvml  # noqa: E402

import shield as S  # noqa: E402

POLICIES = ["baseline", "drlcap", "drlcap_shield", "lowest_safe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workloads", nargs="+", default=W.TEST)
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    args = ap.parse_args()

    print("GPU:", G.init())
    if G.gpu_busy_by_others():
        raise SystemExit(f"GPU {G.GPU_ID} busy: {G.gpu_busy_by_others()}")
    handle = pynvml.nvmlDeviceGetHandleByIndex(G.GPU_ID)
    out = os.path.join(HERE, "results", f"budget_d{args.delta:g}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out)
    path = os.path.abspath(RT.F2_MODEL)
    variant, kind, settings, actions = E.load_model_spec(path)
    f = open(os.path.join(out, "trace.csv"), "w", newline="")
    wr = csv.DictWriter(f, fieldnames=RT.FIELDS)
    wr.writeheader()
    log = RT.Logger(wr, handle)
    steps_f = open(os.path.join(out, "steps.csv"), "w", newline="")
    steps = csv.DictWriter(steps_f, fieldnames=D.LOG_FIELDS)
    steps.writeheader()
    log.start()
    rows, job, pol = [], 0, args.policies
    try:
        for rep in range(args.repeats):
            for wi, name in enumerate(args.workloads):
                k = (rep + wi) % len(pol)
                for policy in pol[k:] + pol[:k]:
                    shared = D.wait_gpu_free()
                    log.tag = {"exp": "budget", "job": job, "policy": policy,
                               "workload": name, "phase": "run"}
                    est = masked = ""
                    if policy == "baseline":
                        res = E.baseline_job(name)
                    else:
                        E.use_settings(settings)
                        budget = S.Budget(args.delta, args.margin)
                        if policy == "lowest_safe":
                            ag = S.LowestSafe(actions, budget, seed=job)
                        else:
                            ag = D.make_agent(kind, len(actions), 1000 + job)
                            ag.load(path)
                            ag.epsilon = A.EPSILON_END
                            ag.learn_start = A.LEARN_START_ONLINE
                            if policy == "drlcap_shield":
                                ag = S.ShieldedAgent(ag, actions, budget)
                        # Algorithm 1's "compute-bound -> max cap" belongs to
                        # DRLCap; the non-learning reference must not get it
                        D.FORCE_CASE_A = policy != "lowest_safe"
                        try:
                            res = D.run_job(ag, name, actions=actions, learn=True,
                                            log=steps, job_id=job)
                        finally:
                            D.FORCE_CASE_A = True
                        if policy != "drlcap":
                            est = round(100 * budget.slowdown(), 2)
                            masked = ag.masked_steps
                    log.tag = dict(log.tag, phase="gap")
                    res.update(policy=policy, repeat=rep, job=job,
                               est_slowdown_pct=est, masked_steps=masked,
                               delta=args.delta, margin=args.margin,
                               gpu_shared=int(bool(shared)))
                    rows.append(res)
                    with open(os.path.join(out, "runs.csv"), "w", newline="") as g:
                        w = csv.DictWriter(g, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)
                    steps_f.flush()
                    print(f"budget rep {rep} {name:12s} {policy:14s} "
                          f"t={res['time_s']:6.1f}s E={res['energy_j']:7.0f}J est={est}",
                          flush=True)
                    job += 1
                    time.sleep(2.0)
    finally:
        log.close()
        f.close()
        steps_f.close()
        G.reset_cap()
    print("saved", out)


if __name__ == "__main__":
    main()
