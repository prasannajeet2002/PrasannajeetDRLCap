"""Fixed-cap sweep of the 10 test programs on GPU 1, logged every 300 ms.

  python sweep.py [--caps 2100 2235 2370 2505 2745] [--repeats 2]

Each program runs under NVIDIA default and under each fixed frequency cap.
Gives (1) the true slowdown and energy of every program at every cap, the
ground truth for the online slowdown estimator, and (2) the best single cap
per program within a slowdown budget, the target for the budgeted
controller. Reuses drlcap_faithful (evaluate.baseline_job) and the
thermal_study logger unchanged. Only frequency caps are changed; every job
ends with the cap reset.
"""

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "thermal_study"))

import run_thermal as RT  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="" first)
from run_thermal import D, E, G, W  # noqa: E402

import pynvml  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", type=int, nargs="+",
                    default=[2100, 2235, 2370, 2505, 2745])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--workloads", nargs="+", default=W.TEST)
    args = ap.parse_args()

    print("GPU:", G.init())
    if G.gpu_busy_by_others():
        raise SystemExit(f"GPU {G.GPU_ID} busy: {G.gpu_busy_by_others()}")
    handle = pynvml.nvmlDeviceGetHandleByIndex(G.GPU_ID)
    out = os.path.join(HERE, "results", f"sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out)
    f = open(os.path.join(out, "trace.csv"), "w", newline="")
    wr = csv.DictWriter(f, fieldnames=RT.FIELDS)
    wr.writeheader()
    log = RT.Logger(wr, handle)
    log.start()
    settings = [None] + args.caps
    rows, job = [], 0
    try:
        for rep in range(args.repeats):
            for wi, name in enumerate(args.workloads):
                k = (rep + wi) % len(settings)
                for cap in settings[k:] + settings[:k]:
                    shared = D.wait_gpu_free()
                    policy = "baseline" if cap is None else "fixed"
                    log.tag = {"exp": "sweep", "job": job, "policy": policy,
                               "workload": name, "phase": "run",
                               "cap_mhz": "" if cap is None else cap}
                    res = E.baseline_job(name, cap)
                    log.tag = dict(log.tag, phase="gap")
                    res.update(policy=policy, cap_mhz=res["mean_cap_mhz"],
                               repeat=rep, job=job, gpu_shared=int(bool(shared)))
                    rows.append(res)
                    with open(os.path.join(out, "runs.csv"), "w", newline="") as g:
                        w = csv.DictWriter(g, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)
                    print(f"sweep rep {rep} {name:12s} cap {str(cap):5s} "
                          f"t={res['time_s']:6.1f}s E={res['energy_j']:7.0f}J",
                          flush=True)
                    job += 1
                    time.sleep(2.0)
    finally:
        log.close()
        f.close()
        G.reset_cap()
    print("saved", out)


if __name__ == "__main__":
    main()
