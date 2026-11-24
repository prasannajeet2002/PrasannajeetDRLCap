"""Progress-signal ladder step 3: CUPTI kernel timing, overhead and accuracy.

  python kernel_sweep.py [--caps 2100 2370 2505] [--repeats 2] [--workloads ...]

Each program runs under NVIDIA default without tracing ("plain"), and with
the CUPTI kernel tracer (cupti/libkernel_trace.so, loaded through
CUDA_INJECTION64_PATH) under NVIDIA default and each fixed cap, in rotating
order. Gives (1) the tracer's overhead: traced vs plain default time, and
(2) per-kernel durations at every cap, so kernel_test.py can check whether
kernel slowdown predicts program slowdown. Same structure and logging as
sweep.py; only frequency caps are changed and every job ends with the cap
reset.
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

LIB = os.path.join(HERE, "cupti", "libkernel_trace.so")


def tracing(ktrace_dir):
    """Workloads inherit os.environ (workloads.gpu_env), so this attaches the
    tracer to the next job without touching drlcap_faithful."""
    if ktrace_dir is None:
        os.environ.pop("CUDA_INJECTION64_PATH", None)
        os.environ.pop("KTRACE_DIR", None)
    else:
        os.makedirs(ktrace_dir)
        os.environ["CUDA_INJECTION64_PATH"] = LIB
        os.environ["KTRACE_DIR"] = ktrace_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", type=int, nargs="+", default=[2100, 2370, 2505])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--workloads", nargs="+", default=W.TEST)
    ap.add_argument("--tag", default="ksweep")
    args = ap.parse_args()

    print("GPU:", G.init())
    if G.gpu_busy_by_others():
        raise SystemExit(f"GPU {G.GPU_ID} busy: {G.gpu_busy_by_others()}")
    handle = pynvml.nvmlDeviceGetHandleByIndex(G.GPU_ID)
    out = os.path.join(HERE, "results", f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out)
    f = open(os.path.join(out, "trace.csv"), "w", newline="")
    wr = csv.DictWriter(f, fieldnames=RT.FIELDS)
    wr.writeheader()
    log = RT.Logger(wr, handle)
    log.start()
    settings = [("plain", None), ("traced", None)] + [("traced", c) for c in args.caps]
    rows, job = [], 0
    try:
        for rep in range(args.repeats):
            for wi, name in enumerate(args.workloads):
                k = (rep + wi) % len(settings)
                for mode, cap in settings[k:] + settings[:k]:
                    shared = D.wait_gpu_free()
                    policy = "baseline" if cap is None else "fixed"
                    log.tag = {"exp": args.tag, "job": job, "policy": policy,
                               "workload": name, "phase": "run",
                               "cap_mhz": "" if cap is None else cap}
                    tracing(os.path.join(out, "ktrace", f"job{job}") if mode == "traced" else None)
                    try:
                        res = E.baseline_job(name, cap)
                    finally:
                        tracing(None)
                    log.tag = dict(log.tag, phase="gap")
                    res.update(policy=policy, cap_mhz=res["mean_cap_mhz"], mode=mode,
                               repeat=rep, job=job, gpu_shared=int(bool(shared)))
                    rows.append(res)
                    with open(os.path.join(out, "runs.csv"), "w", newline="") as g:
                        w = csv.DictWriter(g, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)
                    print(f"{args.tag} rep {rep} {name:12s} {mode:6s} cap {str(cap):5s} "
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
