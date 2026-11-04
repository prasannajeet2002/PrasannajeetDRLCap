"""Run each registered benchmark once on the RTX 4070 Ti and record what the
code reviewer asked for (item 5): the exact command, input, runtime, exit
status, and confirmation that it used the intended GPU.

GPU confirmation: while the program runs, NVML is polled every 0.2 s; the
program's process tree must appear in GPU 1's compute-process list and in
no other GPU's list. GPU 1 utilisation, memory utilisation and power are
recorded too, as a first look at whether the program is compute- or
memory-heavy.

    python bench_probe.py                 # every workload
    python bench_probe.py pb-sgemm lulesh # selected ones
    python bench_probe.py --job ...       # whole job: all `repeat` runs
Output: results/bench_probe_<timestamp>.csv and .../logs/<name>.txt
"""

import csv
import os
import subprocess
import sys
import time

import psutil
import pynvml as N

import workloads as W

TIMEOUT_S = 1800


def tree_pids(pid):
    try:
        p = psutil.Process(pid)
        return {pid} | {c.pid for c in p.children(recursive=True)}
    except psutil.NoSuchProcess:
        return set()


def gpu_pids(h):
    try:
        return {p.pid for p in N.nvmlDeviceGetComputeRunningProcesses(h)}
    except N.NVMLError:
        return set()


def probe(name, handles, logdir, whole_job=False):
    w = W.WORKLOADS[name]
    cmd = w["cmd"]
    if whole_job and w["repeat"] > 1:
        # all repeats of one training job, back to back, as one process tree
        cmd = ["bash", "-c", f'for i in $(seq {w["repeat"]}); do "$@" || exit $?; done',
               "_", *cmd]
    W.prepare(name)
    target = handles[W.GPU_ID]
    others = [h for i, h in enumerate(handles) if i != W.GPU_ID]
    util, mem, power = [], [], []
    on_target = on_other = False
    t0 = time.perf_counter()
    with open(os.path.join(logdir, f"{name}.txt"), "w") as log:
        log.write(f"cwd: {w['cwd']}\ncmd: {' '.join(cmd)}\n\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=w["cwd"], env={**W.gpu_env(), **w["env"]},
                             stdout=log, stderr=subprocess.STDOUT)
        timed_out = False
        while p.poll() is None:
            if time.perf_counter() - t0 > TIMEOUT_S:
                p.kill()
                timed_out = True
                break
            pids = tree_pids(p.pid)
            on_target |= bool(pids & gpu_pids(target))
            on_other |= any(pids & gpu_pids(h) for h in others)
            u = N.nvmlDeviceGetUtilizationRates(target)
            util.append(u.gpu)
            mem.append(u.memory)
            power.append(N.nvmlDeviceGetPowerUsage(target) / 1000.0)
            time.sleep(0.2)
        p.wait()
    dt = time.perf_counter() - t0
    # some programs (miniFE) report a CUDA failure but still exit 0
    with open(os.path.join(logdir, f"{name}.txt"), errors="replace") as log:
        text = log.read().lower()
    failed_output = any(s in text for s in
                        ("out of memory", "cuda failure", "threw an exception"))
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else ""
    return {
        "name": name, "suite": w["suite"], "split": w["split"], "input": w["input"],
        "cmd": " ".join(w["cmd"]), "cwd": w["cwd"],
        "runs": w["repeat"] if whole_job else 1,
        "exit": "timeout" if timed_out else
                ("error-in-output" if failed_output and p.returncode == 0
                 else (0 if p.returncode in w.get("ok_exit", (0,)) else p.returncode)),
        "runtime_s": round(dt, 2), "samples": len(util),
        "on_gpu1": on_target, "on_other_gpu": on_other,
        "mean_util": mean(util), "max_util": max(util, default=""),
        "mean_mem_util": mean(mem), "max_mem_util": max(mem, default=""),
        "mean_power_w": mean(power), "max_power_w": round(max(power), 1) if power else "",
    }


def main():
    args = sys.argv[1:]
    whole_job = "--job" in args
    names = [a for a in args if a != "--job"] or list(W.WORKLOADS)
    N.nvmlInit()
    handles = [N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())]
    if "4070 Ti" not in N.nvmlDeviceGetName(handles[W.GPU_ID]):
        raise SystemExit("GPU index 1 is not the RTX 4070 Ti")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    logdir = os.path.join("results", f"bench_probe_{stamp}_logs")
    os.makedirs(logdir)
    out = os.path.join("results", f"bench_probe_{stamp}.csv")
    rows = []
    for name in names:
        r = probe(name, handles, logdir, whole_job)
        rows.append(r)
        print(f"{name:24s} exit={r['exit']!s:7s} {r['runtime_s']:8.2f}s "
              f"gpu1={r['on_gpu1']!s:5s} other={r['on_other_gpu']!s:5s} "
              f"util={r['mean_util']} mem={r['mean_mem_util']} P={r['mean_power_w']}",
              flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
