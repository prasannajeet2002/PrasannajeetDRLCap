"""Benchmark table for the reviewer (item 5): one row per application with
suite, input, execution command, single-run runtime, repeats per job, job
length, GPU confirmation, mean utilisation / memory utilisation / power,
the compute / memory / mixed label derived from them, and the split.

    python bench_table.py single1.csv [single2.csv ...] --job job.csv
Single-run probe files give the single-run time and exit status (later
files override earlier ones). The --job file (bench_probe.py --job: all
repeats of a training job back to back) gives the job length, the GPU
confirmation and the utilisation / power means for training and
validation programs, since single runs of < 1 s are mostly start-up. Writes
results/benchmark_table.csv and prints a summary per suite and split.
"""

import csv
import sys

import workloads as W


def label(row):
    """Coarse workload character from the probe's NVML means (a guide for
    coverage, not a classification the controller uses)."""
    try:
        u, m = float(row["mean_util"]), float(row["mean_mem_util"])
    except (TypeError, ValueError):
        return "unknown"
    if u < 30:
        return "low-GPU / host-bound"
    if m >= 40:
        return "memory-heavy"
    if m < 15:
        return "compute-heavy"
    return "mixed"


def main():
    args = sys.argv[1:]
    job_rows = {}
    if "--job" in args:
        i = args.index("--job")
        with open(args[i + 1]) as f:
            job_rows = {r["name"]: r for r in csv.DictReader(f)}
        args = args[:i] + args[i + 2:]
    rows = {}
    for path in args:
        with open(path) as f:
            for r in csv.DictReader(f):
                rows[r["name"]] = r
    out = []
    for name, w in W.WORKLOADS.items():
        r = rows.get(name)
        if r is not None and r["exit"] not in ("0", "timeout", "error-in-output") \
                and int(r["exit"]) in w.get("ok_exit", (0,)):
            r = {**r, "exit": "0"}  # documented normal finish (QE: exit 2)
        single = float(r["runtime_s"]) if r and r["exit"] == "0" else None
        j = job_rows.get(name)
        job_s = None
        if j is not None and j["exit"] == "0":
            job_s = float(j["runtime_s"])
            r = {**r, **{k: j[k] for k in ("on_gpu1", "on_other_gpu", "mean_util",
                                              "mean_mem_util", "mean_power_w")}}
        out.append({
            "application": name, "suite": w["suite"], "split": w["split"],
            "input": w["input"], "command": " ".join(w["cmd"]),
            "cwd": w["cwd"],
            "status": "not probed" if r is None else
                      ("ok" if r["exit"] == "0" else f"failed (exit {r['exit']})"),
            "single_run_s": "" if single is None else round(single, 2),
            "repeat": w["repeat"],
            "job_s": round(job_s, 1) if job_s is not None else
                     ("" if single is None else round(single * w["repeat"], 1)),
            "job_s_source": "measured (whole job)" if job_s is not None else
                            "single run x repeat",
            "on_gpu1": "" if r is None else r["on_gpu1"],
            "on_other_gpu": "" if r is None else r["on_other_gpu"],
            "mean_util": "" if r is None else r["mean_util"],
            "mean_mem_util": "" if r is None else r["mean_mem_util"],
            "mean_power_w": "" if r is None else r["mean_power_w"],
            "character": "" if r is None else label(r),
        })
    with open("results/benchmark_table.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    for split in ("train", "val", "test"):
        sel = [o for o in out if o["split"] == split]
        ok = [o for o in sel if o["status"] == "ok"]
        print(f"{split}: {len(ok)}/{len(sel)} run OK")
        for suite in sorted({o["suite"] for o in sel}):
            s = [o for o in sel if o["suite"] == suite]
            chars = {}
            for o in s:
                chars[o["character"]] = chars.get(o["character"], 0) + 1
            print(f"   {suite:10s} {sum(o['status'] == 'ok' for o in s)}/{len(s)}  {chars}")
    print("saved results/benchmark_table.csv")


if __name__ == "__main__":
    main()
