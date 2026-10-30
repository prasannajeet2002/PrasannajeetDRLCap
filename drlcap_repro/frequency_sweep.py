"""Static frequency-cap sweep on the RTX 4070 Ti (DRLCap Sec. III / Fig. 1).

For each workload, run it under the NVIDIA default policy and under a set
of fixed frequency caps, measuring runtime, energy (NVML energy counter)
and EDP. The results show (a) how much energy capping can save on this GPU
and (b) where the slowdown becomes large, which is how the paper picks the
lower end of the action range (Sec. V-C, Table IV).

Usage:
    python frequency_sweep.py --workloads hotspot gaussian --caps 1200 1800 2400
"""

import argparse
import csv
import os
import threading
import time

import gpu_control as G
import workloads as W

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def measure(name, cap):
    """Run one job under `cap` (None = NVIDIA default). Returns a row dict."""
    if cap is None:
        G.reset_cap()
    else:
        G.set_cap(cap)
    time.sleep(1.0)

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(G.read())
            time.sleep(0.1)

    th = threading.Thread(target=sampler, daemon=True)
    e0 = G.energy_j()
    th.start()
    elapsed = W.run_blocking(name)
    stop.set()
    th.join()
    energy = G.energy_j() - e0
    G.reset_cap()

    busy = [s for s in samples if s["util"] > 0] or samples
    return {
        "workload": name,
        "cap_mhz": "default" if cap is None else cap,
        "time_s": round(elapsed, 3),
        "energy_j": round(energy, 2),
        "edp": round(energy * elapsed, 2),
        "avg_power_w": round(energy / elapsed, 2),
        "mean_busy_clock_mhz": round(
            sum(s["clock"] for s in busy) / len(busy), 1),
        "max_clock_mhz": max(s["clock"] for s in samples),
        "mean_util": round(sum(s["util"] for s in samples) / len(samples), 1),
        "mean_mem_util": round(
            sum(s["mem_util"] for s in samples) / len(samples), 1),
        "max_temp_c": max(s["temp"] for s in samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=W.TRAIN)
    ap.add_argument("--caps", nargs="+", type=int,
                    default=[1200, 1500, 1800, 2100, 2400, 2700])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("GPU:", G.init())
    if G.gpu_busy_by_others():
        raise SystemExit(f"GPU {G.GPU_ID} busy: {G.gpu_busy_by_others()}")

    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(
        RESULTS, f"frequency_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    rows = []
    try:
        for name in args.workloads:
            for rep in range(args.repeats):
                for cap in [None] + args.caps:
                    row = measure(name, cap)
                    row["repeat"] = rep
                    rows.append(row)
                    print(row, flush=True)
                    with open(out, "w", newline="") as f:
                        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                        wr.writeheader()
                        wr.writerows(rows)
    finally:
        G.reset_cap()
    print("saved", out)


if __name__ == "__main__":
    main()
