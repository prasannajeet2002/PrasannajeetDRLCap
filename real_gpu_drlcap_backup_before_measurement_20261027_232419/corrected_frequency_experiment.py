import csv
import time
import threading
import subprocess
import statistics
from datetime import datetime

import tensorflow as tf

GPU_ID = 1
FIXED_CLOCK = 1800
ITERATIONS = 100
REPEATS = 3
SAMPLE_INTERVAL = 0.25

SUMMARY = "corrected_frequency_summary.csv"
RAW = "corrected_frequency_telemetry.csv"

def command(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n{r.stderr}"
        )
    return r.stdout.strip()

def set_clock(mode):
    if mode == "AUTO":
        command(["sudo", "nvidia-smi", "-i", str(GPU_ID), "-rgc"])
    else:
        command([
            "sudo", "nvidia-smi", "-i", str(GPU_ID),
            "-lgc", f"{FIXED_CLOCK},{FIXED_CLOCK}"
        ])

def read_gpu():
    out = command([
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=power.draw,utilization.gpu,temperature.gpu,clocks.gr",
        "--format=csv,noheader,nounits"
    ])
    return [float(v.strip()) for v in out.split(",")]

print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("No visible TensorFlow GPU.")

# Fixed input matrices; synchronize every operation.
with tf.device("/GPU:0"):
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    _ = tf.matmul(a, b).numpy()

plan = [
    ("AUTO", 1), ("FIXED_1800", 1),
    ("FIXED_1800", 2), ("AUTO", 2),
    ("AUTO", 3), ("FIXED_1800", 3)
]

summaries = []
raw_samples = []

try:
    for mode, rep in plan:
        trial = f"{mode}_rep{rep}"
        print(f"\nStarting {trial}")

        set_clock(mode)
        time.sleep(2)

        # Warm-up occurs before the measured interval.
        with tf.device("/GPU:0"):
            for _ in range(5):
                _ = tf.matmul(a, b).numpy()

        window = {"start": None, "end": None}
        stop = threading.Event()
        samples = []

        def monitor():
            while not stop.is_set():
                sample_start = time.perf_counter()
                try:
                    p, u, temp, clock = read_gpu()
                    start = window["start"]
                    end = window["end"]

                    # Keep only samples initiated inside timed execution.
                    if start is not None and sample_start >= start and (
                        end is None or sample_start <= end
                    ):
                        samples.append({
                            "timestamp": datetime.now().isoformat(
                                timespec="milliseconds"
                            ),
                            "elapsed_s": sample_start - start,
                            "trial": trial,
                            "mode": mode,
                            "power_w": p,
                            "gpu_util_percent": u,
                            "temperature_c": temp,
                            "clock_mhz": clock
                        })
                except Exception as exc:
                    print("Telemetry warning:", exc)

                stop.wait(SAMPLE_INTERVAL)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        time.sleep(0.2)

        start = time.perf_counter()
        window["start"] = start

        try:
            with tf.device("/GPU:0"):
                for _ in range(ITERATIONS):
                    _ = tf.matmul(a, b).numpy()
        finally:
            end = time.perf_counter()
            window["end"] = end

        # Let any in-flight telemetry query finish.
        time.sleep(0.3)
        stop.set()
        thread.join(timeout=5)

        duration = end - start
        raw_samples.extend(samples)

        if len(samples) < 5:
            raise RuntimeError(
                f"Only {len(samples)} in-window telemetry samples for {trial}; "
                "results are insufficient. Check the GPU and rerun."
            )

        powers = [s["power_w"] for s in samples]
        utils = [s["gpu_util_percent"] for s in samples]
        clocks = [s["clock_mhz"] for s in samples]
        temps = [s["temperature_c"] for s in samples]

        avg_power = statistics.mean(powers)
        energy = avg_power * duration

        row = {
            "trial": trial,
            "mode": mode,
            "repetition": rep,
            "iterations": ITERATIONS,
            "execution_time_s": duration,
            "iteration_time_ms": duration * 1000 / ITERATIONS,
            "telemetry_samples": len(samples),
            "average_power_w": avg_power,
            "peak_sampled_power_w": max(powers),
            "average_gpu_util_percent": statistics.mean(utils),
            "average_clock_mhz": statistics.mean(clocks),
            "average_temperature_c": statistics.mean(temps),
            "estimated_energy_j": energy,
            "estimated_edp_j_s": energy * duration
        }
        summaries.append(row)

        print(
            f"{trial}: time={duration:.3f}s, "
            f"samples={len(samples)}, power={avg_power:.2f}W, "
            f"clock={statistics.mean(clocks):.0f}MHz, "
            f"energy~{energy:.2f}J, EDP~{energy*duration:.2f}"
        )

finally:
    try:
        set_clock("AUTO")
        print("\nAutomatic clock management restored.")
    except Exception as exc:
        print("\nWARNING: Clock restoration failed:", exc)
        print("Run: sudo nvidia-smi -i 1 -rgc")

# Save results, including partial results if a later trial fails.
if summaries:
    with open(SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

if raw_samples:
    fields = [
        "timestamp", "elapsed_s", "trial", "mode", "power_w",
        "gpu_util_percent", "temperature_c", "clock_mhz"
    ]
    with open(RAW, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_samples)

print("\n========== CORRECTED SUMMARY ==========")
for mode in ("AUTO", "FIXED_1800"):
    rows = [r for r in summaries if r["mode"] == mode]
    if not rows:
        continue
    print(f"\n{mode}: {len(rows)} trials")
    for field in (
        "execution_time_s", "average_power_w",
        "estimated_energy_j", "estimated_edp_j_s",
        "average_clock_mhz"
    ):
        values = [r[field] for r in rows]
        print(f"  {field}: mean={statistics.mean(values):.3f}")

print("\nSaved:", SUMMARY)
print("Saved:", RAW)
print("Energy and EDP are estimates from sampled power, not direct energy-counter readings.")
