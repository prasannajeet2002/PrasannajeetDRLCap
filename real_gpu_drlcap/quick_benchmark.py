import os
import csv
import time
import threading
import subprocess
import statistics
from datetime import datetime

import tensorflow as tf

GPU_ID = 1
OUTPUT = "quick_benchmark_4070ti.csv"
SAMPLE_INTERVAL = 0.5
IDLE_SECONDS = 5
WORK_SECONDS = 20

# Verify TensorFlow is using the intended GPU.
print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("TensorFlow cannot detect a GPU.")

# Sample GPU telemetry in a separate thread.
samples = []
stop_event = threading.Event()
phase = {"value": "idle"}

def read_telemetry():
    cmd = [
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=power.draw,utilization.gpu,temperature.gpu,clocks.gr",
        "--format=csv,noheader,nounits"
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=5
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    values = [float(v.strip()) for v in result.stdout.strip().split(",")]
    return values

def monitor():
    while not stop_event.is_set():
        now = time.perf_counter()
        try:
            power, util, temp, clock = read_telemetry()
            samples.append({
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_s": now - start_all,
                "phase": phase["value"],
                "power_w": power,
                "gpu_util_percent": util,
                "temperature_c": temp,
                "clock_mhz": clock
            })
        except Exception as e:
            print("Telemetry warning:", e)
        stop_event.wait(SAMPLE_INTERVAL)

# Explicitly place computation on TensorFlow's visible GPU.
with tf.device("/GPU:0"):
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    _ = tf.matmul(a, b).numpy()  # Warm-up and synchronize

print("Warm-up completed.")
print(f"Collecting idle telemetry for {IDLE_SECONDS} seconds...")

start_all = time.perf_counter()
thread = threading.Thread(target=monitor, daemon=True)
thread.start()

time.sleep(IDLE_SECONDS)
idle_end = time.perf_counter()

phase["value"] = "workload"
print(f"Running workload for {WORK_SECONDS} seconds...")

work_start = time.perf_counter()
iterations = 0

while time.perf_counter() - work_start < WORK_SECONDS:
    with tf.device("/GPU:0"):
        c = tf.matmul(a, b)
        _ = c.numpy()  # Synchronize GPU execution
    iterations += 1

work_end = time.perf_counter()
phase["value"] = "finished"
time.sleep(1)
stop_event.set()
thread.join(timeout=3)

# Save all recorded samples.
with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "timestamp", "elapsed_s", "phase", "power_w",
        "gpu_util_percent", "temperature_c", "clock_mhz"
    ])
    writer.writeheader()
    writer.writerows(samples)

idle = [s for s in samples if s["phase"] == "idle"]
work = [
    s for s in samples
    if s["phase"] == "workload"
    and work_start <= start_all + s["elapsed_s"] <= work_end
]

duration = work_end - work_start

print("\n========== PRELIMINARY RESULTS ==========")
print("GPU: NVIDIA GeForce RTX 4070 Ti")
print("Completed matrix multiplications:", iterations)
print(f"Workload duration: {duration:.3f} seconds")
print("Telemetry samples during workload:", len(work))
print("CSV saved to:", OUTPUT)

if idle and work:
    idle_power = statistics.mean(s["power_w"] for s in idle)
    avg_power = statistics.mean(s["power_w"] for s in work)
    peak_power = max(s["power_w"] for s in work)
    avg_util = statistics.mean(s["gpu_util_percent"] for s in work)
    avg_clock = statistics.mean(s["clock_mhz"] for s in work)

    # Approximate energy using sampled average power.
    energy_j = avg_power * duration
    dynamic_energy_j = max(0, avg_power - idle_power) * duration
    edp = energy_j * duration

    print(f"Idle power: {idle_power:.2f} W")
    print(f"Average workload power: {avg_power:.2f} W")
    print(f"Peak sampled power: {peak_power:.2f} W")
    print(f"Average GPU utilization: {avg_util:.2f}%")
    print(f"Average graphics clock: {avg_clock:.0f} MHz")
    print(f"Approx. total energy: {energy_j:.2f} J")
    print(f"Approx. dynamic energy above idle: {dynamic_energy_j:.2f} J")
    print(f"Approx. EDP: {edp:.2f} J*s")
else:
    print("INSUFFICIENT TELEMETRY: rerun and check CSV samples.")
    print("Do not report energy or EDP from this run.")

print("=========================================")
