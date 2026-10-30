import csv
import time
import threading
import subprocess
import statistics
from datetime import datetime

import tensorflow as tf

GPU_ID = 1
OUTPUT = "clean_benchmark_4070ti.csv"
INTERVAL = 0.5
COOLDOWN = 10
IDLE_SECONDS = 10
WORK_SECONDS = 20

def telemetry():
    cmd = [
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=power.draw,utilization.gpu,temperature.gpu,clocks.gr",
        "--format=csv,noheader,nounits"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return [float(x.strip()) for x in r.stdout.strip().split(",")]

print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

# Prepare and warm up the workload before collecting baseline data.
with tf.device("/GPU:0"):
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    _ = tf.matmul(a, b).numpy()

print("\nGPU processes before measurement:")
subprocess.run(["nvidia-smi", "-i", str(GPU_ID)])

print(f"\nCooldown: {COOLDOWN} seconds")
time.sleep(COOLDOWN)

samples = []
phase = {"name": "idle"}
stop = threading.Event()

def monitor():
    while not stop.is_set():
        t = time.perf_counter()
        try:
            p, u, temp, clock = telemetry()
            samples.append({
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "t": t,
                "phase": phase["name"],
                "power_w": p,
                "gpu_util_percent": u,
                "temperature_c": temp,
                "clock_mhz": clock
            })
        except Exception as e:
            print("Telemetry error:", e)
        stop.wait(INTERVAL)

thread = threading.Thread(target=monitor, daemon=True)
thread.start()

print(f"Measuring idle for {IDLE_SECONDS} seconds...")
time.sleep(IDLE_SECONDS)

# Use only the last five seconds of the idle window.
idle_cutoff = time.perf_counter() - 5
idle_samples = [
    s for s in samples
    if s["phase"] == "idle" and s["t"] >= idle_cutoff
]

phase["name"] = "workload"
print(f"Running workload for {WORK_SECONDS} seconds...")
work_start = time.perf_counter()
iterations = 0

while time.perf_counter() - work_start < WORK_SECONDS:
    with tf.device("/GPU:0"):
        _ = tf.matmul(a, b).numpy()
    iterations += 1

work_end = time.perf_counter()
phase["name"] = "finished"
time.sleep(1)
stop.set()
thread.join(timeout=3)

work_samples = [
    s for s in samples
    if s["phase"] == "workload"
    and work_start <= s["t"] <= work_end
]

with open(OUTPUT, "w", newline="") as f:
    fields = [
        "timestamp", "t", "phase", "power_w",
        "gpu_util_percent", "temperature_c", "clock_mhz"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(samples)

print("\n========== CLEAN BENCHMARK ==========")
print("Iterations:", iterations)
duration = work_end - work_start
print(f"Execution window: {duration:.3f} s")
print("Idle samples used:", len(idle_samples))
print("Workload samples used:", len(work_samples))
print("CSV:", OUTPUT)

if len(idle_samples) < 5 or len(work_samples) < 10:
    print("WARNING: Too few samples; do not treat this run as final.")
else:
    idle_power = statistics.mean(s["power_w"] for s in idle_samples)
    powers = [s["power_w"] for s in work_samples]
    avg_power = statistics.mean(powers)
    energy = avg_power * duration
    edp = energy * duration

    print(f"Settled idle power: {idle_power:.2f} W")
    print(f"Average workload power: {avg_power:.2f} W")
    print(f"Peak sampled power: {max(powers):.2f} W")
    print(f"Average GPU utilization: {statistics.mean(s['gpu_util_percent'] for s in work_samples):.2f}%")
    print(f"Average graphics clock: {statistics.mean(s['clock_mhz'] for s in work_samples):.0f} MHz")
    print(f"Approximate workload energy: {energy:.2f} J")
    print(f"Approximate EDP: {edp:.2f} J*s")
    print(f"Dynamic energy above idle (rough): {max(0, avg_power-idle_power)*duration:.2f} J")

print("=====================================")
