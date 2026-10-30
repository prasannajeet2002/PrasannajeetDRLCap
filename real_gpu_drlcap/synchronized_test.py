import csv
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

GPU_ID = "1"
telemetry_file = "synchronized_telemetry.csv"
workload_file = "synchronized_workload.log"

stop_event = threading.Event()

def collect_telemetry():
    query = (
        "timestamp,utilization.gpu,power.draw,"
        "clocks.gr,temperature.gpu"
    )

    with open(telemetry_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "gpu_util_percent",
            "power_w", "graphics_clock_mhz", "temperature_c"
        ])

        while not stop_event.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi", "-i", GPU_ID,
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                fields = [x.strip() for x in result.stdout.strip().split(",")]
                writer.writerow(fields)
                f.flush()
            else:
                print("Telemetry warning:", result.stderr.strip())

            stop_event.wait(0.25)

print("Starting synchronized GPU diagnostic...")
thread = threading.Thread(target=collect_telemetry, daemon=True)
thread.start()

start = time.time()
try:
    with open(workload_file, "w") as log:
        result = subprocess.run(
            [
                sys.executable, "-u", "gpu_diagnostic.py"
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    elapsed = time.time() - start
finally:
    stop_event.set()
    thread.join()

print(f"Workload exit code: {result.returncode}")
print(f"Total elapsed time: {elapsed:.2f} seconds")
print(f"Workload log: {workload_file}")
print(f"Telemetry log: {telemetry_file}")

with open(workload_file) as f:
    print("\n--- Workload output ---")
    print(f.read())
