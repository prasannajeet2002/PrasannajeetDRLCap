import argparse
import csv
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import tensorflow as tf

GPU_ID = 1
MATRIX_SIZE = 4096
ITERATIONS = 30
SAMPLE_INTERVAL = 0.1


def read_gpu():
    cmd = [
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=timestamp,power.draw,temperature.gpu,"
        "utilization.gpu,utilization.memory,clocks.gr,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=5
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    values = [x.strip() for x in result.stdout.strip().split(",")]
    if len(values) != 7 or any(
        x in ("", "N/A", "[Not Supported]") for x in values
    ):
        raise RuntimeError(f"Invalid telemetry: {values}")

    return {
        "timestamp": values[0],
        "power_w": float(values[1]),
        "temperature_c": float(values[2]),
        "gpu_util_percent": float(values[3]),
        "memory_util_percent": float(values[4]),
        "graphics_clock_mhz": float(values[5]),
        "memory_used_mib": float(values[6]),
    }


def sampler(stop_event, samples, start_time, errors):
    while not stop_event.is_set():
        try:
            row = read_gpu()
            row["elapsed_s"] = time.perf_counter() - start_time
            samples.append(row)
        except Exception as exc:
            errors.append(str(exc))
        stop_event.wait(SAMPLE_INTERVAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="auto")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(f"controlled_{args.label}_{timestamp}.csv")

    print("TensorFlow:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
    print("Target physical GPU:", GPU_ID)
    print(f"Workload: {ITERATIONS} x {MATRIX_SIZE} matrix multiplications")
    print("CSV:", output)

    with tf.device("/GPU:0"):
        a = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
        b = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])

        # Warm up TensorFlow before starting the timed run.
        for _ in range(3):
            tf.matmul(a, b).numpy()

        samples = []
        errors = []
        stop_event = threading.Event()

        start_time = time.perf_counter()
        thread = threading.Thread(
            target=sampler,
            args=(stop_event, samples, start_time, errors),
            daemon=True,
        )
        thread.start()

        iteration_times = []
        try:
            for i in range(ITERATIONS):
                iteration_start = time.perf_counter()
                tf.matmul(a, b).numpy()
                iteration_times.append(
                    time.perf_counter() - iteration_start
                )
                print(
                    f"Iteration {i + 1:02d}/{ITERATIONS}: "
                    f"{iteration_times[-1]:.4f} s"
                )
        finally:
            end_time = time.perf_counter()
            stop_event.set()
            thread.join(timeout=3)

    runtime_s = end_time - start_time

    fields = [
        "timestamp", "elapsed_s", "power_w", "temperature_c",
        "gpu_util_percent", "memory_util_percent",
        "graphics_clock_mhz", "memory_used_mib",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)

    energy_j = 0.0
    for left, right in zip(samples, samples[1:]):
        dt = right["elapsed_s"] - left["elapsed_s"]
        if dt > 0:
            energy_j += (
                (left["power_w"] + right["power_w"]) / 2
            ) * dt

    powers = [s["power_w"] for s in samples]
    utils = [s["gpu_util_percent"] for s in samples]
    clocks = [s["graphics_clock_mhz"] for s in samples]

    print("\n--- Benchmark summary ---")
    print(f"Runtime: {runtime_s:.3f} s")
    print(f"Mean iteration time: {sum(iteration_times)/len(iteration_times):.4f} s")
    print(f"Telemetry samples: {len(samples)}")
    if samples:
        print(f"Mean sampled power: {sum(powers)/len(powers):.2f} W")
        print(f"Mean sampled GPU utilization: {sum(utils)/len(utils):.1f}%")
        print(f"Mean sampled graphics clock: {sum(clocks)/len(clocks):.0f} MHz")
        print(f"Approx. energy over sampled interval: {energy_j:.2f} J")
    if errors:
        print("Telemetry errors:", errors[:3])
    print("Saved:", output)


if __name__ == "__main__":
    main()
