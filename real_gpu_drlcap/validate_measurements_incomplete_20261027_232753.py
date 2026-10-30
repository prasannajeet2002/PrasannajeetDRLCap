import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import tensorflow as tf

GPU_ID = 1
MATRIX_SIZE = 4096
ITERATIONS = 10
SAMPLE_INTERVAL = 0.1

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = Path(f"measurement_validation_{timestamp}.csv")


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

    values = [v.strip() for v in result.stdout.strip().split(",")]

    if len(values) != 7 or any(v in ("", "[Not Supported]", "N/A")
                                for v in values):
        raise RuntimeError(f"Invalid GPU telemetry: {values}")

    return [
        values[0],
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
        float(values[6]),
    ]


def main():
    print(f"TensorFlow: {tf.__version__}")
    print(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Target physical GPU: {GPU_ID}")
    print(f"Output: {output_file}")

    # Warm-up and allocate tensors before timed measurement.
    with tf.device("/GPU:0"):
        a = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
        b = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])

        for _ in range(2):
            c = tf.matmul(a, b)
            c.numpy()

        samples = []
        start = time.perf_counter()
        next_sample = start

        for i in range(ITERATIONS):
            iteration_start = time.perf_counter()

            c = tf.matmul(a, b)
            c.numpy()  # Synchronize so execution time includes GPU work.

            iteration_end = time.perf_counter()

            samples.append({
                "event": "iteration",
                "iteration": i + 1,
                "elapsed_s": iteration_end - start,
                "iteration_time_s": iteration_end - iteration_start,
            })

            print(
                f"Iteration {i + 1}/{ITERATIONS}: "
                f"{iteration_end - iteration_start:.4f} s"
            )

            # Sample telemetry during the workload, not just afterward.
            while time.perf_counter() >= next_sample:
                gpu = read_gpu()
                samples.append({
                    "event": "telemetry",
                    "timestamp": gpu[0],
                    "elapsed_s": time.perf_counter() - start,
                    "power_w": gpu[1],
                    "temperature_c": gpu[2],
                    "gpu_util_percent": gpu[3],
                    "memory_util_percent": gpu[4],
                    "graphics_clock_mhz": gpu[5],
                    "memory_used_mib": gpu[6],
                })
                next_sample += SAMPLE_INTERVAL

        total_time = time.perf_counter() - start

    with output_file.open("w", newline="") as f:
        fields = [
            "event", "iteration", "timestamp", "elapsed_s",
            "iteration_time_s", "power_w", "temperature_c",
            "gpu_util_percent", "memory_util_percent",
            "graphics_clock_mhz", "memory_used_mib",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)

    telemetry = [s for s in samples if s["event"] == "telemetry"]

    # Approximate energy from measured power samples using trapezoidal
    # integration over elapsed time. This is GPU-board energy, not
    # whole-system energy, and depends on telemetry sampling resolution.
    energy_j = 0.0
    if len(telemetry) >= 2:
        for left, right in zip(telemetry, telemetry[1:]):
            dt = right["elapsed_s"] - left["elapsed_s"]
            energy_j += (left["power_w"] + right["power_w"]) * 0.5 * dt

    print("\nMeasurement validation complete")
    print(f"Workload runtime: {total_time:.3f} s")
    print(f"Iterations: {ITERATIONS}")
    print(f"Telemetry samples: {len(telemetry)}")
    print(f"Approx. measured GPU energy: {energy_j:.2f} J")
    print(f"CSV saved to: {output_file}")


if __name__ == "__main__":
    main()
