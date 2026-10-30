
import csv
import time
import subprocess
import tensorflow as tf

GPU_ID = 1
WARMUP_RUNS = 3
MEASURED_RUNS = 10
MATRIX_SIZE = 2048
OUTPUT_FILE = "real_gpu_baseline.csv"


def read_gpu_telemetry():
    cmd = [
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=utilization.gpu,power.draw,clocks.gr,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True
    )

    values = [float(x.strip()) for x in result.stdout.strip().split(",")]

    return {
        "gpu_util_pct": values[0],
        "power_w": values[1],
        "clock_mhz": values[2],
        "temperature_c": values[3],
    }


def run_workload():
    # CUDA_VISIBLE_DEVICES=1 makes this physical GPU appear as /GPU:0.
    with tf.device("/GPU:0"):
        a = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
        b = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])

        # Synchronize GPU execution before timing ends.
        result = tf.matmul(a, b)
        _ = result.numpy()

        start = time.perf_counter()

        for _ in range(3):
            result = tf.matmul(a, b)

        _ = result.numpy()
        elapsed = time.perf_counter() - start

    return elapsed


def main():
    print("TensorFlow:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
    print("GPU:", GPU_ID)
    print("Clock policy: default automatic management")
    print("No clock or power settings will be changed.")

    rows = []

    for i in range(WARMUP_RUNS + MEASURED_RUNS):
        elapsed = run_workload()
        telemetry = read_gpu_telemetry()

        row = {
            "run": i + 1,
            "phase": "warmup" if i < WARMUP_RUNS else "measured",
            "workload_time_s": elapsed,
            **telemetry,
        }

        if i >= WARMUP_RUNS:
            rows.append(row)

        print(
            f"Run {i + 1:02d} | "
            f"time={elapsed:.4f}s | "
            f"power={telemetry['power_w']:.2f}W | "
            f"clock={telemetry['clock_mhz']:.0f}MHz | "
            f"temp={telemetry['temperature_c']:.0f}C"
        )

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    times = [r["workload_time_s"] for r in rows]

    print(f"\nMeasured runs: {len(times)}")
    print(f"Mean execution time: {sum(times) / len(times):.4f}s")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
