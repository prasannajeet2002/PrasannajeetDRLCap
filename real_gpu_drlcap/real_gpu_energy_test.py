
import csv
import subprocess
import threading
import time
import statistics
import tensorflow as tf

GPU_ID = 1
MATRIX_SIZE = 2048
MATMUL_ITERATIONS = 200
SAMPLE_INTERVAL = 0.10
OUTPUT_FILE = "real_gpu_energy_samples.csv"


def read_telemetry():
    command = [
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=utilization.gpu,power.draw,clocks.gr,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True
    )
    values = [float(x.strip()) for x in result.stdout.strip().split(",")]

    return {
        "time_s": time.perf_counter(),
        "gpu_util_pct": values[0],
        "power_w": values[1],
        "clock_mhz": values[2],
        "temperature_c": values[3],
    }


def run_test():
    samples = []
    stop_sampling = threading.Event()

    def sampler():
        while not stop_sampling.is_set():
            try:
                samples.append(read_telemetry())
            except (ValueError, subprocess.SubprocessError) as exc:
                print("Telemetry warning:", exc)

            stop_sampling.wait(SAMPLE_INTERVAL)

    # Warm up TensorFlow and the GPU before collecting measurements.
    with tf.device("/GPU:0"):
        a = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
        b = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
        _ = tf.matmul(a, b).numpy()

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    try:
        # Allow telemetry sampling to begin before computation.
        time.sleep(0.3)

        start = time.perf_counter()

        with tf.device("/GPU:0"):
            for _ in range(MATMUL_ITERATIONS):
                result = tf.matmul(a, b)

            # Synchronize: wait for the queued GPU work to finish.
            _ = result.numpy()

        elapsed = time.perf_counter() - start

    finally:
        stop_sampling.set()
        thread.join()

    # Sort samples and estimate energy with trapezoidal integration.
    samples.sort(key=lambda row: row["time_s"])
    energy_j = 0.0
    integrated_duration_s = 0.0

    for previous, current in zip(samples, samples[1:]):
        dt = current["time_s"] - previous["time_s"]

        if 0 < dt <= 1.0:
            energy_j += (
                (previous["power_w"] + current["power_w"]) / 2.0
            ) * dt
            integrated_duration_s += dt

    if samples:
        power_values = [s["power_w"] for s in samples]
        clock_values = [s["clock_mhz"] for s in samples]
        util_values = [s["gpu_util_pct"] for s in samples]

        print(f"Workload time: {elapsed:.4f} s")
        print(f"Matmuls completed: {MATMUL_ITERATIONS}")
        print(f"Mean power during sampling: {statistics.mean(power_values):.2f} W")
        print(f"Maximum sampled power: {max(power_values):.2f} W")
        print(f"Maximum sampled clock: {max(clock_values):.0f} MHz")
        print(f"Maximum sampled GPU utilization: {max(util_values):.1f}%")
        print(f"Integrated sampling duration: {integrated_duration_s:.3f} s")
        print(f"Estimated energy over sampled interval: {energy_j:.2f} J")
        print(f"Telemetry samples: {len(samples)}")

        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "time_s", "gpu_util_pct", "power_w",
                "clock_mhz", "temperature_c"
            ])
            writer.writeheader()
            writer.writerows(samples)

        print(f"Samples saved to {OUTPUT_FILE}")
    else:
        print("No telemetry samples were collected.")


if __name__ == "__main__":
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("Using default automatic GPU clock management.")
    run_test()
