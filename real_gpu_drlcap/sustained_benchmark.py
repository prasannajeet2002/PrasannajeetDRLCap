#!/usr/bin/env python3
"""Sustained GPU benchmark with timestamped nvidia-smi telemetry."""
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
SAMPLE_INTERVAL = 0.2


def read_gpu():
    cmd = [
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=timestamp,power.draw,temperature.gpu,"
        "utilization.gpu,utilization.memory,clocks.current.graphics,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    values = [x.strip() for x in result.stdout.strip().split(",")]
    if len(values) != 7 or any(x in ("", "N/A", "[Not Supported]") for x in values):
        raise RuntimeError(f"Invalid telemetry row: {values}")
    return {
        "wall_timestamp": values[0],
        "power_w": float(values[1]),
        "temperature_c": float(values[2]),
        "gpu_util_percent": float(values[3]),
        "memory_util_percent": float(values[4]),
        "graphics_clock_mhz": float(values[5]),
        "memory_used_mib": float(values[6]),
    }


def sampler(stop_event, samples, origin, errors):
    while not stop_event.is_set():
        sample_time = time.perf_counter()
        try:
            row = read_gpu()
            row["elapsed_s"] = sample_time - origin
            samples.append(row)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        stop_event.wait(SAMPLE_INTERVAL)


def integrate_power(samples, start_s, end_s):
    inside = [r for r in samples if start_s <= r["elapsed_s"] <= end_s]
    if len(inside) < 2:
        return float("nan"), len(inside)
    energy = 0.0
    for a, b in zip(inside, inside[1:]):
        dt = b["elapsed_s"] - a["elapsed_s"]
        if dt > 0:
            energy += 0.5 * (a["power_w"] + b["power_w"]) * dt
    return energy, len(inside)


def main():
    parser = argparse.ArgumentParser(description="Sustained GPU matrix-multiplication benchmark")
    parser.add_argument("--label", default="run")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--matrix-size", type=int, default=MATRIX_SIZE)
    args = parser.parse_args()
    if args.duration < 5:
        parser.error("--duration must be at least 5 seconds")
    if args.matrix_size < 512:
        parser.error("--matrix-size must be at least 512")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(f"sustained_{args.label}_{run_id}.csv")
    summary_path = Path(f"sustained_{args.label}_{run_id}_summary.txt")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Target physical GPU: {GPU_ID}")
    print(f"Workload: repeated {args.matrix_size}x{args.matrix_size} matrix multiplication")
    print(f"Target duration: {args.duration:.1f} seconds")
    print(f"Telemetry interval target: {SAMPLE_INTERVAL:.1f} seconds")
    print(f"CSV: {csv_path}")

    with tf.device("/GPU:0"):
        a = tf.random.normal([args.matrix_size, args.matrix_size])
        b = tf.random.normal([args.matrix_size, args.matrix_size])
        print("Warming up GPU (excluded from measurement)...")
        for _ in range(3):
            tf.matmul(a, b).numpy()

        samples, errors = [], []
        stop_event = threading.Event()
        origin = time.perf_counter()
        thread = threading.Thread(target=sampler, args=(stop_event, samples, origin, errors), daemon=True)
        thread.start()
        time.sleep(0.5)  # collect a pre-workload sample
        start_s = time.perf_counter() - origin
        start_wall = datetime.now().isoformat(timespec="milliseconds")
        iteration_times = []
        completed = 0
        try:
            while (time.perf_counter() - origin) - start_s < args.duration:
                t0 = time.perf_counter()
                tf.matmul(a, b).numpy()  # synchronizes GPU execution
                iteration_times.append(time.perf_counter() - t0)
                completed += 1
        finally:
            end_s = time.perf_counter() - origin
            end_wall = datetime.now().isoformat(timespec="milliseconds")
            time.sleep(SAMPLE_INTERVAL)
            stop_event.set()
            thread.join(timeout=5)

    fields = [
        "wall_timestamp", "elapsed_s", "power_w", "temperature_c",
        "gpu_util_percent", "memory_util_percent", "graphics_clock_mhz", "memory_used_mib"
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)

    energy_j, n_energy = integrate_power(samples, start_s, end_s)
    in_window = [r for r in samples if start_s <= r["elapsed_s"] <= end_s]
    def mean(key):
        vals = [r[key] for r in in_window]
        return sum(vals) / len(vals) if vals else float("nan")

    runtime = end_s - start_s
    mean_iter = sum(iteration_times) / len(iteration_times) if iteration_times else float("nan")
    lines = [
        f"Label: {args.label}",
        f"TensorFlow version: {tf.__version__}",
        f"Target physical GPU: {GPU_ID}",
        f"Matrix size: {args.matrix_size}x{args.matrix_size}",
        f"Requested duration: {args.duration:.3f} s",
        f"Measured workload runtime: {runtime:.3f} s",
        f"Workload start: {start_wall}",
        f"Workload end: {end_wall}",
        f"Completed matrix multiplications: {completed}",
        f"Mean iteration time: {mean_iter:.5f} s",
        f"Total telemetry samples: {len(samples)}",
        f"Samples inside workload interval: {n_energy}",
        f"Mean sampled power: {mean('power_w'):.2f} W",
        f"Mean sampled GPU utilization: {mean('gpu_util_percent'):.1f}%",
        f"Mean sampled memory utilization: {mean('memory_util_percent'):.1f}%",
        f"Mean sampled graphics clock: {mean('graphics_clock_mhz'):.0f} MHz",
        f"Mean sampled temperature: {mean('temperature_c'):.1f} C",
        f"Approx. energy from in-window samples: {energy_j:.2f} J",
        f"CSV: {csv_path}",
    ]
    if errors:
        lines.append(f"Telemetry errors ({len(errors)}): {' | '.join(errors[:5])}")
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n--- Sustained benchmark summary ---")
    print("\n".join(lines))
    print(f"Summary saved: {summary_path}")
    if n_energy < 10:
        print("WARNING: Fewer than 10 in-window samples; energy/power estimates are low-confidence.")
    if errors:
        print("WARNING: Some telemetry reads failed; inspect the CSV and summary.")


if __name__ == "__main__":
    main()
