import csv
import time
import threading
import subprocess
import statistics
from datetime import datetime

import tensorflow as tf

GPU_ID = 1
FIXED_CLOCK_MHZ = 1800
ITERATIONS = 100
REPEATS = 3
SAMPLE_INTERVAL = 0.25
COOLDOWN_SECONDS = 5

SUMMARY_FILE = "frequency_comparison_summary.csv"
TELEMETRY_FILE = "frequency_comparison_telemetry.csv"

def run_command(args):
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=20
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()

def set_clock(mode):
    if mode == "AUTO":
        run_command([
            "sudo", "nvidia-smi", "-i", str(GPU_ID), "-rgc"
        ])
        print("Automatic graphics clock management restored.")
    else:
        run_command([
            "sudo", "nvidia-smi", "-i", str(GPU_ID),
            "-lgc", f"{FIXED_CLOCK_MHZ},{FIXED_CLOCK_MHZ}"
        ])
        print(f"Requested graphics clock lock: {FIXED_CLOCK_MHZ} MHz")

def read_telemetry():
    output = run_command([
        "nvidia-smi", "-i", str(GPU_ID),
        "--query-gpu=power.draw,utilization.gpu,temperature.gpu,clocks.gr",
        "--format=csv,noheader,nounits"
    ])
    return [float(x.strip()) for x in output.split(",")]

print("TensorFlow:", tf.__version__)
print("Visible GPU:", tf.config.list_physical_devices("GPU"))

if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("TensorFlow GPU not detected.")

# Fixed workload reused for every trial.
with tf.device("/GPU:0"):
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    _ = tf.matmul(a, b).numpy()

# Alternate conditions to reduce order-related bias.
trial_plan = [
    ("AUTO", 1), ("FIXED_1800", 1),
    ("AUTO", 2), ("FIXED_1800", 2),
    ("AUTO", 3), ("FIXED_1800", 3),
]

summaries = []
all_samples = []

try:
    for mode, repetition in trial_plan:
        print(f"\nPreparing {mode}, repetition {repetition}")

        set_clock(mode)
        time.sleep(2)

        # Warm-up outside the timed interval.
        with tf.device("/GPU:0"):
            for _ in range(5):
                _ = tf.matmul(a, b).numpy()

        trial_id = f"{mode}_rep{repetition}"
        samples = []
        stop_event = threading.Event()
        measurement_start = {"t": None}

        def monitor():
            while not stop_event.is_set():
                t = time.perf_counter()
                try:
                    power, util, temp, clock = read_telemetry()
                    samples.append({
                        "timestamp": datetime.now().isoformat(
                            timespec="milliseconds"
                        ),
                        "elapsed_s": (
                            t - measurement_start["t"]
                            if measurement_start["t"] is not None
                            else 0.0
                        ),
                        "trial": trial_id,
                        "mode": mode,
                        "power_w": power,
                        "gpu_util_percent": util,
                        "temperature_c": temp,
                        "clock_mhz": clock
                    })
                except Exception as exc:
                    print("Telemetry warning:", exc)
                stop_event.wait(SAMPLE_INTERVAL)

        monitor_thread = threading.Thread(
            target=monitor, daemon=True
        )
        monitor_thread.start()

        # Allow the monitor to start before timing.
        time.sleep(0.5)
        start = time.perf_counter()
        measurement_start["t"] = start

        with tf.device("/GPU:0"):
            for _ in range(ITERATIONS):
                result = tf.matmul(a, b)
                _ = result.numpy()  # Synchronize every iteration

        end = time.perf_counter()
        duration = end - start

        # Capture a final sample while the trial is still labelled.
        try:
            power, util, temp, clock = read_telemetry()
            samples.append({
                "timestamp": datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                "elapsed_s": end - start,
                "trial": trial_id,
                "mode": mode,
                "power_w": power,
                "gpu_util_percent": util,
                "temperature_c": temp,
                "clock_mhz": clock
            })
        except Exception as exc:
            print("Final telemetry warning:", exc)

        stop_event.set()
        monitor_thread.join(timeout=5)
        all_samples.extend(samples)

        if len(samples) < 3:
            raise RuntimeError(
                f"Insufficient telemetry for {trial_id}: {len(samples)} samples"
            )

        powers = [s["power_w"] for s in samples]
        utils = [s["gpu_util_percent"] for s in samples]
        clocks = [s["clock_mhz"] for s in samples]
        temps = [s["temperature_c"] for s in samples]

        avg_power = statistics.mean(powers)
        energy_j = avg_power * duration
        edp = energy_j * duration

        row = {
            "trial": trial_id,
            "mode": mode,
            "repetition": repetition,
            "iterations": ITERATIONS,
            "execution_time_s": duration,
            "iteration_time_ms": duration * 1000 / ITERATIONS,
            "telemetry_samples": len(samples),
            "average_power_w": avg_power,
            "peak_sampled_power_w": max(powers),
            "average_gpu_util_percent": statistics.mean(utils),
            "average_clock_mhz": statistics.mean(clocks),
            "average_temperature_c": statistics.mean(temps),
            "estimated_energy_j": energy_j,
            "estimated_edp_j_s": edp
        }
        summaries.append(row)

        print(
            f"{trial_id}: time={duration:.3f}s, "
            f"power={avg_power:.2f}W, "
            f"energy~{energy_j:.2f}J, "
            f"EDP~{edp:.2f}J*s, "
            f"clock~{statistics.mean(clocks):.0f}MHz, "
            f"samples={len(samples)}"
        )

        time.sleep(COOLDOWN_SECONDS)

finally:
    # Always attempt to remove the clock lock, including on errors.
    try:
        set_clock("AUTO")
        print("Clock lock cleanup attempted.")
    except Exception as exc:
        print("CRITICAL: Could not restore automatic clocks:", exc)
        print("Run manually: sudo nvidia-smi -i 1 -rgc")

# Write results only after the trials finish.
if summaries:
    with open(SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    with open(TELEMETRY_FILE, "w", newline="") as f:
        fields = [
            "timestamp", "elapsed_s", "trial", "mode", "power_w",
            "gpu_util_percent", "temperature_c", "clock_mhz"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_samples)

    print("\n========== EXPERIMENT SUMMARY ==========")
    for mode in ("AUTO", "FIXED_1800"):
        rows = [r for r in summaries if r["mode"] == mode]
        print(f"\n{mode}: {len(rows)} trials")
        for field in (
            "execution_time_s", "average_power_w",
            "estimated_energy_j", "estimated_edp_j_s",
            "average_clock_mhz"
        ):
            values = [r[field] for r in rows]
            print(f"  {field}: mean={statistics.mean(values):.3f}")

    print("\nSaved:", SUMMARY_FILE)
    print("Saved:", TELEMETRY_FILE)
    print("Remember: energy and EDP are estimates from sampled power.")
