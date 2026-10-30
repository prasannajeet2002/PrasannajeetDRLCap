import tensorflow as tf
import time
import subprocess
import csv


GPU_ID = 1

ITERATIONS = 100


def get_gpu():

    cmd = [
        "nvidia-smi",
        "-i", str(GPU_ID),
        "--query-gpu="
        "power.draw,"
        "temperature.gpu,"
        "utilization.gpu,"
        "clocks.gr",
        "--format=csv,noheader,nounits"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    values = [
        float(x.strip())
        for x in result.stdout.strip().split(",")
    ]

    return values


print("=" * 70)
print("BASELINE GPU EXPERIMENT")
print("=" * 70)

# Make sure automatic clock control is active
subprocess.run([
    "sudo",
    "nvidia-smi",
    "-i", str(GPU_ID),
    "-rgc"
])

with tf.device("/GPU:0"):

    A = tf.random.normal(
        [8192, 8192]
    )

    B = tf.random.normal(
        [8192, 8192]
    )

    # warmup
    for _ in range(5):

        C = tf.matmul(A, B)
        C.numpy()

    rows = []

    start = time.time()

    for i in range(ITERATIONS):

        iteration_start = time.time()

        C = tf.matmul(A, B)

        C.numpy()

        iteration_time = (
            time.time()
            -
            iteration_start
        )

        power, temp, util, clock = get_gpu()

        rows.append([
            i + 1,
            power,
            temp,
            util,
            clock,
            iteration_time
        ])

        print(
            f"{i+1:03d} | "
            f"Power {power:.2f} W | "
            f"Temp {temp:.1f} C | "
            f"GPU {util:.1f}% | "
            f"Clock {clock:.0f} MHz | "
            f"Time {iteration_time:.4f}s"
        )


total_time = time.time() - start

with open(
    "baseline_results.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "iteration",
        "power_w",
        "temperature_c",
        "gpu_util_percent",
        "clock_mhz",
        "iteration_time_s"
    ])

    writer.writerows(rows)


print()
print("=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)

print(
    f"Total runtime: {total_time:.3f} seconds"
)

print(
    f"Average iteration: "
    f"{total_time / ITERATIONS:.5f} seconds"
)

print(
    "Saved: baseline_results.csv"
)
