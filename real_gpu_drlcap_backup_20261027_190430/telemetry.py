import subprocess
import csv
import time
from datetime import datetime


GPU_ID = 1

OUTPUT_FILE = "gpu_telemetry.csv"

INTERVAL = 0.3


def get_gpu_data():

    command = [
        "nvidia-smi",
        "-i", str(GPU_ID),
        "--query-gpu="
        "timestamp,"
        "power.draw,"
        "power.limit,"
        "temperature.gpu,"
        "utilization.gpu,"
        "utilization.memory,"
        "clocks.gr,"
        "clocks.sm,"
        "clocks.mem,"
        "memory.used",
        "--format=csv,noheader,nounits"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    line = result.stdout.strip()

    values = [x.strip() for x in line.split(",")]

    return values


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "power_w",
        "power_limit_w",
        "temperature_c",
        "gpu_util_percent",
        "memory_util_percent",
        "graphics_clock_mhz",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "memory_used_mib"
    ])

    print("Telemetry logging started...")
    print("Output:", OUTPUT_FILE)

    try:

        while True:

            values = get_gpu_data()

            writer.writerow(values)

            f.flush()

            print(
                f"{values[0]} | "
                f"Power {values[1]} W | "
                f"Temp {values[3]} C | "
                f"GPU {values[4]} % | "
                f"Clock {values[6]} MHz"
            )

            time.sleep(INTERVAL)

    except KeyboardInterrupt:

        print("\nTelemetry stopped.")
