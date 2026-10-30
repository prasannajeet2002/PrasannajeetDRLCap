import subprocess
import csv
import time
from datetime import datetime


GPU_ID = 1

OUTPUT_FILE = "real_gpu_experiments/4070ti_monitor.csv"

SAMPLE_INTERVAL = 0.2


def get_gpu_data():

    command = [
        "nvidia-smi",
        "-i",
        str(GPU_ID),
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
        "memory.used,"
        "memory.total",
        "--format=csv,noheader,nounits"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    return [
        x.strip()
        for x in result.stdout.strip().split(",")
    ]


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "power_draw_W",
        "power_limit_W",
        "temperature_C",
        "gpu_util_percent",
        "memory_util_percent",
        "graphics_clock_MHz",
        "sm_clock_MHz",
        "memory_clock_MHz",
        "memory_used_MB",
        "memory_total_MB"
    ])

    print(
        "Monitoring GPU",
        GPU_ID
    )

    print(
        "Saving to:",
        OUTPUT_FILE
    )

    print(
        "Press Ctrl+C to stop."
    )

    try:

        while True:

            data = get_gpu_data()

            writer.writerow(data)

            f.flush()

            print(
                datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "| Power:",
                data[1],
                "W | Temp:",
                data[3],
                "C | GPU:",
                data[4],
                "%"
            )

            time.sleep(
                SAMPLE_INTERVAL
            )

    except KeyboardInterrupt:

        print(
            "\nMonitoring stopped."
        )
