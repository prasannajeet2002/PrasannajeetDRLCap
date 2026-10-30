import subprocess
import time
import atexit
import signal
import sys


# ============================================================
# GPU CONFIGURATION
# ============================================================

GPU_ID = 1

# Minimum delay between hardware control operations.
# This prevents the RL agent from issuing commands too quickly.
MIN_CONTROL_INTERVAL = 0.5


# ============================================================
# INTERNAL STATE
# ============================================================

ORIGINAL_POWER_LIMIT = None
ORIGINAL_MIN_POWER_LIMIT = None
ORIGINAL_MAX_POWER_LIMIT = None

LAST_CONTROL_TIME = 0.0
RESTORED = False


# ============================================================
# RUN SYSTEM COMMAND
# ============================================================

def run_command(command):

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        print("\nCOMMAND FAILED")
        print("Command:", " ".join(command))
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        raise RuntimeError(
            "Command failed: " + " ".join(command)
        )

    return result.stdout.strip()


# ============================================================
# READ GPU TELEMETRY
# ============================================================

def get_gpu_info():

    command = [
        "nvidia-smi",
        "-i",
        str(GPU_ID),
        "--query-gpu="
        "name,"
        "power.draw,"
        "power.limit,"
        "power.default_limit,"
        "power.min_limit,"
        "power.max_limit,"
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

    output = run_command(command)

    values = [x.strip() for x in output.split(",")]

    return {
        "name": values[0],
        "power_draw": float(values[1]),
        "power_limit": float(values[2]),
        "default_power_limit": float(values[3]),
        "min_power_limit": float(values[4]),
        "max_power_limit": float(values[5]),
        "temperature": float(values[6]),
        "gpu_utilization": float(values[7]),
        "memory_utilization": float(values[8]),
        "graphics_clock": float(values[9]),
        "sm_clock": float(values[10]),
        "memory_clock": float(values[11]),
        "memory_used": float(values[12]),
        "memory_total": float(values[13])
    }


# ============================================================
# INITIALIZE GPU
# ============================================================

def initialize():

    global ORIGINAL_POWER_LIMIT
    global ORIGINAL_MIN_POWER_LIMIT
    global ORIGINAL_MAX_POWER_LIMIT

    print("\n==============================================")
    print("INITIALIZING RTX 4070 Ti CONTROLLER")
    print("==============================================")

    info = get_gpu_info()

    print("GPU:", info["name"])
    print("Current power:", info["power_draw"], "W")
    print("Current power limit:", info["power_limit"], "W")
    print("Default power limit:", info["default_power_limit"], "W")
    print("Minimum power limit:", info["min_power_limit"], "W")
    print("Maximum power limit:", info["max_power_limit"], "W")
    print("Temperature:", info["temperature"], "C")
    print("GPU utilization:", info["gpu_utilization"], "%")
    print("Memory utilization:", info["memory_utilization"], "%")
    print("Graphics clock:", info["graphics_clock"], "MHz")

    ORIGINAL_POWER_LIMIT = info["power_limit"]
    ORIGINAL_MIN_POWER_LIMIT = info["min_power_limit"]
    ORIGINAL_MAX_POWER_LIMIT = info["max_power_limit"]

    print("\nOriginal settings saved.")

    # Ctrl+C handler
    signal.signal(signal.SIGINT, emergency_shutdown)

    # SIGTERM handler
    signal.signal(signal.SIGTERM, emergency_shutdown)

    # Normal Python exit backup
    atexit.register(restore)


# ============================================================
# VALIDATE POWER LIMIT
# ============================================================

def validate_power_limit(power_limit):

    if ORIGINAL_MIN_POWER_LIMIT is None:
        raise RuntimeError(
            "GPU controller has not been initialized."
        )

    if power_limit < ORIGINAL_MIN_POWER_LIMIT:

        raise ValueError(
            f"{power_limit} W is below GPU minimum "
            f"{ORIGINAL_MIN_POWER_LIMIT} W"
        )

    if power_limit > ORIGINAL_MAX_POWER_LIMIT:

        raise ValueError(
            f"{power_limit} W is above GPU maximum "
            f"{ORIGINAL_MAX_POWER_LIMIT} W"
        )


# ============================================================
# SET POWER LIMIT
# ============================================================

def set_power_limit(power_limit):

    global LAST_CONTROL_TIME

    power_limit = float(power_limit)

    validate_power_limit(power_limit)

    current_time = time.time()

    elapsed = current_time - LAST_CONTROL_TIME

    if elapsed < MIN_CONTROL_INTERVAL:

        time.sleep(
            MIN_CONTROL_INTERVAL - elapsed
        )

    print(
        f"[GPU] Setting power limit to "
        f"{power_limit:.1f} W"
    )

    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        str(GPU_ID),
        "-pl",
        str(power_limit)
    ]

    run_command(command)

    LAST_CONTROL_TIME = time.time()


# ============================================================
# LOCK GRAPHICS CLOCK
# ============================================================

def lock_graphics_clock(clock_mhz):

    print(
        f"[GPU] Locking graphics clock to "
        f"{clock_mhz} MHz"
    )

    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        str(GPU_ID),
        "-lgc",
        str(int(clock_mhz))
    ]

    run_command(command)


# ============================================================
# RELEASE GRAPHICS CLOCK
# ============================================================

def release_graphics_clock():

    print("[GPU] Releasing graphics clock lock...")

    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        str(GPU_ID),
        "-rgc"
    ]

    try:

        run_command(command)

    except Exception as e:

        print(
            "[WARNING] Could not release graphics clock:",
            e
        )


# ============================================================
# RESTORE ORIGINAL GPU STATE
# ============================================================

def restore():

    global RESTORED

    if RESTORED:
        return

    RESTORED = True

    print("\n==============================================")
    print("RESTORING GPU")
    print("==============================================")

    # Release any graphics-clock lock
    release_graphics_clock()

    # Restore original power limit
    if ORIGINAL_POWER_LIMIT is not None:

        try:

            print(
                "Restoring power limit to:",
                ORIGINAL_POWER_LIMIT,
                "W"
            )

            command = [
                "sudo",
                "nvidia-smi",
                "-i",
                str(GPU_ID),
                "-pl",
                str(ORIGINAL_POWER_LIMIT)
            ]

            run_command(command)

        except Exception as e:

            print(
                "[CRITICAL WARNING] "
                "Could not restore power limit."
            )

            print(e)

    # Verify
    try:

        info = get_gpu_info()

        print("\nGPU AFTER RESTORATION")
        print("---------------------")
        print(
            "Power limit:",
            info["power_limit"],
            "W"
        )

        print(
            "Temperature:",
            info["temperature"],
            "C"
        )

        print(
            "Graphics clock:",
            info["graphics_clock"],
            "MHz"
        )

    except Exception as e:

        print(
            "[WARNING] Could not verify restoration."
        )

        print(e)


# ============================================================
# CTRL+C / TERMINATION
# ============================================================

def emergency_shutdown(signum, frame):

    print("\n")
    print("==============================================")
    print("INTERRUPT RECEIVED")
    print("==============================================")

    restore()

    sys.exit(1)
