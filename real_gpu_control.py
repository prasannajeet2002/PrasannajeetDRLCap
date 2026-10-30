import subprocess
import time
import atexit
import signal
import sys


# ============================================================
# CONFIGURATION
# ============================================================

GPU_ID = 0

# ------------------------------------------------------------
# IMPORTANT:
# Do NOT hard-code the original power limit here.
# It will be detected automatically.
# ------------------------------------------------------------

SAFETY_MAX_POWER_LIMIT = None

# Minimum allowed power limit for your experiment.
# CHANGE THIS according to your GPU's actual supported range.
MIN_EXPERIMENT_POWER = 150

# Maximum power limit that the RL agent is allowed to request.
# This is NOT necessarily the GPU's physical maximum.
# Set this conservatively for initial experiments.
MAX_EXPERIMENT_POWER = 350

# Minimum time between hardware-control commands.
# Prevents the agent from repeatedly changing hardware settings.
MIN_CONTROL_INTERVAL = 0.5

# Whether frequency control is enabled.
USE_CLOCK_CONTROL = False

# If frequency control is enabled, change these after
# determining supported clock values for your GPU.
MIN_CLOCK = None
MAX_CLOCK = None


# ============================================================
# INTERNAL STATE
# ============================================================

original_power_limit = None
original_power_default_limit = None
original_power_min_limit = None
original_power_max_limit = None

last_control_time = 0
settings_changed = False
restoration_done = False


# ============================================================
# COMMAND EXECUTION
# ============================================================

def run_command(command):
    """
    Execute a system command and return stdout.

    Raises RuntimeError if the command fails.
    """

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        return result.stdout.strip()

    except subprocess.CalledProcessError as e:

        print("\n[GPU CONTROL ERROR]")
        print("Command:", " ".join(command))
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)

        raise RuntimeError(
            f"GPU command failed: {' '.join(command)}"
        )


# ============================================================
# GPU INFORMATION
# ============================================================

def get_gpu_information():

    command = [
        "nvidia-smi",
        "-i", str(GPU_ID),
        "--query-gpu="
        "index,"
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
        "clocks.mem,"
        "memory.used,"
        "memory.total",
        "--format=csv,noheader,nounits"
    ]

    output = run_command(command)

    values = [x.strip() for x in output.split(",")]

    return {
        "index": int(values[0]),
        "name": values[1],
        "power_draw": float(values[2]),
        "power_limit": float(values[3]),
        "power_default_limit": float(values[4]),
        "power_min_limit": float(values[5]),
        "power_max_limit": float(values[6]),
        "temperature": float(values[7]),
        "utilization_gpu": float(values[8]),
        "utilization_memory": float(values[9]),
        "clock_graphics": float(values[10]),
        "clock_memory": float(values[11]),
        "memory_used": float(values[12]),
        "memory_total": float(values[13])
    }


# ============================================================
# CHECK GPU
# ============================================================

def check_gpu():

    print("\n================================================")
    print("CHECKING GPU")
    print("================================================")

    info = get_gpu_information()

    print("GPU ID          :", info["index"])
    print("GPU             :", info["name"])
    print("Power Draw      :", info["power_draw"], "W")
    print("Power Limit     :", info["power_limit"], "W")
    print("Default Limit   :", info["power_default_limit"], "W")
    print("Minimum Limit   :", info["power_min_limit"], "W")
    print("Maximum Limit   :", info["power_max_limit"], "W")
    print("Temperature     :", info["temperature"], "°C")
    print("GPU Utilization :", info["utilization_gpu"], "%")
    print("Memory Util     :", info["utilization_memory"], "%")
    print("Graphics Clock  :", info["clock_graphics"], "MHz")
    print("Memory Clock    :", info["clock_memory"], "MHz")

    return info


# ============================================================
# CHECK ACTIVE PROCESSES
# ============================================================

def check_processes():

    print("\n================================================")
    print("CHECKING GPU PROCESSES")
    print("================================================")

    command = [
        "nvidia-smi",
        "-i", str(GPU_ID)
    ]

    output = run_command(command)

    print(output)

    print("\nIMPORTANT:")
    print("Make sure no other student's workload is using GPU",
          GPU_ID, "before changing the power limit.")


# ============================================================
# SAVE ORIGINAL GPU SETTINGS
# ============================================================

def save_original_settings():

    global original_power_limit
    global original_power_default_limit
    global original_power_min_limit
    global original_power_max_limit

    info = get_gpu_information()

    original_power_limit = info["power_limit"]
    original_power_default_limit = info["power_default_limit"]
    original_power_min_limit = info["power_min_limit"]
    original_power_max_limit = info["power_max_limit"]

    print("\n================================================")
    print("ORIGINAL GPU SETTINGS SAVED")
    print("================================================")

    print("Original Power Limit :",
          original_power_limit, "W")

    print("Default Power Limit  :",
          original_power_default_limit, "W")

    print("Minimum Power Limit  :",
          original_power_min_limit, "W")

    print("Maximum Power Limit  :",
          original_power_max_limit, "W")

    # Automatically make the maximum safe experiment
    # range dependent on the actual GPU.

    global SAFETY_MAX_POWER_LIMIT

    SAFETY_MAX_POWER_LIMIT = min(
        MAX_EXPERIMENT_POWER,
        original_power_max_limit
    )

    print(
        "RL Maximum Power Limit:",
        SAFETY_MAX_POWER_LIMIT,
        "W"
    )


# ============================================================
# SAFETY VALIDATION
# ============================================================

def validate_power_limit(power_limit):

    if original_power_min_limit is None:
        raise RuntimeError(
            "Original GPU settings have not been saved."
        )

    if power_limit < original_power_min_limit:

        raise ValueError(
            f"Requested power limit {power_limit} W "
            f"is below GPU minimum "
            f"{original_power_min_limit} W."
        )

    if power_limit > SAFETY_MAX_POWER_LIMIT:

        raise ValueError(
            f"Requested power limit {power_limit} W "
            f"exceeds experiment safety limit "
            f"{SAFETY_MAX_POWER_LIMIT} W."
        )


# ============================================================
# SET POWER LIMIT
# ============================================================

def set_power_limit(power_limit):

    global last_control_time
    global settings_changed

    power_limit = float(power_limit)

    validate_power_limit(power_limit)

    current_time = time.time()

    if (
        current_time - last_control_time
        < MIN_CONTROL_INTERVAL
    ):

        time.sleep(
            MIN_CONTROL_INTERVAL
            - (current_time - last_control_time)
        )

    print(
        f"[GPU CONTROL] Setting power limit "
        f"to {power_limit:.1f} W"
    )

    command = [
        "sudo",
        "nvidia-smi",
        "-i", str(GPU_ID),
        "-pl", str(int(power_limit))
    ]

    run_command(command)

    last_control_time = time.time()

    settings_changed = True


# ============================================================
# FREQUENCY CONTROL
# ============================================================

def set_graphics_clock(clock):

    if not USE_CLOCK_CONTROL:

        print(
            "[GPU CONTROL] Frequency control disabled."
        )

        return

    if MIN_CLOCK is not None:

        if clock < MIN_CLOCK:

            raise ValueError(
                "Requested clock below experiment minimum."
            )

    if MAX_CLOCK is not None:

        if clock > MAX_CLOCK:

            raise ValueError(
                "Requested clock above experiment maximum."
            )

    print(
        f"[GPU CONTROL] Setting graphics clock "
        f"to {clock} MHz"
    )

    command = [
        "sudo",
        "nvidia-smi",
        "-i", str(GPU_ID),
        "-lgc", str(int(clock))
    ]

    run_command(command)


# ============================================================
# RELEASE CLOCK LOCK
# ============================================================

def release_graphics_clock():

    if not USE_CLOCK_CONTROL:
        return

    print(
        "[GPU CONTROL] Releasing graphics clock lock..."
    )

    command = [
        "sudo",
        "nvidia-smi",
        "-i", str(GPU_ID),
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
# RESTORE GPU
# ============================================================

def restore_gpu():

    global restoration_done

    if restoration_done:
        return

    restoration_done = True

    print("\n")
    print("================================================")
    print("RESTORING GPU SETTINGS")
    print("================================================")

    # --------------------------------------------------------
    # Release clock lock first
    # --------------------------------------------------------

    release_graphics_clock()

    # --------------------------------------------------------
    # Restore original power limit
    # --------------------------------------------------------

    if original_power_limit is not None:

        try:

            print(
                "Restoring power limit to:",
                original_power_limit,
                "W"
            )

            command = [
                "sudo",
                "nvidia-smi",
                "-i", str(GPU_ID),
                "-pl", str(int(original_power_limit))
            ]

            run_command(command)

        except Exception as e:

            print(
                "[CRITICAL WARNING] "
                "Could not restore power limit."
            )

            print(e)

    # --------------------------------------------------------
    # Verify restoration
    # --------------------------------------------------------

    try:

        info = get_gpu_information()

        print("\nGPU AFTER RESTORATION")
        print("---------------------")

        print(
            "Power Limit:",
            info["power_limit"],
            "W"
        )

        print(
            "Temperature:",
            info["temperature"],
            "°C"
        )

        print(
            "Graphics Clock:",
            info["clock_graphics"],
            "MHz"
        )

    except Exception as e:

        print(
            "[WARNING] Could not verify GPU restoration:"
        )

        print(e)

    print("\nGPU restoration procedure completed.")


# ============================================================
# EMERGENCY SIGNAL HANDLER
# ============================================================

def emergency_shutdown(signum, frame):

    print("\n")
    print("================================================")
    print("INTERRUPT RECEIVED")
    print("================================================")

    print(
        "Restoring GPU before terminating program..."
    )

    restore_gpu()

    sys.exit(1)


# ============================================================
# REGISTER SAFETY HANDLERS
# ============================================================

def initialize_gpu_safety():

    print("\n")
    print("================================================")
    print("INITIALIZING GPU SAFETY SYSTEM")
    print("================================================")

    check_gpu()

    check_processes()

    save_original_settings()

    # Handle Ctrl+C
    signal.signal(
        signal.SIGINT,
        emergency_shutdown
    )

    signal.signal(
        signal.SIGTERM,
        emergency_shutdown
    )

    # Backup restoration if Python exits normally.
    atexit.register(restore_gpu)

    print("\nGPU safety system initialized.")