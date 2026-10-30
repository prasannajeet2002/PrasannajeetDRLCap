"""Workload monitor + power controller for the RTX 4070 Ti (DRLCap Fig. 4).

Telemetry comes from NVML (pynvml), which is what nvidia-smi itself uses,
without the cost of spawning a process every 300 ms. Frequency capping uses
`nvidia-smi -lgc <min>,<cap>`: the GPU may run anywhere up to the cap and
the hardware still lowers clocks on its own below it, which is what the
paper means by a frequency *cap* (as opposed to a fixed clock).

Safety: the clock limit is always reset (`-rgc`) on normal exit, on an
exception, on Ctrl-C and on SIGTERM.
"""

import atexit
import signal
import subprocess
import sys

import pynvml

# The RTX 4070 Ti is physical GPU 1 (nvidia-smi / NVML index order).
GPU_ID = 1
NVIDIA_SMI = "/usr/bin/nvidia-smi"

# Lowest clock the 4070 Ti supports; used as the floor of every cap range.
MIN_CLOCK_MHZ = 210
# Highest supported graphics clock (NVML supported-clocks list).
MAX_CLOCK_MHZ = 3105
# Board power limit (W); used to normalise the power feature.
POWER_LIMIT_W = 285.0

_handle = None
_cap_applied = False


def init():
    """Initialise NVML and check we are talking to the RTX 4070 Ti."""
    global _handle
    pynvml.nvmlInit()
    _handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_ID)
    name = pynvml.nvmlDeviceGetName(_handle)
    if "4070 Ti" not in name:
        raise RuntimeError(f"GPU {GPU_ID} is {name!r}, expected RTX 4070 Ti")
    atexit.register(reset_cap)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    return name


def _on_signal(signum, frame):
    reset_cap()
    sys.exit(128 + signum)


def read():
    """One telemetry sample: the five DRLCap state metrics plus energy.

    Returns a dict with util (%), mem_util (%), power (W), clock (MHz,
    actual SM clock, not the requested cap), temp (C) and energy_j
    (cumulative board energy counter, joules).
    """
    u = pynvml.nvmlDeviceGetUtilizationRates(_handle)
    return {
        "util": float(u.gpu),
        "mem_util": float(u.memory),
        "power": pynvml.nvmlDeviceGetPowerUsage(_handle) / 1000.0,
        "clock": float(pynvml.nvmlDeviceGetClockInfo(
            _handle, pynvml.NVML_CLOCK_SM)),
        "temp": float(pynvml.nvmlDeviceGetTemperature(
            _handle, pynvml.NVML_TEMPERATURE_GPU)),
        "energy_j": pynvml.nvmlDeviceGetTotalEnergyConsumption(_handle)
        / 1000.0,
    }


def energy_j():
    """Cumulative board energy counter in joules (NVML, mJ resolution)."""
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(_handle) / 1000.0


def _smi(args):
    result = subprocess.run(
        ["sudo", "-n", NVIDIA_SMI, "-i", str(GPU_ID)] + args,
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi {' '.join(args)} failed: "
            f"{result.stdout.strip()} {result.stderr.strip()}")


def set_cap(cap_mhz):
    """Cap the graphics clock at cap_mhz (hardware may still go lower)."""
    global _cap_applied
    _smi(["-lgc", f"{MIN_CLOCK_MHZ},{int(cap_mhz)}"])
    _cap_applied = True


def reset_cap():
    """Return clock management to the NVIDIA default (the paper baseline)."""
    global _cap_applied
    if _handle is None:
        return
    _smi(["-rgc"])
    _cap_applied = False


def gpu_busy_by_others():
    """PIDs of compute processes already on the 4070 Ti (should be none)."""
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_handle)
    return [p.pid for p in procs]
