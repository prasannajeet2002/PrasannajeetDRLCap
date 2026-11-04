"""Workload monitor + power controller for the RTX 4070 Ti (DRLCap Fig. 4).

Telemetry comes from NVML (pynvml), which is what nvidia-smi itself uses,
without the cost of spawning a process every 300 ms. Frequency capping uses
`nvidia-smi -lgc <min>,<cap>`: the GPU may run anywhere up to the cap and
the hardware still lowers clocks on its own below it, which is what the
paper means by a frequency *cap* (as opposed to a fixed clock).

The power-capping variant uses `nvidia-smi -pl <W>` (100-285 W).

Safety: the clock limit (`-rgc`) and the power limit (back to 285 W) are
always reset on normal exit, on an exception, on Ctrl-C and on SIGTERM.
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
# Board power limit (W); used to normalise the power feature. It is also
# the default (and maximum) power cap; the driver accepts 100-285 W.
POWER_LIMIT_W = 285.0
MIN_POWER_LIMIT_W = 100

_handle = None
_cap_applied = False
# Assume a power cap may be left over from an aborted run, so the first
# reset always restores the default limit.
_power_applied = True


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


def set_power_cap(watts):
    """Cap board power (paper's power-capping variant, `nvidia-smi -pl`)."""
    global _power_applied
    _smi(["-pl", str(int(watts))])
    _power_applied = True


def apply(action, previous=(None, None)):
    """Apply one DRLCap action (freq cap MHz, power cap W); None means that
    knob's default. Only knobs that differ from `previous` are touched,
    since each nvidia-smi call costs tens of ms of the 300 ms period."""
    freq_cap, power_cap = action
    if freq_cap != previous[0]:
        if freq_cap is None:
            reset_clocks()
        else:
            set_cap(freq_cap)
    if power_cap != previous[1]:
        if power_cap is None:
            reset_power()
        else:
            set_power_cap(power_cap)


def reset_clocks():
    global _cap_applied
    if _handle is not None:
        _smi(["-rgc"])
        _cap_applied = False


def reset_power():
    global _power_applied
    if _handle is not None and _power_applied:
        _smi(["-pl", str(int(POWER_LIMIT_W))])
        _power_applied = False


def reset_cap():
    """Return clock and power management to the NVIDIA default (the
    paper baseline)."""
    reset_clocks()
    reset_power()


def gpu_busy_by_others():
    """PIDs of compute processes already on the 4070 Ti (should be none)."""
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_handle)
    return [p.pid for p in procs]
