"""Benchmark registry for the DRLCap reproduction.

Each workload is an unmodified program (DRLCap is application-transparent:
the agent never touches the program). Rodinia inputs are enlarged and
repeated so one job lasts roughly 20-60 s at default clocks, long enough
for the 300 ms DRLCap control loop to act on it. Repeating short programs
also produces idle gaps between launches, i.e. real phase changes.

Split (paper Sec. V-B trains on one set of benchmarks and tests on unseen
ones). The paper's training suites (Parboil, SHOC, PolyBench, CUDA SDK)
and test apps (LULESH, CloverLeaf, ...) are not installed here, so we
split Rodinia itself. The paper's CNN tests (VGG-16, GoogleNet, and
ResNet50 from its RTX 2080 Ti experiment) are test-only workloads run by
cnn_workload.py. TensorFlow 2.21 needs CUDA 12 / cuDNN 9 libraries; the
ones pip put in prasannajeet_env are CUDA 13 builds that driver 575
(CUDA 12.9) cannot run, so the CNN jobs load the CUDA 12.9 / cuDNN 9.26
libraries already present in the base miniconda install, read-only, via
LD_LIBRARY_PATH (nothing is installed or modified).
"""

import glob
import sys

import os
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RODINIA = os.path.join(PROJECT_ROOT, "benchmarks", "rodinia", "cuda")
DATA = os.path.join(PROJECT_ROOT, "benchmarks", "rodinia", "data")
HERE = os.path.dirname(os.path.abspath(__file__))

# The RTX 4070 Ti is physical GPU 1. Every workload runs only there.
GPU_ID = 1


# CUDA 12 runtime + cuDNN 9 for TensorFlow (see module docstring).
CUDA12_SITE = "/home/drl/miniconda3/lib/python3.13/site-packages/nvidia"
CUDA12_LIBS = sorted(glob.glob(os.path.join(CUDA12_SITE, "*", "lib")))


def _w(name, subdir, cmd, repeat, category, split, env=None):
    return {
        "name": name,
        "cwd": os.path.join(RODINIA, subdir) if subdir else HERE,
        "cmd": cmd,
        "repeat": repeat,
        "category": category,
        "split": split,
        "env": env or {},
    }


# Steps chosen so each job trains for ~30 s after TF start-up (start-up is
# ~23 / 60 / 37 s for vgg16 / googlenet / resnet50, measured 2026-10-30;
# the low-GPU start-up phase is kept as a real phase change).
def _cnn(name, steps):
    return _w(name, None,
              [sys.executable, os.path.join(HERE, "cnn_workload.py"), name,
               "--steps", str(steps)], 1, "compute", "test",
              env={"LD_LIBRARY_PATH": ":".join(CUDA12_LIBS)})


WORKLOADS = {w["name"]: w for w in [
    # ---------------- training set ----------------
    _w("backprop", "backprop", ["./backprop", "524288"], 60, "compute", "train"),
    _w("bfs", "bfs", ["./bfs", f"{DATA}/bfs/graph1MW_6.txt"], 25, "memory", "train"),
    _w("gaussian", "gaussian", ["./gaussian", "-s", "4096", "-q"], 10, "compute", "train"),
    _w("hotspot", "hotspot",
       ["./hotspot", "1024", "2", "100000", f"{DATA}/hotspot/temp_1024",
        f"{DATA}/hotspot/power_1024", "/dev/null"], 3, "compute", "train"),
    _w("huffman", "huffman",
       ["./pavle", f"{DATA}/huffman/test1024_H2.206587175259.in"], 100, "memory", "train"),
    _w("nw", "nw", ["./needle", "16384", "10"], 8, "memory", "train"),
    _w("particlefilter", "particlefilter",
       ["./particlefilter_float", "-x", "128", "-y", "128", "-z", "100",
        "-np", "200000"], 3, "compute", "train"),
    _w("pathfinder", "pathfinder",
       ["./pathfinder", "100000", "5000", "20"], 1, "memory", "train"),
    # ---------------- test set (never seen in training) ----------------
    _w("b+tree", "b+tree",
       ["./b+tree.out", "file", f"{DATA}/b+tree/mil.txt",
        "command", f"{DATA}/b+tree/command.txt"], 30, "memory", "test"),
    _w("dwt2d", "dwt2d",
       ["./dwt2d", "rgb.bmp", "-d", "1024x1024", "-f", "-5", "-l", "3"],
       100, "memory", "test"),
    _w("heartwall", "heartwall",
       ["./heartwall", f"{DATA}/heartwall/test.avi", "104"], 40, "compute", "test"),
    _w("lavaMD", "lavaMD", ["./lavaMD", "-boxes1d", "60"], 3, "compute", "test"),
    _w("lud", "lud", ["cuda/lud_cuda", "-s", "32768"], 3, "compute", "test"),
    _w("myocyte", "myocyte", ["./myocyte.out", "6000", "1", "0"], 1, "compute", "test"),
    _w("streamcluster", "streamcluster",
       ["./sc_gpu", "10", "20", "256", "524288", "524288", "2000",
        "none", "/dev/null", "1"], 1, "memory", "test"),
    _cnn("vgg16", 60),
    _cnn("googlenet", 400),
    _cnn("resnet50", 150),
]}

# Rodinia test programs only (the CNN jobs need the CUDA 12 libraries).
TEST_RODINIA = [k for k, w in WORKLOADS.items()
                if w["split"] == "test" and w["cwd"] != HERE]

TRAIN = [k for k, w in WORKLOADS.items() if w["split"] == "train"]
TEST = [k for k, w in WORKLOADS.items() if w["split"] == "test"]


def gpu_env():
    """Environment that pins a child process to the RTX 4070 Ti."""
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return env


def run_blocking(name, timeout=None):
    """Run all repetitions of a workload; returns elapsed seconds."""
    w = WORKLOADS[name]
    t0 = time.perf_counter()
    for _ in range(w["repeat"]):
        p = subprocess.run(
            w["cmd"], cwd=w["cwd"], env={**gpu_env(), **w["env"]},
            timeout=timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"{name} exited {p.returncode}: {p.stderr[-500:]}")
    return time.perf_counter() - t0
