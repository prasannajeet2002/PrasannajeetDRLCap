"""Benchmark registry for the paper-faithful DRLCap reproduction.

Paper (Wang et al., IEEE TSUSC 2024), Sec. V-B: DRLCap is trained offline on
five GPU suites -- Rodinia, Parboil, SHOC, PolyBench and the NVIDIA CUDA SDK
-- and tested on the ten unseen programs of Table II (LULESH, CloverLeaf,
TeaLeaf, miniBUDE, CoMD, Quantum ESPRESSO, GROMACS, miniFE, VGG-16 and
GoogLeNet). The paper does not list which programs of each suite it used,
so every program of each suite that builds and runs on the RTX 4070 Ti with
CUDA 12.3 is registered here. Build scripts and logs: benchmarks/build_scripts,
benchmarks/build_logs.

Every program runs unmodified (DRLCap is application-transparent) on the
RTX 4070 Ti only (physical GPU 1). Fields per entry:
    name, suite, cwd, cmd, input (human-readable input/size), split
    (train / val / test), repeat (runs per job, set after timing them).
"""

import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benchmarks")
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, "tmp")     # program output files we discard

# The RTX 4070 Ti is physical GPU 1. Every workload runs only there.
GPU_ID = 1

RODINIA = os.path.join(BENCH, "rodinia", "cuda")
RODINIA_DATA = os.path.join(BENCH, "rodinia", "data")
PARBOIL = os.path.join(BENCH, "parboil-2.5")
SHOC = os.path.join(BENCH, "shoc", "build", "src", "cuda")
POLY = os.path.join(BENCH, "polybench-gpu", "CUDA")
SDK = os.path.join(BENCH, "cuda-samples", "Samples")


def _w(name, suite, cwd, cmd, input_desc, split="train", repeat=1, env=None):
    return {"name": name, "suite": suite, "cwd": cwd, "cmd": cmd,
            "input": input_desc, "split": split, "repeat": repeat,
            "env": env or {}}


# ---------------------------------------------------------------- Rodinia 3.1
# Same programs and inputs as drlcap_repro (built by drlcap_repro/build_rodinia.sh).
def _rod(name, sub, cmd, inp):
    return _w(name, "Rodinia", os.path.join(RODINIA, sub), cmd, inp)


D = RODINIA_DATA
RODINIA_PROGS = [
    _rod("backprop", "backprop", ["./backprop", "524288"], "524288 input units"),
    _rod("bfs", "bfs", ["./bfs", f"{D}/bfs/graph1MW_6.txt"], "graph1MW_6 (1M nodes)"),
    _rod("gaussian", "gaussian", ["./gaussian", "-s", "4096", "-q"], "4096x4096 matrix"),
    _rod("hotspot", "hotspot", ["./hotspot", "1024", "2", "100000",
          f"{D}/hotspot/temp_1024", f"{D}/hotspot/power_1024", "/dev/null"],
         "1024x1024 grid, 100000 iterations"),
    _rod("huffman", "huffman", ["./pavle", f"{D}/huffman/test1024_H2.206587175259.in"],
         "test1024_H2 input"),
    _rod("nw", "nw", ["./needle", "16384", "10"], "16384x16384, penalty 10"),
    _rod("particlefilter", "particlefilter", ["./particlefilter_float", "-x", "128",
          "-y", "128", "-z", "100", "-np", "200000"], "128x128x100, 200000 particles"),
    _rod("pathfinder", "pathfinder", ["./pathfinder", "100000", "5000", "20"],
         "100000 cols x 5000 rows, pyramid 20"),
    _rod("b+tree", "b+tree", ["./b+tree.out", "file", f"{D}/b+tree/mil.txt",
          "command", f"{D}/b+tree/command.txt"], "mil.txt (1M keys)"),
    _rod("dwt2d", "dwt2d", ["./dwt2d", "rgb.bmp", "-d", "1024x1024", "-f", "-5",
          "-l", "3"], "1024x1024 rgb.bmp, 5/3 forward, 3 levels"),
    _rod("heartwall", "heartwall", ["./heartwall", f"{D}/heartwall/test.avi", "104"],
         "test.avi, 104 frames"),
    _rod("lavaMD", "lavaMD", ["./lavaMD", "-boxes1d", "60"], "60 boxes per dim"),
    _rod("lud", "lud", ["cuda/lud_cuda", "-s", "32768"], "32768x32768 matrix"),
    _rod("myocyte", "myocyte", ["./myocyte.out", "6000", "1", "0"], "6000 ms, 1 instance"),
    _rod("streamcluster", "streamcluster", ["./sc_gpu", "10", "20", "256", "524288",
          "524288", "2000", "none", "/dev/null", "1"], "524288 points, 256 dims"),
]


# ---------------------------------------------------------------- Parboil 2.5
# Official pb2.5 sources and "standard" datasets; the largest dataset of each
# benchmark. Command line = binary -i <inputs> -o <output> -- <parameters>,
# exactly as the Parboil driver builds it from datasets/*/input/DESCRIPTION.
def _pb(name, dataset, inputs, params=(), inp=""):
    d = os.path.join(PARBOIL, "datasets", name, dataset, "input")
    cmd = [os.path.join(PARBOIL, "benchmarks", name, "build", "cuda_default", name)]
    if inputs:
        cmd += ["-i", ",".join(os.path.join(d, f) for f in inputs)]
    cmd += ["-o", os.path.join(SCRATCH, f"parboil_{name}.out")]
    if params:
        cmd += ["--", *params]
    return _w(f"pb-{name}", "Parboil", os.path.join(PARBOIL, "benchmarks", name),
              cmd, inp or f"dataset {dataset}")


PARBOIL_PROGS = [
    _pb("cutcp", "large", ["watbox.sl100.pqr"], inp="large: watbox.sl100.pqr"),
    _pb("histo", "large", ["img.bin"], ["10000", "4"], "large: img.bin, 10000 iters"),
    _pb("lbm", "long", ["120_120_150_ldc.of"], ["3000"], "long: 120x120x150, 3000 steps"),
    _pb("mri-gridding", "small", ["small.uks"], ["32", "0"], "small: small.uks"),
    _pb("mri-q", "large", ["64_64_64_dataset.bin"], inp="large: 64x64x64"),
    _pb("sgemm", "medium", ["matrix1.txt", "matrix2t.txt", "matrix2t.txt"],
        inp="medium: matrix1 x matrix2t"),
    # DESCRIPTION gives only the parameters; the input file is the one in
    # the dataset's input directory (the driver passes it)
    _pb("stencil", "default", ["512x512x64x100.bin"], ["512", "512", "64", "100"],
        "default: 512x512x64, 100 iters"),
    _pb("tpacf", "large", ["Datapnts.1"] + [f"Randompnts.{i}" for i in range(1, 101)],
        ["-n", "100", "-p", "10391"], "large: 100 random sets, 10391 points"),
]


# ---------------------------------------------------------------- SHOC
# vetter/shoc master, CUDA, serial. -s 4 = largest predefined problem size.
def _shoc(name, level, sub, size="4"):
    return _w(f"shoc-{name}", "SHOC", os.path.join(SHOC, level, sub),
              [f"./{name}", "-s", size], f"size class {size}, default passes")


SHOC_PROGS = [
    _shoc("BusSpeedDownload", "level0", ""),
    _shoc("BusSpeedReadback", "level0", ""),
    _shoc("MaxFlops", "level0", ""),   # --passes 1 set below (107 s -> 5.5 s)
    _shoc("BFS", "level1", "bfs"),
    _shoc("FFT", "level1", "fft"),
    _shoc("GEMM", "level1", "gemm"),
    _shoc("MD5Hash", "level1", "md5hash"),
    _shoc("NeuralNet", "level1", "neuralnet"),
    _shoc("Reduction", "level1", "reduction"),
    _shoc("Scan", "level1", "scan"),
    _shoc("Sort", "level1", "sort"),
    _shoc("Stencil2D", "level1", "stencil2d"),
    _shoc("Triad", "level1", "triad"),
    _shoc("S3D", "level2", "s3d"),
]


# ---------------------------------------------------------------- PolyBench/GPU
# socal-ucr/polybench-gpu (paper ref [50]); built-in default sizes.
def _poly(d, exe, inp):
    return _w(f"poly-{d.lower()}", "PolyBench", os.path.join(POLY, d), [f"./{exe}"], inp)


POLY_PROGS = [
    _poly("2DCONV", "2DConvolution.exe", "default 4096x4096"),
    _poly("2MM", "2mm.exe", "default 2048^3"),
    _poly("3DCONV", "3DConvolution.exe", "default 256^3"),
    _poly("3MM", "3mm.exe", "default 512^3"),
    _poly("ATAX", "atax.exe", "default 4096x4096"),
    _poly("BICG", "bicg.exe", "default 4096x4096"),
    _poly("CORR", "correlation.exe", "default 2048x2048"),
    _poly("COVAR", "covariance.exe", "default 2048x2048"),
    _poly("FDTD-2D", "fdtd2d.exe", "default 2048x2048, 500 steps"),
    _poly("GEMM", "gemm.exe", "default 512^3"),
    _poly("GESUMMV", "gesummv.exe", "default 4096"),
    _poly("GRAMSCHM", "gramschmidt.exe", "default 2048x2048"),
    _poly("MVT", "mvt.exe", "default 4096"),
    _poly("SYR2K", "syr2k.exe", "default 2048x2048"),
    _poly("SYRK", "syrk.exe", "default 1024x1024"),
]


# ---------------------------------------------------------------- CUDA SDK
# NVIDIA/cuda-samples tag v12.3 (matches the installed CUDA 12.3); default
# arguments; run from each sample's own directory so it finds its data.
def _sdk(cat, name, args=(), inp="default problem size"):
    d = os.path.join(SDK, cat, name)
    return _w(f"sdk-{name}", "CUDA SDK", d, [f"./{name}", *args], inp)


SDK_PROGS = [
    _sdk("5_Domain_Specific", "BlackScholes"),
    _sdk("5_Domain_Specific", "binomialOptions"),
    _sdk("5_Domain_Specific", "convolutionFFT2D"),
    _sdk("5_Domain_Specific", "fastWalshTransform"),
    _sdk("5_Domain_Specific", "MonteCarloMultiGPU"),
    _sdk("2_Concepts_and_Techniques", "sortingNetworks"),
    _sdk("6_Performance", "transpose"),
    _sdk("2_Concepts_and_Techniques", "reduction"),
    _sdk("0_Introduction", "mergeSort"),
    _sdk("2_Concepts_and_Techniques", "histogram"),
    _sdk("2_Concepts_and_Techniques", "scalarProd"),
    _sdk("5_Domain_Specific", "quasirandomGenerator"),
    _sdk("5_Domain_Specific", "SobolQRNG"),
    _sdk("0_Introduction", "matrixMul"),
    _sdk("3_CUDA_Features", "cudaTensorCoreGemm"),
    _sdk("2_Concepts_and_Techniques", "dct8x8"),
    _sdk("2_Concepts_and_Techniques", "convolutionSeparable"),
    _sdk("2_Concepts_and_Techniques", "eigenvalues"),
    _sdk("5_Domain_Specific", "dxtc"),
    _sdk("2_Concepts_and_Techniques", "interval"),
    _sdk("5_Domain_Specific", "FDTD3d"),
    _sdk("4_CUDA_Libraries", "lineOfSight"),
    _sdk("2_Concepts_and_Techniques", "radixSortThrust"),
]


# ---------------------------------------------------------------- test (Table II)
GMX = os.path.join(BENCH, "gromacs", "bin", "gmx")
GMX_RUN = os.path.join(SCRATCH, "gromacs_0768")

TEST_PROGS = [
    _w("lulesh", "LLNL", os.path.join(BENCH, "lulesh", "cuda", "src"),
       ["./lulesh", "-s", "150"], "-s 150 (Table II)", "test"),
    _w("cloverleaf", "UK-MAC", os.path.join(BENCH, "cloverleaf-cuda"),
       ["./clover_leaf"], "clover.in 3840x3840, 2955 steps (Table II: 3840)", "test"),
    _w("tealeaf", "UK-MAC", os.path.join(BENCH, "tealeaf-cuda"),
       ["./tea_leaf"], "tea.in 4000x4000, 10 steps (Table II: 4000)", "test"),
    _w("minibude", "UoB-HPC", os.path.join(BENCH, "minibude", "build"),
       ["./cuda-bude", "--deck", os.path.join(BENCH, "minibude", "data", "bm1"),
        "-n", "65536"], "bm1 deck, 65536 poses (Table II); one run is "
       "1.6 s, so a job repeats it (see TEST_REPEAT)", "test"),
    _w("comd", "NVIDIA", os.path.join(BENCH, "comd-cuda", "bin"),
       ["./CoMD-cuda", "-e", "-d", os.path.join(BENCH, "comd-cuda", "pots"),
        "-x", "90", "-y", "90", "-z", "90"],
       "EAM, 90x90x90 cells (Table II: 100; 100^3 runs out of the 12 GB of the "
       "4070 Ti, the paper's V100S has 32 GB)", "test"),
    _w("gromacs", "GROMACS", GMX_RUN,
       [GMX, "mdrun", "-s", "topol.tpr", "-nb", "gpu", "-pme", "gpu",
        "-ntmpi", "1", "-ntomp", "16", "-nsteps", "20000", "-noconfout",
        "-g", "md.log"], "water-cut1.0 0768 (Table II), PME, 20000 steps", "test",
       # mdrun keeps #md.log.N# backups and aborts after 99 of them
       # (run 2026-11-06); -1 just overwrites the discarded output files
       env={"GMX_MAXBACKUP": "-1"}),
    _w("minife", "Mantevo", os.path.join(BENCH, "minife", "cuda", "src"),
       ["./miniFE.x", "-nx", "300", "-ny", "300", "-nz", "300"],
       "300x300x300 (Table II: 400; 320^3 and larger run out of memory "
       "on this machine)", "test"),
]

# Paper Table II CNNs: VggNet-16 and GoogleNet trained on CIFAR-100
# (cnn_workload.py). TensorFlow needs the CUDA 12 / cuDNN 9 libraries of the
# base miniconda install (the prasannajeet_env pip ones are CUDA 13 builds the
# 575 driver cannot run); they are only put on LD_LIBRARY_PATH.
CUDA12_SITE = "/home/drl/miniconda3/lib/python3.13/site-packages/nvidia"
CUDA12_LIBS = sorted(glob.glob(os.path.join(CUDA12_SITE, "*", "lib")))


def _cnn(name, steps):
    return _w(name, "CNN", HERE,
              [sys.executable, os.path.join(HERE, "cnn_workload.py"), name,
               "--steps", str(steps)], f"CIFAR-100, batch 128, {steps} steps",
              "test", env={"LD_LIBRARY_PATH": ":".join(CUDA12_LIBS)})


TEST_PROGS += [_cnn("vgg16", 400), _cnn("googlenet", 200)]

# Quantum ESPRESSO 7.2 pw.x, GPU build with nvfortran from the NVIDIA HPC
# SDK 23.11 (benchmarks/build_scripts/build_qe.sh); input AUSURF112 from
# github.com/QEF/benchmarks (paper Table II "ausurf112"): scf, 112 Au atoms,
# electron_maxstep 2. ONE INPUT CHANGE: diagonalization 'david' -> 'cg'.
# With Davidson the run needs > 12 GB and dies with "Out of memory" on the
# 4070 Ti (the paper's V100S has 32 GB); with CG it uses ~7.5 GB and takes
# 12 min 13 s per run (2 SCF iterations, as the deck specifies).
# Serial build, 16 OpenMP host threads.
QE_RUN = os.path.join(SCRATCH, "qe_ausurf112")
TEST_PROGS.append(
    _w("qe", "Quantum ESPRESSO", QE_RUN,
       [os.path.join(BENCH, "qe-src", "bin", "pw.x"), "-in", "ausurf.in"],
       "AUSURF112 (Table II: ausurf112), diagonalization='cg' (Davidson "
       "runs out of the 12 GB)", "test",
       env={"OMP_NUM_THREADS": "16"}))
# The deck stops after 2 SCF iterations by design (electron_maxstep = 2);
# pw.x then prints JOB DONE and exits with 2 = "SCF not converged"
# (PW/src/run_pwscf.f90 lines 195-202). That is a normal finish here.
TEST_PROGS[-1]["ok_exit"] = (0, 2)

WORKLOADS = {w["name"]: w for w in
             RODINIA_PROGS + PARBOIL_PROGS + SHOC_PROGS + POLY_PROGS + SDK_PROGS
             + TEST_PROGS}

WORKLOADS["shoc-MaxFlops"]["cmd"] += ["--passes", "1"]
WORKLOADS["shoc-MaxFlops"]["input"] = "size class 4, 1 pass"

# Runs per job so that one job lasts ~10 s at default clocks (single-run
# times from bench_probe_20261104_164051.csv: repeat = round(10 s / time),
# at least 1). Short programs repeated back to back also give real idle
# gaps between launches. One pass over all 75 suite programs is ~3650
# 300 ms steps, so the paper's 5000 training iterations cover every
# training program at least once.
REPEAT = {
    "backprop": 22, "bfs": 9, "gaussian": 3, "hotspot": 1, "huffman": 45,
    "nw": 3, "particlefilter": 1, "pathfinder": 1, "b+tree": 11, "dwt2d": 45,
    "heartwall": 15, "lavaMD": 1, "lud": 1, "myocyte": 1, "streamcluster": 1,
    "pb-cutcp": 6, "pb-histo": 2, "pb-lbm": 3, "pb-mri-gridding": 1,
    "pb-mri-q": 42, "pb-sgemm": 9, "pb-stencil": 15, "pb-tpacf": 15,
    "shoc-BusSpeedDownload": 6, "shoc-BusSpeedReadback": 5,
    "shoc-MaxFlops": 2, "shoc-BFS": 22, "shoc-FFT": 6, "shoc-GEMM": 2,
    "shoc-MD5Hash": 15, "shoc-NeuralNet": 22, "shoc-Reduction": 6,
    "shoc-Scan": 3, "shoc-Sort": 7, "shoc-Stencil2D": 1, "shoc-Triad": 5,
    "shoc-S3D": 21, "poly-2dconv": 15, "poly-2mm": 1, "poly-3dconv": 22,
    "poly-3mm": 9, "poly-atax": 22, "poly-bicg": 22, "poly-corr": 1,
    "poly-covar": 1, "poly-fdtd-2d": 3, "poly-gemm": 22, "poly-gesummv": 21,
    "poly-gramschm": 1, "poly-mvt": 22, "poly-syr2k": 1, "poly-syrk": 9,
    "sdk-BlackScholes": 11, "sdk-binomialOptions": 2,
    "sdk-convolutionFFT2D": 3, "sdk-fastWalshTransform": 3,
    "sdk-MonteCarloMultiGPU": 45, "sdk-sortingNetworks": 3,
    "sdk-transpose": 21, "sdk-reduction": 15, "sdk-mergeSort": 22,
    "sdk-histogram": 11, "sdk-scalarProd": 38, "sdk-quasirandomGenerator": 11,
    "sdk-SobolQRNG": 22, "sdk-matrixMul": 45, "sdk-cudaTensorCoreGemm": 7,
    "sdk-dct8x8": 22, "sdk-convolutionSeparable": 6, "sdk-eigenvalues": 15,
    "sdk-dxtc": 45, "sdk-interval": 3, "sdk-FDTD3d": 1, "sdk-lineOfSight": 45,
    "sdk-radixSortThrust": 11,
}
for _k, _r in REPEAT.items():
    WORKLOADS[_k]["repeat"] = _r

# Test programs run once per job, as in the paper, except miniBUDE whose
# single run (1.6 s) spans only ~5 controller decisions.
TEST_REPEAT = {"minibude": 10}
for _k, _r in TEST_REPEAT.items():
    WORKLOADS[_k]["repeat"] = _r

# Validation set (PROPOSAL for review): held out of offline training and
# used only for tuning decisions (e.g. comparing action sets or variants),
# never reported as unseen tests. Two or so per suite, spread over
# compute-heavy, memory-heavy and low-GPU programs.
VAL = ["lud", "myocyte", "heartwall", "pb-lbm", "pb-histo", "shoc-Scan",
       "shoc-GEMM", "poly-fdtd-2d", "poly-corr", "sdk-FDTD3d",
       "sdk-binomialOptions"]
for _k in VAL:
    WORKLOADS[_k]["split"] = "val"

TRAIN = [k for k, w in WORKLOADS.items() if w["split"] == "train"]
TEST = [k for k, w in WORKLOADS.items() if w["split"] == "test"]


def gpu_env():
    """Environment that pins a child process to the RTX 4070 Ti."""
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return env


def prepare(name):
    """One-time input preparation a program needs before it can run."""
    os.makedirs(SCRATCH, exist_ok=True)
    if name == "qe" and not os.path.exists(os.path.join(QE_RUN, "ausurf.in")):
        src = os.path.join(BENCH, "qe-benchmarks", "AUSURF112")
        os.makedirs(QE_RUN, exist_ok=True)
        shutil.copy(os.path.join(src, "Au.pbe-nd-van.UPF"), QE_RUN)
        with open(os.path.join(src, "ausurf.in")) as f:
            deck = f.read()
        with open(os.path.join(QE_RUN, "ausurf.in"), "w") as f:
            f.write(deck.replace("diagonalization='david'", "diagonalization='cg'"))
    if name == "gromacs" and not os.path.exists(os.path.join(GMX_RUN, "topol.tpr")):
        src = os.path.join(BENCH, "gromacs-inputs", "water-cut1.0_GMX50_bare", "0768")
        os.makedirs(GMX_RUN, exist_ok=True)
        subprocess.run([GMX, "grompp", "-f", os.path.join(src, "pme.mdp"),
                        "-c", os.path.join(src, "conf.gro"),
                        "-p", os.path.join(src, "topol.top"),
                        "-o", "topol.tpr", "-po", "mdout.mdp"],
                       cwd=GMX_RUN, check=True, capture_output=True)


def run_blocking(name, timeout=None):
    """Run all repetitions of a workload; returns elapsed seconds."""
    w = WORKLOADS[name]
    prepare(name)
    t0 = time.perf_counter()
    for _ in range(w["repeat"]):
        p = subprocess.run(w["cmd"], cwd=w["cwd"], env={**gpu_env(), **w["env"]},
                           timeout=timeout, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True)
        if p.returncode not in w.get("ok_exit", (0,)):
            raise RuntimeError(f"{name} exited {p.returncode}: {p.stderr[-500:]}")
    return time.perf_counter() - t0
