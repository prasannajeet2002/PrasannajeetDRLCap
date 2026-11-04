"""Write BENCHMARKS.md (review item 5) from results/benchmark_table.csv plus
the per-suite source / version / location facts and the build failures."""

import csv

SOURCES = [
    ("Rodinia", "Rodinia 3.1 (rodinia.cs.virginia.edu)", "benchmarks/rodinia",
     "drlcap_repro/build_rodinia.sh"),
    ("Parboil", "Parboil 2.5, official pb2.5driver/benchmarks/datasets_standard "
     "archives (impact.crhc.illinois.edu)", "benchmarks/parboil-2.5",
     "build_scripts/build_parboil.sh"),
    ("SHOC", "github.com/vetter/shoc, master 00b25e2 (2020-04-17)",
     "benchmarks/shoc (built in shoc/build)", "build_scripts/build_shoc.sh"),
    ("PolyBench", "PolyBench/GPU, github.com/socal-ucr/polybench-gpu 08b2fb0 "
     "(paper ref. [50])", "benchmarks/polybench-gpu", "build_scripts/build_polybench.sh"),
    ("CUDA SDK", "github.com/NVIDIA/cuda-samples tag v12.3", "benchmarks/cuda-samples",
     "build_scripts/build_cuda_sdk.sh"),
    ("LULESH", "github.com/LLNL/LULESH branch 2.0.2-dev (46c0ffc), cuda/src",
     "benchmarks/lulesh", "build_scripts/build_test_apps.sh"),
    ("CloverLeaf", "github.com/UK-MAC/CloverLeaf_CUDA (03c7803)",
     "benchmarks/cloverleaf-cuda", "build_scripts/build_test_apps.sh"),
    ("TeaLeaf", "github.com/UK-MAC/TeaLeaf_CUDA (2ac2460)", "benchmarks/tealeaf-cuda",
     "build_scripts/build_test_apps.sh"),
    ("miniBUDE", "github.com/UoB-HPC/miniBUDE (2b23c27), CUDA model",
     "benchmarks/minibude", "build_scripts/build_test_apps.sh"),
    ("CoMD", "github.com/NVIDIA/CoMD-CUDA (c3b39ed), src-mpi/Makefile.EAM",
     "benchmarks/comd-cuda", "build_scripts/build_test_apps.sh"),
    ("miniFE", "github.com/Mantevo/miniFE (c79e0f7), cuda/", "benchmarks/minife",
     "build_scripts/build_test_apps.sh"),
    ("GROMACS", "GROMACS 2024.4 (git tag v2024.4), CUDA, own FFTW 3.3.10; input "
     "water_GMX50_bare 0768 (ftp.gromacs.org)", "benchmarks/gromacs",
     "build_scripts/build_gromacs.sh"),
    ("Quantum ESPRESSO", "QE 7.2 (gitlab QEF/q-e tag qe-7.2) with NVIDIA HPC SDK 23.11 "
     "nvfortran; input AUSURF112 (github.com/QEF/benchmarks)", "benchmarks/qe-src, "
     "benchmarks/nvhpc", "build_scripts/build_qe.sh"),
    ("CNN", "VGG-16 / GoogLeNet (Inception v1) on CIFAR-100 (cs.toronto.edu/~kriz, "
     "md5 eb9058c3...), TensorFlow 2.21", "drlcap_faithful/cnn_workload.py, "
     "benchmarks/cifar-100", "-"),
]

FAILED = [
    ("Parboil", "bfs, sad, spmv",
     "`texture is not a template` / `cudaUnbindTexture undefined`: the code uses "
     "CUDA texture references, removed in CUDA 12. Tried the cuda_base versions; "
     "they use texture references too. Not fixable without editing the source."),
    ("SHOC", "DeviceMemory, MD, Spmv, QTClustering", "same texture-reference errors."),
    ("Rodinia", "cfd, kmeans, hybridsort, srad, hotspot3D, leukocyte",
     "do not build with CUDA 12.3 / need the CUDA 5.5 runtime (unchanged from "
     "drlcap_repro)."),
]

CHANGED_INPUTS = [
    ("CoMD", "100 -> 90 cells per side", "100^3 runs out of the 12 GB of GPU memory "
     "(`CUDA error 2: out of memory`); the paper's V100S has 32 GB."),
    ("miniFE", "400 -> 300 per side", "350^3, 330^3 and 320^3 fail with an "
     "out-of-memory exception in generate_matrix_structure."),
    ("Quantum ESPRESSO", "diagonalization david -> cg", "Davidson needs > 12 GB "
     "(`Out of memory allocating 1220940800 bytes`); CG uses ~7.5 GB, 12 min 13 s per run."),
    ("SHOC MaxFlops", "--passes 1", "default 10 passes take 107 s; 1 pass 5.5 s, "
     "job = 2 runs."),
    ("miniBUDE", "10 runs per job", "one run is 1.6 s (~5 controller decisions)."),
]


def main():
    rows = list(csv.DictReader(open("results/benchmark_table.csv")))
    out = ["# Expanded benchmark set (review item 5)", "",
           "All programs run unmodified on the RTX 4070 Ti (GPU 1, "
           "`CUDA_VISIBLE_DEVICES=1`). Every row was run by `bench_probe.py`, which "
           "polls NVML every 0.2 s and checks that the program's processes appear in "
           "GPU 1's process list and in no other GPU's.", "",
           "## Sources", "", "| Suite / app | Source and version | Location | Build |",
           "|---|---|---|---|"]
    out += [f"| {a} | {b} | `{c}` | `{d}` |" for a, b, c, d in SOURCES]
    out += ["", "## Programs that could not be used", "",
            "| Suite | Programs | Error and attempted resolution |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in FAILED]
    out += ["", "## Inputs changed from the default / paper", "",
            "| Program | Change | Reason |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in CHANGED_INPUTS]
    for split, title in (("train", "Training set"), ("val", "Validation set (tuning only, "
                         "never reported as unseen tests)"),
                         ("test", "Test set (paper Table II)")):
        sel = [r for r in rows if r["split"] == split]
        out += ["", f"## {title}: {len(sel)} programs", "",
                "| Application | Suite | Input | Runs / job | Job (s) | GPU 1 | Util % | "
                "Mem util % | Power W | Character |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sel:
            gpu = "yes" if r["on_gpu1"] == "True" and r["on_other_gpu"] == "False" else \
                  ("not seen in 0.2 s polls" if r["on_other_gpu"] == "False" else "CHECK")
            out.append(f"| {r['application']} | {r['suite']} | {r['input']} | {r['repeat']} | "
                       f"{r['job_s']} | {gpu} | {r['mean_util']} | {r['mean_mem_util']} | "
                       f"{r['mean_power_w']} | {r['character']} |")
        counts = {}
        for r in sel:
            counts[r["character"]] = counts.get(r["character"], 0) + 1
        out += ["", "Character counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    out += ["", "Execution commands and working directories for every program are in "
            "`results/benchmark_table.csv` (columns `command`, `cwd`) and in "
            "`workloads.py`. Character = coarse label from the job's mean NVML "
            "utilisation (< 30 % low-GPU / host-bound; memory util >= 40 % memory-heavy; "
            "< 15 % compute-heavy; else mixed)."]
    open("BENCHMARKS.md", "w").write("\n".join(out) + "\n")
    print("wrote BENCHMARKS.md")


if __name__ == "__main__":
    main()
