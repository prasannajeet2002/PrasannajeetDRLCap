# Expanded benchmark set (review item 5)

All programs run unmodified on the RTX 4070 Ti (GPU 1, `CUDA_VISIBLE_DEVICES=1`). Every row was run by `bench_probe.py`, which polls NVML every 0.2 s and checks that the program's processes appear in GPU 1's process list and in no other GPU's.

## Sources

| Suite / app | Source and version | Location | Build |
|---|---|---|---|
| Rodinia | Rodinia 3.1 (rodinia.cs.virginia.edu) | `benchmarks/rodinia` | `drlcap_repro/build_rodinia.sh` |
| Parboil | Parboil 2.5, official pb2.5driver/benchmarks/datasets_standard archives (impact.crhc.illinois.edu) | `benchmarks/parboil-2.5` | `build_scripts/build_parboil.sh` |
| SHOC | github.com/vetter/shoc, master 00b25e2 (2020-04-17) | `benchmarks/shoc (built in shoc/build)` | `build_scripts/build_shoc.sh` |
| PolyBench | PolyBench/GPU, github.com/socal-ucr/polybench-gpu 08b2fb0 (paper ref. [50]) | `benchmarks/polybench-gpu` | `build_scripts/build_polybench.sh` |
| CUDA SDK | github.com/NVIDIA/cuda-samples tag v12.3 | `benchmarks/cuda-samples` | `build_scripts/build_cuda_sdk.sh` |
| LULESH | github.com/LLNL/LULESH branch 2.0.2-dev (46c0ffc), cuda/src | `benchmarks/lulesh` | `build_scripts/build_test_apps.sh` |
| CloverLeaf | github.com/UK-MAC/CloverLeaf_CUDA (03c7803) | `benchmarks/cloverleaf-cuda` | `build_scripts/build_test_apps.sh` |
| TeaLeaf | github.com/UK-MAC/TeaLeaf_CUDA (2ac2460) | `benchmarks/tealeaf-cuda` | `build_scripts/build_test_apps.sh` |
| miniBUDE | github.com/UoB-HPC/miniBUDE (2b23c27), CUDA model | `benchmarks/minibude` | `build_scripts/build_test_apps.sh` |
| CoMD | github.com/NVIDIA/CoMD-CUDA (c3b39ed), src-mpi/Makefile.EAM | `benchmarks/comd-cuda` | `build_scripts/build_test_apps.sh` |
| miniFE | github.com/Mantevo/miniFE (c79e0f7), cuda/ | `benchmarks/minife` | `build_scripts/build_test_apps.sh` |
| GROMACS | GROMACS 2024.4 (git tag v2024.4), CUDA, own FFTW 3.3.10; input water_GMX50_bare 0768 (ftp.gromacs.org) | `benchmarks/gromacs` | `build_scripts/build_gromacs.sh` |
| Quantum ESPRESSO | QE 7.2 (gitlab QEF/q-e tag qe-7.2) with NVIDIA HPC SDK 23.11 nvfortran; input AUSURF112 (github.com/QEF/benchmarks) | `benchmarks/qe-src, benchmarks/nvhpc` | `build_scripts/build_qe.sh` |
| CNN | VGG-16 / GoogLeNet (Inception v1) on CIFAR-100 (cs.toronto.edu/~kriz, md5 eb9058c3...), TensorFlow 2.21 | `drlcap_faithful/cnn_workload.py, benchmarks/cifar-100` | `-` |

## Programs that could not be used

| Suite | Programs | Error and attempted resolution |
|---|---|---|
| Parboil | bfs, sad, spmv | `texture is not a template` / `cudaUnbindTexture undefined`: the code uses CUDA texture references, removed in CUDA 12. Tried the cuda_base versions; they use texture references too. Not fixable without editing the source. |
| SHOC | DeviceMemory, MD, Spmv, QTClustering | same texture-reference errors. |
| Rodinia | cfd, kmeans, hybridsort, srad, hotspot3D, leukocyte | do not build with CUDA 12.3 / need the CUDA 5.5 runtime (unchanged from drlcap_repro). |

## Inputs changed from the default / paper

| Program | Change | Reason |
|---|---|---|
| CoMD | 100 -> 90 cells per side | 100^3 runs out of the 12 GB of GPU memory (`CUDA error 2: out of memory`); the paper's V100S has 32 GB. |
| miniFE | 400 -> 300 per side | 350^3, 330^3 and 320^3 fail with an out-of-memory exception in generate_matrix_structure. |
| Quantum ESPRESSO | diagonalization david -> cg | Davidson needs > 12 GB (`Out of memory allocating 1220940800 bytes`); CG uses ~7.5 GB, 12 min 13 s per run. |
| SHOC MaxFlops | --passes 1 | default 10 passes take 107 s; 1 pass 5.5 s, job = 2 runs. |
| miniBUDE | 10 runs per job | one run is 1.6 s (~5 controller decisions). |

## Training set: 64 programs

| Application | Suite | Input | Runs / job | Job (s) | GPU 1 | Util % | Mem util % | Power W | Character |
|---|---|---|---|---|---|---|---|---|---|
| backprop | Rodinia | 524288 input units | 22 | 9.0 | yes | 4.8 | 0.5 | 37.2 | low-GPU / host-bound |
| bfs | Rodinia | graph1MW_6 (1M nodes) | 9 | 9.1 | yes | 4.7 | 0.1 | 40.4 | low-GPU / host-bound |
| gaussian | Rodinia | 4096x4096 matrix | 3 | 9.0 | yes | 87.4 | 7.8 | 87.9 | compute-heavy |
| hotspot | Rodinia | 1024x1024 grid, 100000 iterations | 1 | 9.3 | yes | 91.7 | 0.1 | 92.1 | compute-heavy |
| huffman | Rodinia | test1024_H2 input | 45 | 9.2 | yes | 11.1 | 0.3 | 49.4 | low-GPU / host-bound |
| nw | Rodinia | 16384x16384, penalty 10 | 3 | 9.7 | yes | 12.8 | 1.1 | 42.6 | low-GPU / host-bound |
| particlefilter | Rodinia | 128x128x100, 200000 particles | 1 | 11.0 | yes | 95.1 | 0.5 | 85.7 | compute-heavy |
| pathfinder | Rodinia | 100000 cols x 5000 rows, pyramid 20 | 1 | 25.7 | yes | 2.6 | 0.0 | 15.2 | low-GPU / host-bound |
| b+tree | Rodinia | mil.txt (1M keys) | 11 | 9.6 | yes | 1.7 | 0.2 | 38.1 | low-GPU / host-bound |
| dwt2d | Rodinia | 1024x1024 rgb.bmp, 5/3 forward, 3 levels | 45 | 9.3 | yes | 3.2 | 0.2 | 44.6 | low-GPU / host-bound |
| lavaMD | Rodinia | 60 boxes per dim | 1 | 10.6 | yes | 81.2 | 0.0 | 72.3 | compute-heavy |
| streamcluster | Rodinia | 524288 points, 256 dims | 1 | 30.6 | yes | 26.8 | 10.4 | 46.5 | low-GPU / host-bound |
| pb-cutcp | Parboil | large: watbox.sl100.pqr | 6 | 9.4 | yes | 0.7 | 0.1 | 41.6 | low-GPU / host-bound |
| pb-mri-gridding | Parboil | small: small.uks | 1 | 6.8 | yes | 0.3 | 0.0 | 39.2 | low-GPU / host-bound |
| pb-mri-q | Parboil | large: 64x64x64 | 42 | 8.4 | yes | 3.8 | 0.6 | 45.0 | low-GPU / host-bound |
| pb-sgemm | Parboil | medium: matrix1 x matrix2t | 9 | 8.8 | yes | 0.6 | 0.1 | 41.0 | low-GPU / host-bound |
| pb-stencil | Parboil | default: 512x512x64, 100 iters | 15 | 9.9 | yes | 2.3 | 0.5 | 45.4 | low-GPU / host-bound |
| pb-tpacf | Parboil | large: 100 random sets, 10391 points | 15 | 10.5 | yes | 37.2 | 0.0 | 70.1 | compute-heavy |
| shoc-BusSpeedDownload | SHOC | size class 4, default passes | 6 | 9.2 | yes | 58.4 | 2.3 | 52.0 | compute-heavy |
| shoc-BusSpeedReadback | SHOC | size class 4, default passes | 5 | 10.2 | yes | 48.9 | 1.6 | 49.0 | compute-heavy |
| shoc-MaxFlops | SHOC | size class 4, 1 pass | 2 | 20.2 | yes | 97.5 | 0.1 | 77.0 | compute-heavy |
| shoc-BFS | SHOC | size class 4, default passes | 22 | 6.5 | yes | 15.8 | 0.0 | 53.5 | low-GPU / host-bound |
| shoc-FFT | SHOC | size class 4, default passes | 6 | 9.7 | yes | 15.4 | 7.3 | 49.3 | low-GPU / host-bound |
| shoc-MD5Hash | SHOC | size class 4, default passes | 15 | 8.8 | yes | 57.4 | 0.2 | 164.1 | compute-heavy |
| shoc-NeuralNet | SHOC | size class 4, default passes | 22 | 5.0 | not seen in 0.2 s polls | 11.1 | 0.5 | 72.9 | low-GPU / host-bound |
| shoc-Reduction | SHOC | size class 4, default passes | 6 | 8.8 | yes | 57.5 | 54.9 | 91.9 | memory-heavy |
| shoc-Sort | SHOC | size class 4, default passes | 7 | 8.1 | yes | 41.9 | 21.8 | 68.4 | mixed |
| shoc-Stencil2D | SHOC | size class 4, default passes | 1 | 89.2 | yes | 13.0 | 12.6 | 40.9 | low-GPU / host-bound |
| shoc-Triad | SHOC | size class 4, default passes | 5 | 9.5 | yes | 21.8 | 5.6 | 52.1 | low-GPU / host-bound |
| shoc-S3D | SHOC | size class 4, default passes | 21 | 7.8 | yes | 30.1 | 19.0 | 67.3 | mixed |
| poly-2dconv | PolyBench | default 4096x4096 | 15 | 9.2 | yes | 2.3 | 0.1 | 44.5 | low-GPU / host-bound |
| poly-2mm | PolyBench | default 2048^3 | 1 | 123.0 | yes | 0.0 | 0.0 | 13.8 | low-GPU / host-bound |
| poly-3dconv | PolyBench | default 256^3 | 22 | 8.8 | yes | 5.7 | 0.4 | 36.9 | low-GPU / host-bound |
| poly-3mm | PolyBench | default 512^3 | 9 | 10.1 | yes | 0.7 | 0.1 | 38.9 | low-GPU / host-bound |
| poly-atax | PolyBench | default 4096x4096 | 22 | 5.9 | yes | 6.3 | 0.8 | 41.6 | low-GPU / host-bound |
| poly-bicg | PolyBench | default 4096x4096 | 22 | 6.8 | yes | 2.5 | 0.4 | 41.5 | low-GPU / host-bound |
| poly-covar | PolyBench | default 2048x2048 | 1 | 14.4 | yes | 1.2 | 0.0 | 41.4 | low-GPU / host-bound |
| poly-gemm | PolyBench | default 512^3 | 22 | 11.1 | yes | 0.2 | 0.0 | 39.5 | low-GPU / host-bound |
| poly-gesummv | PolyBench | default 4096 | 21 | 6.6 | yes | 4.4 | 0.4 | 41.5 | low-GPU / host-bound |
| poly-gramschm | PolyBench | default 2048x2048 | 1 | 112.1 | yes | 1.0 | 0.0 | 15.8 | low-GPU / host-bound |
| poly-mvt | PolyBench | default 4096 | 22 | 8.3 | yes | 5.0 | 2.4 | 36.9 | low-GPU / host-bound |
| poly-syr2k | PolyBench | default 2048x2048 | 1 | 15.6 | yes | 0.0 | 0.0 | 40.0 | low-GPU / host-bound |
| poly-syrk | PolyBench | default 1024x1024 | 9 | 9.9 | yes | 0.5 | 0.1 | 38.8 | low-GPU / host-bound |
| sdk-BlackScholes | CUDA SDK | default problem size | 11 | 8.4 | yes | 6.1 | 5.2 | 58.6 | low-GPU / host-bound |
| sdk-convolutionFFT2D | CUDA SDK | default problem size | 3 | 7.9 | yes | 0.8 | 0.0 | 41.0 | low-GPU / host-bound |
| sdk-fastWalshTransform | CUDA SDK | default problem size | 3 | 11.7 | yes | 1.0 | 0.2 | 37.8 | low-GPU / host-bound |
| sdk-MonteCarloMultiGPU | CUDA SDK | default problem size | 45 | 9.2 | yes | 9.3 | 0.5 | 49.9 | low-GPU / host-bound |
| sdk-sortingNetworks | CUDA SDK | default problem size | 3 | 9.7 | yes | 0.4 | 0.0 | 39.8 | low-GPU / host-bound |
| sdk-transpose | CUDA SDK | default problem size | 21 | 5.7 | yes | 5.2 | 0.5 | 43.5 | low-GPU / host-bound |
| sdk-reduction | CUDA SDK | default problem size | 15 | 8.6 | yes | 6.1 | 3.0 | 44.5 | low-GPU / host-bound |
| sdk-mergeSort | CUDA SDK | default problem size | 22 | 8.1 | yes | 4.5 | 1.4 | 42.4 | low-GPU / host-bound |
| sdk-histogram | CUDA SDK | default problem size | 11 | 9.0 | yes | 1.0 | 0.3 | 40.6 | low-GPU / host-bound |
| sdk-scalarProd | CUDA SDK | default problem size | 38 | 8.1 | yes | 3.0 | 0.7 | 43.2 | low-GPU / host-bound |
| sdk-quasirandomGenerator | CUDA SDK | default problem size | 11 | 7.4 | yes | 2.4 | 0.0 | 40.4 | low-GPU / host-bound |
| sdk-SobolQRNG | CUDA SDK | default problem size | 22 | 7.2 | yes | 2.1 | 0.0 | 41.5 | low-GPU / host-bound |
| sdk-matrixMul | CUDA SDK | default problem size | 45 | 9.1 | yes | 18.1 | 0.9 | 48.1 | low-GPU / host-bound |
| sdk-cudaTensorCoreGemm | CUDA SDK | default problem size | 7 | 9.4 | yes | 0.8 | 0.0 | 40.7 | low-GPU / host-bound |
| sdk-dct8x8 | CUDA SDK | default problem size | 22 | 9.2 | yes | 7.7 | 0.8 | 41.1 | low-GPU / host-bound |
| sdk-convolutionSeparable | CUDA SDK | default problem size | 6 | 9.2 | yes | 1.2 | 0.2 | 40.0 | low-GPU / host-bound |
| sdk-eigenvalues | CUDA SDK | default problem size | 15 | 7.2 | yes | 56.5 | 0.3 | 55.5 | compute-heavy |
| sdk-dxtc | CUDA SDK | default problem size | 45 | 8.4 | yes | 4.4 | 0.6 | 47.7 | low-GPU / host-bound |
| sdk-interval | CUDA SDK | default problem size | 3 | 11.2 | yes | 2.5 | 0.3 | 39.2 | low-GPU / host-bound |
| sdk-lineOfSight | CUDA SDK | default problem size | 45 | 9.3 | yes | 3.9 | 0.9 | 43.0 | low-GPU / host-bound |
| sdk-radixSortThrust | CUDA SDK | default problem size | 11 | 7.5 | yes | 0.8 | 0.1 | 48.5 | low-GPU / host-bound |

Character counts: compute-heavy 10, low-GPU / host-bound 51, memory-heavy 1, mixed 2

## Validation set (tuning only, never reported as unseen tests): 11 programs

| Application | Suite | Input | Runs / job | Job (s) | GPU 1 | Util % | Mem util % | Power W | Character |
|---|---|---|---|---|---|---|---|---|---|
| heartwall | Rodinia | test.avi, 104 frames | 15 | 10.4 | yes | 34.3 | 2.4 | 68.6 | compute-heavy |
| lud | Rodinia | 32768x32768 matrix | 1 | 11.7 | yes | 75.0 | 67.7 | 163.9 | memory-heavy |
| myocyte | Rodinia | 6000 ms, 1 instance | 1 | 24.6 | yes | 65.0 | 0.8 | 58.5 | compute-heavy |
| pb-histo | Parboil | large: img.bin, 10000 iters | 2 | 9.4 | yes | 23.4 | 2.9 | 68.9 | low-GPU / host-bound |
| pb-lbm | Parboil | long: 120x120x150, 3000 steps | 3 | 10.4 | yes | 70.8 | 69.0 | 116.9 | memory-heavy |
| shoc-GEMM | SHOC | size class 4, default passes | 2 | 9.5 | yes | 69.0 | 6.0 | 89.3 | compute-heavy |
| shoc-Scan | SHOC | size class 4, default passes | 3 | 10.8 | yes | 72.7 | 72.7 | 98.2 | memory-heavy |
| poly-corr | PolyBench | default 2048x2048 | 1 | 13.5 | yes | 6.7 | 0.0 | 43.2 | low-GPU / host-bound |
| poly-fdtd-2d | PolyBench | default 2048x2048, 500 steps | 3 | 10.6 | yes | 7.4 | 0.1 | 49.1 | low-GPU / host-bound |
| sdk-binomialOptions | CUDA SDK | default problem size | 2 | 7.9 | yes | 0.4 | 0.0 | 39.9 | low-GPU / host-bound |
| sdk-FDTD3d | CUDA SDK | default problem size | 1 | 12.6 | yes | 0.4 | 0.1 | 39.4 | low-GPU / host-bound |

Character counts: compute-heavy 3, low-GPU / host-bound 5, memory-heavy 3

## Test set (paper Table II): 10 programs

| Application | Suite | Input | Runs / job | Job (s) | GPU 1 | Util % | Mem util % | Power W | Character |
|---|---|---|---|---|---|---|---|---|---|
| lulesh | LLNL | -s 150 (Table II) | 1 | 153.9 | yes | 98.1 | 39.5 | 111.2 | mixed |
| cloverleaf | UK-MAC | clover.in 3840x3840, 2955 steps (Table II: 3840) | 1 | 213.2 | yes | 99.3 | 83.2 | 143.8 | memory-heavy |
| tealeaf | UK-MAC | tea.in 4000x4000, 10 steps (Table II: 4000) | 1 | 81.7 | yes | 98.7 | 98.8 | 153.8 | memory-heavy |
| minibude | UoB-HPC | bm1 deck, 65536 poses (Table II); one run is 1.6 s, so a job repeats it (see TEST_REPEAT) | 10 | 15.8 | yes | 86.9 | 28.6 | 115.8 | mixed |
| comd | NVIDIA | EAM, 90x90x90 cells (Table II: 100; 100^3 runs out of the 12 GB of the 4070 Ti, the paper's V100S has 32 GB) | 1 | 27.7 | yes | 87.5 | 2.1 | 84.8 | compute-heavy |
| gromacs | GROMACS | water-cut1.0 0768 (Table II), PME, 20000 steps | 1 | 88.5 | yes | 71.1 | 20.5 | 201.3 | mixed |
| minife | Mantevo | 300x300x300 (Table II: 400; 320^3 and larger run out of memory on this machine) | 1 | 9.6 | yes | 69.4 | 62.0 | 119.0 | memory-heavy |
| vgg16 | CNN | CIFAR-100, batch 128, 400 steps | 1 | 36.0 | yes | 18.3 | 12.3 | 73.0 | low-GPU / host-bound |
| googlenet | CNN | CIFAR-100, batch 128, 200 steps | 1 | 90.8 | yes | 35.7 | 31.6 | 100.3 | mixed |
| qe | Quantum ESPRESSO | AUSURF112 (Table II: ausurf112), diagonalization='cg' (Davidson runs out of the 12 GB) | 1 | 733.6 | yes | 92.6 | 87.6 | 146.4 | memory-heavy |

Character counts: compute-heavy 1, low-GPU / host-bound 1, memory-heavy 4, mixed 4

Execution commands and working directories for every program are in `results/benchmark_table.csv` (columns `command`, `cwd`) and in `workloads.py`. Character = coarse label from the job's mean NVML utilisation (< 30 % low-GPU / host-bound; memory util >= 40 % memory-heavy; < 15 % compute-heavy; else mixed).
