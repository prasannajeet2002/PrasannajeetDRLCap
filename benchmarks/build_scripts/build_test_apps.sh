#!/bin/bash
# Build the DRLCap paper's CUDA test applications (paper Table II) for the
# RTX 4070 Ti (sm_89), single GPU. Sources are not modified: every change
# is a make/cmake variable given on the command line, listed per app.
# Usage: build_test_apps.sh [app ...]   (default: all)
B=$(cd "$(dirname "$0")/.." && pwd)
LOGS=$B/build_logs
CUDA=/usr/local/cuda-12.3
export PATH=$CUDA/bin:$PATH
ARCH="-gencode arch=compute_89,code=sm_89"
run() {  # run <name> <command...>: log, report OK/FAIL
    local name=$1; shift
    if "$@" > "$LOGS/test_$name.log" 2>&1; then echo "OK   $name"; else echo "FAIL $name"; fi
}

lulesh() {
    # LLNL/LULESH branch 2.0.2-dev, cuda/src (official LULESH 2.0 CUDA port).
    # Only build/Makefile.CRAY exists (Cray + MPI, sm_35); we compile the
    # same files with its release flags (-O3 -DNDEBUG) minus -DUSE_MPI.
    cd "$B/lulesh/cuda/src" && rm -f *.o lulesh &&
    for f in allocator lulesh lulesh-comms lulesh-comms-gpu; do
        nvcc -O3 -DNDEBUG -arch=sm_89 $f.cu -I ./ -c -o $f.o || return 1
    done &&
    nvcc allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o -o lulesh
}

cloverleaf() {
    # UK-MAC/CloverLeaf_CUDA master. Makefile has NV_ARCH presets up to
    # AMPERE; we add an ADA preset via the command line.
    cd "$B/cloverleaf-cuda" && make clean >/dev/null 2>&1
    make COMPILER=GNU CUDA_HOME=$CUDA NV_ARCH=ADA CODE_GEN_ADA="$ARCH" -j16
}

tealeaf() {
    # UK-MAC/TeaLeaf_CUDA (default branch). Same NV_ARCH mechanism.
    cd "$B/tealeaf-cuda" && make clean >/dev/null 2>&1
    # serial make: the Fortran modules must be compiled in Makefile order
    make COMPILER=GNU CUDA_HOME=$CUDA NV_ARCH=ADA CODE_GEN_ADA="$ARCH"
}

minibude() {
    # UoB-HPC/miniBUDE main, CUDA model via its CMake build.
    cd "$B/minibude" && rm -rf build &&
    cmake -Bbuild -H. -DMODEL=cuda -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCUDA_ARCH=sm_89 &&
    cmake --build build -j16
}

comd() {
    # NVIDIA/CoMD-CUDA, src-mpi/Makefile.EAM (paper runs CoMD with EAM? the
    # paper only gives input "100"; EAM is the README's documented case).
    # Single GPU: DO_MPI=OFF, CC=gcc. NVCC_FLAGS is the Makefile's own list
    # (minus --ptxas-options=-v) with one change to the gencodes:
    #  * The code uses legacy __shfl/__shfl_xor/__shfl_down, which nvcc only
    #    accepts for compute capability < 7.0. Instead of editing the source
    #    we emit compute_60 PTX only; the driver JIT-compiles it for sm_89
    #    at first launch.
    #  * Its cub submodule (2016) clashes with the CUB that CUDA 12.3's
    #    Thrust headers pull in, so comd-cuda/cub/cub is a symlink to
    #    CUDA 12.3's own CUB (the submodule checkout is not used).
    # The README says to use Makefile.EAM "then make", and its CoMD_info.h
    # rule depends on a file literally named Makefile, so copy it.
    cd "$B/comd-cuda" && if [ ! -L cub/cub ]; then
        rm -rf cub && mkdir cub && ln -s $CUDA/include/cub cub/cub; fi
    cd "$B/comd-cuda/src-mpi" && cp Makefile.EAM Makefile && make clean >/dev/null 2>&1
    make CUDA_HOME=$CUDA DO_MPI=OFF CC=gcc \
        NVCC_FLAGS="-DMAXATOMS=64 -lineinfo -gencode arch=compute_60,code=compute_60 -DCOMD_DOUBLE -O2 -DNDEBUG" -j16
}

minife() {
    # Mantevo/miniFE, cuda/src (MPI build, run as a single rank). Only the
    # gencode in NVCCFLAGS is changed (sm_35 -> sm_89).
    cd "$B/minife/cuda/src" && make clean >/dev/null 2>&1
    # The Makefile expects OpenMPI under $MPI_HOME; this machine has MPICH,
    # so MPICFLAGS/MPILDFLAGS are set to what `mpicxx -show` reports.
    make NVCCFLAGS="-lineinfo $ARCH" \
        MPICFLAGS="-I/usr/include/x86_64-linux-gnu/mpich -pthread" \
        MPILDFLAGS="-L/usr/lib/x86_64-linux-gnu -lmpichcxx -lmpich" -j16
}

apps=("$@"); [ ${#apps[@]} -eq 0 ] && apps=(lulesh cloverleaf tealeaf minibude comd minife)
for a in "${apps[@]}"; do run "$a" "$a"; done
