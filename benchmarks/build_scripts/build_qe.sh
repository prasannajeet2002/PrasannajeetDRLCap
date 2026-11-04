#!/bin/bash
# Quantum ESPRESSO 7.2 (gitlab.com/QEF/q-e, tag qe-7.2), GPU build of pw.x.
# The paper (Sec. VI-E) notes that the GPU version of QE needs nvfortran from
# the NVIDIA HPC SDK. We install HPC SDK 23.11 (bundled CUDA 12.3, the same
# CUDA as the system toolkit) into benchmarks/nvhpc, user space only, no
# root, nothing outside the project folder, then build pw.x for sm_89.
# Input: AUSURF112 from github.com/QEF/benchmarks (paper Table II "ausurf112").
B=$(cd "$(dirname "$0")/.." && pwd)
LOGS=$B/build_logs
TARBALL=$B/downloads/nvhpc_2023_2311_Linux_x86_64_cuda_12.3.tar.gz
NVHPC=$B/nvhpc
NVROOT=$NVHPC/Linux_x86_64/23.11

if [ ! -x "$NVROOT/compilers/bin/nvfortran" ]; then
    mkdir -p "$B/nvhpc_src" && tar xzf "$TARBALL" -C "$B/nvhpc_src" || exit 1
    (cd "$B/nvhpc_src"/nvhpc_2023_2311_Linux_x86_64_cuda_12.3 &&
     NVHPC_SILENT=true NVHPC_INSTALL_DIR="$NVHPC" NVHPC_INSTALL_TYPE=single \
         ./install) > "$LOGS/nvhpc_install.log" 2>&1 || { echo "nvhpc install FAILED"; exit 1; }
    rm -rf "$B/nvhpc_src"
fi
export PATH=$NVROOT/compilers/bin:$NVROOT/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=$NVROOT/compilers/lib:$LD_LIBRARY_PATH

cd "$B/qe-src" || exit 1
make veryclean > /dev/null 2>&1
# serial (no MPI) GPU build, OpenMP on the host, CUDA from the HPC SDK
./configure --with-cuda="$NVROOT/cuda/12.3" --with-cuda-cc=89 \
    --with-cuda-runtime=12.3 --enable-openmp --disable-parallel \
    --with-scalapack=no > "$LOGS/qe_configure.log" 2>&1 || { echo "configure FAILED"; exit 1; }
make -j32 pw > "$LOGS/qe_make.log" 2>&1 || { echo "make FAILED"; exit 1; }
echo "OK   qe -> $B/qe-src/bin/pw.x"
