#!/bin/bash
# Build GROMACS 2024.4 (git tag v2024.4) with CUDA for sm_89, single
# precision, thread-MPI (no external MPI), installed to benchmarks/gromacs.
# GROMACS builds its own FFTW; cmake cannot download through the campus
# proxy, so FFTW 3.3.10 is fetched beforehand into downloads/ (md5
# 8ccbf6a5ea78a16dbc3e1306e234cc5c, the official checksum) and passed as
# a local URL.
B=$(cd "$(dirname "$0")/.." && pwd)
CUDA=/usr/local/cuda-12.3
SRC=$B/gromacs-src
rm -rf "$SRC/build" && mkdir -p "$SRC/build" && cd "$SRC/build" || exit 1
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DGMX_GPU=CUDA -DCUDA_TOOLKIT_ROOT_DIR=$CUDA \
    -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DGMX_CUDA_TARGET_SM=89 \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DGMX_BUILD_OWN_FFTW_URL="file://$B/downloads/fftw-3.3.10.tar.gz" \
    -DGMX_BUILD_OWN_FFTW_MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c \
    -DREGRESSIONTEST_DOWNLOAD=OFF \
    -DCMAKE_INSTALL_PREFIX="$B/gromacs" \
    > "$B/build_logs/gromacs_cmake.log" 2>&1 || { echo "cmake FAILED"; exit 1; }
make -j48 > "$B/build_logs/gromacs_make.log" 2>&1 || { echo "make FAILED"; exit 1; }
make install > "$B/build_logs/gromacs_install.log" 2>&1 || { echo "install FAILED"; exit 1; }
echo "OK   gromacs -> $B/gromacs/bin/gmx"
