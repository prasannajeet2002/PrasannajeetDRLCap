#!/bin/bash
# Build the official Parboil 2.5 CUDA benchmarks (impact.crhc.illinois.edu,
# pb2.5driver/benchmarks/datasets_standard.tgz) for sm_89.
# The Parboil driver script is Python 2, which this machine lacks, so we
# call its own top-level makefile exactly as driver/benchmark.py does
# (cwd = benchmark dir, SRCDIR=src/cuda, BUILDDIR=build/cuda_default,
# BIN=build/cuda_default/<name>), only overriding the CUDA path and adding
# -arch=sm_89 to the platform's -O3.
#
# Five app Makefiles hard-code "-arch compute_12" / "compute_20", which
# CUDA 12 rejects. For those we pass APP_CUDACFLAGS on the command line
# with that one flag removed and every other app flag kept (sources and
# Makefiles stay unmodified):
#   bfs          -Xptxas -dlcm=cg -arch compute_12 --use_fast_math
#   histo        -arch compute_20
#   mri-gridding -arch=compute_20
#   tpacf        -arch compute_12
#   spmv         -I$(TOOLS_SRC)   (no arch flag; listed only for clarity)
PB=$(cd "$(dirname "$0")/../parboil-2.5" && pwd)
LOGS=$PB/../build_logs
CUDA=/usr/local/cuda-12.3
sed -e "s|^CUDA_PATH=.*|CUDA_PATH=$CUDA|" \
    -e "s|^CUDA_LIB_PATH=.*|CUDA_LIB_PATH=$CUDA/lib64|" \
    "$PB/common/Makefile.conf.example-nvidia" > "$PB/common/Makefile.conf"

declare -A APPFLAGS=(
    [bfs]="-Xptxas -dlcm=cg --use_fast_math"
    [histo]=""
    [mri-gridding]=""
    [tpacf]=""
)

for b in bfs cutcp histo lbm mri-gridding mri-q sad sgemm spmv stencil tpacf; do
    # the "cuda" version of each benchmark is the optimised one
    extra=()
    if [[ -v APPFLAGS[$b] ]]; then
        extra=(APP_CUDACFLAGS="${APPFLAGS[$b]}")
    fi
    rm -rf "$PB/benchmarks/$b/build/cuda_default"
    if (cd "$PB/benchmarks/$b" && make -f "$PB/common/mk/Makefile" \
            PARBOIL_ROOT="$PB" PLATFORM=default BUILD=cuda \
            SRCDIR=src/cuda BUILDDIR=build/cuda_default \
            BIN=build/cuda_default/$b \
            CUDAHOME=$CUDA PLATFORM_CUDACFLAGS="-O3 -arch=sm_89" \
            "${extra[@]}") > "$LOGS/parboil_$b.log" 2>&1; then
        echo "OK   $b"
    else
        echo "FAIL $b"
    fi
done
