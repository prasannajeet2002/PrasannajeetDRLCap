#!/usr/bin/env bash
# Build the Rodinia 3.1 CUDA benchmarks for the RTX 4070 Ti (Ada, sm_89).
# benchmarks/ is not tracked in git, so this script records exactly how the
# binaries used by workloads.py were produced.
#
# nvcc 12.3 supports gcc <= 12, but prasannajeet_env puts gcc 15 (without
# cc1plus) first on PATH, so the system gcc 11 is forced here.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R="$ROOT/benchmarks/rodinia"
export PATH=/usr/local/cuda-12.3/bin:/usr/bin:$PATH
export CC=/usr/bin/gcc CXX=/usr/bin/g++ LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

cd "$R"
# Old Fermi/Tesla arch flags -> Ada; toolkit path -> CUDA 12.3.
sed -i -E 's/-arch[= ]sm_[0-9]+/-arch=sm_89/g; s/compute_20/compute_89/g' \
    cuda/*/Makefile* cuda/lavaMD/makefile
sed -i 's#^CUDA_DIR = /usr/local/cuda$#CUDA_DIR = /usr/local/cuda-12.3#' \
    common/make.config
# Inputs live in the separately downloaded rodinia-data tree.
ln -sfn ../rodinia-data/rodinia-data data

for d in backprop bfs b+tree dwt2d gaussian heartwall hotspot huffman \
         lavaMD lud myocyte nn nw particlefilter pathfinder streamcluster; do
    if (cd "cuda/$d" && make clean >/dev/null 2>&1; make >"/tmp/rodinia_$d.log" 2>&1); then
        echo "OK   $d"
    else
        echo "FAIL $d (see /tmp/rodinia_$d.log)"
    fi
done
