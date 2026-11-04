#!/bin/bash
# Build PolyBench/GPU (socal-ucr/polybench-gpu, Grauer-Gray et al. 2012 = the
# paper's ref [50]) CUDA versions for sm_89. Sources are not modified; the
# repo's common.mk only lacks an -arch flag, so we call nvcc directly with
# the same -O3 plus -arch=sm_89. Each program uses device 0, which is the
# RTX 4070 Ti once CUDA_VISIBLE_DEVICES=1 is set at run time.
cd "$(dirname "$0")/../polybench-gpu/CUDA" || exit 1
LOGS=../../build_logs
for d in */; do
    d=${d%/}
    exe=$(sed -n 's/^EXECUTABLE := //p' "$d/Makefile")
    src=$(sed -n 's/^CUFILES := //p' "$d/Makefile")
    if (cd "$d" && /usr/local/cuda-12.3/bin/nvcc -O3 -arch=sm_89 $src -o $exe) \
            > "$LOGS/polybench_$d.log" 2>&1; then
        echo "OK   $d ($exe)"
    else
        echo "FAIL $d"
    fi
done
