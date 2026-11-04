#!/bin/bash
# Build SHOC (vetter/shoc, master 00b25e2) CUDA benchmarks for sm_89,
# serial (no MPI), no OpenCL. Out-of-tree build in shoc/build.
SHOC=$(cd "$(dirname "$0")/../shoc" && pwd)
LOGS=$SHOC/../build_logs
export PATH=/usr/local/cuda-12.3/bin:$PATH
rm -rf "$SHOC/build"; mkdir -p "$SHOC/build" && cd "$SHOC/build" || exit 1
"$SHOC/configure" --with-cuda --without-opencl --without-mpi \
    CUDA_CPPFLAGS="-gencode=arch=compute_89,code=sm_89" \
    LDFLAGS="-L/usr/local/cuda-12.3/lib64" LIBS="-lcudart" \
    > "$LOGS/shoc_configure.log" 2>&1 || { echo "configure FAILED"; exit 1; }
# -k: keep going so one broken benchmark does not hide the others
make -k -j32 > "$LOGS/shoc_make.log" 2>&1
echo "make exit $?"
find "$SHOC/build/src/cuda" -type f -perm -u+x -name "[A-Z]*" | sort
