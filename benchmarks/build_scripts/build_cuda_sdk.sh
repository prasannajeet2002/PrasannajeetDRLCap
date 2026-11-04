#!/bin/bash
# Build the chosen NVIDIA CUDA SDK samples (cuda-samples tag v12.3) for the
# RTX 4070 Ti (sm_89). One log per sample in ../build_logs/.
cd "$(dirname "$0")/../cuda-samples" || exit 1
LOGS=../build_logs
mkdir -p $LOGS
for s in BlackScholes binomialOptions convolutionFFT2D fastWalshTransform \
         MonteCarloMultiGPU sortingNetworks transpose reduction mergeSort \
         histogram scalarProd quasirandomGenerator SobolQRNG matrixMul \
         cudaTensorCoreGemm dct8x8 convolutionSeparable eigenvalues dxtc \
         interval FDTD3d lineOfSight radixSortThrust; do
    d=$(find Samples -maxdepth 2 -type d -name "$s")
    if make -C "$d" SMS=89 CUDA_PATH=/usr/local/cuda-12.3 -j16 \
            > "$LOGS/sdk_$s.log" 2>&1; then
        echo "OK   $s"
    else
        echo "FAIL $s"
    fi
done
