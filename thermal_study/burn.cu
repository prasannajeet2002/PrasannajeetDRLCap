// Steady compute load for the leakage experiment: runs FMA-heavy kernels
// back to back for a fixed wall-clock time, so GPU activity stays constant
// while the die heats up.  Usage: ./burn <seconds>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

__global__ void fma_kernel(float *out, int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = idx * 1e-7f, y = 1.0f;
    for (int i = 0; i < iters; i++) {
        x = fmaf(x, 0.9999991f, y);
        y = fmaf(y, 0.9999993f, x);
    }
    out[idx] = x + y;
}

int main(int argc, char **argv)
{
    double seconds = argc > 1 ? atof(argv[1]) : 60.0;
    const int blocks = 60 * 64, threads = 256;   // 60 SMs on the 4070 Ti
    float *out;
    cudaMalloc(&out, blocks * threads * sizeof(float));
    auto t0 = std::chrono::steady_clock::now();
    long launches = 0;
    while (std::chrono::duration<double>(
               std::chrono::steady_clock::now() - t0).count() < seconds) {
        fma_kernel<<<blocks, threads>>>(out, 20000);
        cudaDeviceSynchronize();
        launches++;
    }
    cudaError_t err = cudaGetLastError();
    cudaFree(out);
    printf("launches %ld status %s\n", launches, cudaGetErrorString(err));
    return err == cudaSuccess ? 0 : 1;
}
