#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>


__global__
void compute_kernel(
    float *A,
    float *B,
    float *C,
    int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float x = A[idx];
        float y = B[idx];

        for (int i = 0; i < 1000; i++)
        {
            x = x * 1.000001f + y * 0.999999f;
            y = y * 1.000002f + x * 0.999998f;
        }

        C[idx] = x + y;
    }
}


int main()
{
    int N = 1 << 24;

    size_t bytes = N * sizeof(float);

    float *A;
    float *B;
    float *C;

    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    for (int i = 0; i < N; i++)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    cudaDeviceSynchronize();

    auto start =
        std::chrono::high_resolution_clock::now();

    for (int iteration = 0;
         iteration < 100;
         iteration++)
    {
        int blockSize = 256;

        int gridSize =
            (N + blockSize - 1)
            / blockSize;

        compute_kernel<<<gridSize, blockSize>>>(
            A,
            B,
            C,
            N
        );

        cudaDeviceSynchronize();
    }

    auto end =
        std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration<double>(
            end - start
        ).count();

    printf(
        "Execution time: %.6f seconds\n",
        elapsed
    );

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
