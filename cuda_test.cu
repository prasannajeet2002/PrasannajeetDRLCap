#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_gpu()
{
    printf("Hello from GPU!\n");
}

int main()
{
    int device_count = 0;

    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess)
    {
        printf(
            "CUDA error: %s\n",
            cudaGetErrorString(err)
        );

        return 1;
    }

    printf(
        "Number of CUDA devices: %d\n",
        device_count
    );

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;

        cudaGetDeviceProperties(
            &prop,
            i
        );

        printf(
            "CUDA Device %d: %s\n",
            i,
            prop.name
        );
    }

    /*
     * We specifically want physical GPU 1,
     * which is your RTX 4070 Ti.
     */
    cudaSetDevice(1);

    hello_gpu<<<1, 1>>>();

    cudaDeviceSynchronize();

    return 0;
}
