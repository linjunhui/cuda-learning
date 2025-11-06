#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

int main() {
    printf("CUDA version: %d\n", CUDART_VERSION);
    printf("CUDA version: %d.%d\n", CUDART_VERSION / 1000, CUDART_VERSION % 100);

    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Number of devices: %d\n", deviceCount);
}