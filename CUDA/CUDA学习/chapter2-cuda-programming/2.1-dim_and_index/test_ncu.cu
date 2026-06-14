#include <iostream>
#include <cuda_runtime.h>

__global__ void simpleKernel() {
    printf("Hello from CUDA kernel! ThreadIdx.x: %d\n", threadIdx.x);
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int result = idx * 2;  // 做一些简单的计算
    printf("Result: %d\n", result);
}

int main() {
    simpleKernel<<<2, 4>>>();
    cudaDeviceSynchronize();  // 等待内核完成
    return 0;
}
