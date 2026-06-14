/*
#### 题目1：第一个CUDA程序
**难度**：★☆☆☆☆

编写你的第一个CUDA程序：
1. 在GPU上打印"Hello CUDA World!"
2. 显示当前线程的ID
3. 显示当前线程块的ID
4. 显示当前网格的ID

**要求**：
- 使用 `__global__` 关键字定义内核函数
- 使用 `threadIdx.x`、`blockIdx.x`、`blockDim.x` 等内置变量
- 在内核函数中使用 `printf` 输出信息
- 使用 `cudaDeviceSynchronize()` 等待GPU完成

**知识点**：
- CUDA程序基本结构
- 内核函数定义和调用
- 线程索引概念
- 主机-设备交互
*/

#include<cuda_runtime.h>
#include<cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>

__global__  void hello_world_from_gpu() {
    printf("Hello CUDA World!\n");
    // 计算每个Block的线程数量
    int threadCountPerBlock = blockDim.x * blockDim.y * blockDim.z;
    // 计算Block的全局索引
    int blockGlobalIdx = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

    int threadGloabalIdx = threadCountPerBlock * blockGlobalIdx \
                        + blockDim.x * blockDim.y * threadIdx.z \
                        + blockDim.x * threadIdx.y + threadIdx.x;
    
    printf("Block->x: %d, Block->y: %d, Block->z: %d \t Thread->x: %d, Thread->y: %d, Thread->z: %d Global Idx: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, threadGloabalIdx);

}

int main() {
    dim3 gridDim(2, 2, 4);
    dim3 blockDim(2, 4, 4);
    hello_world_from_gpu<<<gridDim, blockDim>>>();
    cudaError_t error = cudaGetLastError();
    printf("cudaError %d\n", error);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    printf("cudaError %d\n", error);
}