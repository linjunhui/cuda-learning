#include <cuda_runtime.h>
#include <cstdio>

/*
题目 5：全局线程 ID 计算（2D）
知识点：二维线程索引
*/
__global__ void print_idx() {
    // Grid 也是 二维
    // 每个block 中的线程数量
    int thread_per_block = blockDim.x * blockDim.y;
    // 每个 grid 中的block数量
    int block_per_grid = gridDim.x * gridDim.y;

    // 计算 block 的线性索引
    int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    // 计算线程在block内的线性索引
    int thread_in_block = blockDim.x * threadIdx.y + threadIdx.x;

    // 计算全局线程索引：块索引 × 每块线程数 + 块内线程索引
    int global_idx = block_idx * thread_per_block + thread_in_block;

    printf("Current Thread Global Idx: %d\n", global_idx);
}

int main() {

    const int N = 1000;

    int block_size_x = 4, block_size_y = 4;
    int block_size = block_size_x * block_size_y;

    int grid_size = (N + block_size - 1) / block_size;
    int grid_size_x = 16;
    int grid_size_y = (grid_size + grid_size_x - 1) / grid_size_x;

    dim3 grid_size_2d(grid_size_x, grid_size_y);
    dim3 block_size_2d(block_size_x, block_size_y);

    print_idx<<<grid_size_2d, block_size_2d>>>();

    cudaDeviceSynchronize();

    return 0;
}