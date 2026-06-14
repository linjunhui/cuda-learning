/*
题目 4：全局线程 ID 计算（1D）
知识点：全局线程索引
完成点评：当前实现通过 `tid = blockIdx.x * blockDim.x + threadIdx.x` 正确计算全局 ID，并在示例中配合 `grid_size(3,1,1)`、`block_size(4,1,1)` 输出 12 个连续线程，满足题目要求；如需更贴合练习表格，可额外打印对应的 `blockIdx`、`threadIdx`。
*/
#include<cuda_runtime.h>
#include<stdio.h>

__global__ void print_idx() {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Current Thread Global Idx: %d\n", tid);
}

int main() {
    dim3 grid_size(3, 1, 1);
    dim3 block_size(4, 1, 1); 

    print_idx<<<grid_size, block_size>>>();

    cudaDeviceSynchronize();

    return 0;
}