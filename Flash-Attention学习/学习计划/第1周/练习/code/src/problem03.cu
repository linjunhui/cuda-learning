/*
题目 3：线程标识符
知识点：threadIdx、blockIdx、blockDim、gridDim
*/

#include<cuda_runtime.h>
#include<stdio.h>

__global__ void print_idx() {


    printf("blockIdx: (%d/%d, %d/%d, %d/%d)\n",
        blockIdx.x, gridDim.x,
        blockIdx.y, gridDim.y,
        blockIdx.z, gridDim.z);
 
    printf("threadIdx: (%d/%d, %d/%d, %d/%d)\n",
            threadIdx.x, blockDim.x,
            threadIdx.y, blockDim.y,
            threadIdx.z, blockDim.z);
}

int main() {

    dim3 block_size(2, 4, 8);
    dim3 grid_size(2, 4, 4);

    print_idx<<<grid_size, block_size>>>();

    cudaDeviceSynchronize();
    return 0;
}