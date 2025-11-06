
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print_idx() {
    // x 还是 变化最快的维度
    printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z);
    // calculate the block thread index, 这里算的只是线程块内的索引
    int idx_in_block = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    printf("Block thread index: %d\n", idx_in_block);

    // 计算 每个 block 有多少线程
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    // 当前是第几个block
    int block_idx =  blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

    printf("Block index: %d\n", block_idx);
    int global_idx = block_idx * block_size + idx_in_block;
    printf("Global thread index: %d\n", global_idx);
}

int main() {

    dim3 grid(3, 4); //it's equivalent to dim3 grid(3, 4, 1);
    dim3 block(4, 2); //it's equivalent to dim3 block(4, 2, 1);

    print_idx<<<grid, block>>>();
    cudaDeviceSynchronize();

}
