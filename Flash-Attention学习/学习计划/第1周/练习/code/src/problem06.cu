/*
题目 6：Warp 概念
知识点：Warp、Lane ID
*/

#include <cstdio>
#include<cuda_runtime.h>
#include <cuda.h>

/*
先分析 逻辑，再抄代码
1. 计算线程属于哪个warp
2. 每个warp的一个线程进行打印，简化打印量
3. 打印每个线程的 全局id， block内的id， laneid
*/
__global__ void printWarpAndLaneInfo() {
    /*
    这里是 一维场景：
        blockDim.x 是一个Block 的线程数量
        blockIdx.x 是 Block 索引

        threadIdx.x 是在当前Block中的索引
    */
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    //int laneId = __laneid();
    // int laneId = threadIdx.x % 32;
    // 32 -> 100 000, 第5位 往前的都是 32 的倍数，只需要保留前5位即可
    // 31 -> 011 111 
    int laneId = threadIdx.x & 31;
    // warp 的 ID
    int warpIdInBlock = threadIdx.x / 32;

    if(laneId == 0){
        printf("Block %d, Warp %d (in block), starts at Global Thread ID %d\n", blockIdx.x, warpIdInBlock, globalThreadId);
    }

    // 线程块 内同步
    __syncthreads();

    printf("Global Thread ID:%d, ThreadIdx.x: %d, Lane ID:%d\n", globalThreadId, threadIdx.x, laneId);
}

int main() {
    // 定义 一个 block 中的线程数量，这里故意没有写成32的倍数
    dim3 blockSize(42);
    dim3 gridSize(2); // 两个线程块

    printWarpAndLaneInfo<<<gridSize, blockSize>>>();

    cudaDeviceSynchronize();

    return 0;
}