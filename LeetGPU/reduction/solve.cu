#include <cuda_runtime.h>

// 存在大数吃小数的问题
__global__ void reduce_sum1(const float *input, float *output, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N) {
        atomicAdd(output, input[tid]);
    }
}

// 核函数声明 - 你自己实现
__global__ void reduce_sum(const float* input, float* block_sums, int N) {
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int local_idx = threadIdx.x;

    // 声明共享内存
    extern __shared__ float shared_input[];

    // 初始化

    shared_input[local_idx] = global_idx < N ? input[global_idx] : 0.0f;
    __syncthreads();

    // 只用计算 blockDim.x 的一半
    if(local_idx < blockDim.x / 2) {
        for(int i = blockDim.x / 2; i > 0; i = i >> 1) {
            shared_input[local_idx] += shared_input[local_idx + i];
            __syncthreads();
        }
    }

    if(local_idx == 0) {
        block_sums[blockIdx.x] = shared_input[0];
    }
    
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    cudaMemset(output, 0, sizeof(float));

    int block_size = 16;
    int grid_size = (block_size + N - 1)/ block_size;

    // 计算 shared 的大小
    int shared_size = sizeof(float) * block_size;

    float* block_sums;
    cudaMalloc((void **)&block_sums, grid_size * sizeof(float));

    // reduce_sum1<<<grid_size, block_size, shared_size>>>(input, output, N);
    reduce_sum<<<grid_size, block_size, shared_size>>>(input, block_sums, N);

    float* h_block_sums;
    float h_sum = 0.0f;

    h_block_sums = (float *)malloc(grid_size * sizeof(float));

    cudaMemcpy(h_block_sums, block_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < grid_size; i++) {
        h_sum += h_block_sums[i];
    }

    cudaMemcpy(output, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

}