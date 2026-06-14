#include "vector_add.cuh"

// CUDA kernel 实现
template <typename T>
__global__ void vectorAdd(T *input1, T* input2, T *output, int64_t N) {
    // 计算 全局索引
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N) {
        output[tid] = input1[tid] + input2[tid];
    }
}

// launch_kernel 函数实现（包含CUDA kernel调用）
template<typename T>
void launch_kernel(T *h_input1, T *h_input2, T *h_output, int64_t N, int64_t M_SIZE) {
    float *d_input1, *d_input2, *d_output;

    cudaMalloc((void **)&d_input1, M_SIZE);
    cudaMalloc((void **)&d_input2, M_SIZE);
    cudaMalloc((void **)&d_output, M_SIZE);

    cudaMemcpy(d_input1, h_input1, M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, M_SIZE, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 64;
    dim3 block_size = BLOCK_SIZE;
    dim3 grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<grid_size, block_size>>>(d_input1, d_input2, d_output, N);

    cudaMemcpy(h_output, d_output, M_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

// 显式实例化模板
template __global__ void vectorAdd<float>(float *input1, float *input2, float *output, int64_t N);
template void launch_kernel<float>(float *h_input1, float *h_input2, float *h_output, int64_t N, int64_t M_SIZE);