#include <cstddef>
#include <cstdio>
#include<cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <float.h>
#include <vector>
/*
softmax 是一个比较综合的题目
1. 找出数组中的最大值
2. 计算分母， 归一化常数
*/

__global__ void max_reduction_kernel(const float *arr, float *output, int N) {
    extern __shared__ float s_max[]; // 

    int block_idx = blockIdx.x;
    int global_tid = block_idx * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if(tid < N) {
        s_max[tid] = arr[global_tid];
    } else {
        s_max[tid] = -FLT_MAX;
    }
    __syncthreads();
    // reduce 计算 max
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        // 比较 arr[tid] 与 arr[tid+offset]
        if(global_tid < N && global_tid + offset < N) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid+offset]);
        }
        __syncthreads();
    }
    if(tid == 0) {
        output[block_idx] = s_max[0];
    }
}


__global__ void softmax_sum_reduction_kernel(const float *arr, float *output_sum, const float g_max, int N) {
    // 在每个block 计算指数 再求和
    extern __shared__ float s_sum[];

    // 加载数据到shared memory
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_x = threadIdx.x;

    if(global_idx < N) {
        //s_sum[tid_x] = arr[global_idx];
        s_sum[tid_x] = __expf(arr[global_idx] - g_max);
    } else {
        s_sum[tid_x] = 0.0f; // 参与求和，但是不影响
    }


    
    __syncthreads();

    // reduction 求和
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if(tid_x < offset) { // 保证这个条件 tid_x + offset 不会越界  blockDim.x
            s_sum[tid_x] += s_sum[tid_x + offset];
        }
        __syncthreads();
    }
    if(tid_x == 0) {
        output_sum[blockIdx.x] = s_sum[0];
    }
}

__global__ void softmax_kernel(const float *arr, float *output, const float g_max, const float S_factor, int N) {
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(global_idx < N) {
        output[global_idx] = __expf(arr[global_idx] - g_max) / S_factor;
    }
}

float max_reduction_wrapper(const float *d_arr, const int N) {

    // 定义一个数组接收每个block的最大值
    float *d_block_max;
    float *h_block_max;

    int block_size = 4;
    int grid_size = (N + block_size - 1) / block_size;
    size_t shared_bytes_size = block_size * sizeof(float);


    h_block_max = (float *)malloc(grid_size * sizeof(float));
    cudaMalloc((void **)&d_block_max, grid_size * sizeof(float));
    max_reduction_kernel<<<grid_size, block_size, shared_bytes_size>>>(d_arr, d_block_max, N);
    cudaMemcpy(h_block_max, d_block_max, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 这里 grid_size 不会特别大，直接求最大值了
    float rlt_max = -FLT_MAX;
    for(int i = 0; i < grid_size; i++) {
        rlt_max = fmaxf(rlt_max, h_block_max[i]);
    }

    return rlt_max;
}

float softmax_sum_reduction_wrapper(const float *d_arr, const float g_max, const int N) {
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;

    // 需要一个数组存储每个block的和
    float *h_block_sum;
    float *d_block_sum;
    h_block_sum = (float *)malloc(grid_size * sizeof(float));
    cudaMalloc((void **)&d_block_sum, grid_size * sizeof(float));

    softmax_sum_reduction_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(d_arr, d_block_sum, g_max, N);
    cudaMemcpy(h_block_sum, d_block_sum, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    float rlt_sum = 0.0f;
    for(int i = 0; i < grid_size; i++) {
        rlt_sum += h_block_sum[i];
    }

    return rlt_sum;
}

void softmax_wrapper(const float *arr, float *output, float g_max, float S_factor, int N) {
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    
    softmax_kernel<<<grid_size, block_size>>>(arr, output, g_max, S_factor, N);
    cudaDeviceSynchronize();
}

void softmax(const float *arr, int N) {
    //
    size_t arr_byte_size = N * sizeof(float);

    float *d_arr, *d_output;
    float *h_output;
    h_output = (float *)malloc(arr_byte_size);

    cudaMalloc((void **)&d_output, arr_byte_size);
    cudaMalloc((void **) &d_arr, arr_byte_size);
    cudaMemcpy(d_arr, arr, arr_byte_size, cudaMemcpyHostToDevice);


    // 计算 x_max
    float g_x_max = max_reduction_wrapper(d_arr, N);
    printf("g_x_max: %f\n", g_x_max);

    // 计算 softmax分母
    float S_factor = softmax_sum_reduction_wrapper(d_arr, g_x_max, N);
    printf("S_factor: %f\n", S_factor);

    softmax_wrapper(d_arr, d_output, g_x_max, S_factor, N);
    cudaMemcpy(h_output, d_output, arr_byte_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; i++) {
        printf("output[%d] = %f\n", i, h_output[i]);
    }
    cudaFree(d_arr);
    cudaFree(d_output);
}

int main() {
    const int N = 3;
    std::vector<float> vec(N);
    vec = {1.0, 2.0, 3.0};
    for(int i = 0; i < N; i++) vec[i] = i + 1.0f;

    softmax(vec.data(), N);

}