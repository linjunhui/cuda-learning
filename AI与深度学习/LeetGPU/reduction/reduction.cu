#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

int main() {
    // 测试数据：N=1024，值都是 1.0f
    const int N = 1024;
    const int block_size = 16;
    
    // 计算 grid_size
    int grid_size = (N + block_size - 1) / block_size;
    
    // 计算共享内存大小
    int shared_size = sizeof(float) * block_size;
    
    // 主机端内存分配
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(sizeof(float));
    float* h_block_sums = (float*)malloc(grid_size * sizeof(float));
    
    // 初始化输入数据：全部为 1.0f
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    
    // 设备端内存分配
    float* d_input;
    float* d_output;
    float* d_block_sums;
    
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    cudaMalloc((void**)&d_block_sums, grid_size * sizeof(float));
    
    // 初始化输出为 0
    cudaMemset(d_output, 0, sizeof(float));
    
    // 将输入数据拷贝到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // ===========================================
    // 启动核函数
    // ===========================================
    reduce_sum<<<grid_size, block_size, shared_size>>>(d_input, d_block_sums, N);
    
    // 等待核函数执行完成
    cudaDeviceSynchronize();
    
    // 将每个 block 的结果拷贝回主机
    cudaMemcpy(h_block_sums, d_block_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 在 CPU 上汇总所有 block 的结果
    float h_sum = 0.0f;
    for (int i = 0; i < grid_size; i++) {
        h_sum += h_block_sums[i];
    }
    
    // 将最终结果拷贝回设备（如果需要）
    cudaMemcpy(d_output, &h_sum, sizeof(float), cudaMemcpyHostToDevice);
    
    // 验证结果
    float expected_sum = N * 1.0f;  // 1024.0f
    printf("计算结果: %.2f\n", h_sum);
    printf("期望结果: %.2f\n", expected_sum);
    
    if (h_sum == expected_sum) {
        printf("✓ 结果正确！\n");
    } else {
        printf("✗ 结果错误！差异: %.2f\n", h_sum - expected_sum);
    }
    
    // 清理内存
    free(h_input);
    free(h_output);
    free(h_block_sums);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);
    
    return 0;
}