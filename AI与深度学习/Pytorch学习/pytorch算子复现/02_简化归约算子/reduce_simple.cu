#include <cuda_runtime.h>
#include <stdio.h>

/*
简化版 CUDA 归约算子（Sum）

去除所有 PyTorch 的封装和抽象，直接展示核心实现：
1. 线程级归约：每个线程归约多个元素
2. Block 内归约：使用共享内存
3. 全局归约：合并多个 Block 的结果

这就是 PyTorch reduce 算子的本质！
*/

// ============ 第一步：线程级归约 ============
// 每个线程加载多个元素并归约它们
__device__ float thread_reduce(const float* input, int N, int thread_id, int total_threads) {
    float sum = 0.0f;
    
    // Grid-Stride Loop：每个线程处理多个元素
    for (int i = thread_id; i < N; i += total_threads) {
        sum += input[i];
    }
    
    return sum;
}

// ============ 第二步：Block 内归约（共享内存） ============
// 使用树形归约在 block 内合并所有线程的结果
__global__ void reduce_kernel_simple(const float* input, float* output, int N) {
    extern __shared__ float sdata[];  // 共享内存
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. 每个线程先做局部归约（加载多个元素）
    float sum = 0.0f;
    int total_threads = gridDim.x * blockDim.x;
    
    for (int i = idx; i < N; i += total_threads) {
        sum += input[i];
    }
    
    // 2. 将局部结果存入共享内存
    sdata[tid] = sum;
    __syncthreads();
    
    // 3. 在 block 内做树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 4. 第一个线程写入 block 的结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============ 第三步：全局归约（如果需要） ============
// 如果数据量很大，需要多个 block，需要再次归约
__global__ void reduce_final_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    int total_threads = gridDim.x * blockDim.x;
    
    for (int i = idx; i < N; i += total_threads) {
        sum += input[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 第一个线程写入最终结果
    if (tid == 0) {
        atomicAdd(output, sdata[0]);  // 原子操作合并多个 block
    }
}

// ============ 主机端封装函数 ============
// 简化版：直接归约，假设结果可以放在一个 block 里
void reduce_simple_host(float* d_input, float* d_output, int N) {
    const int threads_per_block = 256;
    
    // 第一轮：每个 block 归约一部分数据
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    num_blocks = num_blocks > 1024 ? 1024 : num_blocks;  // 限制最大 blocks
    
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    // 如果只需要一个 block，直接输出到 d_output[0]
    if (num_blocks == 1) {
        reduce_kernel_simple<<<1, threads_per_block, shared_mem_size>>>(
            d_input, d_output, N);
    } else {
        // 需要多个 block：先归约到临时数组，再归约一次
        float* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(float));
        
        // 第一轮：归约到临时数组
        reduce_kernel_simple<<<num_blocks, threads_per_block, shared_mem_size>>>(
            d_input, d_temp, N);
        
        // 第二轮：归约临时数组到最终结果
        cudaMemset(d_output, 0, sizeof(float));
        reduce_final_kernel<<<1, threads_per_block, shared_mem_size>>>(
            d_temp, d_output, num_blocks);
        
        cudaFree(d_temp);
    }
    
    cudaDeviceSynchronize();
}

// ============ 测试代码 ============
int main() {
    const int N = 10000;
    const size_t size = N * sizeof(float);
    
    // 分配主机内存
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(sizeof(float));
    
    // 初始化输入数据
    float expected_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i + 1);
        expected_sum += h_input[i];
    }
    
    printf("输入数据大小: %d\n", N);
    printf("期望结果: %.2f\n", expected_sum);
    
    // 分配设备内存
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // 执行归约
    reduce_simple_host(d_input, d_output, N);
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // 复制结果回主机
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("计算结果: %.2f\n", h_output[0]);
    printf("误差: %.6f\n", fabs(h_output[0] - expected_sum));
    
    if (fabs(h_output[0] - expected_sum) < 1e-5) {
        printf("✓ 测试通过！\n");
    } else {
        printf("✗ 测试失败！\n");
    }
    
    // 清理
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}

