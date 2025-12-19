#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <float.h>

// --- CUDA Kernel 实现 ---
__global__ void alibi_kernel(const float* Q, const float* K, const float* V, float* output, 
                             int M, int N, int d, float alpha) {
    // 每个 Block 处理 Output 的一行 (m)
    int m = blockIdx.x; 
    if (m >= M) return;

    // Shared Memory 存储这一行的 Attention Logits (S = QK^T + Bias)
    extern __shared__ float S[]; 

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 1. 计算 Q[m] * K^T + ALiBi Bias
    // 每个线程负责计算该行的若干个列 (j)
    for (int j = tid; j < N; j += num_threads) {
        float sum = 0.0f;
        for (int k = 0; k < d; ++k) {
            sum += Q[m * d + k] * K[j * d + k];
        }
        S[j] = sum + alpha * (float)(m - j);
    }
    __syncthreads();

    // 2. Row-wise Softmax
    // 2a. 找最大值 (Safe Softmax)
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += num_threads) {
        if (S[j] > local_max) local_max = S[j];
    }
    
    // 使用原子操作在 Shared Memory 中归约最大值 (简单实现)
    __shared__ float block_max;
    if (tid == 0) block_max = -FLT_MAX;
    __syncthreads();
    atomicMax((int*)&block_max, __float_as_int(local_max)); 
    __syncthreads();
    float final_max = block_max;

    // 2b. 计算 Exp Sum (分母)
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += num_threads) {
        S[j] = expf(S[j] - final_max);
        local_sum += S[j];
    }
    
    __shared__ float block_denom;
    if (tid == 0) block_denom = 0.0f;
    __syncthreads();
    atomicAdd(&block_denom, local_sum);
    __syncthreads();
    float denom = block_denom;

    // 3. 计算 Softmax(S) * V
    // 每个线程计算输出行 Output[m] 的若干列 (k)
    for (int k = tid; k < d; k += num_threads) {
        float res = 0.0f;
        for (int j = 0; j < N; ++j) {
            res += (S[j] / denom) * V[j * d + k];
        }
        output[m * d + k] = res;
    }
}

// --- 封装好的 solve 函数 ---
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, 
                      int M, int N, int d, float alpha) {
    int threads = 256; 
    int blocks = M;
    size_t shared_mem_size = N * sizeof(float);

    alibi_kernel<<<blocks, threads, shared_mem_size>>>(Q, K, V, output, M, N, d, alpha);
    cudaDeviceSynchronize();
}

// --- Main 测试程序 ---
int main() {
    // 设定维度 (符合约束条件)
    const int M = 32;
    const int N = 32;
    const int d = 64;
    const float alpha = 0.5f;

    size_t size_q = M * d * sizeof(float);
    size_t size_k = N * d * sizeof(float);
    size_t size_v = N * d * sizeof(float);
    size_t size_out = M * d * sizeof(float);

    // 1. 分配 Host 内存并初始化
    std::vector<float> h_Q(M * d), h_K(N * d), h_V(N * d), h_O(M * d);
    for(int i=0; i < M*d; ++i) h_Q[i] = (float)rand() / RAND_MAX;
    for(int i=0; i < N*d; ++i) h_K[i] = (float)rand() / RAND_MAX;
    for(int i=0; i < N*d; ++i) h_V[i] = (float)rand() / RAND_MAX;

    // 2. 分配 Device 内存
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, size_q);
    cudaMalloc(&d_K, size_k);
    cudaMalloc(&d_V, size_v);
    cudaMalloc(&d_O, size_out);

    // 3. 拷贝数据到 Device
    cudaMemcpy(d_Q, h_Q.data(), size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), size_k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), size_v, cudaMemcpyHostToDevice);

    // 4. 执行计算
    std::cout << "Running ALiBi kernel..." << std::endl;
    solve(d_Q, d_K, d_V, d_O, M, N, d, alpha);

    // 5. 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 6. 拷贝结果回 Host
    cudaMemcpy(h_O.data(), d_O, size_out, cudaMemcpyDeviceToHost);

    // 7. 打印部分结果验证
    std::cout << "Output[0][0..4]: ";
    for(int i=0; i<5; ++i) std::cout << h_O[i] << " ";
    std::cout << "\nSuccess!" << std::endl;

    // 8. 释放资源
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    return 0;
}