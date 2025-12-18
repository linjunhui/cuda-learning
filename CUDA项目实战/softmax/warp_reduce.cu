#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

// 定义 Warp Size (通常为 32)
const int WARP_SIZE = 32;

// 您的 Warp 归约函数
template <const int kWarpSize = WARP_SIZE>
__device__ float warp_reduce_sum_f32(float val) {
    // 蝶形归约：利用 __shfl_xor_sync 进行无共享内存的线程间通信
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        // val = val + 从目标线程获取的值
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// 核心核函数：演示如何使用 Warp 归约
__global__ void reduce_kernel(const float* input, float* output, int n) {
    // 1. 计算全局索引和线程块内的线程索引
    // threadIdx.x 是当前线程在块内的索引 (0 到 BlockSize - 1)
    unsigned int tid = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + tid;
    
    // 2. 每个线程读取其对应的数据
    float thread_sum = 0.0f;
    if (global_idx < n) {
        thread_sum = input[global_idx];
    }

    // --- 在这里使用 Warp 归约 ---
    
    // 3. 执行 Warp 归约
    // 将线程的私有求和值在 Warp 内进行归约
    thread_sum = warp_reduce_sum_f32(thread_sum);

    // 4. 将归约结果写回输出数组
    // 归约的总和只存在于每个 Warp 的第一个线程（即 Warp 内索引为 0 的线程）中。
    // threadIdx.x & (WARP_SIZE - 1) 等价于 threadIdx.x % WARP_SIZE
    if ((tid & (WARP_SIZE - 1)) == 0) {
        // 只有 Warp 内的第一个线程（Warp Leader）执行写入操作
        // blockIdx.x 是当前线程块的索引
        // tid / WARP_SIZE 是当前线程在块内属于第几个 Warp 的索引
        unsigned int warp_leader_idx = blockIdx.x * (blockDim.x / WARP_SIZE) + (tid / WARP_SIZE);
        
        // 将 Warp 的结果（局部总和）写入到输出数组
        output[warp_leader_idx] = thread_sum;
    }
}

/*
使用Warp规约，计算大型浮点数数组的总和
*/

// 主函数
int main() {
    const int N = 1024; // 数组大小
    const int BLOCK_SIZE = 256; // 线程块大小 (必须是 WARP_SIZE 的倍数)
    
    // --- 1. CPU 数据准备 ---
    std::vector<float> h_input(N);
    // 初始化为 1.0f，因此期望的总和是 N
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f; 
    }
    float expected_sum = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
    
    // 计算需要的线程块数量
    int num_warps_per_block = BLOCK_SIZE / WARP_SIZE; // 每个block中的warp数量
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int output_size = num_blocks * num_warps_per_block; // 每个 Warp 产生一个结果

    // --- 2. GPU 内存分配与数据传输 ---
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // --- 3. 核函数启动 ---
    reduce_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    
    // --- 4. 结果回传 ---
    std::vector<float> h_output(output_size);
    cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // --- 5. 最终求和与验证 ---
    // 由于每个 Warp Leader 都将结果写入了 h_output，现在我们只需要在 CPU 上将这些 Warp 结果加起来
    float final_sum = 0.0f;
    for (float val : h_output) {
        final_sum += val;
    }

    std::cout << "期望的总和 (CPU): " << expected_sum << std::endl;
    std::cout << "实际计算的总和 (GPU): " << final_sum << std::endl;

    // 简单的误差检查
    if (std::abs(final_sum - expected_sum) < 1e-5) {
        std::cout << "✅ 结果验证成功！" << std::endl;
    } else {
        std::cout << "❌ 结果验证失败！" << std::endl;
    }
    
    // --- 6. 清理 ---
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}