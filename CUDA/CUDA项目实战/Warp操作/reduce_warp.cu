#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h> // 虽然这个头文件在这里用不到，但保留原样
#include <driver_types.h>
#include <iterator>
#include <vector>
#include <iostream>
#include <numeric> // 用于 C++ Host 端的 std::accumulate

const int WARP_SIZE = 32;


// 检查CUDA API调用错误的宏，提供更健壮的错误处理
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


/*
* 初始:
* mask = 16 (0b10000) 第5位
* 当lane id < 16时 第5位 是0， 和mask异或变成1，相当于 + 16 (从 0 得到 16)
* 当lane id >= 16时 第5位 是1，和mask异或变成0了，相当于 -16 (从 16 得到 0)
*
* __shfl_xor_sync(mask, val, mask_xor) 
* - mask: 参与同步的线程掩码，0xffffffff 表示整个 Warp 的 32 个线程。
* - val: 当前线程的值。
* - mask_xor: 目标线程 ID = 当前线程 ID 异或 mask_xor。
* 它会从目标线程 ID 处获取 val 的值。
*/

// Warp 级规约求和模板函数，使用 Warp Shuffle 指令
template <const int kWarpSize = WARP_SIZE>
__device__ float warp_reduce_sum_f32(float val) {
    // 确保整个 Warp (32个线程) 都参与同步和数据交换
    unsigned int mask = 0xffffffff; 
    
    // 循环 log2(kWarpSize) 次。对于 32 个线程，循环 5 次 (16, 8, 4, 2, 1)
    for(int mask_xor = kWarpSize >> 1; mask_xor >= 1; mask_xor >>= 1) {
        // val += 从目标线程取来的值
        // 第一次 (mask_xor=16): lane 0 从 lane 16 取值，lane 1 从 lane 17 取值...
        // 最终，规约结果会累积到 Warp 的第一个线程 (lane 0) 上。
        val += __shfl_xor_sync(mask, val, mask_xor);
    }
    
    // 最终结果存储在 Warp 的 lane 0 上
    return val;
}


/*
* Block 规约 Kernel 函数：将一个线程块 (Block) 内的所有元素规约求和
* d_input: 原始输入数组 (全局内存)
* d_output: 每个 Block 的规约结果 (全局内存)，大小为 grid_size
* N: 原始输入数组的总大小
*/
__global__ void reduce_kernel(const float *d_input, float *d_output, const int N) {
    // 定义动态共享内存。大小在调用 Kernel 时传入。
    extern __shared__ float s_warp_sums[];
    
    // 线程在 Block 内的索引 (0 到 blockDim.x - 1)
    int tid = threadIdx.x; 
    // 线程在整个 Grid 中的全局索引
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算当前线程在 Block 内的 Warp ID 和 Lane ID
    int warp_id = tid / WARP_SIZE;          // Block 内 Warp 的索引
    int lane_id = tid % WARP_SIZE;          // Warp 内线程的索引 (0 to 31)
    
    // --- 1. 每个线程读取并计算自己的初始和 (读取分块数据) ---
    float thread_sum = 0.0f;
    if(global_idx < N) {
        thread_sum = d_input[global_idx]; // 读取当前线程需要处理的单个值
    }

    // --- 2. Warp 规约 (Block 内规约的第一阶段) ---
    // 每个 Warp 将其 32 个线程的 thread_sum 累加，结果返回到该 Warp 的 lane 0 线程
    float warp_sum = warp_reduce_sum_f32(thread_sum);

    // --- 3. 存储 Warp 规约结果到共享内存 ---
    // 只有每个 Warp 的 lane 0 线程 (tid % WARP_SIZE == 0) 持有该 Warp 的最终和 (warp_sum)
    if (lane_id == 0) {
        // 每个 Warp 的 lane 0 线程将 warp_sum 写入共享内存
        // 存储位置的索引就是 Warp 在 Block 内的 ID (warp_id)
        s_warp_sums[warp_id] = warp_sum;
    }

    // 同步 Block 内所有线程，确保所有 Warp 的结果都已写入共享内存
    __syncthreads();

    // --- 4. Block 最终规约 (Block 内规约的第二阶段) ---
    // Block 的第一个 Warp (warp_id == 0) 负责将共享内存中的所有 Warp 结果累加起来。
    if (warp_id == 0) {
        // Block 内的 Warp 数量
        int num_warps_per_block = blockDim.x / WARP_SIZE;
        
        // Block 0 ~ num_warps_per_block-1 的线程读取共享内存中的 warp_sum
        float block_sum = 0.0f;
        if (lane_id < num_warps_per_block) {
            // lane_id 作为索引，读取共享内存中的 warp_sum
            block_sum = s_warp_sums[lane_id];
        }
        
        // 对 Block 内所有 Warp 的和 (s_warp_sums) 进行第二次 Warp 规约
        // 注意：即使 num_warps_per_block < 32，我们仍然使用 WARP_SIZE (32) 作为模板参数
        // 因为 __shfl_xor_sync 是基于 32 线程的 warp 设计的
        // 只有前 num_warps_per_block 个线程有有效数据，其他线程的 block_sum 为 0
        float total_block_sum = warp_reduce_sum_f32<WARP_SIZE>(block_sum);
        
        // Block 的第一个线程 (tid == 0 且 warp_id == 0 且 lane_id == 0) 持有最终的 Block 总和
        if (tid == 0) {
            // 将 Block 结果写入全局内存 d_output。每个 Block 写入一个结果。
            d_output[blockIdx.x] = total_block_sum;
        }
    }
}


int main() {
    const int N = 100;

    // --- Host 端数据初始化 ---
    std::vector<float> vec(N);
    for(int i = 0; i < N; i++) vec[i] = i + 1.0f; // 1.0 + 2.0 + ... + 100.0

    // 计算预期结果进行验证
    float expected_sum = std::accumulate(vec.begin(), vec.end(), 0.0f);
    std::cout << "Expected Sum: " << expected_sum << std::endl;

    // --- Device 端内存分配和数据传输 ---
    float *d_arr; // 设备输入数组
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_arr, vec.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const int BLOCK_SIZE = 256; // 线程块大小 (必须是 WARP_SIZE 的整数倍，如 32, 64, 128, 256...)
    
    // --- Kernel 启动参数计算 ---
    int num_warps_per_block = BLOCK_SIZE / WARP_SIZE; // 每个 Block 有多少 Warp
    // 计算 Grid 大小 (需要的 Block 数量)
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 存储每个 Block 规约结果的数组 (全局内存)，大小等于 grid_size
    float *d_block_sums;
    size_t block_sums_size = grid_size * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&d_block_sums, block_sums_size));
    
    // 共享内存大小：需要存储每个 Warp 的结果 (num_warps_per_block 个 float)
    size_t shmem_size = num_warps_per_block * sizeof(float);

    // --- 第一次 Kernel 调用：Block 规约 ---
    // 将 N 个元素的数组规约成 grid_size 个元素的 Block 规约结果 (d_block_sums)
    // 启动参数: (Block 数量, 线程数量/Block, 共享内存大小/Block)
    reduce_kernel<<<grid_size, BLOCK_SIZE, shmem_size>>>(d_arr, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError()); // 检查 Kernel 启动时的异步错误
    
    // --- 第二次 Kernel 调用：Grid 规约 (可选，如果 grid_size > 1) ---
    float *d_final_sum; // 最终结果指针
    CUDA_CHECK(cudaMalloc((void **)&d_final_sum, sizeof(float)));
    
    // 如果 Block 数量大于 1，则需要进行第二次规约
    if (grid_size > 1) {
        // 使用一个 Block 来规约 d_block_sums 中的 grid_size 个结果
        
        // 为了确保第二次规约也能使用 Warp 规约的模式，我们设置：
        int final_block_size = BLOCK_SIZE;
        int final_grid_size = (grid_size + final_block_size - 1) / final_block_size;
        
        // 计算第二次规约所需的共享内存大小
        int final_num_warps_per_block = final_block_size / WARP_SIZE;
        size_t final_shmem_size = final_num_warps_per_block * sizeof(float);
        
        // 再次调用 reduce_kernel，输入是 d_block_sums，输出是 d_final_sum
        // N 变成了 grid_size (要规约的 Block 和的数量)
        reduce_kernel<<<final_grid_size, final_block_size, final_shmem_size>>>(d_block_sums, d_final_sum, grid_size);
        CUDA_CHECK(cudaGetLastError());
        
        // 如果 final_grid_size > 1，理论上需要继续循环规约直到 final_grid_size == 1。
        // 但对于 N=100, BLOCK_SIZE=256 的小规模问题，grid_size=1，所以这个 if 分支通常不执行。
        // 假设 N=100000, BLOCK_SIZE=256，grid_size约400，final_grid_size=2，需要第三次规约...
        // 这里为简化，只写了两步。
    } else {
        // 如果只有一个 Block (grid_size == 1)，d_block_sums[0] 即为最终结果
        // 需要将 d_block_sums[0] 的值拷贝到 d_final_sum
        CUDA_CHECK(cudaMemcpy(d_final_sum, d_block_sums, sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // --- 结果拷贝回 Host ---
    float final_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&final_sum, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Computed Sum: " << final_sum << std::endl;
    
    // --- 验证结果 ---
    if (std::abs(final_sum - expected_sum) < 1e-5) {
        std::cout << "Result Verified: Success!" << std::endl;
    } else {
        std::cout << "Result Verified: Failure!" << std::endl;
    }

    // --- 清理 Device 端内存 ---
    CUDA_CHECK(cudaFree(d_arr));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_final_sum)); // d_final_sum 总是独立分配的，需要释放

    return 0;
}