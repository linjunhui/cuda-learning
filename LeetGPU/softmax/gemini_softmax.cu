#include <cuda_runtime.h>
#include <float.h>

// 1. 定义 Online Softmax 所需的元数据结构
struct MD {
    float max_val;
    float sum;
};

// 2. 核心数学转换函数：数值稳定的在线更新
__device__ __forceinline__ MD update_md(MD a, MD b) {
    if (a.max_val > b.max_val) {
        return {a.max_val, a.sum + b.sum * __expf(b.max_val - a.max_val)};
    } else {
        return {b.max_val, b.sum + a.sum * __expf(a.max_val - b.max_val)};
    }
}

// 3. Warp 级蝴蝶规约实现
__device__ __forceinline__ MD warp_reduce(MD local) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o_max = __shfl_xor_sync(0xffffffff, local.max_val, offset);
        float o_sum = __shfl_xor_sync(0xffffffff, local.sum, offset);
        local = update_md(local, {o_max, o_sum});
    }
    return local;
}

// --- 阶段 A：分块局部规约 (Map Phase) ---
__global__ void softmax_partial_reduce(const float* __restrict__ input, MD* block_results, int N) {
    int tid = threadIdx.x;
    MD thread_md = {-FLT_MAX, 0.0f};

    // 使用 float4 向量化读取，压榨显存带宽
    const float4* input_v4 = reinterpret_cast<const float4*>(input);
    int vec_N = N / 4;

    // Grid-Stride Loop: 即使 N 极大，也能保证负载均衡
    for (int i = blockIdx.x * blockDim.x + tid; i < vec_N; i += blockDim.x * gridDim.x) {
        float4 v = input_v4[i];
        thread_md = update_md(thread_md, {v.x, 1.0f});
        thread_md = update_md(thread_md, {v.y, 1.0f});
        thread_md = update_md(thread_md, {v.z, 1.0f});
        thread_md = update_md(thread_md, {v.w, 1.0f});
    }

    // 处理 N % 4 的残余部分
    if (blockIdx.x == 0) { // 仅由第一个 Block 处理余数，减少判断开销
        for (int i = vec_N * 4 + tid; i < N; i += blockDim.x) {
            thread_md = update_md(thread_md, {input[i], 1.0f});
        }
    }

    // Block 内部规约
    static __shared__ float s_max[32];
    static __shared__ float s_sum[32];
    int lane = tid % 32;
    int wid = tid / 32;

    thread_md = warp_reduce(thread_md);
    if (lane == 0) { s_max[wid] = thread_md.max_val; s_sum[wid] = thread_md.sum; }
    __syncthreads();

    if (wid == 0) {
        MD final_md = { (tid < (blockDim.x / 32)) ? s_max[lane] : -FLT_MAX, 
                        (tid < (blockDim.x / 32)) ? s_sum[lane] : 0.0f };
        final_md = warp_reduce(final_md);
        if (lane == 0) {
            block_results[blockIdx.x] = final_md;
        }
    }
}

// --- 阶段 B：全局汇总并同步写回 (Reduce & Scale Phase) ---
__global__ void softmax_final_scale(const float* __restrict__ input, float* output, const MD* block_results, int num_blocks, int N) {
    int tid = threadIdx.x;
    
    // 1. 单 Block 汇总所有局部 MD (数据量小，L2 缓存命中率极高)
    __shared__ float g_max, g_sum;
    MD local_md = {-FLT_MAX, 0.0f};
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_md = update_md(local_md, block_results[i]);
    }
    local_md = warp_reduce(local_md); // 这里假设 block_size >= num_blocks/32

    if (tid == 0) { g_max = local_md.max_val; g_sum = local_md.sum; }
    __syncthreads();

    // 2. 再次使用向量化写回结果
    const float4* input_v4 = reinterpret_cast<const float4*>(input);
    float4* output_v4 = reinterpret_cast<float4*>(output);
    int vec_N = N / 4;
    float inv_sum = 1.0f / g_sum;

    for (int i = blockIdx.x * blockDim.x + tid; i < vec_N; i += blockDim.x * gridDim.x) {
        float4 v = input_v4[i];
        float4 out;
        out.x = __expf(v.x - g_max) * inv_sum;
        out.y = __expf(v.y - g_max) * inv_sum;
        out.z = __expf(v.z - g_max) * inv_sum;
        out.w = __expf(v.w - g_max) * inv_sum;
        output_v4[i] = out;
    }

    // 处理余数
    for (int i = vec_N * 4 + (blockIdx.x * blockDim.x + tid); i < N; i += blockDim.x * gridDim.x) {
        output[i] = __expf(input[i] - g_max) * inv_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm_count * 2; // 刚好填满 SM 波次，实现最佳负载

    MD* d_block_results;
    cudaMalloc(&d_block_results, blocks * sizeof(MD));

    softmax_partial_reduce<<<blocks, threads>>>(input, d_block_results, N);
    softmax_final_scale<<<blocks, threads>>>(input, output, d_block_results, blocks, N);

    cudaFree(d_block_results);
}