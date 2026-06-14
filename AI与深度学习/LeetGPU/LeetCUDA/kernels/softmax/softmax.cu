/**
 * CUDA Softmax 实现
 * 
 * 本文件实现了多种优化的 Softmax CUDA kernel，包括：
 * 1. 基础 Softmax（FP32）
 * 2. 向量化 Softmax（FP32x4）
 * 3. Safe Softmax（数值稳定版本，先减最大值再计算）
 * 4. Online Softmax（在线归一化算法）
 * 5. FP16 混合精度版本
 */

#include <algorithm>
#include <cuda_bf16.h>      // CUDA bfloat16 类型支持
#include <cuda_fp16.h>       // CUDA half (FP16) 类型支持
#include <cuda_fp8.h>        // CUDA FP8 类型支持
#include <cuda_runtime.h>    // CUDA 运行时 API
#include <float.h>           // 浮点数常量定义（如 FLT_MAX）
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h> // PyTorch C++ 扩展接口
#include <torch/types.h>
#include <vector>

// CUDA WARP 大小：一个 warp 包含 32 个线程
#define WARP_SIZE 32

// 类型转换宏：用于向量化内存访问
// 将值重新解释为 int4（128位，4个int）
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
// 将值重新解释为 float4（128位，4个float）
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
// 将值重新解释为 half2（32位，2个half）
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
// 将值重新解释为 bfloat162（32位，2个bfloat16）
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
// 用于 128 位对齐的内存加载/存储（通常用于向量化访问）
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

/**
 * Online Softmax 所需的数据结构
 * 
 * MD 结构体用于在线归一化算法（Online Normalizer）：
 * - m: 最大值（max value）
 * - d: 归一化因子（denominator），即 sum(exp(x_i - m))
 * 
 * 使用 8 字节对齐以优化内存访问
 */
struct __align__(8) MD {
  float m;  // 最大值
  float d;  // 归一化因子（分母）
};

/**
 * Warp 级别的 MD 归约操作（用于 Online Softmax）
 * 
 * 该函数在 warp 内对 MD 结构进行归约，使用 butterfly 模式（XOR shuffle）
 * 实现高效的并行归约。
 * 
 * 算法原理：
 * 1. 使用 XOR shuffle 在 warp 内交换数据
 * 2. 每次迭代比较两个 MD 值，保留较大的 m，合并 d
 * 3. 合并公式：d_new = d_bigger + d_smaller * exp(m_smaller - m_bigger)
 * 
 * @param value: 输入的 MD 值
 * @return: 归约后的 MD 值（包含全局最大值和归一化因子）
 */
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  unsigned int mask = 0xffffffff;  // 全 warp 掩码
#pragma unroll  // 循环展开优化
  // Butterfly 归约：从 stride=16 开始，每次减半，直到 stride=1
  for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    // 使用 XOR shuffle 获取距离当前线程 stride 位置的线程的值
    other.m = __shfl_xor_sync(mask, value.m, stride);
    other.d = __shfl_xor_sync(mask, value.d, stride);

    // 比较两个值，确定哪个更大
    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;   // 较大的值
    MD smaller_m = value_bigger ? other : value;    // 较小的值

    // 合并归一化因子：d_new = d_big + d_small * exp(m_small - m_big)
    // 这确保了数值稳定性
    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;  // 保留最大值
  }
  return value;
}

/**
 * Warp 级别的求和归约（FP32）
 * 
 * 使用 butterfly 模式在 warp 内对 32 个 float 值进行求和归约。
 * 使用 shuffle 指令避免共享内存访问，提高性能。
 * 
 * 算法流程：
 * 1. 每个线程持有自己的值
 * 2. 通过 XOR shuffle 与距离为 mask 的线程交换值并累加
 * 3. 经过 log2(32) = 5 次迭代后，所有线程都得到 warp 内的总和
 * 
 * @param val: 当前线程的输入值
 * @return: warp 内所有值的总和（所有线程都得到相同结果）
 */
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll  // 循环展开优化
  // Butterfly 归约模式：mask 从 16 -> 8 -> 4 -> 2 -> 1
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    // 与距离为 mask 的线程交换值并累加
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

/**
 * Warp 级别的最大值归约（FP32）
 * 
 * 使用 butterfly 模式在 warp 内对 32 个 float 值进行最大值归约。
 * 
 * @param val: 当前线程的输入值
 * @return: warp 内所有值的最大值（所有线程都得到相同结果）
 */
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll  // 循环展开优化
  // Butterfly 归约模式：mask 从 16 -> 8 -> 4 -> 2 -> 1
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    // 与距离为 mask 的线程交换值并取最大值
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

/**
 * Block 级别的求和归约（FP32）
 * 
 * 在 thread block 内对所有线程的值进行求和归约。
 * 采用两级归约策略：
 * 1. 第一级：每个 warp 内部归约（使用 shuffle）
 * 2. 第二级：将各 warp 的结果写入共享内存，然后在第一个 warp 内再次归约
 * 
 * 注意：每个 block 最多 32 个 warps（受限于每个 block 最多 1024 个线程）
 * 
 * @param val: 当前线程的输入值
 * @return: block 内所有值的总和（所有线程都得到相同结果）
 */
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  // 计算该 block 包含的 warp 数量（向上取整）
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;  // 当前线程所在的 warp ID
  int lane = threadIdx.x % WARP_SIZE;   // 当前线程在 warp 内的 lane ID
  static __shared__ float shared[NUM_WARPS];  // 共享内存，存储每个 warp 的归约结果

  // 第一级：warp 内归约
  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  
  // 每个 warp 的第 0 号线程将结果写入共享内存
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();  // 同步，确保所有 warp 都完成了第一级归约

  // 第二级：在第一个 warp 内对共享内存中的值进行归约
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);
  
  // 将最终结果广播到所有线程（从 lane 0 广播到整个 warp）
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

/**
 * Block 级别的最大值归约（FP32）
 * 
 * 在 thread block 内对所有线程的值进行最大值归约。
 * 采用两级归约策略，与 block_reduce_sum_f32 类似。
 * 
 * @param val: 当前线程的输入值
 * @return: block 内所有值的最大值（所有线程都得到相同结果）
 */
template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
  // 计算该 block 包含的 warp 数量（向上取整）
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;  // 当前线程所在的 warp ID
  int lane = threadIdx.x % WARP_SIZE;   // 当前线程在 warp 内的 lane ID
  static __shared__ float shared[NUM_WARPS];  // 共享内存，存储每个 warp 的归约结果

  // 第一级：warp 内归约
  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  
  // 每个 warp 的第 0 号线程将结果写入共享内存
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();  // 同步，确保所有 warp 都完成了第一级归约

  // 第二级：在第一个 warp 内对共享内存中的值进行归约
  // 对于超出范围的 lane，使用 -FLT_MAX 作为初始值
  value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max_f32<NUM_WARPS>(value);
  
  // 将最终结果广播到所有线程（从 lane 0 广播到整个 warp）
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// Softmax x: N, y: N
// grid(N/256), block(K=256)
// template<const int NUM_THREADS = 256>
// __global__ void softmax_f32_kernel(float* x, float* y, float* total, int N) {

//   const int tid = threadIdx.x;
//   const int idx = blockIdx.x * blockDim.x + tid;

//   float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
//   float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
//   // get the total sum of all blocks.
//   if (tid == 0) atomicAdd(total, exp_sum);
//   __threadfence(); // grid level memory fence
//   // e^x_i/sum(e^x_0,...,e^x_n-1)
//   // printf("N: %d, idx: %d, bid: %d, tid: %d, exp_val: %f, exp_sum: %f,
//   total: %f\n",
//   //         N,     idx, blockIdx.x,  tid,     exp_val,     exp_sum, *total);
//   if (idx < N) y[idx] = exp_val / (*total);
// }

// // Softmax Vec4 x: N, y: N
// // grid(N/256), block(256/4)
// template<const int NUM_THREADS = 256/4>
// __global__ void softmax_f32x4_kernel(float* x, float* y, float* total, int N)
// {
//   const int tid = threadIdx.x;
//   const int idx = (blockIdx.x * blockDim.x + tid) * 4;

//   float4 reg_x = FLOAT4(x[idx]);
//   float4 reg_exp;
//   reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
//   reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
//   reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
//   reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;
//   float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
//   float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
//   // get the total sum of all blocks.
//   if (tid == 0) atomicAdd(total, exp_sum);
//   __threadfence(); // grid level memory fence
//   // e^x_i/sum(e^x_0,...,e^x_n-1)
//   if (idx + 3 < N) {
//     float4 reg_y;
//     reg_y.x = reg_exp.x / (*total);
//     reg_y.y = reg_exp.y / (*total);
//     reg_y.z = reg_exp.z / (*total);
//     reg_y.w = reg_exp.w / (*total);
//     FLOAT4(y[idx]) = reg_y;
//   }
// }

/**
 * 基础 Softmax Kernel（FP32，Per-Token）
 * 
 * 对每个 token 独立计算 softmax。
 * 
 * 输入输出格式：
 * - x: (S, H) 输入张量，S 为序列长度，H 为头大小/KV长度
 * - y: (S, H) 输出张量
 * 
 * Grid/Block 配置：
 * - grid(S): 每个 token 对应一个 block
 * - block(H): 每个 block 的线程数等于头大小 H
 * - 假设 H <= 1024，且 H 为 2 的幂次（64, 128, 256, 512, 1024）
 * 
 * 注意：此版本没有数值稳定性保护，可能在大值输入时溢出
 * 
 * @param x: 输入数据指针
 * @param y: 输出数据指针
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x;  // 线程在 block 内的索引
  const int idx = blockIdx.x * blockDim.x + tid;  // 全局索引

  // 计算 exp(x[i])，如果索引越界则设为 0
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  
  // Block 内求和归约，得到 sum(exp(x))
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  
  // Softmax 公式：y[i] = exp(x[i]) / sum(exp(x))
  if (idx < N)
    y[idx] = exp_val / exp_sum;
}

/**
 * 向量化 Softmax Kernel（FP32x4，Per-Token）
 * 
 * 使用 float4 向量化加载/存储，每个线程处理 4 个元素，提高内存带宽利用率。
 * 
 * Grid/Block 配置：
 * - grid(S): 每个 token 对应一个 block
 * - block(H/4): 每个 block 的线程数为 H/4（因为每个线程处理 4 个元素）
 * 
 * 优势：
 * - 减少内存事务数量（128 位对齐访问）
 * - 提高指令吞吐量
 * - 减少线程数量，降低调度开销
 * 
 * @param x: 输入数据指针
 * @param y: 输出数据指针
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256 / 4>
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x;  // 线程在 block 内的索引
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;  // 全局索引（每个线程处理 4 个元素）

  // 向量化加载：一次加载 4 个 float（128 位）
  float4 reg_x = FLOAT4(x[idx]);
  
  // 计算每个元素的 exp 值
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;

  // 计算这 4 个元素的 exp 值之和
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  
  // Block 内求和归约，得到全局 sum(exp(x))
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  
  // Softmax 计算：y[i] = exp(x[i]) / sum(exp(x))
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;
    reg_y.z = reg_exp.z / exp_sum;
    reg_y.w = reg_exp.w / exp_sum;
    
    // 向量化存储：一次存储 4 个 float（128 位）
    FLOAT4(y[idx]) = reg_y;
  }
}

/**
 * Safe Softmax Kernel（FP32，Per-Token）
 * 
 * 数值稳定的 Softmax 实现，使用 "减最大值" 技巧避免溢出。
 * 
 * 算法步骤：
 * 1. 找到输入的最大值 max_val
 * 2. 计算 exp(x[i] - max_val) 而不是 exp(x[i])
 * 3. 归一化：y[i] = exp(x[i] - max_val) / sum(exp(x[j] - max_val))
 * 
 * 数学原理：
 * softmax(x[i]) = exp(x[i]) / sum(exp(x[j]))
 *               = exp(x[i] - max_val) / sum(exp(x[j] - max_val))
 * 
 * 这样做的优势：
 * - 避免 exp(x[i]) 溢出（当 x[i] 很大时）
 * - 提高数值精度
 * 
 * @param x: 输入数据指针
 * @param y: 输出数据指针
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x;  // 线程在 block 内的索引
  const int idx = blockIdx.x * blockDim.x + tid;  // 全局索引

  // 第一步：找到最大值（对于越界索引使用 -FLT_MAX）
  float val = (idx < N) ? x[idx] : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val);  // Block 内最大值归约
  
  // 第二步：计算 exp(x[i] - max_val)，避免溢出
  float exp_val = (idx < N) ? expf(x[idx] - max_val) : 0.0f;
  
  // 第三步：计算归一化因子 sum(exp(x[j] - max_val))
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  
  // 第四步：归一化得到最终的 softmax 值
  if (idx < N)
    y[idx] = exp_val / exp_sum;
}

/**
 * Safe Softmax Kernel（FP32x4 向量化，Per-Token）
 * 
 * 结合数值稳定性和向量化优化的版本。
 * 使用 float4 向量化加载/存储，每个线程处理 4 个元素。
 * 
 * Grid/Block 配置：
 * - grid(S): 每个 token 对应一个 block
 * - block(H/4): 每个 block 的线程数为 H/4（因为每个线程处理 4 个元素）
 * 
 * 优势：
 * - 数值稳定：使用 Safe Softmax 技巧避免溢出
 * - 高性能：向量化访问提高内存带宽利用率
 * - 减少线程数量：降低调度开销
 * 
 * @param x: 输入数据指针
 * @param y: 输出数据指针
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256 / 4>
__global__ void safe_softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
  const int tid = threadIdx.x;  // 线程在 block 内的索引
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;  // 全局索引（每个线程处理 4 个元素）

  // 向量化加载：一次加载 4 个 float（128 位）
  float4 reg_x = FLOAT4(x[idx]);
  
  // 边界处理：对于越界元素，设置为 -FLT_MAX（不影响最大值计算）
  reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
  reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
  reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
  reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;
  
  // 第一步：找到这 4 个元素的局部最大值
  float val = reg_x.x;
  val = fmaxf(val, reg_x.y);
  val = fmaxf(val, reg_x.z);
  val = fmaxf(val, reg_x.w);
  
  // Block 内最大值归约
  float max_val = block_reduce_max_f32<NUM_THREADS>(val);

  // 第二步：计算 exp(x[i] - max_val)，避免溢出
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z - max_val) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w - max_val) : 0.0f;

  // 计算这 4 个元素的 exp 值之和
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  
  // 第三步：Block 内求和归约，得到全局归一化因子
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  
  // 第四步：计算 softmax 并向量化存储
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;
    reg_y.z = reg_exp.z / exp_sum;
    reg_y.w = reg_exp.w / exp_sum;
    
    // 向量化存储：一次存储 4 个 float（128 位）
    FLOAT4(y[idx]) = reg_y;
  }
}

/**
 * Safe Softmax Kernel（FP16 输入/输出，FP32 中间计算）
 * 
 * 混合精度实现：
 * - 输入/输出：FP16（half）
 * - 中间计算：FP32（float）
 * 
 * 优势：
 * - 减少内存带宽（FP16 占用内存是 FP32 的一半）
 * - 保持计算精度（使用 FP32 进行中间计算）
 * - 适合深度学习推理场景
 * 
 * @param x: 输入数据指针（FP16）
 * @param y: 输出数据指针（FP16）
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half *x, half *y, int N) {
  const int tid = threadIdx.x;  // 线程在 block 内的索引
  const int idx = blockIdx.x * blockDim.x + tid;  // 全局索引

  // 将 FP16 转换为 FP32 进行计算
  float val = (idx < N) ? __half2float(x[idx]) : -FLT_MAX;
  
  // 找到最大值
  float max_val = block_reduce_max_f32<NUM_THREADS>(val);
  
  // 计算 exp(x[i] - max_val)
  float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
  
  // 计算归一化因子
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  
  // 计算 softmax 并转换回 FP16
  if (idx < N)
    y[idx] = __float2half_rn(exp_val / exp_sum);  // 使用 round-to-nearest 模式
}

/**
 * Safe Softmax Kernel（FP16x2 向量化，FP32 中间计算）
 * 
 * 使用 half2 向量化，每个线程处理 2 个 FP16 元素。
 * 
 * 优势：
 * - 向量化内存访问（half2 是 32 位对齐）
 * - 减少线程数量
 * - 提高内存带宽利用率
 * 
 * @param x: 输入数据指针（FP16）
 * @param y: 输出数据指针（FP16）
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half *x, half *y,
                                                        int N) {
  const int tid = threadIdx.x;  // 线程在 block 内的索引
  const int idx = (blockIdx.x * blockDim.x + tid) * 2;  // 全局索引（每个线程处理 2 个元素）

  // 向量化加载：一次加载 2 个 half（32 位），并转换为 float2
  float2 reg_x = __half22float2(HALF2(x[idx]));
  
  // 找到这两个元素的最大值
  float max_val = -FLT_MAX;
  max_val = ((idx + 0) < N) ? fmaxf(reg_x.x, max_val) : -FLT_MAX;
  max_val = ((idx + 1) < N) ? fmaxf(reg_x.y, max_val) : -FLT_MAX;
  
  // Block 内最大值归约
  max_val = block_reduce_max_f32<NUM_THREADS>(max_val);

  // 计算 exp(x[i] - max_val)
  float2 reg_exp;
  reg_exp.x = ((idx + 0) < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = ((idx + 1) < N) ? expf(reg_x.y - max_val) : 0.0f;

  // 计算这两个元素的 exp 值之和
  float exp_val = reg_exp.x + reg_exp.y;
  
  // Block 内求和归约
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

  // 计算 softmax
  float2 reg_y;
  reg_y.x = reg_exp.x / exp_sum;
  reg_y.y = reg_exp.y / exp_sum;

  // 向量化存储：将 float2 转换回 half2 并存储
  if ((idx + 1) < N)
    HALF2(y[idx]) = __float22half2_rn(reg_y);
}

/**
 * Safe Softmax Kernel（FP16x8 打包，FP32 中间计算）
 * 
 * 使用 128 位对齐的向量化加载/存储，每个线程处理 8 个 FP16 元素。
 * 
 * 关键优化：
 * - 使用寄存器数组存储 8 个 half（128 位 = 8 * 16 位）
 * - 通过 float4 重新解释实现 128 位对齐的内存事务
 * - 减少内存事务数量（1 次加载 8 个元素，1 次存储 8 个元素）
 * 
 * 优势：
 * - 最大化内存带宽利用率
 * - 减少线程数量（线程数 = H / 8）
 * - 适合大尺寸的 head size（H >= 256）
 * 
 * @param x: 输入数据指针（FP16）
 * @param y: 输出数据指针（FP16）
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half *x, half *y,
                                                             int N) {
  const int tid = threadIdx.x;  // 线程在 block 内的索引
  const int idx = (blockIdx.x * blockDim.x + tid) * 8;  // 全局索引（每个线程处理 8 个元素）
  
  // 寄存器数组：临时存储 8 个 half（128 位，在 PTX 中对应 .local 空间，可寻址）
  half pack_x[8], pack_y[8];  // 8 * 16 bits = 128 bits
  
  // 向量化加载：将 8 个 half 重新解释为 float4，一次加载 128 位
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

  // 第一步：找到这 8 个元素的最大值
  float max_val = -FLT_MAX;
#pragma unroll  // 循环展开优化
  for (int i = 0; i < 8; ++i) {
    max_val = fmaxf(__half2float(pack_x[i]), max_val);
  }
  // Block 内最大值归约
  max_val = block_reduce_max_f32<NUM_THREADS>(max_val);

  // 第二步：计算 sum(exp(x[i] - max_val))
  float exp_sum = 0.0f;
#pragma unroll  // 循环展开优化
  for (int i = 0; i < 8; ++i) {
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    exp_sum += (((idx + i) < N) ? exp_val : 0.0f);  // 边界检查
  }
  // Block 内求和归约
  exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_sum);

  // 第三步：计算 softmax 并存储到 pack_y
#pragma unroll  // 循环展开优化
  for (int i = 0; i < 8; ++i) {
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    pack_y[i] = __float2half_rn(exp_val / exp_sum);
  }
  
  // 向量化存储：将 8 个 half 重新解释为 float4，一次存储 128 位
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
  // TODO: 支持非 8 的倍数的 K（head size）
}

/**
 * Online Safe Softmax Kernel（FP32，Per-Token）
 * 
 * 使用在线归一化算法（Online Normalizer）实现数值稳定的 Softmax。
 * 
 * 参考论文：https://arxiv.org/pdf/1805.02867
 * "Online normalizer calculation for softmax"
 * 
 * 算法优势：
 * - 单次遍历即可同时计算最大值和归一化因子
 * - 数值稳定，避免溢出
 * - 适合流式处理场景
 * 
 * MD 结构：
 * - m: 当前看到的最大值
 * - d: 归一化因子 sum(exp(x[i] - m))
 * 
 * 合并规则：
 * 当合并两个 MD 值 (m1, d1) 和 (m2, d2) 时：
 * - m_new = max(m1, m2)
 * - d_new = d_bigger + d_smaller * exp(m_smaller - m_bigger)
 * 
 * @param x: 输入数据指针
 * @param y: 输出数据指针
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256>
__global__ void online_safe_softmax_f32_per_token_kernel(const float *x,
                                                         float *y, int N) {
  int local_tid = threadIdx.x;   // 线程在 block 内的索引
  int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;  // 全局索引
  const int WARP_NUM = NUM_THREADS / WARP_SIZE;  // Block 内的 warp 数量
  int warp_id = local_tid / WARP_SIZE;   // 当前线程所在的 warp ID
  int lane_id = local_tid % WARP_SIZE;    // 当前线程在 warp 内的 lane ID
  
  // 初始化 MD 值：m 为当前元素值（或 -FLT_MAX），d 为 1.0（或 0.0）
  MD val;
  val.m = global_tid < N ? x[global_tid] : -FLT_MAX;
  val.d = global_tid < N ? 1.0f : 0.0f;

  // 共享内存：存储每个 warp 的归约结果
  __shared__ MD shared[WARP_NUM];
  
  // 第一级：warp 内 MD 归约
  MD res = warp_reduce_md_op<WARP_SIZE>(val);

  // 每个 warp 的第 0 号线程将结果写入共享内存
  if (lane_id == 0)
    shared[warp_id] = res;
  __syncthreads();

  // 第二级：在第一个 warp 内对共享内存中的 MD 值进行归约
  if (local_tid < WARP_SIZE) {
    MD block_res = shared[local_tid];
    block_res = warp_reduce_md_op<WARP_NUM>(block_res);
    if (local_tid == 0) {
      shared[0] = block_res;  // 存储最终的 MD 值
    }
  }
  __syncthreads();

  // 获取最终的 MD 值（包含全局最大值和归一化因子）
  MD final_res = shared[0];
  
  // 计算归一化因子的倒数（使用快速除法）
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  
  // 计算最终的 softmax 值：exp(x[i] - m_max) / d_total
  if (global_tid < N) {
    y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
  }
}

/**
 * Online Safe Softmax Kernel（FP32x4 向量化打包，Per-Token）
 * 
 * 结合在线归一化算法和向量化优化的版本。
 * 每个线程处理 4 个 float 元素，使用 float4 向量化加载/存储。
 * 
 * 参考论文：https://arxiv.org/pdf/1805.02867
 * 
 * 优势：
 * - 在线归一化算法：数值稳定，单次遍历
 * - 向量化访问：提高内存带宽利用率
 * - 减少线程数量：降低调度开销
 * 
 * @param x: 输入数据指针
 * @param y: 输出数据指针
 * @param N: 当前 token 的元素数量（等于 H）
 */
template <const int NUM_THREADS = 256 / 4>
__global__ void
online_safe_softmax_f32x4_pack_per_token_kernel(float *x, float *y, int N) {
  int local_tid = threadIdx.x;   // 线程在 block 内的索引
  int global_tid = (blockIdx.x * NUM_THREADS + local_tid) * 4;  // 全局索引（每个线程处理 4 个元素）

  const int WARP_NUM = NUM_THREADS / WARP_SIZE;  // Block 内的 warp 数量
  int warp_id = local_tid / WARP_SIZE;   // 当前线程所在的 warp ID
  int lane_id = local_tid % WARP_SIZE;    // 当前线程在 warp 内的 lane ID
  
  // 向量化加载：一次加载 4 个 float（128 位）
  float4 val = FLOAT4((x)[global_tid]);
  
  // 计算这 4 个元素的局部最大值
  float local_m = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
  
  // 计算这 4 个元素的局部归一化因子：sum(exp(x[i] - local_m))
  float local_d = __expf(val.x - local_m) + __expf(val.y - local_m) +
                  __expf(val.z - local_m) + __expf(val.w - local_m);

  // 构建局部 MD 值
  MD local_md = {local_m, local_d};
  
  // 第一级：warp 内 MD 归约
  MD res = warp_reduce_md_op<WARP_SIZE>(local_md);
  __shared__ MD shared[WARP_NUM];

  // 每个 warp 的第 0 号线程将结果写入共享内存
  if (lane_id == 0)
    shared[warp_id] = res;
  __syncthreads();
  
  // 第二级：在第一个 warp 内对共享内存中的 MD 值进行归约
  if (local_tid < WARP_SIZE) {
    MD block_res = shared[local_tid];
    block_res = warp_reduce_md_op<WARP_NUM>(block_res);
    if (local_tid == 0)
      shared[0] = block_res;
  }
  __syncthreads();
  
  // 获取最终的 MD 值
  MD final_res = shared[0];
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  
  // 计算最终的 softmax 值并向量化存储
  if (global_tid < N) {
    float4 reg_y;
    reg_y.x = __expf(val.x - final_res.m) * d_total_inverse;
    reg_y.y = __expf(val.y - final_res.m) * d_total_inverse;
    reg_y.z = __expf(val.z - final_res.m) * d_total_inverse;
    reg_y.w = __expf(val.w - final_res.m) * d_total_inverse;
    
    // 向量化存储：一次存储 4 个 float（128 位）
    FLOAT4((y)[global_tid]) = reg_y;
  }
}

/**
 * PyTorch 绑定辅助宏
 */

// 将宏参数转换为字符串
#define STRINGFY(str) #str

// 定义 PyTorch 扩展函数的通用宏
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// 检查张量数据类型是否匹配
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// 检查两个张量的形状是否匹配
#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }

// grid memory fence
#define TORCH_BINDING_SOFTMAX(packed_type, th_type, element_type, n_elements)  \
  void softmax_##packed_type(torch::Tensor x, torch::Tensor y) {               \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype((th_type)).device(torch::kCUDA, 0);       \
    const int N = x.size(0);                                                   \
    CHECK_TORCH_TENSOR_SHAPE(x, y)                                             \
    auto total = torch::zeros({1}, options);                                   \
    dim3 block(256);                                                           \
    dim3 grid(((N + 256 - 1) / 256) / (n_elements));                           \
    softmax_##packed_type##_kernel<256><<<grid, block>>>(                      \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()),                        \
        reinterpret_cast<element_type *>(total.data_ptr()), N);                \
  }

// softmax per token
#define LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                                 \
  softmax_f32_per_token_kernel<(H)>                                            \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                            \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                                    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                                    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                                   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                                   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                                   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                                  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                               \
  softmax_f32x4_per_token_kernel<(H) / 4>                                      \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                          \
  const int NT = (H) / 4;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                           \
  case 64:                                                                     \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                           \
  case 128:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;                          \
  case 256:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;                          \
  case 512:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;                          \
  case 1024:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;                         \
  case 2048:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;                         \
  case 4096:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;                         \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*4");             \
    break;                                                                     \
  }

// safe softmax per token
#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                            \
  safe_softmax_f32_per_token_kernel<(H)>                                       \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                       \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                               \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                               \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                              \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                              \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                              \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                             \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                          \
  online_safe_softmax_f32_per_token_kernel<(H)>                                \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                     \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                             \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                             \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                            \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                            \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                            \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                           \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(H)                   \
  online_safe_softmax_f32x4_pack_per_token_kernel<(H / 4)>                     \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)              \
  dim3 block((H / 4));                                                         \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(128)                     \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(256)                     \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(512)                     \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(1024)                    \
    break;                                                                     \
  case 2048:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(2048)                    \
    break;                                                                     \
  case 4096:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(4096)                    \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 128/256/.../4096;");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                          \
  safe_softmax_f32x4_per_token_kernel<(H) / 4>                                 \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                     \
  const int NT = (H) / 4;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                      \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                      \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;                    \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;                    \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;                    \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*4");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(H)                        \
  safe_softmax_f16_f32_per_token_kernel<(H)>                                   \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)                   \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(32)                           \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(64)                           \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(128)                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(256)                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(512)                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(1024)                         \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(H)                      \
  safe_softmax_f16x2_f32_per_token_kernel<(H) / 2>                             \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)                 \
  const int NT = (H) / 2;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(32) break;                  \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(64) break;                  \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(128) break;                 \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(256) break;                 \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(512) break;                 \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(1024) break;                \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(2048) break;                \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*2");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(H)                 \
  safe_softmax_f16x8_pack_f32_per_token_kernel<(H) / 8>                        \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)            \
  const int NT = (H) / 8;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(32) break;             \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(64) break;             \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(128) break;            \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(256) break;            \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(512) break;            \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(1024) break;           \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(2048) break;           \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(4096) break;           \
  case 8192:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(8192) break;           \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*8");             \
    break;                                                                     \
  }

/**
 * PyTorch 接口函数：基础 Softmax（FP32，Per-Token）
 * 
 * 输入输出格式：x, y 都是 (S, H) 形状的 FP32 张量
 * - S: 序列长度（sequence length）
 * - H: 头大小/KV长度（head size/key-value length）
 * 
 * 每个 token 独立计算 softmax。
 */
void softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)  // 检查输入数据类型
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)  // 检查输出数据类型
  CHECK_TORCH_TENSOR_SHAPE(x, y)                 // 检查形状匹配
  const int S = x.size(0);  // 序列长度
  const int H = x.size(1);  // 头大小/KV长度
  const int N = S * H;       // 总元素数
  DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)   // 根据 H 的大小分派到对应的 kernel
}

void softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

// per token fp16
void safe_softmax_f16_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x2_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x8_pack_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32x4_pack_per_token(torch::Tensor x,
                                              torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0);
  const int H = x.size(1);
  const int N = S * H;
  DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)
}

// grid memory fence fp32
// TORCH_BINDING_SOFTMAX(f32,   torch::kFloat32, float, 1)
// TORCH_BINDING_SOFTMAX(f32x4, torch::kFloat32, float, 4)

/**
 * PyTorch 扩展模块注册
 * 
 * 将所有 CUDA kernel 函数注册为 Python 可调用的函数。
 * 这些函数可以在 Python 中通过 lib.函数名() 的方式调用。
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 基础 Softmax（已注释，使用 per-token 版本）
  // TORCH_BINDING_COMMON_EXTENSION(softmax_f32)
  // TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4)
  
  // Per-Token Softmax 函数
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)              // 基础 FP32
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token)             // 向量化 FP32x4
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)          // Safe FP32
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token)         // Safe FP32x4
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_f32_per_token)      // Safe FP16（FP32 中间计算）
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x2_f32_per_token)    // Safe FP16x2（FP32 中间计算）
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x8_pack_f32_per_token) // Safe FP16x8 打包（FP32 中间计算）
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32_per_token)   // Online Safe FP32
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32x4_pack_per_token) // Online Safe FP32x4 打包
}
