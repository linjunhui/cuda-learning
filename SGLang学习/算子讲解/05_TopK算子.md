# TopK 算子详解

## 📖 算子概述

**TopK** 是 LLM 推理中的关键算子，用于从概率分布中选择概率最大的 K 个 token。

**用途**：
- **采样**：从词汇表的概率分布中选择 token
- **Beam Search**：选择 top K 路径
- **KV Cache 管理**：选择 top K 的 KV cache

**特点**：
- **计算密集**：需要排序或部分排序
- **内存访问复杂**：不规则访问模式
- **性能关键**：影响生成速度

---

## 🔢 公式与算法

### 数学公式

**TopK 问题**：
给定数组 `A = [a_0, a_1, ..., a_{n-1}]`，找到最大的 K 个元素及其索引。

```
TopK(A, K) = {(value_i, index_i) | value_i 在最大的 K 个值中}
```

**排序版本**（完整排序）：
```
sort(A) = [a_{i_0}, a_{i_1}, ..., a_{i_{n-1}}]  (降序)
TopK(A, K) = [(a_{i_0}, i_0), (a_{i_1}, i_1), ..., (a_{i_{K-1}}, i_{K-1})]
```

**部分排序版本**（只找 TopK，不完整排序）：
- 只需要前 K 个最大元素
- 不需要完整排序
- 更高效

---

## 🧠 算法原理

### 标准算法对比

| 算法 | 时间复杂度 | 空间复杂度 | 特点 |
|------|-----------|-----------|------|
| **完整排序** | O(n log n) | O(n) | 简单但慢 |
| **堆排序** | O(n log K) | O(K) | 中等效率 |
| **快速选择** | O(n) 平均 | O(1) | 平均快但不稳定 |
| **基数排序** | O(n) | O(n) | 稳定，适合 GPU |

### SGLang 使用的算法：基数排序（Radix Sort）

**核心思想**：
1. **粗筛**：使用高位快速筛选
2. **细筛**：对候选元素精细排序
3. **二分查找**：快速定位 TopK 的阈值

**步骤**：

#### 阶段 1：8位粗筛（Coarse Histogram）

```
1. 将 float 转为 uint8（保留高位）
2. 构建直方图：histogram[bin] = count
3. 计算累积和：cumsum[bin] = sum(histogram[0..bin])
4. 找到阈值 bin：cumsum[threshold_bin+1] <= K < cumsum[threshold_bin]
```

**为什么用 uint8？**
- **快速**：只需要检查 256 个 bin
- **粗筛**：快速排除大部分元素
- **位操作**：利用浮点数的位表示

#### 阶段 2：16位精细排序

```
对于 threshold_bin 中的元素：
1. 提取 16 位（更精细）
2. 构建 16 位直方图
3. 递归处理直到找到 TopK
```

#### 阶段 3：完整排序（如果需要）

```
对于最终候选元素：
1. 使用完整 32 位排序
2. 选择前 K 个
```

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/elementwise/topk.cu`

### 核心代码分析

#### 1. Float 转 Uint8（粗筛）

```64:69:SGLang学习/sglang/sgl-kernel/csrc/elementwise/topk.cu
__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}
```

**关键技巧**：
1. **转 half**：先转成 `half`（16 位）
2. **位操作**：提取关键位
3. **符号处理**：负数取反，确保排序正确
4. **提取高位**：`>> 8` 提取高 8 位

**为什么这样？**
- 浮点数的二进制表示是有序的（对于同符号数）
- 使用位操作比比较快得多
- 只需要检查 256 个 bin，非常快

#### 2. 粗筛直方图

```95:103:SGLang学习/sglang/sgl-kernel/csrc/elementwise/topk.cu
  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();
```

**关键点**：
- **协作构建**：所有线程协作构建直方图
- **原子操作**：`atomicAdd` 确保线程安全
- **共享内存**：直方图在共享内存中，快速访问

#### 3. 累积和（Prefix Sum）

```105:120:SGLang学习/sglang/sgl-kernel/csrc/elementwise/topk.cu
  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };
```

**算法**：**并行前缀和（Parallel Prefix Sum）**

**步骤**（对于 256 个元素）：
```
迭代 0: stride = 1  → 每个元素 += 前1个元素
迭代 1: stride = 2  → 每个元素 += 前2个元素
迭代 2: stride = 4  → 每个元素 += 前4个元素
...
迭代 7: stride = 128 → 每个元素 += 前128个元素
```

**结果**：
```
cumsum[0] = histogram[0]
cumsum[1] = histogram[0] + histogram[1]
cumsum[2] = histogram[0] + histogram[1] + histogram[2]
...
```

**可视化**（8 元素示例）：
```
输入: [3, 1, 4, 1, 5, 9, 2, 6]
      ↓
stride=1: [3, 4, 5, 5, 6, 10, 11, 8]
      ↓
stride=2: [3, 4, 8, 9, 6, 10, 16, 17]
      ↓
stride=4: [3, 4, 8, 9, 14, 14, 20, 26]
      ↓
stride=8: [3, 4, 8, 9, 14, 14, 20, 31]  ← 累积和
```

#### 4. 阈值查找

```122:128:SGLang学习/sglang/sgl-kernel/csrc/elementwise/topk.cu
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();
```

**逻辑**：
- 找到第一个 bin，使得 `cumsum[bin+1] <= K < cumsum[bin]`
- 这意味着 TopK 包含在 `bin` 和 `bin+1` 中
- `threshold_bin` 是需要精细处理的 bin

**示例**：
- `K = 1000`
- `cumsum[50] = 800`, `cumsum[51] = 1200`
- → `threshold_bin = 50`
- → TopK 包含 bin 50 的所有元素 + bin 51 的一部分

#### 5. 精细排序（16 位）

```170:237:SGLang学习/sglang/sgl-kernel/csrc/elementwise/topk.cu
  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              /// NOTE: (dark) fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
```

**关键步骤**：

**Round 0-3**（每次检查 8 位）：
- Round 0：检查 bits [31:24]（最高 8 位）
- Round 1：检查 bits [23:16]
- Round 2：检查 bits [15:8]
- Round 3：检查 bits [7:0]（最低 8 位）

**逻辑**：
1. 对候选元素构建更精细的直方图
2. 找到新的阈值 bin
3. 如果 `topk == 0`，已经找到所有 TopK，退出
4. 否则，继续下一轮

**优化**：
- **融合计算**：在检查的同时计算下一轮的直方图
- **共享内存复用**：使用两个缓冲区交替使用

---

## 📐 算法流程图

```
输入数组 [N 个元素]，K = 1000
    ↓
[阶段 1: 8位粗筛]
构建直方图 → 累积和 → 找到阈值 bin
    ↓
假设：bin 50 有 200 个元素，bin 51 有 1500 个元素
cumsum[50] = 800, cumsum[51] = 2500
阈值 bin = 50
    ↓
[阶段 2: 16位精细排序]
只处理 bin 50 和 bin 51 的元素
    ↓
Round 0: 检查 bits [31:24]
构建直方图 → 累积和 → 找到新阈值
    ↓
Round 1: 检查 bits [23:16]
...（递归）
    ↓
Round 3: 检查 bits [7:0]
找到最终的 TopK
    ↓
输出 TopK 个索引
```

---

## 🎯 关键优化技巧

### 1. 位操作优化

**Float 转 Uint**：
```cpp
uint32_t bits = __float_as_uint(x);
uint32_t key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
```

**为什么这样？**
- 负数取反，确保排序正确（负数 < 正数）
- 使用位操作比比较快得多
- 不需要实际排序，只需要分类到 bin

### 2. 共享内存优化

**直方图缓冲区**：
```cpp
__shared__ int s_histogram_buf[2][RADIX + 128];
```

**双缓冲区**：
- 缓冲区 0：当前轮次
- 缓冲区 1：下一轮次
- 交替使用，避免等待

### 3. 原子操作

**原子累加**：
```cpp
::atomicAdd(&s_histogram[bin], 1);
::atomicAdd(&s_counter, 1);
```

**性能考虑**：
- 原子操作有开销
- 但如果冲突少（元素分散到不同 bin），性能可接受
- 使用共享内存的原子操作比全局内存快

### 4. 提前退出

**优化**：
```cpp
if (topk == 0) {
    // 已经找到所有 TopK，提前退出
    break;
}
```

**效果**：
- 不需要处理所有元素
- 一旦找到 TopK，立即退出
- 大幅减少计算量

---

## 📊 复杂度分析

### 时间复杂度

**阶段 1（8位粗筛）**：
- 构建直方图：O(N) / threads
- 累积和：O(log RADIX) = O(8)
- 总复杂度：O(N) / threads

**阶段 2（精细排序）**：
- 最多 4 轮（每轮检查 8 位）
- 每轮：O(candidates) / threads
- 候选数量逐渐减少
- 总复杂度：O(N) / threads（平均）

**总体复杂度**：O(N)（并行后）

### 空间复杂度

```
共享内存：
  - 直方图：O(RADIX) = O(256)
  - 候选索引：O(SMEM_INPUT_SIZE)
  
总共享内存：约 128 KB
```

---

## 💡 简化版本（理解核心逻辑）

如果你想理解核心逻辑，这里是简化版本：

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

const int RADIX = 256;
const int TOPK = 1000;

// 简化的 TopK（只展示核心逻辑）
__global__ void topk_simple_kernel(
    const float* input,
    int* indices,
    int N,
    int K) {
    
    extern __shared__ int smem[];
    int* histogram = smem;
    int* cumsum = smem + RADIX;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 阶段 1: 构建直方图
    if (tid < RADIX) {
        histogram[tid] = 0;
    }
    __syncthreads();
    
    // 协作构建直方图
    for (int i = bid * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) {
        uint8_t bin = convert_to_uint8(input[i]);
        atomicAdd(&histogram[bin], 1);
    }
    __syncthreads();
    
    // 阶段 2: 计算累积和
    for (int stride = 1; stride < RADIX; stride *= 2) {
        if (tid < RADIX) {
            if (tid >= stride) {
                cumsum[tid] = histogram[tid] + histogram[tid - stride];
            } else {
                cumsum[tid] = histogram[tid];
            }
        }
        __syncthreads();
        // 交换缓冲区
        int* temp = histogram;
        histogram = cumsum;
        cumsum = temp;
    }
    
    // 阶段 3: 找到阈值 bin
    int threshold_bin = 0;
    if (tid == 0) {
        for (int i = RADIX - 1; i >= 0; i--) {
            if (cumsum[i] <= K) {
                threshold_bin = i;
                break;
            }
        }
    }
    __syncthreads();
    
    // 阶段 4: 收集 TopK
    int counter = 0;
    if (tid == 0) {
        for (int i = N - 1; i >= 0 && counter < K; i--) {
            uint8_t bin = convert_to_uint8(input[i]);
            if (bin >= threshold_bin) {
                indices[counter++] = i;
            }
        }
    }
}
```

---

## 📝 总结

### 核心概念

1. **基数排序**：按位分类，从高位到低位
2. **粗筛 + 精细排序**：先快速筛选，再精细处理
3. **并行前缀和**：O(log n) 的累积和算法
4. **原子操作**：线程安全的累加

### 关键技巧

- ✅ **位操作**：利用浮点数的位表示
- ✅ **分层筛选**：8位 → 16位 → 完整
- ✅ **共享内存**：快速访问直方图
- ✅ **提前退出**：找到 TopK 后立即退出

### 学习价值

TopK 展示了：
- 复杂算法在 GPU 上的实现
- 基数排序的实际应用
- 并行前缀和算法
- 共享内存和原子操作的使用

---

## 🔗 相关资源

- **下一个算子**：[06_Fused_Add_RMSNorm.md](./06_Fused_Add_RMSNorm.md)
- **基数排序**：Radix Sort 算法详解
- **并行前缀和**：Parallel Prefix Sum 算法

