# Fused Add RMSNorm 算子详解

## 📖 算子概述

**Fused Add RMSNorm** 是 LLM 中的融合操作，将两个常见操作合并：
1. **Add（残差连接）**：`x = input + residual`
2. **RMSNorm（归一化）**：`x = x / sqrt(mean(x²) + eps) * weight`

**用途**：
- Transformer 的层归一化
- 残差连接的融合实现
- 减少内存访问和 kernel 启动次数

**特点**：
- **融合操作**：两个操作合并为一个 kernel
- **减少内存访问**：中间结果保留在寄存器中
- **性能优化**：减少 kernel 启动开销

---

## 🔢 公式与算法

### 数学公式

#### 步骤 1：残差连接

```
x = input + residual
```

**含义**：将输入与残差相加。

#### 步骤 2：RMSNorm

**RMS（Root Mean Square）**：
```
rms = sqrt(mean(x²) + eps)
```

其中：
- `mean(x²) = sum(x_i²) / n`
- `eps`：防止除零的小常数（如 1e-6）

**归一化**：
```
x_norm = x / rms
```

**缩放**：
```
output = x_norm * weight
```

#### 完整公式

```
x = input + residual
rms = sqrt(sum(x²) / n + eps)
output = (x / rms) * weight
```

**向量形式**：
```
x_i = input_i + residual_i
rms = sqrt((1/n) * Σ(x_i²) + eps)
output_i = (x_i / rms) * weight_i
```

---

## 🧠 算法原理

### 基本原理

RMSNorm 是 LayerNorm 的简化版本：

| 归一化方式 | 公式 | 特点 |
|-----------|------|------|
| **LayerNorm** | `(x - mean(x)) / std(x) * γ + β` | 减均值，除标准差 |
| **RMSNorm** | `x / rms(x) * weight` | 只除 RMS，不减均值 |

**为什么用 RMSNorm？**
- **计算更快**：不需要计算均值
- **效果相近**：在很多情况下性能相似
- **数值稳定**：避免减均值带来的精度问题

### 算法流程

```
1. Add（残差连接）
   x = input + residual
   ↓
2. 计算平方和（并行归约）
   sum_sq = Σ(x_i²)
   ↓
3. 计算 RMS
   rms = sqrt(sum_sq / n + eps)
   ↓
4. 归一化和缩放
   output_i = (x_i / rms) * weight_i
```

### 融合操作的优势

**分离版本**：
```
1. add_kernel<<<...>>>(input, residual, temp)      // Kernel 1
   cudaDeviceSynchronize()
2. rmsnorm_kernel<<<...>>>(temp, weight, output)   // Kernel 2
```

**问题**：
- 需要中间内存 `temp`（O(n)）
- 两次内存读写（写入 temp，读取 temp）
- 两次 kernel 启动开销

**融合版本**：
```
fused_add_rmsnorm_kernel<<<...>>>(input, residual, weight, output)
```

**优势**：
- ✅ **无中间内存**：中间结果保留在寄存器中
- ✅ **一次内存读写**：减少内存访问
- ✅ **一次 kernel 启动**：降低开销
- ✅ **更好的缓存**：数据在缓存中复用

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu`

### 主机端调用

```24:59:SGLang学习/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu
void sgl_fused_add_rmsnorm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps, bool enable_pdl) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  auto device = input.device();
  CHECK_EQ(residual.device(), device);
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
  CHECK_DIM(2, residual);  // residual: (batch_size, hidden_size)
  CHECK_DIM(1, weight);    // weight: (hidden_size)
  CHECK_EQ(input.size(0), residual.size(0));
  CHECK_EQ(input.size(1), residual.size(1));
  CHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  // support float16, bfloat16 and float32
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::FusedAddRMSNorm(
        static_cast<c_type*>(input.data_ptr()),
        static_cast<c_type*>(residual.data_ptr()),
        static_cast<c_type*>(weight.data_ptr()),
        batch_size,
        hidden_size,
        input.stride(0),
        residual.stride(0),
        eps,
        enable_pdl,
        torch_current_stream);
    TORCH_CHECK(
        status == cudaSuccess, "FusedAddRMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
}
```

**关键参数**：
- `input`: 输入张量 `[batch_size, hidden_size]`
- `residual`: 残差张量 `[batch_size, hidden_size]`
- `weight`: 权重 `[hidden_size]`
- `eps`: 防止除零的小常数

**注意**：SGLang 使用了 FlashInfer 库的实现，实际的 kernel 在 FlashInfer 中。

### 简化实现（展示核心逻辑）

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

template<typename T>
__global__ void fused_add_rmsnorm_kernel(
    const T* input,        // [batch, hidden]
    const T* residual,     // [batch, hidden]
    const T* weight,       // [hidden]
    T* output,             // [batch, hidden]
    float eps,
    int batch_size,
    int hidden_size) {
    
    extern __shared__ float smem[];
    float* sum_sq = smem;  // 用于存储平方和
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;  // batch 索引
    
    if (bid >= batch_size) return;
    
    // 1. Add（残差连接）并计算平方（并行归约）
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = bid * hidden_size + i;
        float x = (float)input[idx] + (float)residual[idx];
        local_sum_sq += x * x;  // 同时计算平方和
        // 暂存在共享内存（如果需要）
        smem[i] = x;  // 假设有足够共享内存
    }
    
    // 2. Block 内归约（计算平方和）
    // 使用树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_sq[tid] += sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // 3. 计算 RMS（第一个线程）
    float rms = 1.0f;
    if (tid == 0) {
        float mean_sq = sum_sq[0] / hidden_size;
        rms = sqrtf(mean_sq + eps);
        sum_sq[0] = rms;  // 存储 RMS 供所有线程使用
    }
    __syncthreads();
    
    rms = sum_sq[0];  // 所有线程读取 RMS
    
    // 4. 归一化和缩放
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = bid * hidden_size + i;
        float x = smem[i];  // 从共享内存读取
        float x_norm = x / rms;
        output[idx] = (T)(x_norm * (float)weight[i]);
    }
}
```

**注意**：这是简化版本，实际的实现更复杂，需要考虑：
- 向量化加载
- 更好的归约方式
- 共享内存大小限制

---

## 📐 完整实现（考虑所有细节）

### 优化版本

```cpp
template<typename T>
__global__ void fused_add_rmsnorm_optimized(
    const T* input,
    const T* residual,
    const T* weight,
    T* output,
    float eps,
    int batch_size,
    int hidden_size) {
    
    extern __shared__ char smem[];
    float* sum_sq_shared = (float*)smem;
    float* x_shared = (float*)(smem + blockDim.x * sizeof(float));
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    // 1. Add 和计算平方和（融合）
    float sum_sq = 0.0f;
    const int vec_size = 4;  // 一次处理 4 个元素
    const int num_vectors = (hidden_size + vec_size - 1) / vec_size;
    
    for (int vec_idx = 0; vec_idx < num_vectors; vec_idx++) {
        int base_idx = vec_idx * vec_size;
        float x_vec[vec_size];
        float sq_vec[vec_size];
        
        // 向量化加载和计算
        for (int i = 0; i < vec_size && base_idx + i < hidden_size; i++) {
            int idx = bid * hidden_size + base_idx + i;
            x_vec[i] = (float)input[idx] + (float)residual[idx];
            sq_vec[i] = x_vec[i] * x_vec[i];
            sum_sq += sq_vec[i];
        }
        
        // 存储到共享内存（用于后续归一化）
        if (vec_idx * blockDim.x + tid < hidden_size) {
            int store_idx = vec_idx * blockDim.x + tid;
            if (store_idx < hidden_size) {
                x_shared[store_idx] = x_vec[tid % vec_size];
            }
        }
    }
    
    // 2. Block 内归约平方和
    sum_sq_shared[tid] = sum_sq;
    __syncthreads();
    
    // 树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_sq_shared[tid] += sum_sq_shared[tid + stride];
        }
        __syncthreads();
    }
    
    // 3. 计算 RMS
    float rms = 1.0f;
    if (tid == 0) {
        float mean_sq = sum_sq_shared[0] / hidden_size;
        rms = sqrtf(mean_sq + eps);
        sum_sq_shared[0] = rms;
    }
    __syncthreads();
    
    rms = sum_sq_shared[0];
    
    // 4. 归一化和缩放
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = bid * hidden_size + i;
        float x = x_shared[i];
        float x_norm = x / rms;
        output[idx] = (T)(x_norm * (float)weight[i]);
    }
}
```

---

## 🎯 设计要点与优化

### 1. 融合操作的优势

**内存访问对比**：

| 操作 | 分离版本 | 融合版本 |
|------|---------|---------|
| **读取** | input, residual, temp, weight | input, residual, weight |
| **写入** | temp, output | output |
| **总访问** | 5 次（4读1写） | 3 次（2读1写） |

**性能提升**：
- 减少 40% 的内存访问
- 无中间内存分配
- 更好的缓存利用率

### 2. 归约优化

**平方和计算**：
- 每个线程计算局部平方和
- 使用共享内存做 block 内归约
- 如果 hidden_size 很大，可能需要多级归约

**优化技巧**：
- 向量化计算：一次处理多个元素
- 多个累加器：减少循环依赖
- 树形归约：O(log n) 复杂度

### 3. 数值稳定性

**EPS 的作用**：
```cpp
rms = sqrt(mean_sq + eps);
```

**为什么需要 eps？**
- 防止 `mean_sq = 0` 时除零
- 提高数值稳定性
- 通常 `eps = 1e-6`

### 4. 向量化实现

**向量化加载**：
```cpp
// 一次加载 4 个 float
float4 vec_input = *((float4*)&input[idx]);
float4 vec_residual = *((float4*)&residual[idx]);
```

**向量化计算**：
```cpp
float4 vec_x;
vec_x.x = vec_input.x + vec_residual.x;
vec_x.y = vec_input.y + vec_residual.y;
// ...
```

---

## 📊 性能分析

### 复杂度

**时间复杂度**：
- Add：O(hidden_size) / threads
- 平方和归约：O(hidden_size) / threads + O(log threads)
- 归一化：O(hidden_size) / threads
- 总复杂度：O(hidden_size) / threads

**空间复杂度**：
- 共享内存：O(threads)（存储归约中间结果）
- 全局内存：O(batch * hidden_size)

### 性能瓶颈

1. **内存访问**：读取 input、residual，写入 output
2. **归约操作**：需要共享内存同步
3. **除法运算**：`x / rms` 相对慢

### 优化建议

1. **向量化**：一次处理多个元素
2. **共享内存**：复用数据，减少全局内存访问
3. **快速数学函数**：使用 `__fdividef`（快速除法）

---

## 📝 总结

### 核心概念

1. **融合操作**：将多个操作合并为一个 kernel
2. **残差连接**：`x = input + residual`
3. **RMSNorm**：`output = (x / rms) * weight`
4. **并行归约**：计算平方和

### 关键优化

- ✅ **减少内存访问**：无中间结果
- ✅ **融合计算**：一次 kernel 启动
- ✅ **共享内存**：快速归约
- ✅ **向量化**：提高带宽利用率

### 学习价值

Fused Add RMSNorm 展示了：
- 融合操作的设计思路
- 并行归约的实现
- 内存访问优化技巧
- 数值稳定性的考虑

---

## 🔗 相关资源

- **RMSNorm 论文**：Root Mean Square Layer Normalization
- **LayerNorm vs RMSNorm**：性能对比
- **下一个算子**：[README.md](./README.md)（目录索引）

