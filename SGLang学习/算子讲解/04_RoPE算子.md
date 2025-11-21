# RoPE (Rotary Position Embedding) 算子详解

## 📖 算子概述

**RoPE (Rotary Position Embedding)** 是 LLM 中最重要的位置编码方式之一，它将位置信息编码到 query 和 key 向量中，使模型能够感知 token 的相对位置。

**用途**：
- Transformer 中的位置编码
- 每个 token 的 q、k 向量应用旋转
- 支持相对位置关系

**特点**：
- **旋转操作**：通过复数旋转编码位置
- **相对位置**：内积只依赖于相对位置
- **高效实现**：使用预计算的 cos/sin 缓存

---

## 🔢 公式与算法

### 数学公式

#### RoPE 的核心公式

对于位置 `m` 的向量 `x`（2D 切片）：

```
RoPE(x, m) = [x_0*cos(mθ) - x_1*sin(mθ), x_0*sin(mθ) + x_1*cos(mθ)]
```

其中 `θ` 是旋转频率：
```
θ_i = base^(-2i/d)
```

#### 矩阵形式

对于 2D 向量 `[x_0, x_1]`：

```
[x_0']   [cos(mθ)  -sin(mθ)] [x_0]
[x_1'] = [sin(mθ)   cos(mθ)] [x_1]
```

这是标准的**旋转矩阵**。

#### 复数形式

将 2D 向量看作复数：
```
z = x_0 + i*x_1
```

RoPE 相当于：
```
z' = z * exp(i*m*θ)
  = z * (cos(m*θ) + i*sin(m*θ))
```

**含义**：将复数旋转 `m*θ` 角度。

#### 完整公式（向量形式）

对于 `d` 维向量 `x`，将其分成 `d/2` 对：

```
对于 i = 0, 2, 4, ..., d-2:
    x'_i   = x_i*cos(m*θ_{i/2}) - x_{i+1}*sin(m*θ_{i/2})
    x'_{i+1} = x_i*sin(m*θ_{i/2}) + x_{i+1}*cos(m*θ_{i/2})
```

**关键点**：
- 向量被分成 `d/2` 对
- 每对独立旋转
- 每对有不同的旋转频率 `θ_i`

### 相对位置关系

RoPE 的关键优势：**内积只依赖于相对位置**。

对于位置 `m` 的 query 和位置 `n` 的 key：

```
<RoPE(q, m), RoPE(k, n)> = <RoPE(q, 0), RoPE(k, n-m)>
```

**含义**：注意力分数只依赖于相对位置 `n-m`，不依赖于绝对位置。

---

## 🧠 算法原理

### 基本思路

1. **预计算 cos/sin 缓存**：
   ```
   cos_cache[pos][i] = cos(pos * θ_i)
   sin_cache[pos][i] = sin(pos * θ_i)
   ```
   - 对所有位置和所有频率预计算
   - 避免运行时计算三角函数

2. **对每个 token**：
   - 获取其位置 `pos`
   - 加载对应的 `cos` 和 `sin` 值
   - 对 q、k 向量应用旋转

3. **旋转操作**：
   - 将向量分成对
   - 每对应用旋转矩阵
   - 使用预计算的 cos/sin 值

### 算法流程

```
对于每个 token:
  1. 获取位置 pos
  2. 加载 cos_sin_cache[pos]
  3. 对于 q 向量的每对元素 [x_i, x_{i+1}]:
      - x'_i = x_i*cos - x_{i+1}*sin
      - x'_{i+1} = x_i*sin + x_{i+1}*cos
  4. 对 k 向量重复步骤 3
```

### 内存布局

#### Cos/Sin Cache

```
cos_sin_cache: [max_seq_len, rotary_dim]
               第一半是 cos，第二半是 sin

访问：
  cos[i] = cos_sin_cache[pos][i]
  sin[i] = cos_sin_cache[pos][rotary_dim/2 + i]
```

#### Interleave 模式

某些实现使用交错布局：
```
cos_sin_cache[pos] = [cos_0, cos_1, ..., cos_n, sin_0, sin_1, ..., sin_n]
```

或交错：
```
cos_sin_cache[pos] = [cos_0, sin_0, cos_1, sin_1, ..., cos_n, sin_n]
```

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/elementwise/rope.cu`  
`SGLang学习/sglang/sgl-kernel/csrc/elementwise/pos_enc.cuh`

### 主机端调用

```27:168:SGLang学习/sglang/sgl-kernel/csrc/elementwise/rope.cu
void apply_rope_pos_ids_cos_sin_cache(
    at::Tensor q,
    at::Tensor k,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    bool enable_pdl,
    const std::optional<at::Tensor>& v,
    const std::optional<at::Tensor>& k_buffer,
    const std::optional<at::Tensor>& v_buffer,
    const std::optional<at::Tensor>& kv_cache_loc) {
  // ... 参数验证 ...

  unsigned int rotary_dim = cos_sin_cache.size(1);
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);

  // ... stride 计算 ...

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(q.scalar_type(), c_type, [&] {
    if (save_kv_cache) {
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCacheEnhanced(
          static_cast<c_type*>(q.data_ptr()),
          static_cast<c_type*>(k.data_ptr()),
          // ... 其他参数 ...
          stream);
    } else {
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache(
          static_cast<c_type*>(q.data_ptr()),
          static_cast<c_type*>(k.data_ptr()),
          // ... 其他参数 ...
          stream);
    }
  });
}
```

**关键参数**：
- `q`, `k`：输入的 query 和 key（形状 `[nnz, heads, head_dim]`）
- `q_rope`, `k_rope`：输出的旋转后的 q、k
- `cos_sin_cache`：预计算的 cos/sin 值（形状 `[max_seq_len, rotary_dim]`）
- `pos_ids`：每个 token 的位置 ID（形状 `[nnz]`）

### Kernel 核心代码（来自 FlashInfer）

虽然 SGLang 使用了 FlashInfer 库，但核心逻辑是：

```cpp
// 伪代码展示核心逻辑
__device__ void apply_rope(
    const float* x,           // 输入向量 [head_dim]
    float* x_rope,            // 输出向量 [head_dim]
    const float* cos,         // cos 值 [rotary_dim/2]
    const float* sin,         // sin 值 [rotary_dim/2]
    int rotary_dim) {
    
    int tid = threadIdx.x;
    int num_pairs = rotary_dim / 2;
    
    // 向量化加载：一次处理多个对
    #pragma unroll
    for (int i = 0; i < num_pairs; i += vec_size) {
        if (tid * vec_size + i < num_pairs) {
            // 加载一对元素
            float x0 = x[2 * (i + tid * vec_size)];
            float x1 = x[2 * (i + tid * vec_size) + 1];
            
            // 加载 cos/sin
            float c = cos[i + tid * vec_size];
            float s = sin[i + tid * vec_size];
            
            // 应用旋转矩阵
            float x0_new = x0 * c - x1 * s;
            float x1_new = x0 * s + x1 * c;
            
            // 存储结果
            x_rope[2 * (i + tid * vec_size)] = x0_new;
            x_rope[2 * (i + tid * vec_size) + 1] = x1_new;
        }
    }
}
```

---

## 📐 简化版本（理解核心逻辑）

### 完整简化实现

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// 简化的 RoPE 实现
__global__ void rope_kernel(
    const float* q,           // [batch*heads, head_dim]
    const float* k,           // [batch*heads, head_dim]
    float* q_rope,            // [batch*heads, head_dim]
    float* k_rope,            // [batch*heads, head_dim]
    const float* cos_cache,   // [max_seq_len, rotary_dim/2]
    const float* sin_cache,   // [max_seq_len, rotary_dim/2]
    const int* pos_ids,       // [batch*heads]
    int head_dim,
    int rotary_dim,
    int num_tokens) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_pairs = rotary_dim / 2;
    
    if (bid >= num_tokens) return;
    
    // 获取位置
    int pos = pos_ids[bid];
    
    // 计算偏移
    int q_offset = bid * head_dim;
    int k_offset = bid * head_dim;
    int cos_offset = pos * num_pairs;
    int sin_offset = pos * num_pairs;
    
    // 对每对元素应用旋转
    for (int i = tid; i < num_pairs; i += blockDim.x) {
        // 加载输入
        float q0 = q[q_offset + 2 * i];
        float q1 = q[q_offset + 2 * i + 1];
        float k0 = k[k_offset + 2 * i];
        float k1 = k[k_offset + 2 * i + 1];
        
        // 加载 cos/sin
        float c = cos_cache[cos_offset + i];
        float s = sin_cache[sin_offset + i];
        
        // 应用旋转矩阵
        // [x0']   [c  -s] [x0]
        // [x1'] = [s   c] [x1]
        float q0_new = q0 * c - q1 * s;
        float q1_new = q0 * s + q1 * c;
        float k0_new = k0 * c - k1 * s;
        float k1_new = k0 * s + k1 * c;
        
        // 存储结果
        q_rope[q_offset + 2 * i] = q0_new;
        q_rope[q_offset + 2 * i + 1] = q1_new;
        k_rope[k_offset + 2 * i] = k0_new;
        k_rope[k_offset + 2 * i + 1] = k1_new;
    }
}

void rope_cuda(
    const float* d_q,
    const float* d_k,
    float* d_q_rope,
    float* d_k_rope,
    const float* d_cos_cache,
    const float* d_sin_cache,
    const int* d_pos_ids,
    int head_dim,
    int rotary_dim,
    int num_tokens) {
    
    const int threads = 128;
    dim3 block(threads);
    dim3 grid(num_tokens);
    
    rope_kernel<<<grid, block>>>(
        d_q, d_k, d_q_rope, d_k_rope,
        d_cos_cache, d_sin_cache, d_pos_ids,
        head_dim, rotary_dim, num_tokens);
    
    cudaDeviceSynchronize();
}
```

---

## 🎯 关键设计要点

### 1. 预计算 Cos/Sin Cache

**为什么预计算？**
- 三角函数计算慢（`cos`、`sin` 函数）
- 相同位置的值被多次使用
- 预计算可以复用，节省计算时间

**计算方式**：
```python
# Python 伪代码
base = 10000.0
for pos in range(max_seq_len):
    for i in range(rotary_dim // 2):
        theta_i = base ** (-2 * i / head_dim)
        cos_cache[pos][i] = cos(pos * theta_i)
        sin_cache[pos][i] = sin(pos * theta_i)
```

### 2. 向量化实现

**为什么向量化？**
- 一次加载/存储多个元素
- 提高内存带宽利用率
- 利用 SIMD 指令

**实现方式**：
```cpp
// 一次加载 4 个 float（float4）
float4 vec = *((float4*)&x[4*i]);
// 处理 4 个元素
```

### 3. Interleave 模式

**什么是 Interleave？**
- Cos 和 Sin 值交错存储
- 方便向量化加载
- 减少内存访问次数

**布局对比**：

**标准布局**：
```
cache[pos] = [cos_0, cos_1, ..., cos_n, sin_0, sin_1, ..., sin_n]
```

**Interleave 布局**：
```
cache[pos] = [cos_0, sin_0, cos_1, sin_1, ..., cos_n, sin_n]
```

**优势**：
- 一次加载一对 `(cos, sin)`
- 减少内存访问
- 更适合向量化

---

## 📊 性能优化

### 1. 内存访问优化

**合并访问**：
- `q`、`k` 向量：连续访问
- `cos_sin_cache`：根据位置访问，可能需要缓存优化

**共享内存**：
- 如果多个 token 共享相同位置，可以缓存 cos/sin 值

### 2. 计算优化

**使用快速数学函数**：
```cpp
float c = __cosf(pos * theta);  // 快速版本
float s = __sinf(pos * theta);  // 快速版本
```

**精度 vs 速度**：
- `__cosf`：快速但精度略低
- `cosf`：标准精度
- 根据需求选择

### 3. 向量化技巧

**一次处理多个对**：
```cpp
// 一次处理 4 对（8 个元素）
for (int i = 0; i < num_pairs; i += 4) {
    float4 q_vec = *((float4*)&q[2*i]);
    float4 cos_sin = *((float4*)&cos_cache[i]);  // 假设 interleave
    
    // 向量化旋转
    // ...
}
```

---

## 📝 总结

### 核心概念

1. **旋转编码**：通过复数旋转编码位置
2. **相对位置**：内积只依赖于相对位置
3. **预计算缓存**：cos/sin 值预计算
4. **向量化**：一次处理多个元素对

### 关键公式

```
[x_0']   [cos(mθ)  -sin(mθ)] [x_0]
[x_1'] = [sin(mθ)   cos(mθ)] [x_1]
```

### 学习价值

RoPE 展示了：
- 复数运算在 GPU 上的实现
- 矩阵运算的优化技巧
- 预计算缓存的使用
- 向量化内存访问

---

## 🔗 相关资源

- **RoPE 论文**：RoFormer: Enhanced Transformer with Rotary Position Embedding
- **下一个算子**：[05_TopK算子.md](./05_TopK算子.md)
- **FlashInfer**：SGLang 使用的实现库

