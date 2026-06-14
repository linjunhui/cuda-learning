# Lightning Attention Decode 算子详解

## 📖 算子概述

**Lightning Attention Decode** 是 LLM **解码阶段**的注意力计算算子。相比预填充阶段的注意力，解码阶段更简单但更频繁（每个 token 都要计算一次）。

**用途**：
- LLM 生成时的逐 token 注意力计算
- KV Cache 的增量更新
- 支持滑动窗口注意力（通过衰减因子）

**特点**：
- **增量计算**：只计算当前 token 的注意力
- **KV Cache 更新**：融合更新操作，减少内存访问
- **共享内存优化**：q、k、v 载入共享内存复用
- **高效设计**：专门为解码阶段优化

---

## 🔢 公式与算法

### 数学公式

#### 标准注意力公式（解码阶段）

对于当前 token `t`：

```
Attention(Q_t, K, V) = softmax(Q_t @ K^T / √d_k) @ V
```

其中：
- `Q_t`：当前 token 的 query（形状 `[1, heads, d]`）
- `K`：所有过去 token 的 key（形状 `[seq_len, heads, d]`）
- `V`：所有过去 token 的 value（形状 `[seq_len, heads, d]`）

#### 滑动窗口注意力（衰减）

Lightning Attention 使用滑动窗口，通过衰减因子：

```
new_kv = ratio * old_kv + k_t @ v_t^T
```

其中：
- `ratio = exp(-slope)`：衰减因子
- `k_t`：当前 token 的 key
- `v_t`：当前 token 的 value
- `old_kv`：过去的 `k @ v^T` 累积

#### 注意力输出

```
output = Q_t @ new_kv
```

**关键点**：
- `new_kv` 的形状是 `[qk_dim, v_dim]`（不是 `[seq_len, qk_dim, v_dim]`）
- 这是一个**矩阵-向量乘法**：`q` (向量) × `kv` (矩阵) = `output` (向量)

---

## 🧠 算法原理

### 核心思想

解码阶段的注意力与预填充阶段不同：

| 阶段 | Query | Key/Value | 输出 |
|------|-------|-----------|------|
| **预填充** | `[seq_len, heads, d]` | `[seq_len, heads, d]` | `[seq_len, heads, d]` |
| **解码** | `[1, heads, d]` | `[seq_len, heads, d]` | `[1, heads, d]` |

**关键差异**：
- 解码阶段每次只处理**一个 token**
- 可以利用这个特点做很多优化

### 算法流程

```
1. 加载当前 token 的 q, k, v 到共享内存
   ↓
2. 计算新的 KV：new_kv = ratio * old_kv + k @ v^T
   ↓
3. 更新 KV Cache：past_kv = new_kv
   ↓
4. 计算注意力输出：output = q @ new_kv
```

### 滑动窗口机制

**传统注意力**：
- 需要存储所有历史 token 的 k、v
- 内存占用：O(seq_len)

**滑动窗口注意力**：
- 只存储累积的 `k @ v^T`
- 通过衰减因子自动"忘记"旧信息
- 内存占用：O(1)

**衰减公式**：
```
new_kv = exp(-slope) * old_kv + k_new @ v_new^T
```

**含义**：
- `slope = 0`：不衰减，记住所有历史
- `slope > 0`：逐渐忘记旧信息
- `slope = ∞`：只记住当前 token（相当于无历史）

### 矩阵乘法优化

**计算**：`output = q @ kv`

```
q:      [qk_dim]           (向量)
kv:     [qk_dim, v_dim]    (矩阵)
output: [v_dim]            (向量)
```

**并行化**：
- 每个线程计算 `output` 的一个元素
- 线程 `i` 计算：`output[i] = Σ(q[j] * kv[j][i])` for all j

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu`

### 完整 Kernel 代码

```25:113:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
template <typename T>
__global__ void lightning_attention_decode_kernel(
    const T* __restrict__ q,            // [b, h, 1, d]
    const T* __restrict__ k,            // [b, h, 1, d]
    const T* __restrict__ v,            // [b, h, 1, e]
    const float* __restrict__ past_kv,  // [b, h, d, e]
    const float* __restrict__ slope,    // [h, 1, 1]
    T* __restrict__ output,             // [b, h, 1, e]
    float* __restrict__ new_kv,         // [b, h, d, e]
    const int batch_size,
    const int num_heads,
    const int qk_dim,
    const int v_dim) {
  extern __shared__ char smem[];
  T* __restrict__ q_shared = reinterpret_cast<T*>(smem);
  T* __restrict__ k_shared = reinterpret_cast<T*>(smem + qk_dim * sizeof(T));
  T* __restrict__ v_shared = reinterpret_cast<T*>(smem + 2 * qk_dim * sizeof(T));
  float* __restrict__ new_kv_shared = reinterpret_cast<float*>(smem + (2 * qk_dim + v_dim) * sizeof(T));
  T* __restrict__ output_shared =
      reinterpret_cast<T*>(smem + (2 * qk_dim + v_dim) * sizeof(T) + qk_dim * (v_dim + 1) * sizeof(float));

  const int32_t tid = threadIdx.x;
  const int32_t current_head = blockIdx.x;
  const int32_t b = current_head / num_heads;
  const int32_t h = current_head % num_heads;

  if (b >= batch_size) return;

  const int32_t qk_offset = b * num_heads * qk_dim + h * qk_dim;
  const int32_t v_offset = b * num_heads * v_dim + h * v_dim;
  const int32_t kv_offset = b * num_heads * qk_dim * v_dim + h * qk_dim * v_dim;

  // Load q, k, v into shared memory
  for (int d = tid; d < qk_dim; d += blockDim.x) {
    q_shared[d] = q[qk_offset + d];
    k_shared[d] = k[qk_offset + d];
  }
  for (int e = tid; e < v_dim; e += blockDim.x) {
    v_shared[e] = v[v_offset + e];
  }

  __syncthreads();

  const float ratio = expf(-1.0f * slope[h]);

  // Compute new_kv
  for (int d = tid; d < qk_dim; d += blockDim.x) {
    const T k_val = k_shared[d];
    for (int e = 0; e < v_dim; ++e) {
      const int past_kv_idx = kv_offset + d * v_dim + e;
      const T v_val = v_shared[e];
      const float new_val = ratio * past_kv[past_kv_idx] + k_val * v_val;
      const int shared_idx = d * (v_dim + 1) + e;
      new_kv_shared[shared_idx] = new_val;
    }
  }

  __syncthreads();

  // Store new_kv to global memory
  for (int idx = tid; idx < qk_dim * v_dim; idx += blockDim.x) {
    const int d = idx / v_dim;
    const int e = idx % v_dim;
    const int shared_idx = d * (v_dim + 1) + e;
    const int global_idx = kv_offset + idx;
    new_kv[global_idx] = new_kv_shared[shared_idx];
  }

  __syncthreads();

  // Compute output
  for (int e = tid; e < v_dim; e += blockDim.x) {
    float sum = 0.0f;
    for (int d = 0; d < qk_dim; ++d) {
      const int shared_idx = d * (v_dim + 1) + e;
      sum += q_shared[d] * new_kv_shared[shared_idx];
    }
    output_shared[e] = static_cast<T>(sum);
  }

  __syncthreads();

  // Store output to global memory
  if (tid == 0) {
    for (int e = 0; e < v_dim; ++e) {
      output[v_offset + e] = output_shared[e];
    }
  }
}
```

---

## 📐 代码详细解析

### 第一步：共享内存布局

```38:44:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
  extern __shared__ char smem[];
  T* __restrict__ q_shared = reinterpret_cast<T*>(smem);
  T* __restrict__ k_shared = reinterpret_cast<T*>(smem + qk_dim * sizeof(T));
  T* __restrict__ v_shared = reinterpret_cast<T*>(smem + 2 * qk_dim * sizeof(T));
  float* __restrict__ new_kv_shared = reinterpret_cast<float*>(smem + (2 * qk_dim + v_dim) * sizeof(T));
  T* __restrict__ output_shared =
      reinterpret_cast<T*>(smem + (2 * qk_dim + v_dim) * sizeof(T) + qk_dim * (v_dim + 1) * sizeof(float));
```

**共享内存布局**（假设 `qk_dim=128`, `v_dim=128`, `T=half`）：

```
共享内存布局：
┌─────────────────────────────────────────────────────────┐
│ q_shared:       [qk_dim] = 128 * 2 = 256 字节           │
├─────────────────────────────────────────────────────────┤
│ k_shared:       [qk_dim] = 128 * 2 = 256 字节           │
├─────────────────────────────────────────────────────────┤
│ v_shared:       [v_dim]  = 128 * 2 = 256 字节           │
├─────────────────────────────────────────────────────────┤
│ new_kv_shared:  [qk_dim, v_dim+1] = 128 * 129 * 4       │
│                = 66,048 字节                            │
├─────────────────────────────────────────────────────────┤
│ output_shared:  [v_dim] = 128 * 2 = 256 字节            │
└─────────────────────────────────────────────────────────┘
总共享内存：约 67 KB
```

**为什么 `new_kv_shared` 是 `[qk_dim, v_dim+1]`？**
- 可能是为了内存对齐
- 或者预留空间用于其他用途

**关键点**：
- **手动布局**：完全控制内存布局
- **类型转换**：使用 `reinterpret_cast` 在不同类型间切换
- **对齐优化**：确保访问对齐

### 第二步：计算线程索引

```46:51:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
  const int32_t tid = threadIdx.x;
  const int32_t current_head = blockIdx.x;
  const int32_t b = current_head / num_heads;
  const int32_t h = current_head % num_heads;

  if (b >= batch_size) return;
```

**设计模式**：
- **每个 head 一个 block**：`grid = batch_size * num_heads`
- **Batch 和 Head 的索引**：
  - `b = blockIdx.x / num_heads`：batch 索引
  - `h = blockIdx.x % num_heads`：head 索引

**示例**：
- `batch_size=2`, `num_heads=4`
- Block 0-3：batch 0 的 4 个 heads
- Block 4-7：batch 1 的 4 个 heads

### 第三步：计算内存偏移

```53:55:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
  const int32_t qk_offset = b * num_heads * qk_dim + h * qk_dim;
  const int32_t v_offset = b * num_heads * v_dim + h * v_dim;
  const int32_t kv_offset = b * num_heads * qk_dim * v_dim + h * qk_dim * v_dim;
```

**张量布局**：
- `q, k`: `[batch, heads, 1, qk_dim]`
- `v`: `[batch, heads, 1, v_dim]`
- `past_kv, new_kv`: `[batch, heads, qk_dim, v_dim]`

**偏移计算**：
- `qk_offset`：定位到 `q[b, h, 0, :]` 或 `k[b, h, 0, :]`
- `v_offset`：定位到 `v[b, h, 0, :]`
- `kv_offset`：定位到 `past_kv[b, h, :, :]` 的起始位置

### 第四步：加载数据到共享内存

```57:65:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
  // Load q, k, v into shared memory
  for (int d = tid; d < qk_dim; d += blockDim.x) {
    q_shared[d] = q[qk_offset + d];
    k_shared[d] = k[qk_offset + d];
  }
  for (int e = tid; e < v_dim; e += blockDim.x) {
    v_shared[e] = v[v_offset + e];
  }

  __syncthreads();
```

**协作加载模式**：
- **多线程协作**：`for (int d = tid; d < qk_dim; d += blockDim.x)`
- **每个线程处理多个元素**：元素间隔是 `blockDim.x`
- **合并访问**：如果 `qk_dim` 是 `blockDim.x` 的倍数，会产生完美的合并访问

**示例**：
- `qk_dim=128`, `blockDim.x=32`
- 线程 0：加载 0, 32, 64, 96
- 线程 1：加载 1, 33, 65, 97
- ...

**为什么需要同步？**
- 确保所有数据加载完成
- 后续计算需要这些数据

### 第五步：计算新的 KV（核心）

```68:80:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
  const float ratio = expf(-1.0f * slope[h]);

  // Compute new_kv
  for (int d = tid; d < qk_dim; d += blockDim.x) {
    const T k_val = k_shared[d];
    for (int e = 0; e < v_dim; ++e) {
      const int past_kv_idx = kv_offset + d * v_dim + e;
      const T v_val = v_shared[e];
      const float new_val = ratio * past_kv[past_kv_idx] + k_val * v_val;
      const int shared_idx = d * (v_dim + 1) + e;
      new_kv_shared[shared_idx] = new_val;
    }
  }
```

**计算公式**：
```cpp
new_kv[d][e] = ratio * old_kv[d][e] + k[d] * v[e]
```

**并行化**：
- 外层循环：每个线程处理 `qk_dim / blockDim.x` 行
- 内层循环：每个线程计算整行（`v_dim` 个元素）

**内存访问**：
- **读取**：`past_kv`（全局内存）、`k_shared`、`v_shared`（共享内存）
- **写入**：`new_kv_shared`（共享内存）

**性能考虑**：
- `past_kv` 是全局内存访问（较慢）
- `k_shared`、`v_shared` 是共享内存（很快）
- 内层循环可能有寄存器压力

### 第六步：写回新的 KV

```82:92:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
  __syncthreads();

  // Store new_kv to global memory
  for (int idx = tid; idx < qk_dim * v_dim; idx += blockDim.x) {
    const int d = idx / v_dim;
    const int e = idx % v_dim;
    const int shared_idx = d * (v_dim + 1) + e;
    const int global_idx = kv_offset + idx;
    new_kv[global_idx] = new_kv_shared[shared_idx];
  }
```

**写回模式**：
- 将 `new_kv_shared` 写回全局内存 `new_kv`
- 使用扁平化的索引：`idx = d * v_dim + e`

**注意**：
- `shared_idx = d * (v_dim + 1) + e`（使用 `v_dim+1`）
- `global_idx = kv_offset + idx = kv_offset + d * v_dim + e`（使用 `v_dim`）
- 这是因为共享内存布局和全局内存布局不同

### 第七步：计算注意力输出

```94:104:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
  __syncthreads();

  // Compute output
  for (int e = tid; e < v_dim; e += blockDim.x) {
    float sum = 0.0f;
    for (int d = 0; d < qk_dim; ++d) {
      const int shared_idx = d * (v_dim + 1) + e;
      sum += q_shared[d] * new_kv_shared[shared_idx];
    }
    output_shared[e] = static_cast<T>(sum);
  }
```

**矩阵向量乘法**：
```
output[e] = Σ(q[d] * new_kv[d][e]) for all d
```

**并行化**：
- 每个线程计算 `output` 的一个元素
- 线程 `i` 计算 `output[i]`

**累加**：
- 使用 `float sum` 在寄存器中累加
- 最后转回类型 `T`（可能是 `half`）

**内存访问**：
- `q_shared`：共享内存，快速
- `new_kv_shared`：共享内存，快速
- 所有数据都在共享内存中，访问很快

### 第八步：写回输出

```105:113:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
  __syncthreads();

  // Store output to global memory
  if (tid == 0) {
    for (int e = 0; e < v_dim; ++e) {
      output[v_offset + e] = output_shared[e];
    }
  }
}
```

**设计选择**：
- 只用第一个线程写回（`tid == 0`）
- 串行写入，但 `v_dim` 通常不大（如 128）

**为什么不并行写回？**
- 可能需要更多的共享内存同步
- 单个线程串行写回简单且足够快
- 如果 `v_dim` 很大，可以考虑并行写回

---

## 🎯 设计要点与优化

### 1. 共享内存的使用

**为什么使用共享内存？**
- **复用**：q、k、v 被多次访问
- **速度**：共享内存比全局内存快 10-100x
- **带宽**：减少全局内存访问次数

**共享内存大小**：
```
总大小 ≈ 2*qk_dim*sizeof(T) + v_dim*sizeof(T) + qk_dim*(v_dim+1)*sizeof(float) + v_dim*sizeof(T)
```

**示例**（`qk_dim=128`, `v_dim=128`, `T=half`）：
```
= 2*128*2 + 128*2 + 128*129*4 + 128*2
= 512 + 256 + 66,048 + 256
≈ 67 KB
```

**限制**：
- 共享内存通常只有 48-164 KB（取决于 GPU）
- 如果太大，可能需要调整 block 大小

### 2. 线程分配策略

**当前策略**：每个 head 一个 block

**优点**：
- 简化同步（block 内同步即可）
- 每个 head 独立处理
- 容易理解和调试

**缺点**：
- 如果 `qk_dim` 或 `v_dim` 很小，block 利用率低
- 如果 batch 很大，需要很多 blocks

**替代方案**：
- 多个 heads 共享一个 block（需要更复杂的同步）
- 动态调整 block 大小

### 3. 内存访问模式

**读取模式**：

| 数据 | 位置 | 访问模式 | 性能 |
|------|------|---------|------|
| `q, k, v` | 全局→共享 | 合并访问 | ⭐⭐⭐⭐⭐ |
| `past_kv` | 全局 | 部分合并 | ⭐⭐⭐ |
| `q_shared` | 共享 | 广播 | ⭐⭐⭐⭐⭐ |
| `new_kv_shared` | 共享 | 随机访问 | ⭐⭐⭐⭐ |

**优化建议**：
- ✅ 使用共享内存复用数据
- ✅ 合并全局内存访问
- ⚠️ `past_kv` 的访问可能不是完全合并的（取决于 `v_dim`）

---

## 📊 算法复杂度分析

### 时间复杂度

```
步骤 1：加载 q, k, v → O(qk_dim + v_dim) / blockDim.x
步骤 2：计算 new_kv  → O(qk_dim * v_dim) / blockDim.x
步骤 3：写回 new_kv  → O(qk_dim * v_dim) / blockDim.x
步骤 4：计算 output  → O(qk_dim * v_dim) / blockDim.x
步骤 5：写回 output  → O(v_dim)

总复杂度：O(qk_dim * v_dim) （并行后）
```

### 空间复杂度

```
共享内存：O(qk_dim + v_dim + qk_dim * v_dim)
全局内存：O(batch * heads * qk_dim * v_dim)
```

---

## 💡 简化版本（理解核心逻辑）

如果你想理解核心逻辑，这里是简化版本：

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// 简化的 Lightning Attention Decode
__global__ void simple_attention_decode_kernel(
    const float* q,           // [qk_dim]
    const float* k,           // [qk_dim]
    const float* v,           // [v_dim]
    const float* past_kv,     // [qk_dim, v_dim]
    float* new_kv,            // [qk_dim, v_dim]
    float* output,            // [v_dim]
    float ratio,              // 衰减因子
    int qk_dim,
    int v_dim) {
    
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* k_shared = smem + qk_dim;
    float* v_shared = smem + 2 * qk_dim;
    float* new_kv_shared = smem + 2 * qk_dim + v_dim;
    
    int tid = threadIdx.x;
    
    // 1. 加载 q, k, v 到共享内存
    for (int i = tid; i < qk_dim; i += blockDim.x) {
        q_shared[i] = q[i];
        k_shared[i] = k[i];
    }
    for (int i = tid; i < v_dim; i += blockDim.x) {
        v_shared[i] = v[i];
    }
    __syncthreads();
    
    // 2. 计算 new_kv = ratio * old_kv + k @ v^T
    for (int d = tid; d < qk_dim; d += blockDim.x) {
        for (int e = 0; e < v_dim; ++e) {
            int idx = d * v_dim + e;
            new_kv_shared[idx] = ratio * past_kv[idx] + k_shared[d] * v_shared[e];
        }
    }
    __syncthreads();
    
    // 3. 写回 new_kv
    for (int idx = tid; idx < qk_dim * v_dim; idx += blockDim.x) {
        new_kv[idx] = new_kv_shared[idx];
    }
    __syncthreads();
    
    // 4. 计算 output = q @ new_kv
    for (int e = tid; e < v_dim; e += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < qk_dim; ++d) {
            int idx = d * v_dim + e;
            sum += q_shared[d] * new_kv_shared[idx];
        }
        output[e] = sum;
    }
}

void simple_attention_decode_host(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    const float* d_past_kv,
    float* d_new_kv,
    float* d_output,
    float ratio,
    int qk_dim,
    int v_dim) {
    
    int threads = 128;
    size_t smem_size = (2 * qk_dim + v_dim + qk_dim * v_dim) * sizeof(float);
    
    simple_attention_decode_kernel<<<1, threads, smem_size>>>(
        d_q, d_k, d_v, d_past_kv, d_new_kv, d_output,
        ratio, qk_dim, v_dim);
    
    cudaDeviceSynchronize();
}
```

---

## 📝 总结

### 核心概念

1. **增量注意力**：只计算当前 token 的注意力
2. **KV Cache 更新**：`new_kv = ratio * old_kv + k @ v^T`
3. **共享内存复用**：q、k、v 载入共享内存，多次访问
4. **矩阵向量乘法**：`output = q @ kv`

### 关键优化

- ✅ **共享内存**：减少全局内存访问
- ✅ **融合操作**：同时计算注意力和更新 cache
- ✅ **滑动窗口**：通过衰减因子实现
- ✅ **每个 head 一个 block**：简化同步

### 学习价值

Lightning Attention Decode 展示了：
- 复杂的共享内存使用
- 多阶段计算流程
- 内存访问模式优化
- 融合操作的设计思路

---

## 🔗 相关资源

- **下一个算子**：[04_RoPE算子.md](./04_RoPE算子.md)
- **Flash Attention**：解码阶段注意力的另一种实现
- **滑动窗口注意力**：SWA 机制详解

