# Merge Attention States 算子详解

## 📖 算子概述

**Merge Attention States** 是用于合并前缀（prefix）和后缀（suffix）注意力状态的算子。这在**推测解码（Speculative Decoding）**和**前缀缓存**等场景中非常重要。

**用途**：
- **推测解码**：合并前缀和后缀的注意力状态
- **前缀缓存**：重用前缀的注意力计算
- **分段注意力**：将多个段的注意力状态合并

**特点**：
- **数值稳定**：使用 log-sum-exp 技巧避免数值溢出
- **向量化**：使用 128 位打包加载/存储
- **精度保证**：在 float32 精度下计算，避免精度损失

---

## 🔢 公式与算法

### 数学公式

**核心公式**（来自论文 Section 2.2）：

给定前缀和后缀的注意力状态：
- 前缀输出：`prefix_output`，LSE：`prefix_lse`
- 后缀输出：`suffix_output`，LSE：`suffix_lse`

合并后的输出：

```
output = (prefix_output * prefix_scale) + (suffix_output * suffix_scale)
```

其中：

```
max_lse = max(prefix_lse, suffix_lse)
prefix_lse_norm = prefix_lse - max_lse
suffix_lse_norm = suffix_lse - max_lse
prefix_scale = exp(prefix_lse_norm) / (exp(prefix_lse_norm) + exp(suffix_lse_norm))
suffix_scale = exp(suffix_lse_norm) / (exp(prefix_lse_norm) + exp(suffix_lse_norm))
output_lse = log(exp(prefix_lse_norm) + exp(suffix_lse_norm)) + max_lse
```

### 算法步骤

```
1. 计算 max_lse = max(prefix_lse, suffix_lse)
2. 归一化 LSE：prefix_lse_norm = prefix_lse - max_lse
                 suffix_lse_norm = suffix_lse - max_lse
3. 计算比例：prefix_scale = exp(prefix_lse_norm) / (exp(prefix_lse_norm) + exp(suffix_lse_norm))
             suffix_scale = exp(suffix_lse_norm) / (exp(prefix_lse_norm) + exp(suffix_lse_norm))
4. 合并输出：output = prefix_output * prefix_scale + suffix_output * suffix_scale
5. 计算新的 LSE：output_lse = log(exp(prefix_lse_norm) + exp(suffix_lse_norm)) + max_lse
```

### 数值稳定性

**问题**：直接计算 `exp(lse)` 可能导致溢出（如果 `lse` 很大）。

**解决方案**：**Log-Sum-Exp 技巧**

```
exp(prefix_lse) + exp(suffix_lse)
= exp(max_lse) * (exp(prefix_lse - max_lse) + exp(suffix_lse - max_lse))
= exp(max_lse) * (exp(prefix_lse_norm) + exp(suffix_lse_norm))
```

**优势**：
- `prefix_lse_norm` 和 `suffix_lse_norm` 都是负数（≤ 0）
- `exp(负数)` 不会溢出
- 保证了数值稳定性

---

## 🧠 算法原理

### 基本原理

在注意力机制中，输出和 LSE（Log-Sum-Exp）的关系：

```
output = value @ softmax(scores)
       = value @ exp(scores - max_score) / sum(exp(scores - max_score))
       = (value @ exp(scores - max_score)) / exp(lse)
```

其中 `lse = log(sum(exp(scores - max_score))) + max_score`。

**合并两个注意力状态**：

```
output = (prefix_output * prefix_scale) + (suffix_output * suffix_scale)
```

其中比例 `prefix_scale` 和 `suffix_scale` 通过归一化的 LSE 计算。

### 为什么需要归一化？

**示例**（简化）：

```
前缀 LSE: 10.0
后缀 LSE: 12.0

直接计算：
  exp(10.0) = 22026  （可以计算）
  exp(12.0) = 162754 （可以计算）

但如果 LSE 很大：
  exp(100.0) = 2.69e+43  （溢出！）
```

**归一化后**：
```
max_lse = 12.0
prefix_lse_norm = 10.0 - 12.0 = -2.0
suffix_lse_norm = 12.0 - 12.0 = 0.0

exp(-2.0) = 0.135  （不会溢出）
exp(0.0) = 1.0     （不会溢出）
```

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/attention/merge_attn_states.cu`

### 核心 Kernel 代码

```31:106:SGLang学习/sglang/sgl-kernel/csrc/attention/merge_attn_states.cu
template <typename scalar_t, const uint NUM_THREADS>
__global__ void merge_attn_states_kernel(
    scalar_t* output,
    float* output_lse,
    const scalar_t* prefix_output,
    const float* prefix_lse,
    const scalar_t* suffix_output,
    const float* suffix_lse,
    const uint num_tokens,
    const uint num_heads,
    const uint head_size) {
  using pack_128b_t = uint4;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;  // (0~15)*8, etc.
  const uint head_offset = token_idx * num_heads * head_size + head_idx * head_size;
  const scalar_t* prefix_head_ptr = prefix_output + head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + head_offset;
  scalar_t* output_head_ptr = output + head_offset;

  // float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  // float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
  float p_lse = prefix_lse[token_idx * num_heads + head_idx];
  float s_lse = suffix_lse[token_idx * num_heads + head_idx];
  p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = fmaxf(p_lse, s_lse);
  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = expf(p_lse);
  const float s_se = expf(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (pack_offset < head_size) {
    // Pack 128b load
    pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[pack_offset / pack_size];
    pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[pack_offset / pack_size];
    pack_128b_t o_out_pack;

#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      // Always use float for FMA to keep high precision.
      // half(uint16_t), bfloat16, float -> float.
      const float p_out_f = to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f = to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
      // fma: a * b + c = p_out_f * p_scale + (s_out_f * s_scale)
      const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
      // float -> half(uint16_t), bfloat16, float.
      from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
    }

    // Pack 128b storage
    reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] = o_out_pack;
  }
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && pack_idx == 0) {
    float out_lse = logf(out_se) + max_lse;
    output_lse[token_idx * num_heads + head_idx] = out_lse;
  }
}
```

### 代码逐行解析

#### 第一步：索引计算

```cpp
const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
const uint token_head_idx = global_idx / threads_per_head;
const uint pack_idx = global_idx % threads_per_head;
const uint token_idx = token_head_idx / num_heads;
const uint head_idx = token_head_idx % num_heads;
```

**线程分配**：
- **扁平化索引**：`global_idx` = 线程在 grid 中的全局索引
- **分解索引**：
  - `token_idx`：token 索引
  - `head_idx`：head 索引
  - `pack_idx`：pack 索引（向量化单位）

**设计模式**：
- 每个线程处理一个 pack（128 位）
- 对于 `half`/`bfloat16`：pack_size = 8（一次处理 8 个元素）
- 对于 `float`：pack_size = 4（一次处理 4 个元素）

#### 第二步：LSE 归一化（数值稳定性）

```cpp
float p_lse = prefix_lse[token_idx * num_heads + head_idx];
float s_lse = suffix_lse[token_idx * num_heads + head_idx];
p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

const float max_lse = fmaxf(p_lse, s_lse);
p_lse = p_lse - max_lse;
s_lse = s_lse - max_lse;
```

**关键点**：
- **处理无穷值**：如果 LSE 是正无穷，转为负无穷（避免溢出）
- **归一化**：减去 `max_lse`，确保两个 LSE 都是负数（≤ 0）

#### 第三步：计算比例

```cpp
const float p_se = expf(p_lse);
const float s_se = expf(s_lse);
const float out_se = p_se + s_se;
const float p_scale = p_se / out_se;
const float s_scale = s_se / out_se;
```

**公式**：
```
p_se = exp(p_lse_norm)
s_se = exp(s_lse_norm)
out_se = p_se + s_se
p_scale = p_se / out_se
s_scale = s_se / out_se
```

**数值稳定性**：
- 由于 `p_lse_norm ≤ 0` 和 `s_lse_norm ≤ 0`
- `exp(负数)` 不会溢出（≤ 1.0）

#### 第四步：向量化合并（关键优化）

```cpp
pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[pack_offset / pack_size];
pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[pack_offset / pack_size];
pack_128b_t o_out_pack;

#pragma unroll
for (uint i = 0; i < pack_size; ++i) {
  const float p_out_f = to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
  const float s_out_f = to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
  const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
  from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
}
```

**关键优化**：
- **128 位打包加载**：一次加载 128 位（16 字节）
  - `half`/`bfloat16`：8 个元素
  - `float`：4 个元素
- **类型转换**：转为 `float32` 计算，保证精度
- **FMA 操作**：`p_out_f * p_scale + (s_out_f * s_scale)`
- **循环展开**：`#pragma unroll` 消除循环开销

#### 第五步：写回结果

```cpp
reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] = o_out_pack;
```

**打包存储**：一次写入 128 位，提高带宽利用率。

#### 第六步：计算新的 LSE

```cpp
if (output_lse != nullptr && pack_idx == 0) {
  float out_lse = logf(out_se) + max_lse;
  output_lse[token_idx * num_heads + head_idx] = out_lse;
}
```

**关键点**：
- **只写一次**：只有 `pack_idx == 0` 的线程写 LSE（避免重复写入）
- **公式**：`out_lse = log(out_se) + max_lse`

---

## 🎯 关键设计要点

### 1. 数值稳定性（Log-Sum-Exp 技巧）

**核心思想**：
```
exp(a) + exp(b) = exp(max(a,b)) * (exp(a-max(a,b)) + exp(b-max(a,b)))
```

**优势**：
- 避免 `exp(a)` 或 `exp(b)` 溢出
- 保证计算的数值稳定性

### 2. 向量化优化（128 位打包）

**设计**：
- 使用 `uint4`（128 位）作为打包类型
- 一次加载/存储多个元素

**优势**：
- 提高内存带宽利用率
- 减少内存访问次数

### 3. 类型转换（精度保证）

**流程**：
```
half/bfloat16 → float → 计算 → half/bfloat16
```

**原因**：
- 在 `float32` 精度下计算，避免精度损失
- 特别是 FMA 操作需要高精度

### 4. 只读一次（LSE 写入）

**设计**：
- 只有第一个 pack 的线程写 LSE
- 避免多个线程写入同一位置

**原因**：
- LSE 是标量（每个 head 一个）
- 只需要写一次

---

## 📊 性能分析

### 复杂度

**时间复杂度**：
```
O(num_tokens × num_heads × head_size)
```

**并行化后**：
```
每个线程: O(1) （向量化后）
```

### 内存访问

**读取**：
- `prefix_output`: 1 次（128 位打包）
- `suffix_output`: 1 次（128 位打包）
- `prefix_lse`: 1 次（标量）
- `suffix_lse`: 1 次（标量）

**写入**：
- `output`: 1 次（128 位打包）
- `output_lse`: 1 次（标量，只有第一个线程）

**总访问**：
- 每个 pack：4 次读取 + 2 次写入
- 使用向量化，实际访问次数更少

---

## 📝 总结

### 核心概念

1. **Log-Sum-Exp 技巧**：数值稳定的合并方法
2. **向量化**：128 位打包加载/存储
3. **类型转换**：float32 精度保证
4. **比例混合**：基于归一化 LSE 的比例

### 关键优化

- ✅ **数值稳定性**：Log-Sum-Exp 技巧避免溢出
- ✅ **向量化**：128 位打包提高带宽利用率
- ✅ **精度保证**：float32 中间计算
- ✅ **循环展开**：消除循环开销

### 学习价值

Merge Attention States 展示了：
- 数值稳定性的处理方法
- 向量化优化的技巧
- 复杂数学运算的实现
- 推测解码中的状态合并

---

## 🔗 相关资源

- **论文参考**：Section 2.2 of https://www.arxiv.org/pdf/2501.01005
- **下一个算子**：[09_Concat_MLA算子.md](./09_Concat_MLA算子.md)
- **推测解码**：Speculative Decoding 技术

