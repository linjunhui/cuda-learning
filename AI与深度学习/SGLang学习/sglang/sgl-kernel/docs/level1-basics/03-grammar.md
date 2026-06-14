# Level 1.3: Grammar 语法约束解码

## 📋 模块概述

Grammar 模块实现了基于语法约束的解码功能，允许 LLM 生成符合特定格式要求（如 JSON、代码等）的文本。这是通过位掩码（bitmask）机制来限制每一步可生成的 token 集合实现的。

**难度等级**：⭐⭐ 初级  
**重要程度**：⭐⭐⭐ 特定应用场景

## 📂 文件结构

```
csrc/grammar/
└── apply_token_bitmask_inplace_cuda.cu  # 应用 token 位掩码
```

## 🎯 算子来源与理论基础

### 来源

**原始实现**：MLC-AI xGrammar 项目
- **代码来源**：`mlc-ai/xgrammar/python/xgrammar/kernels/apply_token_bitmask_inplace_cuda.cu`
- **版本**：v0.1.18
- **适配说明**：从 xGrammar 项目适配而来，支持 CUDA 12.4+

### 理论基础

**语法约束解码（Grammar Constrained Decoding）**：
- **问题**：如何让 LLM 生成符合特定语法规则的文本？
- **解决方案**：使用上下文无关文法（CFG）或 JSON Schema 定义允许的 token 序列
- **实现方式**：每一步解码时，用位掩码标记哪些 token 是合法的，将非法 token 的 logits 设为 -∞

## 🔬 算法原理详解

### 1. 位掩码表示

**核心思想**：
- 词汇表大小为 V，每个 token 对应一个位（bit）
- 位掩码是长度为 `V / 32` 的 int32 数组
- 位为 1 表示该 token 是合法的，为 0 表示非法

**示例**：
```python
# 假设词汇表大小 V = 128
# token 0-63 合法，token 64-127 非法
bitmask = [
    0xFFFFFFFFFFFFFFFF,  # 前 64 位全 1
    0x0000000000000000   # 后 64 位全 0
]
```

### 2. Logits 掩码应用

**算法步骤**：
1. **读取位掩码**：从全局内存读取当前 token 位置的位掩码
2. **向量化处理**：使用打包数据类型（如 `float4`）进行向量化
3. **条件掩码**：如果对应的位为 0，将 logit 设为 -∞
4. **批量处理**：支持批处理和索引映射

**核心操作**：
```cpp
// 伪代码
for each token position in batch:
    for each bit in bitmask:
        if bit == 0:
            logits[token_idx] = -infinity
        else:
            logits[token_idx] = original_logits[token_idx]
```

### 3. 内存布局优化

**打包访问模式**：
- 使用 `float4` 或 `int4` 等打包类型进行向量化访问
- 每个线程处理多个 token（对齐到打包类型大小）
- 减少全局内存访问次数

## 💡 应用场景

### 1. JSON 生成

```python
# 确保生成有效的 JSON
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    }
}

# 每一步解码时，根据当前状态更新位掩码
# 例如，在键的位置，只允许字符串 token
# 在值的位置，根据属性类型允许不同的 token
```

### 2. 代码生成

```python
# 确保生成语法正确的 Python 代码
# 例如，在 "if " 之后，必须跟随条件表达式
# 在 "def " 之后，必须跟随函数名
```

### 3. 结构化输出

```python
# 生成特定格式的输出
# 例如，API 响应、配置文件、SQL 查询等
```

## 💻 代码实现分析

### 核心内核实现

```84:100:csrc/grammar/apply_token_bitmask_inplace_cuda.cu
template <typename T, typename PackedT, int32_t kBitsPerThread>
__global__ void __launch_bounds__(THREADS_PER_THREAD_BLOCK) LogitsBitmaskKernel(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t logits_stride,
    int32_t bitmask_stride) {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  constexpr uint32_t kPackedMask = (1 << kAlignment) - 1;

  const int batch_idx = (indices == nullptr) ? blockIdx.y : indices[blockIdx.y];

  const int block_offset = blockIdx.x * THREADS_PER_THREAD_BLOCK * kBitsPerThread;
  T* logits_gmem_ptr = logits + batch_idx * logits_stride + block_offset;
  const int32_t* bitmask_gmem_ptr = bitmask + batch_idx * bitmask_stride + block_offset / BITS_PER_BLOCK;
```

**关键设计点**：

1. **模板参数**：
   - `T`: 数据类型（float, half, bfloat16）
   - `PackedT`: 打包类型（float4, int4 等）
   - `kBitsPerThread`: 每个线程处理的 token 数量

2. **批次索引映射**：
   - 支持可选的索引映射（`indices` 参数）
   - 允许重新排序或选择特定的批次项

3. **块级偏移**：
   - `block_offset`: 当前块处理的起始位置
   - 确保所有线程处理连续的 token 范围

### 位掩码读取和应用

```cpp
// 读取位掩码块
const int bitmask_block_idx = block_offset / BITS_PER_BLOCK;
const int bitmask_inner_idx = threadIdx.x % (BITS_PER_BLOCK / kAlignment);
const int32_t bitmask_word = bitmask_gmem_ptr[bitmask_block_idx];

// 提取对应的位
const uint32_t bit_offset = (block_offset % BITS_PER_BLOCK) + threadIdx.x * kAlignment;
const uint32_t bitmask_bits = (bitmask_word >> bit_offset) & kPackedMask;

// 应用掩码
PackedT packed_logits = *reinterpret_cast<PackedT*>(logits_ptr);
PackedT packed_neg_inf = PackedNegativeInfinity<T, PackedT>();

// 如果位为 0，设为 -∞；否则保持原值
PackedT masked_logits = (bitmask_bits & kPackedMask) ? packed_logits : packed_neg_inf;
*reinterpret_cast<PackedT*>(logits_ptr) = masked_logits;
```

**掩码应用逻辑**：
- 从位掩码中提取对应的位
- 使用条件运算符选择保留原值或设为 -∞
- 向量化操作提高效率

### 负无穷值处理

```50:71:csrc/grammar/apply_token_bitmask_inplace_cuda.cu
template <typename T>
__device__ T NegativeInfinity() {
  return -INFINITY;
}

template <>
__device__ __half NegativeInfinity<__half>() {
#ifdef USE_ROCM
  return __float2half(-INFINITY);
#else
  return -CUDART_INF_FP16;
#endif
}

template <>
__device__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
#ifdef USE_ROCM
  return __nv_bfloat16(-INFINITY);
#else
  return -CUDART_INF_BF16;
#endif
}
```

**类型特化**：
- 不同浮点类型需要不同的负无穷表示
- FP16 和 BF16 有特定的位模式
- 使用模板特化处理不同类型

## ⚡ 性能优化技巧

### 1. 向量化内存访问

- 使用打包类型（`float4`, `int4`）减少内存事务
- 对齐到 128 位边界以最大化带宽

### 2. 位操作优化

- 使用位掩码和位运算快速检查合法性
- 批量处理多个 token

### 3. 合并内存访问

- 确保同一 warp 的线程访问连续内存
- 使用 `__restrict__` 关键字帮助编译器优化

### 4. 共享内存缓存

- 可以缓存常用的位掩码块到共享内存
- 减少全局内存访问

## 🔍 关键接口说明

| 接口名称 | 功能 | 输入维度 | 说明 |
|---------|------|---------|------|
| `apply_token_bitmask_inplace_cuda` | 应用 token 位掩码到 logits | `logits: (B, V), bitmask: (B, V//32), indices: (B,) optional` | 原地修改 logits，非法 token 设为 -∞ |

**参数说明**：
- `logits`: 原始 logits 张量 `(batch_size, vocab_size)`
- `bitmask`: 位掩码张量，每个 int32 表示 32 个 token
- `indices`: 可选的批次索引映射，用于重新排序

## 📊 性能特征

### 内存访问模式

- **读取**：`logits` + `bitmask` → 每个 token 2 次读取
- **写入**：`logits`（原地修改）→ 每个 token 1 次写入
- **带宽需求**：相对较低，主要是条件掩码操作

### 计算复杂度

- **时间复杂度**：O(B × V)，其中 B 是批次大小，V 是词汇表大小
- **并行度**：高，每个 token 可以独立处理

## 🎓 学习建议

1. **理解位操作**：掌握位掩码的基本操作
2. **学习 CFG**：了解上下文无关文法的基本概念
3. **实践应用**：尝试实现简单的 JSON 约束解码
4. **性能分析**：使用 Nsight 分析内存访问模式

## 📚 参考资料

1. **xGrammar 项目**：https://github.com/mlc-ai/xgrammar
2. **Grammar Constrained Decoding 论文**：相关研究论文
3. **CUDA 位操作指南**：NVIDIA CUDA Programming Guide
4. **JSON Schema 规范**：https://json-schema.org/

---

**进入下一级别**：[Level 2: 中级模块](../level2-intermediate/README.md)

