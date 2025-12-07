# Concat MLA 算子详解

## 📖 算子概述

**Concat MLA (Multi-Line Attention)** 是用于合并 **无 RoPE 的 K 向量**和**有 RoPE 的 K 向量**的算子。这在某些 MLA (Multi-Line Attention) 架构中使用，将两部分拼接成完整的 K 向量。

**用途**：
- **MLA 架构**：Multi-Line Attention 模型
- **RoPE 分离**：将 K 向量分成两部分（无 RoPE 和有 RoPE）
- **向量拼接**：将两部分拼接成完整的 K 向量

**特点**：
- **向量化优化**：使用 128 位打包加载/存储
- **Warp 级并行**：使用 Warp 作为基本单位
- **高效内存访问**：使用非对齐全局内存访问优化

---

## 🔢 公式与算法

### 数学公式

**拼接公式**：
```
k = [k_nope, k_rope]
```

其中：
- `k_nope`：无 RoPE 的 K 向量（维度：`QK_NOPE_HEAD_DIM = 128`）
- `k_rope`：有 RoPE 的 K 向量（维度：`QK_ROPE_HEAD_DIM = 64`）
- `k`：完整的 K 向量（维度：`K_HEAD_DIM = 128 + 64 = 192`）

**向量形式**：
```
对于每个 token t，每个 head h，每个维度 d：
  if d < QK_NOPE_HEAD_DIM:
    k[t][h][d] = k_nope[t][h][d]
  else:
    k[t][h][d] = k_rope[t][0][d - QK_NOPE_HEAD_DIM]
```

### 算法步骤

```
对于每个 token t，每个 head chunk：
  1. 加载 k_nope[t][h]（128 个元素，打包为 int2）
  2. 加载 k_rope[t]（64 个元素，打包为 int）
  3. 写入 k[t][h][:128] = k_nope[t][h][:128]
  4. 写入 k[t][h][128:] = k_rope[t][:64]
```

**复杂度**：
- **时间复杂度**：O(num_tokens × num_heads × head_dim)
- **空间复杂度**：O(1)
- **并行度**：num_tokens × num_heads（Warp 级并行）

---

## 🧠 算法原理

### 基本原理

**MLA 架构中的 K 向量分离**：

在某些 MLA 模型中，K 向量被分成两部分：
1. **无 RoPE 部分**（`k_nope`）：128 维，不应用 RoPE
2. **有 RoPE 部分**（`k_rope`）：64 维，应用 RoPE

**拼接目的**：
- 将两部分拼接成完整的 K 向量（192 维）
- 用于后续的注意力计算

**内存布局**：
```
输入 k_nope: [num_tokens, num_heads, 128]
输入 k_rope: [num_tokens, 1, 64]  （所有 head 共享）
输出 k:      [num_tokens, num_heads, 192]
```

**关键点**：
- `k_rope` 是所有 head 共享的（只有 1 个 head）
- `k_nope` 是每个 head 独立的

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/elementwise/concat_mla.cu`

### 核心 Kernel 代码

```16:75:SGLang学习/sglang/sgl-kernel/csrc/elementwise/concat_mla.cu
__global__ void concat_mla_k_kernel(
    nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ k_nope,
    const nv_bfloat16* __restrict__ k_rope,
    const int num_tokens,
    const int64_t k_stride_0,
    const int k_stride_1,
    const int64_t k_nope_stride_0,
    const int k_nope_stride_1,
    const int64_t k_rope_stride_0) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
  const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
  const int lane_id = get_lane_id();
  if (token_id >= num_tokens) return;

  using NopeVec = int2;  // 8B/thread，32 thread = 256B/row
  using RopeVec = int;   // 4B/thread，32 thread = 128B/row
  static_assert(sizeof(NopeVec) * 32 == QK_NOPE_HEAD_DIM * sizeof(nv_bfloat16), "nope vec mismatch");
  static_assert(sizeof(RopeVec) * 32 == QK_ROPE_HEAD_DIM * sizeof(nv_bfloat16), "rope vec mismatch");

  const int head_row0 = head_chunk_id * HEAD_CHUNK_SIZE;

  const int2* __restrict__ nope_src =
      reinterpret_cast<const int2*>(k_nope + token_id * k_nope_stride_0 + head_row0 * k_nope_stride_1) + lane_id;

  int2* __restrict__ nope_dst = reinterpret_cast<int2*>(k + token_id * k_stride_0 + head_row0 * k_stride_1) + lane_id;

  int* __restrict__ rope_dst =
      reinterpret_cast<int*>(k + token_id * k_stride_0 + head_row0 * k_stride_1 + QK_NOPE_HEAD_DIM) + lane_id;

  const int nope_src_stride_v = (k_nope_stride_1 >> 2);  // int2 covers 4 bf16
  const int nope_dst_stride_v = (k_stride_1 >> 2);
  const int rope_dst_stride_v = (k_stride_1 >> 1);  // int covers 2 bf16

  const int* rope_base = reinterpret_cast<const int*>(k_rope + token_id * k_rope_stride_0);
  const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);

  prefetch_L2(nope_src);
  NopeVec cur = ld_na_global_v2(nope_src);

#pragma unroll
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    NopeVec next;
    if (i + 1 < HEAD_CHUNK_SIZE) {
      const int2* next_src = nope_src + nope_src_stride_v;
      prefetch_L2(next_src);
      next = ld_na_global_v2(next_src);
    }

    st_na_global_v2(nope_dst, cur);
    st_na_global_v1(rope_dst, rope_val);

    nope_src += nope_src_stride_v;
    nope_dst += nope_dst_stride_v;
    rope_dst += rope_dst_stride_v;

    cur = next;
  }
}
```

### 代码逐行解析

#### 第一步：Warp 级索引计算

```cpp
const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
const int lane_id = get_lane_id();
```

**关键点**：
- **Warp 作为基本单位**：32 个线程为一个 Warp
- **flat_warp_id**：Warp 在 grid 中的索引
- **token_id**：token 索引
- **head_chunk_id**：head chunk 索引（每个 chunk 16 个 heads）
- **lane_id**：线程在 Warp 内的索引（0-31）

**设计模式**：
- **每个 Warp 处理一个 head chunk**（16 个 heads）
- 所有线程协作处理一个 head chunk

#### 第二步：向量类型定义

```cpp
using NopeVec = int2;  // 8B/thread，32 thread = 256B/row
using RopeVec = int;   // 4B/thread，32 thread = 128B/row
```

**关键点**：
- **NopeVec (int2)**：8 字节，一次处理 4 个 `bfloat16`
  - 4 × 2 = 8 字节
  - 32 线程 × 8 字节 = 256 字节 = 128 × 2 字节（128 个 `bfloat16`）
- **RopeVec (int)**：4 字节，一次处理 2 个 `bfloat16`
  - 2 × 2 = 4 字节
  - 32 线程 × 4 字节 = 128 字节 = 64 × 2 字节（64 个 `bfloat16`）

#### 第三步：计算指针和步长

```cpp
const int2* __restrict__ nope_src = 
    reinterpret_cast<const int2*>(k_nope + token_id * k_nope_stride_0 + head_row0 * k_nope_stride_1) + lane_id;

int2* __restrict__ nope_dst = 
    reinterpret_cast<int2*>(k + token_id * k_stride_0 + head_row0 * k_stride_1) + lane_id;

int* __restrict__ rope_dst = 
    reinterpret_cast<int*>(k + token_id * k_stride_0 + head_row0 * k_stride_1 + QK_NOPE_HEAD_DIM) + lane_id;
```

**关键点**：
- **nope_src**：源指针（`k_nope`）
- **nope_dst**：目标指针（`k` 的前半部分）
- **rope_dst**：目标指针（`k` 的后半部分，偏移 `QK_NOPE_HEAD_DIM`）

**步长计算**：
```cpp
const int nope_src_stride_v = (k_nope_stride_1 >> 2);  // int2 覆盖 4 个 bf16
const int nope_dst_stride_v = (k_stride_1 >> 2);
const int rope_dst_stride_v = (k_stride_1 >> 1);      // int 覆盖 2 个 bf16
```

#### 第四步：加载 RoPE 向量（一次）

```cpp
const int* rope_base = reinterpret_cast<const int*>(k_rope + token_id * k_rope_stride_0);
const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);
```

**关键点**：
- **只加载一次**：`k_rope` 是所有 head 共享的
- **非对齐访问**：`ld_na_global_v1` 用于非对齐全局内存访问
- **广播到所有 head**：同一个 `rope_val` 用于所有 head

#### 第五步：循环处理每个 Head（优化）

```cpp
#pragma unroll
for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
  NopeVec next;
  if (i + 1 < HEAD_CHUNK_SIZE) {
    const int2* next_src = nope_src + nope_src_stride_v;
    prefetch_L2(next_src);
    next = ld_na_global_v2(next_src);
  }

  st_na_global_v2(nope_dst, cur);
  st_na_global_v1(rope_dst, rope_val);

  nope_src += nope_src_stride_v;
  nope_dst += nope_dst_stride_v;
  rope_dst += rope_dst_stride_v;

  cur = next;
}
```

**关键优化**：
- **预取（Prefetch）**：在处理当前数据时，预取下一个数据
- **双缓冲**：使用 `cur` 和 `next` 两个缓冲区
- **非对齐访问**：`ld_na_global_v2` 和 `st_na_global_v2` 用于非对齐访问
- **循环展开**：`#pragma unroll` 消除循环开销

**数据流**：
1. **预取下一个**：`prefetch_L2(next_src)`
2. **加载下一个**：`next = ld_na_global_v2(next_src)`
3. **写入当前**：`st_na_global_v2(nope_dst, cur)` 和 `st_na_global_v1(rope_dst, rope_val)`
4. **更新指针**：移动到下一个 head

---

## 🎯 关键设计要点

### 1. Warp 级并行

**设计**：
- 每个 Warp（32 线程）处理一个 head chunk（16 个 heads）
- Warp 内的线程协作处理

**优势**：
- 简化同步（Warp 内隐式同步）
- 高效的 Warp 内通信

### 2. 向量化优化（128 位打包）

**设计**：
- 使用 `int2`（64 位）和 `int`（32 位）打包
- 一次加载/存储多个元素

**优势**：
- 提高内存带宽利用率
- 减少内存访问次数

### 3. 预取优化（Prefetch）

**设计**：
- 在处理当前数据时，预取下一个数据
- 隐藏内存延迟

**优势**：
- 提高缓存命中率
- 减少内存访问延迟

### 4. 非对齐全局内存访问

**设计**：
- 使用 `ld_na_global_v*` 和 `st_na_global_v*`
- 支持非对齐的内存访问

**优势**：
- 更灵活的内存布局
- 在某些情况下性能更好

---

## 📊 性能分析

### 复杂度

**时间复杂度**：
```
O(num_tokens × num_heads × head_dim)
```

**并行化后**：
```
每个 Warp: O(HEAD_CHUNK_SIZE) （HEAD_CHUNK_SIZE = 16）
```

### 内存访问

**读取**：
- `k_nope`: HEAD_CHUNK_SIZE 次（每个 head 一次）
- `k_rope`: 1 次（所有 head 共享）

**写入**：
- `k`: HEAD_CHUNK_SIZE 次（每个 head 一次）

**总访问**：
- 每个 head chunk：HEAD_CHUNK_SIZE + 1 次读取 + HEAD_CHUNK_SIZE 次写入
- 使用向量化和预取，实际延迟更低

---

## 📝 总结

### 核心概念

1. **向量拼接**：将两部分 K 向量拼接
2. **Warp 级并行**：使用 Warp 作为基本单位
3. **向量化**：128 位打包加载/存储
4. **预取优化**：隐藏内存延迟

### 关键优化

- ✅ **Warp 级并行**：简化同步和通信
- ✅ **向量化**：提高内存带宽利用率
- ✅ **预取**：隐藏内存延迟
- ✅ **非对齐访问**：更灵活的内存布局

### 学习价值

Concat MLA 展示了：
- Warp 级并行的设计
- 向量化优化的技巧
- 预取优化的方法
- 非对齐内存访问的使用

---

## 🔗 相关资源

- **MLA 架构**：Multi-Line Attention 技术
- **RoPE**：[04_RoPE算子.md](./04_RoPE算子.md)
- **下一个算子**：参考 README 了解其他算子

