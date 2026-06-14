# concat_mla.cu 文档

## 📋 文件概述

`concat_mla.cu` 实现了 Multi-head Latent Attention (MLA) 相关的数据连接操作。这些操作用于高效地重组和连接注意力机制中的数据，特别是用于 RoPE 和非 RoPE 部分的连接。

## 🎯 主要功能

### 1. concat_mla_k - 连接 K 向量的 RoPE 部分

将非 RoPE 的 K 向量（k_nope）和 RoPE 的 K 向量（k_rope）连接起来。

### 2. concat_mla_absorb_q - 连接 Q 向量

将两个 bfloat16 张量连接起来，用于 MLA 注意力机制。

## 🔬 实现原理

### 常量定义

```cpp
constexpr int NUM_LOCAL_HEADS = 128;        // 本地头数
constexpr int QK_NOPE_HEAD_DIM = 128;       // 非 RoPE 部分维度
constexpr int QK_ROPE_HEAD_DIM = 64;        // RoPE 部分维度
constexpr int K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;  // 总维度 = 192
constexpr int HEAD_CHUNK_SIZE = 16;         // 每个 chunk 的头数
constexpr int NUM_HEAD_CHUNKS = NUM_LOCAL_HEADS / HEAD_CHUNK_SIZE;  // chunk 数量 = 8
```

### concat_mla_k 内核

```cpp
__global__ void concat_mla_k_kernel(
    nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ k_nope,
    const nv_bfloat16* __restrict__ k_rope,
    // ...
) {
  // 1. 计算 warp 和 token 索引
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
  const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
  const int lane_id = get_lane_id();
  
  // 2. 定义向量类型
  using NopeVec = int2;  // 8 字节/线程，32 线程 = 256 字节/行
  using RopeVec = int;   // 4 字节/线程，32 线程 = 128 字节/行
  
  // 3. 计算源地址和目标地址
  const int head_row0 = head_chunk_id * HEAD_CHUNK_SIZE;
  const int2* nope_src = ...;
  int2* nope_dst = ...;
  int* rope_dst = ...;
  
  // 4. 预取和加载
  const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);
  prefetch_L2(nope_src);
  NopeVec cur = ld_na_global_v2(nope_src);
  
  // 5. 循环处理每个 head
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    // 加载下一个（预取）
    NopeVec next = ...;
    
    // 写入当前
    st_na_global_v2(nope_dst, cur);      // 写入非 RoPE 部分
    st_na_global_v1(rope_dst, rope_val); // 写入 RoPE 部分
    
    // 更新指针
    nope_src += nope_src_stride_v;
    nope_dst += nope_dst_stride_v;
    rope_dst += rope_dst_stride_v;
    
    cur = next;
  }
}
```

**关键优化**：
- **Warp 级处理**：每个 warp 处理一个 head chunk
- **向量化访问**：使用 `int2` 和 `int` 向量类型
- **预取优化**：使用 L2 预取隐藏内存延迟
- **双缓冲**：在处理当前时预取下一个

### concat_mla_absorb_q 内核

```cpp
__global__ void concat_mla_absorb_q_kernel(
    nv_bfloat16* a,
    nv_bfloat16* b,
    nv_bfloat16* out,
    // ...
) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = get_lane_id();
  
  const int idx_0 = flat_warp_id / dim_1;
  const int idx_1 = flat_warp_id % dim_1;
  
  // 定义向量类型
  using ABufType = int4;  // 16 字节/线程
  using BBufType = int;   // 4 字节/线程
  
  // 加载 B（较小的张量）
  BBufType b_buf;
  const BBufType* base_addr = reinterpret_cast<BBufType*>(b + idx_0 * b_stride_0 + idx_1 * b_stride_1);
  b_buf = *(base_addr + lane_id);
  
  // 加载 A（较大的张量，需要多次迭代）
  ABufType a_buf[A_NUM_UNROLL];
  for (int i = 0; i < A_NUM_UNROLL; ++i) {
    const ABufType* base_addr = reinterpret_cast<ABufType*>(a + idx_0 * a_stride_0 + idx_1 * a_stride_1);
    a_buf[i] = *(base_addr + i * 32 + lane_id);
  }
  
  // 写入输出：先写 B，再写 A
  {
    BBufType* base_addr = reinterpret_cast<BBufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1 + A_LAST_DIM);
    *(base_addr + lane_id) = b_buf;
  }
  
  for (int i = 0; i < A_NUM_UNROLL; ++i) {
    ABufType* base_addr = reinterpret_cast<ABufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1);
    *(base_addr + i * 32 + lane_id) = a_buf[i];
  }
}
```

**内存布局**：
```
输出 = [A (512 dim) | B (64 dim)]
总长度 = 576 dim
```

## 💡 算法详解

### 数据流

```
输入 k_nope: (num_tokens, 128 heads, 128 dim)
输入 k_rope: (num_tokens, 1, 64 dim)
输出 k:      (num_tokens, 128 heads, 192 dim)
                     ↓
        [k_nope (128 dim) | k_rope (64 dim)]
```

### 处理模式

1. **Chunk 处理**：将 128 个头分成 8 个 chunk，每个 chunk 16 个头
2. **Warp 协作**：每个 warp（32 线程）处理一个 chunk
3. **向量化**：使用 `int2` (8 字节) 和 `int` (4 字节) 进行向量化访问
4. **预取**：在处理当前行时预取下一行数据

## ⚡ 性能优化技巧

### 1. 向量化内存访问

```cpp
using NopeVec = int2;  // 8 字节/线程
using RopeVec = int;   // 4 字节/线程
```

- 每次访问 8 字节或 4 字节
- 减少内存事务数量
- 提高带宽利用率

### 2. L2 预取

```cpp
prefetch_L2(nope_src);  // 预取到 L2 缓存
```

- 提前加载数据到 L2 缓存
- 隐藏内存访问延迟
- 提高缓存命中率

### 3. 双缓冲

```cpp
NopeVec cur = ld_na_global_v2(nope_src);
// 处理 cur 的同时
NopeVec next = ld_na_global_v2(next_src);  // 预取下一个
```

- 重叠计算和内存访问
- 最大化内存带宽利用率

### 4. Warp 级协作

- 32 个线程协作处理
- 充分利用 SIMT 特性
- 减少 warp 调度开销

## 🔍 接口说明

### concat_mla_k

```cpp
void concat_mla_k(
    at::Tensor k,        // 输出: (num_tokens, 128, 192)
    at::Tensor k_nope,   // 输入: (num_tokens, 128, 128)
    at::Tensor k_rope    // 输入: (num_tokens, 1, 64)
)
```

**维度要求**：
- 所有张量必须是 bfloat16 类型
- 最后一维必须是连续的（stride(2) == 1）
- 必须是 16 字节对齐

### concat_mla_absorb_q

```cpp
void concat_mla_absorb_q(
    at::Tensor a,   // 输入: (*, *, 512)
    at::Tensor b,   // 输入: (*, *, 64)
    at::Tensor out  // 输出: (*, *, 576)
)
```

**维度要求**：
- 所有张量必须是 bfloat16 类型
- a 和 b 的前两维必须匹配
- 最后一维必须是连续的

## 📚 应用场景

### MLA 注意力机制中的使用

```python
# 在 MLA 注意力中
# 1. 分离 RoPE 和非 RoPE 部分
k_nope = compute_k_nope(...)  # 非 RoPE 部分
k_rope = compute_k_rope(...)  # RoPE 部分

# 2. 连接两部分
k = torch.empty(num_tokens, 128, 192, dtype=torch.bfloat16, device='cuda')
sgl_kernel.concat_mla_k(k, k_nope, k_rope)

# 3. 用于后续的注意力计算
scores = compute_attention(q, k, v)
```

## 🔗 相关文件

- `csrc/elementwise/rope.cu` - RoPE 实现
- `csrc/elementwise/utils.cuh` - 工具函数（ld_na_global_v1/v2, st_na_global_v1/v2）
- `csrc/attention/` - 注意力机制实现

## 📖 参考资料

- **MLA 论文**：Multi-head Latent Attention
- **DeepEP 项目**：工具函数来源

