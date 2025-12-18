# CUDA Softmax 实现详解

## 目录

1. [Softmax 基础](#softmax-基础)
2. [CUDA 实现架构](#cuda-实现架构)
3. [核心算法](#核心算法)
4. [优化技术](#优化技术)
5. [实现版本对比](#实现版本对比)
6. [性能分析](#性能分析)
7. [使用指南](#使用指南)

---

## Softmax 基础

### 数学定义

Softmax 函数将任意实数值向量转换为概率分布：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

其中：
- $x_i$ 是输入向量的第 $i$ 个元素
- $n$ 是向量的长度
- 输出满足：$\sum_{i=1}^{n} \text{softmax}(x_i) = 1$

### 数值稳定性问题

直接计算 $e^{x_i}$ 可能导致：
1. **溢出**：当 $x_i$ 很大时，$e^{x_i}$ 可能超出浮点数表示范围
2. **精度损失**：大值之间的差异可能被舍入误差掩盖

### Safe Softmax 技巧

使用"减最大值"技巧提高数值稳定性：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$

其中 $m = \max(x_1, x_2, ..., x_n)$。

**优势**：
- 避免溢出：$e^{x_i - m} \leq 1$，因为 $x_i - m \leq 0$
- 保持数学等价性：分子分母同时除以 $e^m$，结果不变
- 提高数值精度

---

## CUDA 实现架构

### 并行化策略：Per-Token Softmax

本实现采用 **Per-Token** 并行化策略：

- **输入格式**：`(S, H)` 形状的张量
  - `S`：序列长度（sequence length）
  - `H`：头大小/KV长度（head size/key-value length）

- **Grid/Block 配置**：
  - `grid(S)`：每个 token 对应一个 thread block
  - `block(H)`：每个 block 的线程数等于头大小 H

- **优势**：
  - 每个 token 的 softmax 计算完全独立，无数据依赖
  - 适合 Transformer 等模型的注意力机制
  - 可以充分利用 GPU 的并行计算能力

### 内存访问模式

```
输入: x[S, H]         输出: y[S, H]
     ┌─────┐                ┌─────┐
     │Token│                │Token│
     │  0  │ ────────────>  │  0  │
     ├─────┤                ├─────┤
     │Token│                │Token│
     │  1  │ ────────────>  │  1  │
     ├─────┤                ├─────┤
     │ ... │                │ ... │
     └─────┘                └─────┘

每个 Block 处理一个 Token 的 Softmax
```

---

## 核心算法

### 1. Warp 级归约（Warp Reduce）

#### 求和归约（Sum Reduction）

使用 **Butterfly 模式**在 warp 内进行归约：

```cuda
template <const int kWarpSize = WARP_SIZE>
__device__ float warp_reduce_sum_f32(float val) {
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}
```

**工作原理**：
- 使用 `__shfl_xor_sync` 在 warp 内交换数据
- 经过 $\log_2(32) = 5$ 次迭代后，所有线程都得到总和
- **优势**：无需共享内存，延迟低，带宽高

#### 最大值归约（Max Reduction）

类似地，使用相同的模式进行最大值归约：

```cuda
template <const int kWarpSize = WARP_SIZE>
__device__ float warp_reduce_max_f32(float val) {
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}
```

### 2. Block 级归约（Block Reduce）

采用**两级归约策略**：

1. **第一级**：每个 warp 内部归约（使用 shuffle）
2. **第二级**：将各 warp 的结果写入共享内存，然后在第一个 warp 内再次归约

```cuda
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  // 第一级：warp 内归约
  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();

  // 第二级：在第一个 warp 内归约
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);
  value = __shfl_sync(0xffffffff, value, 0, 32);  // 广播结果
  return value;
}
```

### 3. Online Softmax 算法

**参考论文**：[Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)

#### 核心思想

使用 `MD` 结构体同时跟踪最大值和归一化因子：

```cuda
struct MD {
  float m;  // 最大值
  float d;  // 归一化因子 sum(exp(x[i] - m))
};
```

#### 合并规则

当合并两个 MD 值 $(m_1, d_1)$ 和 $(m_2, d_2)$ 时：

- $m_{new} = \max(m_1, m_2)$
- $d_{new} = d_{bigger} + d_{smaller} \cdot e^{m_{smaller} - m_{bigger}}$

**优势**：
- 单次遍历即可同时计算最大值和归一化因子
- 数值稳定，避免溢出
- 适合流式处理场景

---

## 优化技术

### 1. 向量化内存访问（Vectorized Memory Access）

#### FP32x4 向量化

使用 `float4` 类型实现 128 位对齐的内存访问：

```cuda
// 向量化加载：一次加载 4 个 float（128 位）
float4 reg_x = FLOAT4(x[idx]);

// 向量化存储：一次存储 4 个 float（128 位）
FLOAT4(y[idx]) = reg_y;
```

**优势**：
- 减少内存事务数量（1 次事务 vs 4 次事务）
- 提高内存带宽利用率
- 减少线程数量，降低调度开销

#### FP16x8 打包

使用寄存器数组和 128 位对齐访问：

```cuda
half pack_x[8], pack_y[8];  // 8 * 16 bits = 128 bits

// 重新解释为 float4，一次加载 128 位
LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
```

**优势**：
- 最大化内存带宽利用率
- 适合大尺寸的 head size（H >= 256）
- 减少线程数量（线程数 = H / 8）

### 2. 混合精度计算（Mixed Precision）

#### FP16 输入/输出，FP32 中间计算

```cuda
// 输入：FP16
half *x, *y;

// 中间计算：FP32
float val = __half2float(x[idx]);
float exp_val = expf(val - max_val);
float result = exp_val / exp_sum;

// 输出：FP16
y[idx] = __float2half_rn(result);
```

**优势**：
- 减少内存带宽（FP16 占用内存是 FP32 的一半）
- 保持计算精度（使用 FP32 进行中间计算）
- 适合深度学习推理场景

### 3. 循环展开（Loop Unrolling）

使用 `#pragma unroll` 指令提示编译器展开循环：

```cuda
#pragma unroll
for (int i = 0; i < 8; ++i) {
  max_val = fmaxf(__half2float(pack_x[i]), max_val);
}
```

**优势**：
- 减少循环控制开销
- 提高指令级并行度（ILP）
- 更好的寄存器利用

### 4. 共享内存优化

- 使用共享内存存储中间归约结果
- 最小化共享内存使用量（只存储每个 warp 的结果）
- 使用 `__syncthreads()` 确保数据一致性

---

## 实现版本对比

### 版本列表

| 版本 | 数据类型 | 向量化 | 数值稳定性 | 适用场景 |
|------|---------|--------|-----------|---------|
| `softmax_f32_per_token` | FP32 | 否 | ❌ | 小值输入 |
| `softmax_f32x4_per_token` | FP32 | 4x | ❌ | 小值输入，需要高性能 |
| `safe_softmax_f32_per_token` | FP32 | 否 | ✅ | 通用场景 |
| `safe_softmax_f32x4_per_token` | FP32 | 4x | ✅ | 通用场景，需要高性能 |
| `safe_softmax_f16_f32_per_token` | FP16 | 否 | ✅ | 内存受限场景 |
| `safe_softmax_f16x2_f32_per_token` | FP16 | 2x | ✅ | 内存受限，中等性能 |
| `safe_softmax_f16x8_pack_f32_per_token` | FP16 | 8x | ✅ | 内存受限，高性能 |
| `online_safe_softmax_f32_per_token` | FP32 | 否 | ✅ | 流式处理 |
| `online_safe_softmax_f32x4_pack_per_token` | FP32 | 4x | ✅ | 流式处理，高性能 |

### 选择建议

1. **小值输入（无溢出风险）**：
   - 优先使用 `softmax_f32x4_per_token`（性能最佳）

2. **通用场景（需要数值稳定性）**：
   - `H <= 256`：`safe_softmax_f32_per_token`
   - `256 < H <= 1024`：`safe_softmax_f32x4_per_token`
   - `H > 1024`：`safe_softmax_f32x4_per_token` 或 `online_safe_softmax_f32x4_pack_per_token`

3. **内存受限场景**：
   - `H <= 256`：`safe_softmax_f16_f32_per_token`
   - `256 < H <= 512`：`safe_softmax_f16x2_f32_per_token`
   - `H > 512`：`safe_softmax_f16x8_pack_f32_per_token`

4. **流式处理场景**：
   - 使用 `online_safe_softmax_f32_per_token` 或 `online_safe_softmax_f32x4_pack_per_token`

---

## 性能分析

### 性能测试结果

基于测试配置：`S=4096`（序列长度），不同 `H`（头大小）

| H | 最快版本 | 时间 (ms) | 相对 PyTorch 加速 |
|---|---------|----------|------------------|
| 256 | `f32x4(safe+online)` | 0.0041 | ~1.6x |
| 512 | `f32x4(safe+online)` | 0.0055 | ~1.2x |
| 1024 | `f32x4(safe+online)` | 0.0093 | ~1.3x |
| 2048 | `f32x4(safe+online)` | 0.0178 | ~3.8x |
| 4096 | `f16x8packf32(safe)` | 0.0223 | ~8.4x |
| 8192 | `f16x8packf32(safe)` | 0.1899 | ~1.0x |

### 性能优化要点

1. **向量化访问**：FP32x4 和 FP16x8 版本显著提升性能
2. **减少线程数量**：向量化版本减少线程数，降低调度开销
3. **内存带宽**：128 位对齐访问最大化内存带宽利用率
4. **数值稳定性**：Safe Softmax 版本在保证精度的同时保持高性能

---

## 使用指南

### 编译

```bash
# 只测试 Ada 架构（不指定默认编译所有架构，耗时较长）
export TORCH_CUDA_ARCH_LIST=Ada
python3 softmax.py
```

### Python 调用示例

```python
import torch
from torch.utils.cpp_extension import load

# 加载 CUDA 扩展
lib = load(
    name="softmax_lib",
    sources=["softmax.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# 准备数据
S, H = 4096, 256  # 序列长度，头大小
x = torch.randn((S, H), device="cuda").float().contiguous()
y = torch.zeros_like(x)

# 调用 CUDA kernel
lib.safe_softmax_f32x4_per_token(x, y)

# 验证结果
expected = torch.softmax(x, dim=1)
print(f"Max error: {(y - expected).abs().max().item()}")
```

### 注意事项

1. **输入要求**：
   - 输入张量必须是连续的（`.contiguous()`）
   - 数据类型必须匹配（FP32 或 FP16）

2. **形状限制**：
   - `H` 必须是 2 的幂次（32, 64, 128, 256, 512, 1024, ...）
   - `H <= 1024`（对于非向量化版本）

3. **数值精度**：
   - Safe Softmax 版本提供更好的数值稳定性
   - FP16 版本在精度和性能之间取得平衡

---

## 总结

本 CUDA Softmax 实现提供了多种优化版本，适用于不同的应用场景：

- **基础版本**：简单直接，适合小值输入
- **Safe 版本**：数值稳定，适合通用场景
- **向量化版本**：高性能，适合大规模计算
- **混合精度版本**：内存高效，适合资源受限场景
- **Online 版本**：单次遍历，适合流式处理

通过合理选择版本，可以在精度、性能和内存使用之间取得最佳平衡。
