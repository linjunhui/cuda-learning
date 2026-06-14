# Level 1.1: Elementwise 逐元素操作

## 📋 模块概述

Elementwise（逐元素操作）模块是 SGL Kernel 中最基础的部分，实现了各种作用于张量元素的简单操作。这些操作虽然简单，但在大语言模型推理中频繁使用，其性能优化直接影响整体推理速度。

**难度等级**：⭐ 入门级  
**重要程度**：⭐⭐⭐⭐⭐ 核心模块，频繁使用

## 📂 文件结构

```
csrc/elementwise/
├── activation.cu           # 激活函数（SiLU, GELU等）
├── cast.cu                 # 类型转换
├── concat_mla.cu          # MLA 连接操作
├── copy.cu                 # 复制操作
├── fused_add_rms_norm_kernel.cu  # 融合的加法 + RMS 归一化
├── rope.cu                 # 旋转位置编码（RoPE）
├── topk.cu                 # Top-K 选择
├── pos_enc.cuh            # 位置编码头文件
└── utils.cuh              # 工具函数头文件
```

## 🎯 算子来源与理论基础

### 1. 激活函数 (Activation Functions)

**来源**：FlashInfer 项目
- **原始实现**：`flashinfer/activation.cuh`
- **适配说明**：从 FlashInfer 适配而来，支持 ROCm 平台

**包含的激活函数**：
- **SiLU (Swish)**：`x / (1 + exp(-x))` - 广泛应用于现代 LLM（如 Llama, Mistral）
- **GELU (Gaussian Error Linear Unit)**：`x * 0.5 * (1 + erf(x/√2))` - BERT, GPT 系列使用
- **GELU Quick**：`x * sigmoid(1.702 * x)` - GELU 的快速近似
- **GELU Tanh**：使用 tanh 的 GELU 近似版本

### 2. RMSNorm（Root Mean Square Layer Normalization）

**来源**：FlashInfer + 自定义融合
- **原始论文**：Root Mean Square Layer Normalization (2019)
- **核心思想**：相比 LayerNorm，RMSNorm 去除了均值中心化，只做缩放

**数学公式**：
```
RMSNorm(x) = (x / RMS(x)) * γ
其中 RMS(x) = sqrt(mean(x²) + ε)
```

### 3. RoPE (Rotary Position Embedding)

**来源**：FlashInfer 实现
- **原始论文**：RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
- **核心思想**：通过旋转矩阵将位置信息编码到注意力机制中

**数学原理**：
对于位置为 m 的查询向量 q 和位置为 n 的键向量 k：
```
RoPE(q, m) = q * R_θ^m
RoPE(k, n) = k * R_θ^n
其中 R_θ = [cos(θ) -sin(θ); sin(θ) cos(θ)]
```

### 4. Top-K 选择

**来源**：TileLang 项目
- **原始实现**：`tile-ai/tilelang/examples/deepseek_v32/topk_selector.py`
- **适配说明**：从 TileLang 适配到纯 CUDA，优化性能并修复潜在的内存访问问题

**算法**：使用 Radix Sort 的变体进行高效的 Top-K 选择

## 🔬 算法原理详解

### 1. 融合激活函数（Fused Activation）

**问题**：在 LLM 的前馈网络（FFN）中，常见模式是：
```python
# 未融合版本（需要两次内存访问）
x1 = silu(x[:, :d])
x2 = x[:, d:]
output = x1 * x2  # 逐元素相乘
```

**优化思路**：将激活和乘法融合为单个内核，减少内存访问。

**实现细节**：
- 使用向量化加载（128 位对齐）
- 每个线程处理多个元素
- 模板化设计支持不同数据类型（FP16, BF16, FP32）

```cpp
template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}
```

### 2. 融合 RMSNorm + Add

**问题**：残差连接后通常紧跟归一化：
```python
# 标准实现
residual = input + residual
output = rmsnorm(residual, weight)
```

**优化思路**：在计算 RMS 时同时进行加法，避免中间结果写回全局内存。

**关键优化点**：
1. 单次内存扫描完成加法和 RMS 计算
2. 使用 Welford's algorithm 的变体计算方差
3. 支持 PDL (Programmatic Dependent Launch) 优化

### 3. Radix Sort Top-K

**算法步骤**：
1. **8位粗分直方图**：将 float32 转换为 uint8，建立直方图
2. **累积求和**：找到 Top-K 所在的 bin
3. **细分处理**：对边界 bin 进行 32 位细粒度排序
4. **多轮精化**：4 轮 8 位 radix pass 精确定位

**性能优势**：
- 时间复杂度：O(n) 而非 O(n log n)
- 使用共享内存减少全局内存访问
- 支持变长序列的批量处理

## 💡 应用场景

### 1. 激活函数在 LLM 中的使用

```python
# FFN 模块中的典型用法
# MLP(x) = (SiLU(xW1 + b1) ⊙ (xW2 + b2)) * W3
# 其中 ⊙ 表示逐元素相乘

# 在 SGLang 中：
input_split = input.view(batch, seq_len, -1, 2, hidden_size)
output = sgl_kernel.silu_and_mul(input_split[..., 0, :], input_split[..., 1, :])
```

### 2. RMSNorm 在 Transformer 中的应用

现代 LLM（如 Llama, Gemma）广泛使用 RMSNorm 替代 LayerNorm：

```python
# Pre-norm 架构
x = input + attention(rmsnorm(input))
x = x + ffn(rmsnorm(x))
```

### 3. RoPE 在注意力机制中的应用

RoPE 是现代 LLM 位置编码的主流方案：

```python
# 在计算 QK^T 之前应用 RoPE
q_rope = apply_rope(q, cos_cache, sin_cache, position_ids)
k_rope = apply_rope(k, cos_cache, sin_cache, position_ids)
scores = q_rope @ k_rope.T
```

### 4. Top-K 在解码中的应用

用于选择候选 token 进行后续处理：

```python
# 从 logits 中选择 Top-K 候选
topk_indices = fast_topk(logits, k=top_k)
# 用于 PagedAttention 的页面表构建
```

## 💻 代码实现分析

### 实现 1: 融合 SiLU 和乘法

```56:105:csrc/elementwise/activation.cu
template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}

// ... 其他激活函数定义 ...

void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}
```

**关键设计点**：
1. **模板化**：支持 FP16, BF16, FP32
2. **向量化**：使用 128 位（16 字节）对齐的向量化操作
3. **块大小自适应**：根据向量大小调整线程块大小

### 实现 2: 融合 RMSNorm

```24:59:csrc/elementwise/fused_add_rms_norm_kernel.cu
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

**关键特性**：
- **融合操作**：加法和归一化在单次内核中完成
- **PDL 支持**：`enable_pdl` 参数启用程序化依赖启动优化
- **类型分发**：自动选择适合数据类型的内核

### 实现 3: Top-K Radix Sort 核心

```76:122:csrc/elementwise/topk.cu
__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();
```

**算法精髓**：
1. **8 位粗分**：将 float 映射到 256 个 bin
2. **直方图累积**：快速定位 Top-K 所在的 bin 范围
3. **共享内存优化**：所有中间结果存储在共享内存中

## ⚡ 性能优化技巧

### 1. 向量化内存访问

```cpp
uint32_t vec_size = 16 / sizeof(c_type);  // 计算向量大小
// 使用 float4, int4 等向量类型进行对齐访问
```

### 2. 共享内存优化

- Top-K 使用 128KB 共享内存存储中间结果
- 避免全局内存访问的延迟

### 3. 融合减少内存访问

- 激活+乘法：减少一次内存写回
- RMSNorm+Add：减少中间张量的分配

### 4. Warp 级操作

- 使用 warp shuffle 进行线程间通信
- 减少共享内存的使用

## 🔍 关键接口说明

| 接口名称 | 功能 | 输入维度 | 输出维度 |
|---------|------|---------|---------|
| `silu_and_mul` | SiLU 激活并逐元素相乘 | `(B, 2*D)` | `(B, D)` |
| `gelu_and_mul` | GELU 激活并逐元素相乘 | `(B, 2*D)` | `(B, D)` |
| `fused_add_rmsnorm` | 融合加法和 RMS 归一化 | `input: (B, D), residual: (B, D), weight: (D)` | `(B, D)` (inplace) |
| `apply_rope_pos_ids_cos_sin_cache` | 应用旋转位置编码 | `q: (N, H, D), k: (N, H, D)` | `q_rope: (N, H, D), k_rope: (N, H, D)` |
| `fast_topk` | 快速 Top-K 选择 | `score: (B, L)` | `indices: (B, K)` |

## 📚 参考资料

1. **RoPE 论文**：RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
2. **RMSNorm 论文**：Root Mean Square Layer Normalization (2019)
3. **FlashInfer 项目**：https://github.com/flashinfer-ai/flashinfer
4. **TileLang 项目**：https://github.com/tile-ai/tilelang

## 🎓 学习建议

1. **先理解数学原理**：理解每个操作的数学定义
2. **查看单元测试**：`tests/` 目录下有详细的测试用例
3. **运行基准测试**：了解性能特征
4. **对比标准实现**：对比 PyTorch 原生实现的性能差异

---

**下一模块**：[1.2 Memory 内存管理](./02-memory.md)

