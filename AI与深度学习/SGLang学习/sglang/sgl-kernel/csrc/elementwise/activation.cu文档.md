# activation.cu 文档

## 📋 文件概述

`activation.cu` 实现了多种激活函数及其融合操作，主要用于大语言模型的前馈网络（FFN）中。这些操作经过高度优化，支持向量化和融合执行以提高性能。

**文件来源**：适配自 FlashInfer 项目  
**原始代码**：https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/csrc/activation.cu

## 🎯 主要功能

### 1. 激活函数实现

#### SiLU (Swish)
```cpp
template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}
```

**数学公式**：`SiLU(x) = x / (1 + exp(-x))`  
**应用**：Llama、Mistral 等现代 LLM 的 FFN 中广泛使用

#### GELU (Gaussian Error Linear Unit)
```cpp
template <typename T>
__device__ __forceinline__ T gelu(const T& x) {
  constexpr float kAlpha = M_SQRT1_2;
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
}
```

**数学公式**：`GELU(x) = x * 0.5 * (1 + erf(x/√2))`  
**应用**：BERT、GPT 系列模型使用

#### GELU Quick（快速近似）
```cpp
template <typename T>
__device__ __forceinline__ T gelu_quick_act(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val * 1.702f)));
}
```

**数学公式**：`GELU_Quick(x) = x * sigmoid(1.702 * x)`  
**优势**：比标准 GELU 更快，精度略低

#### GELU Tanh（Tanh 近似）
```cpp
template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  float f32_val = detail::to_f32(x);
  const float cdf = 0.5f * (1.0f + tanhf((kBeta * (f32_val + kAlpha * f32_val * f32_val * f32_val))));
  return detail::from_f32<T>(f32_val * cdf);
}
```

**数学公式**：使用 tanh 的三次多项式近似

### 2. 融合操作

#### silu_and_mul
```cpp
void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;  // 输入被分成两半
  // 执行：out = SiLU(input[:, :d]) * input[:, d:]
}
```

**优化原理**：
- **未融合版本**：需要 3 次内存访问
  1. 读取 `input[:, :d]`
  2. 应用 SiLU 后写入中间结果
  3. 读取 `input[:, d:]` 并相乘
  
- **融合版本**：只需要 2 次内存访问
  1. 读取整个 `input`
  2. 计算后直接写入 `out`

**性能提升**：减少约 33% 的内存带宽使用

#### gelu_and_mul / gelu_tanh_and_mul
类似 SiLU，融合 GELU 激活和乘法操作。

## 💻 代码实现详解

### 类型转换辅助函数

```cpp
namespace detail {
template <typename T>
__device__ __forceinline__ float to_f32(const T& x) {
  return static_cast<float>(x);  // 统一转换为 float32 计算
}

template <typename T>
__device__ __forceinline__ T from_f32(float f32) {
  return static_cast<T>(f32);  // 转换回原始类型
}
}
```

**设计原因**：
- GPU 上的数学函数（如 `expf`, `erf`）通常需要 float32 精度
- 统一在 float32 上计算，然后转换回原始类型
- 保证精度和性能的平衡

### 内核启动参数

```cpp
void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);  // 每个 token 一个 block

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);  // 向量化大小
    dim3 block(std::min(d / vec_size, 1024U));  // 自适应块大小
    
    flashinfer::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(...);
  });
}
```

**关键点**：
- **Grid 维度**：`num_tokens` - 每个 token 独立处理
- **Block 维度**：根据向量化大小自适应
- **向量化**：使用 16 字节（128 位）对齐的向量类型

## 🔍 接口说明

| 函数名 | 输入维度 | 输出维度 | 说明 |
|--------|---------|---------|------|
| `silu_and_mul` | `(B, 2*D)` | `(B, D)` | SiLU 激活前半部分，与后半部分相乘 |
| `gelu_and_mul` | `(B, 2*D)` | `(B, D)` | GELU 激活前半部分，与后半部分相乘 |
| `gelu_tanh_and_mul` | `(B, 2*D)` | `(B, D)` | GELU Tanh 激活前半部分，与后半部分相乘 |
| `gelu_quick` | `(B, D)` | `(B, D)` | GELU Quick 激活（仅 ROCm） |

## ⚡ 性能优化技巧

### 1. 向量化内存访问
- 使用 `float4`、`int4` 等向量类型
- 128 位对齐以最大化内存带宽

### 2. 融合操作
- 将激活和乘法融合为单个内核
- 减少中间结果的写回

### 3. 模板特化
- 为不同数据类型生成优化的代码
- 编译时多态，零运行时开销

### 4. 类型统一
- 所有计算在 float32 上进行
- 只在实际存储时转换回原始类型

## 📚 在 LLM 中的应用

### FFN 模块中的典型用法

```python
# MLP(x) = (SiLU(xW1 + b1) ⊙ (xW2 + b2)) * W3
# 其中 ⊙ 表示逐元素相乘

# 在 Transformer 中：
def ffn(x):
    # x: (batch, seq_len, hidden_size)
    gate_up = linear_gate_up(x)  # (batch, seq_len, 2 * ffn_size)
    
    # 使用融合操作
    activated = sgl_kernel.silu_and_mul(gate_up)  # (batch, seq_len, ffn_size)
    
    # 再经过输出投影
    output = linear_out(activated)
    return output
```

**优势**：
- 减少内存带宽压力
- 提高缓存利用率
- 降低延迟

## 🔗 相关文件

- `csrc/common_extension.cc` - PyTorch 扩展注册
- `include/sgl_kernel_ops.h` - 函数声明
- FlashInfer 的 `activation.cuh` - 实际内核实现

## 📖 参考资料

1. **FlashInfer 项目**：https://github.com/flashinfer-ai/flashinfer
2. **SiLU 激活函数**：Swish: a Self-Gated Activation Function (2017)
3. **GELU 激活函数**：Gaussian Error Linear Units (GELUs) (2016)

