# Activation 算子详解（SiLU / GELU）

## 📖 算子概述

**Activation（激活函数）** 是神经网络中最基础的算子之一。在 LLM 中，最常用的激活函数包括：

- **SiLU (Swish)**：`silu(x) = x / (1 + exp(-x))`
- **GELU**：`gelu(x) = x * 0.5 * (1 + erf(x / √2))`
- **GELU-Tanh**：GELU 的近似实现
- **GELU-Quick**：`gelu_quick(x) = x * sigmoid(1.702 * x)`

**用途**：
- Transformer 中的 FFN（前馈网络）
- 每个 token 的激活函数计算
- 大量并行计算，适合 GPU

**特点**：
- 逐元素操作（element-wise）
- 每个元素独立计算
- 计算简单但计算量大

---

## 🔢 公式与算法

### 1. SiLU (Swish) 激活函数

#### 数学公式

```
SiLU(x) = x / (1 + exp(-x))
```

**等价形式**：
```
SiLU(x) = x * sigmoid(x)
SiLU(x) = x * (1 / (1 + exp(-x)))
```

#### 函数图像

```
      |
   y  |      /
      |     /
  0.5 |----/-------
      |   /
      |  /
      | /
  0.0 +----------------> x
      -3  -2  -1   0   1   2   3
```

**特点**：
- 平滑、非单调
- 在负值区域也有输出（与 ReLU 不同）
- 零值处导数为 0.5

#### 导数

```
d/dx SiLU(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

### 2. GELU 激活函数

#### 数学公式

```
GELU(x) = x * 0.5 * (1 + erf(x / √2))
```

其中 `erf` 是误差函数：
```
erf(x) = (2/√π) ∫[0 to x] e^(-t²) dt
```

#### 近似公式

**GELU-Tanh**：
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**GELU-Quick**：
```
GELU(x) ≈ x * sigmoid(1.702 * x)
```

### 3. SiLU and Mul（融合操作）

在 LLM 中，经常需要计算：
```
out = silu(x[:d]) * x[d:]
```

**含义**：
- 输入数组前一半：应用 SiLU
- 输入数组后一半：保持不变
- 结果：逐元素相乘

**用途**：
- **SwiGLU** 激活函数
- 减少 kernel 启动次数
- 提高缓存利用率

---

## 🧠 算法原理

### 基本算法

对于逐元素激活函数：

```
for each element x in input:
    output = activation_function(x)
```

**并行化**：
- 每个线程处理一个或多个元素
- 元素间无依赖，完全并行
- 使用 Grid-Stride Loop 处理任意大小

### 数值稳定性

**问题**：计算 `exp(-x)` 时，如果 `x` 很大（如 `x > 88`），`exp(-x)` 会下溢到 0。

**解决方案**：
- 使用 `expf`（单精度版本）
- 对于 `half` 类型，先转 `float` 计算，再转回 `half`
- 避免精度损失

### 类型转换技巧

```cpp
float f32_val = detail::to_f32(x);  // 转为 float32 计算
float result = f32_val / (1.0f + expf(-f32_val));
return detail::from_f32<T>(result);  // 转回原类型
```

**为什么这样？**
- **精度**：`float32` 计算精度更高
- **范围**：`float32` 的数值范围更大
- **硬件**：现代 GPU 的 `float32` 运算更快

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/elementwise/activation.cu`

### 1. SiLU 函数实现

```56:60:SGLang学习/sglang/sgl-kernel/csrc/elementwise/activation.cu
template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}
```

#### 代码解析

**第 1 行：模板函数**
```cpp
template <typename T>
__device__ __forceinline__ T silu(const T& x)
```
- **模板**：支持多种类型（`half`、`float`、`bfloat16`）
- **`__device__`**：在 GPU 上执行
- **`__forceinline__`**：强制内联，减少函数调用开销

**第 2 行：类型转换**
```cpp
float f32_val = detail::to_f32(x);
```
- 将输入转为 `float32`
- `detail::to_f32` 处理不同类型的转换

**第 3 行：计算**
```cpp
return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
```
- **`expf(-f32_val)`**：计算 `exp(-x)`，使用单精度版本
- **`1.0f + ...`**：计算 `1 + exp(-x)`
- **`f32_val / ...`**：计算 `x / (1 + exp(-x))`
- **`detail::from_f32<T>`**：转回原类型

### 2. GELU 函数实现

```62:67:SGLang学习/sglang/sgl-kernel/csrc/elementwise/activation.cu
template <typename T>
__device__ __forceinline__ T gelu(const T& x) {
  constexpr float kAlpha = M_SQRT1_2;
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
}
```

**关键点**：
- **`M_SQRT1_2`**：`1/√2 = 0.7071067811865476`
- **`erf`**：误差函数，CUDA 内置函数
- 公式：`x * 0.5 * (1 + erf(x / √2))`

### 3. GELU-Tanh 实现（更快）

```76:83:SGLang学习/sglang/sgl-kernel/csrc/elementwise/activation.cu
template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  float f32_val = detail::to_f32(x);
  const float cdf = 0.5f * (1.0f + tanhf((kBeta * (f32_val + kAlpha * f32_val * f32_val * f32_val))));
  return detail::from_f32<T>(f32_val * cdf);
}
```

**近似公式**：
```
cdf = 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
GELU(x) ≈ x * cdf
```

**优势**：
- `tanh` 比 `erf` 计算更快
- 精度足够（误差 < 0.003）

### 4. SiLU and Mul（融合操作）

```85:104:SGLang学习/sglang/sgl-kernel/csrc/elementwise/activation.cu
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

#### 代码解析

**配置参数**：
```cpp
int d = input.size(-1) / 2;              // 一半维度
int64_t num_tokens = input.numel() / input.size(-1);  // token 数量
dim3 grid(num_tokens);                    // 每个 token 一个 block
```

**向量化计算**：
```cpp
uint32_t vec_size = 16 / sizeof(c_type);  // 向量大小
dim3 block(std::min(d / vec_size, 1024U));  // 根据向量大小调整 block
```

**向量化示例**：
- `half` 类型：`sizeof(half) = 2` → `vec_size = 16/2 = 8`
- `float` 类型：`sizeof(float) = 4` → `vec_size = 16/4 = 4`
- 一次加载/存储 8 个 `half` 或 4 个 `float`

---

## 🎯 完整示例：SiLU 实现

### 简化版（不依赖外部库）

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>

// SiLU 函数（支持 half 和 float）
template<typename T>
__device__ __forceinline__ float silu_impl(float x) {
    return x / (1.0f + expf(-x));
}

template<typename T>
__device__ __forceinline__ T silu(const T& x) {
    if constexpr (sizeof(T) == 2) {
        // half 类型：转 float 计算
        float f32_val = __half2float((__half)x);
        float result = silu_impl<T>(f32_val);
        return (T)__float2half(result);
    } else {
        // float 类型：直接计算
        return (T)silu_impl<T>((float)x);
    }
}

// Kernel：SiLU 激活
template<typename T>
__global__ void silu_kernel(const T* input, T* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-Stride Loop
    for (int i = idx; i < N; i += stride) {
        output[i] = silu<T>(input[i]);
    }
}

// 主机端调用
template<typename T>
void silu_cuda(const T* d_input, T* d_output, int N) {
    const int threads_per_block = 256;
    const int max_blocks = 1024;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    blocks = blocks < max_blocks ? blocks : max_blocks;
    
    silu_kernel<T><<<blocks, threads_per_block>>>(
        d_input, d_output, N);
    
    cudaDeviceSynchronize();
}

int main() {
    const int N = 10000;
    
    // 主机端数据
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    
    // 初始化输入
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i - N/2) / (N/10.0f);  // 范围 [-5, 5]
    }
    
    // 设备端数据
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // 复制到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 执行 SiLU
    silu_cuda<float>(d_input, d_output, N);
    
    // 复制回主机
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("Input -> Output:\n");
    for (int i = 0; i < 10; i++) {
        float expected = h_input[i] / (1.0f + expf(-h_input[i]));
        printf("  %.2f -> %.4f (expected: %.4f)\n", 
               h_input[i], h_output[i], expected);
    }
    
    // 清理
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
```

---

## 📊 性能优化技巧

### 1. 向量化加载/存储

```cpp
// 一次加载 4 个 float 或 8 个 half
using vec_t = float4;  // 或 half8

__device__ void silu_vectorized(const vec_t* input, vec_t* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        vec_t vec_input = input[i];
        // 对每个元素应用 SiLU
        vec_t vec_output;
        // ... 处理 ...
        output[i] = vec_output;
    }
}
```

**优势**：
- 减少内存访问次数
- 提高内存带宽利用率
- 降低指令开销

### 2. 使用快速数学函数

```cpp
// 使用 __expf（快速版本）
float fast_silu(float x) {
    return x / (1.0f + __expf(-x));
}
```

**权衡**：
- 速度更快（约 2x）
- 精度略低（通常足够）

### 3. 数值稳定性优化

```cpp
__device__ float silu_stable(float x) {
    // 对于大负数，避免 exp 溢出
    if (x < -20.0f) {
        return x;  // 近似：silu(x) ≈ x when x << 0
    }
    // 对于大正数，避免计算 exp(-x)
    if (x > 20.0f) {
        return x;  // 近似：silu(x) ≈ x when x >> 0
    }
    // 正常计算
    return x / (1.0f + expf(-x));
}
```

---

## 🔍 与其他激活函数对比

### 常见激活函数

| 激活函数 | 公式 | 特点 |
|---------|------|------|
| **ReLU** | `max(0, x)` | 简单、快速，但不可微在 0 处 |
| **SiLU** | `x / (1 + exp(-x))` | 平滑、可微，性能好 |
| **GELU** | `x * 0.5 * (1 + erf(x/√2))` | 更平滑，但计算慢 |
| **Sigmoid** | `1 / (1 + exp(-x))` | 输出 [0, 1]，易饱和 |

### 在 LLM 中的应用

- **Llama 系列**：使用 SiLU（SwiGLU）
- **BERT**：使用 GELU
- **GPT-2**：使用 GELU
- **GPT-3/4**：使用 GELU

---

## 📝 总结

### 核心概念

1. **逐元素操作**：每个元素独立计算
2. **类型转换**：`half` → `float` → 计算 → `half`
3. **数值稳定性**：处理极端值（大正数/大负数）
4. **向量化**：一次处理多个元素

### 关键点

- ✅ **简单但重要**：激活函数是神经网络的基础
- ✅ **计算密集**：LLM 中需要处理大量 token
- ✅ **并行友好**：完全独立，无依赖
- ✅ **数值精度**：需要注意类型转换

### 学习价值

激活函数是学习 CUDA 的**第二个重要算子**，因为它：
- 展示了设备端函数的使用
- 说明了类型转换技巧
- 演示了简单的数学运算
- 为理解更复杂的算子打下基础

---

## 🔗 相关资源

- **下一个算子**：[03_Lightning_Attention_Decode.md](./03_Lightning_Attention_Decode.md)
- **SwiGLU 论文**：GLU Variants Improve Transformer
- **GELU 论文**：Gaussian Error Linear Units (GELUs)

