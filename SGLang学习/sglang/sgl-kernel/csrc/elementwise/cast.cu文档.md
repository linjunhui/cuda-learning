# cast.cu 文档

## 📋 文件概述

`cast.cu` 实现了类型转换操作，主要用于将高精度数据类型（BFloat16、Half）转换为 FP8 格式，用于 KV 缓存的压缩存储。

## 🎯 主要功能

### 1. FP8 下转换（Downcast to FP8）

将 BFloat16 或 Half 转换为 FP8 (E4M3) 格式，用于减少 KV 缓存的内存占用。

**应用场景**：
- KV 缓存的量化存储
- 减少内存带宽需求
- 支持更大的上下文长度

## 🔬 实现原理

### 类型转换模板

```cpp
template <typename T>
struct ConvertToFP8 {
  static __device__ __nv_fp8_storage_t convert_to_fp8(T value) {
    return 0;  // 默认实现（未使用）
  }
};

// BFloat16 特化
template <>
struct ConvertToFP8<__nv_bfloat16> {
  static __device__ __nv_fp8_storage_t convert_to_fp8(__nv_bfloat16 value) {
    return __nv_cvt_bfloat16raw_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
};

// Half 特化
template <>
struct ConvertToFP8<half> {
  static __device__ __nv_fp8_storage_t convert_to_fp8(half value) {
    return __nv_cvt_halfraw_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
};
```

**关键特性**：
- 使用 CUDA 内置的 FP8 转换函数
- `__NV_SATFINITE`：饱和并处理无穷值
- `__NV_E4M3`：使用 E4M3 格式（4 位指数，3 位尾数）

### 从 Float 转换

```cpp
template <typename T>
struct ConvertFromFloat {
  static __device__ T convert_from_float(float value) {
    return 0;
  }
};

template <>
struct ConvertFromFloat<__nv_bfloat16> {
  static __device__ __nv_bfloat16 convert_from_float(float value) {
    return __float2bfloat16(value);
  }
};
```

### 融合下转换内核

```cpp
template <typename T>
__global__ void fused_downcast_kernel(
    const T* cache_k,
    const T* cache_v,
    const float* k_scale,
    const float* v_scale,
    __nv_fp8_storage_t* output_k,
    __nv_fp8_storage_t* output_v,
    // ... 其他参数
) {
  // 1. 计算缩放因子的倒数
  T k_scale_val = ConvertFromFloat<T>::convert_from_float(k_scale[0]);
  T v_scale_val = ConvertFromFloat<T>::convert_from_float(v_scale[0]);
  T k_scale_inv = static_cast<T>(1.f) / k_scale_val;
  T v_scale_inv = static_cast<T>(1.f) / v_scale_val;
  
  // 2. 定义裁剪函数
  auto clamp = [&](T val) { 
    return val > max_fp8 ? max_fp8 : (min_fp8 > val ? min_fp8 : val); 
  };
  
  // 3. 对每个 token 进行处理
  int token_idx = blockIdx.x;
  
  if (token_idx < input_sl) {
    int out_seq_idx = loc[token_idx];  // 输出位置索引
    
    // 4. 并行处理所有 head 和 dim
    for (int i = thread_idx; i < head * dim; i += total_threads) {
      // K 的转换
      T k_val = cache_k[in_idx] * k_scale_inv;
      k_val = clamp(k_val);  // 裁剪到 FP8 范围
      output_k[out_idx] = ConvertToFP8<T>::convert_to_fp8(k_val);
      
      // V 的转换（类似）
      // ...
    }
  }
}
```

## 💡 算法详解

### 量化流程

1. **缩放**：将输入值除以缩放因子
   ```
   scaled_value = original_value / scale
   ```

2. **裁剪**：限制到 FP8 的有效范围
   ```
   clamped_value = clamp(scaled_value, min_fp8, max_fp8)
   ```
   - `max_fp8 = 448.0`（E4M3 格式的最大值）
   - `min_fp8 = -448.0`（E4M3 格式的最小值）

3. **类型转换**：转换为 FP8 存储格式
   ```
   fp8_value = convert_to_fp8(clamped_value)
   ```

### 索引映射

```cpp
int out_seq_idx = loc[token_idx];  // 输入 token 在输出中的位置
int out_idx = (out_seq_idx * mult + offset) * head * dim + i;
```

**说明**：
- `loc`：输入 token 到输出位置的映射
- `mult` 和 `offset`：输出位置的偏移计算
- 支持稀疏的 KV 缓存存储

## 💻 主接口函数

```cpp
void downcast_fp8(
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& k_out,
    at::Tensor& v_out,
    at::Tensor& k_scale,
    at::Tensor& v_scale,
    at::Tensor& loc,
    int64_t mult,
    int64_t offset)
```

**参数说明**：
- `k, v`: 输入的 K 和 V（BFloat16/Half）
- `k_out, v_out`: 输出的 K 和 V（FP8）
- `k_scale, v_scale`: 量化缩放因子
- `loc`: 位置映射索引
- `mult, offset`: 输出位置偏移参数

**维度要求**：
- 输入：`(input_sl, head, dim)`
- 输出：`(output_sl, head, dim)`（通过 `loc` 映射）

## ⚡ 性能优化

### 1. 融合操作
- 同时处理 K 和 V
- 减少内核启动开销

### 2. 并行化
- 每个 token 一个 block
- 线程处理不同的 head*dim 维度

### 3. 向量化
- 使用向量类型提高内存访问效率
- `vec_size = 8` 的向量化处理

### 4. 裁剪优化
- 使用 Lambda 函数内联裁剪操作
- 避免分支预测失败

## 📊 内存节省

**量化效果**：
- **原始**：BFloat16 (2 bytes) 或 Half (2 bytes)
- **量化后**：FP8 (1 byte)
- **压缩比**：2:1

**示例**：
- 原始 KV 缓存：128 GB
- 量化后：64 GB
- **节省 64 GB 内存**

## 🔍 使用场景

### 1. 大上下文长度支持

```python
# 使用 FP8 量化 KV 缓存
k_fp8, v_fp8 = sgl_kernel.downcast_fp8(
    k=k_bf16,
    v=v_bf16,
    k_out=k_fp8_buffer,
    v_out=v_fp8_buffer,
    k_scale=k_scale,
    v_scale=v_scale,
    loc=cache_positions,
    mult=1,
    offset=0
)
```

### 2. 内存受限环境

- 在 GPU 内存有限时使用
- 支持更大的批次大小
- 延长上下文窗口

## 📚 参考资料

1. **FP8 格式规范**：NVIDIA FP8 Format
2. **量化技术**：Post-Training Quantization

## 🔗 相关文件

- `csrc/elementwise/activation.cu` - 激活函数（使用类似类型转换）
- `csrc/gemm/` - 量化矩阵乘法
- `csrc/quantization/` - 其他量化实现

