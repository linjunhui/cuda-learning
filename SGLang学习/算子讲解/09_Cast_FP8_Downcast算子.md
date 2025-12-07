# Cast (FP8 Downcast) 算子详解

## 📖 算子概述

**Cast (FP8 Downcast)** 是将 KV Cache 从高精度（`half`/`bfloat16`）转换为低精度（FP8）的算子。这用于**节省内存**和**提高缓存利用率**。

**用途**：
- **KV Cache 量化**：将 KV Cache 从 16 位转换为 8 位
- **内存优化**：减少 KV Cache 的内存占用（节省 50%）
- **量化推理**：支持 FP8 推理加速

**特点**：
- **量化感知**：使用 scale 进行量化
- **饱和截断**：限制在 FP8 的数值范围内
- **索引重映射**：支持位置索引的重映射（用于 KV Cache）

---

## 🔢 公式与算法

### 数学公式

**量化公式**：
```
output = clamp(input / scale, min_fp8, max_fp8)
output_fp8 = quantize_to_fp8(output)
```

其中：
- `scale`：量化缩放因子
- `min_fp8 = -448.0`：FP8 (E4M3) 的最小值
- `max_fp8 = 448.0`：FP8 (E4M3) 的最大值
- `quantize_to_fp8`：转换为 FP8 格式

**完整流程**：
```
1. 计算 scale_inv = 1.0 / scale
2. scaled_value = input * scale_inv
3. clamped_value = clamp(scaled_value, min_fp8, max_fp8)
4. output = convert_to_fp8(clamped_value)
```

### 算法步骤

```
对于每个 token t，每个维度 d：
  1. 读取 input[t][d]
  2. 计算 scaled = input[t][d] * scale_inv
  3. 截断：clamped = clamp(scaled, -448.0, 448.0)
  4. 转换为 FP8：output[t][d] = to_fp8(clamped)
```

**复杂度**：
- **时间复杂度**：O(num_tokens × num_heads × head_dim)
- **空间复杂度**：O(1)
- **并行度**：num_tokens × num_heads × head_dim（完全并行）

---

## 🧠 算法原理

### 基本原理

**量化原理**：

FP8 (E4M3) 格式：
- **符号位**：1 位
- **指数位**：4 位
- **尾数位**：3 位
- **数值范围**：[-448.0, 448.0]（约）

**量化步骤**：
1. **缩放**：将输入除以 scale，映射到 FP8 范围
2. **截断**：限制在 [-448.0, 448.0] 范围内
3. **转换**：转换为 FP8 格式

**为什么需要 scale？**

```
输入范围可能是 [-100, 100]
FP8 范围是 [-448, 448]

如果直接转换：
  100 → FP8(100)  （可以）
  1000 → FP8(1000)  （溢出，需要截断）

使用 scale：
  scale = 2.0  →  scaled = 1000 / 2.0 = 500  →  clamp(500, -448, 448) = 448
  这样可以将任意范围映射到 FP8 范围
```

### 索引重映射

**为什么需要索引重映射？**

在 KV Cache 中，token 的位置可能不是连续的，需要根据 `loc` 数组重新映射：

```
原始位置：0, 1, 2, 3, ...
映射位置：loc[0], loc[1], loc[2], loc[3], ...

output[loc[t] * mult + offset] = quantize(input[t])
```

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/elementwise/cast.cu`

### 核心 Kernel 代码

```45:91:SGLang学习/sglang/sgl-kernel/csrc/elementwise/cast.cu
template <typename T>
__global__ void fused_downcast_kernel(
    const T* cache_k,
    const T* cache_v,
    const float* k_scale,
    const float* v_scale,
    __nv_fp8_storage_t* output_k,
    __nv_fp8_storage_t* output_v,
    const int input_sl,
    const int head,
    const int dim,
    const T max_fp8,
    const T min_fp8,
    const int64_t mult,
    const int64_t offset,
    const int64_t* loc) {
  // TODO: change name
  int token_idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int total_threads = blockDim.x;

  T k_scale_val = ConvertFromFloat<T>::convert_from_float(k_scale[0]);
  T v_scale_val = ConvertFromFloat<T>::convert_from_float(v_scale[0]);

  T k_scale_inv = static_cast<T>(1.f) / k_scale_val;
  T v_scale_inv = static_cast<T>(1.f) / v_scale_val;

  auto clamp = [&](T val) { return val > max_fp8 ? max_fp8 : (min_fp8 > val ? min_fp8 : val); };

  if (token_idx < input_sl) {
    int out_seq_idx = loc[token_idx];

#pragma unroll
    for (int i = thread_idx; i < head * dim; i += total_threads) {
      int in_idx = token_idx * head * dim + i;
      int out_idx = (out_seq_idx * mult + offset) * head * dim + i;

      T k_val = cache_k[in_idx] * k_scale_inv;
      k_val = clamp(k_val);
      output_k[out_idx] = ConvertToFP8<T>::convert_to_fp8(k_val);

      T v_val = cache_v[in_idx] * v_scale_inv;
      v_val = clamp(v_val);
      output_v[out_idx] = ConvertToFP8<T>::convert_to_fp8(v_val);
    }
  }
}
```

### 代码逐行解析

#### 第一步：加载 Scale 并计算倒数

```cpp
T k_scale_val = ConvertFromFloat<T>::convert_from_float(k_scale[0]);
T v_scale_val = ConvertFromFloat<T>::convert_from_float(v_scale[0]);

T k_scale_inv = static_cast<T>(1.f) / k_scale_val;
T v_scale_inv = static_cast<T>(1.f) / v_scale_val;
```

**关键点**：
- **Scale 加载**：从全局内存加载 scale（每个 token 共享）
- **计算倒数**：`scale_inv = 1.0 / scale`（避免除法）
- **类型转换**：转为输入类型 `T`（`half` 或 `bfloat16`）

**为什么用倒数？**
- 乘法比除法快（约 2x）
- 倒数只需要计算一次（在循环外）

#### 第二步：定义 Clamp 函数

```cpp
auto clamp = [&](T val) { 
  return val > max_fp8 ? max_fp8 : (min_fp8 > val ? min_fp8 : val); 
};
```

**作用**：
- 限制值在 FP8 范围内：`[min_fp8, max_fp8]`
- 防止溢出和下溢

**FP8 范围**：
- `max_fp8 = 448.0`（FP8 E4M3 的最大值）
- `min_fp8 = -448.0`（FP8 E4M3 的最小值）

#### 第三步：索引重映射

```cpp
if (token_idx < input_sl) {
  int out_seq_idx = loc[token_idx];
  // ...
  int out_idx = (out_seq_idx * mult + offset) * head * dim + i;
}
```

**关键点**：
- **位置映射**：`out_seq_idx = loc[token_idx]`
- **输出索引**：`out_idx = (out_seq_idx * mult + offset) * head * dim + i`
- **灵活性**：支持多种索引模式（通过 `mult` 和 `offset`）

#### 第四步：量化流程（Grid-Stride Loop）

```cpp
#pragma unroll
for (int i = thread_idx; i < head * dim; i += total_threads) {
  int in_idx = token_idx * head * dim + i;
  int out_idx = (out_seq_idx * mult + offset) * head * dim + i;

  // K 的量化
  T k_val = cache_k[in_idx] * k_scale_inv;
  k_val = clamp(k_val);
  output_k[out_idx] = ConvertToFP8<T>::convert_to_fp8(k_val);

  // V 的量化
  T v_val = cache_v[in_idx] * v_scale_inv;
  v_val = clamp(v_val);
  output_v[out_idx] = ConvertToFP8<T>::convert_to_fp8(v_val);
}
```

**关键优化**：
- **Grid-Stride Loop**：每个线程处理多个元素
- **循环展开**：`#pragma unroll` 消除循环开销
- **融合操作**：同时处理 K 和 V

**量化步骤**：
1. **缩放**：`value * scale_inv`
2. **截断**：`clamp(value)`
3. **转换**：`convert_to_fp8(value)`

### ConvertToFP8 模板特化

```3:22:SGLang学习/sglang/sgl-kernel/csrc/elementwise/cast.cu
template <typename T>
struct ConvertToFP8 {
  static __device__ __nv_fp8_storage_t convert_to_fp8(T value) {
    return 0;
  }
};

template <>
struct ConvertToFP8<__nv_bfloat16> {
  static __device__ __nv_fp8_storage_t convert_to_fp8(__nv_bfloat16 value) {
    return __nv_cvt_bfloat16raw_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
};

template <>
struct ConvertToFP8<half> {
  static __device__ __nv_fp8_storage_t convert_to_fp8(half value) {
    return __nv_cvt_halfraw_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
};
```

**关键点**：
- **模板特化**：为不同类型提供专门的转换函数
- **CUDA 内置函数**：使用 `__nv_cvt_*_to_fp8` 进行转换
- **参数**：
  - `__NV_SATFINITE`：饱和模式（超出范围时截断）
  - `__NV_E4M3`：FP8 E4M3 格式（4 位指数，3 位尾数）

---

## 🎯 关键设计要点

### 1. 量化 Scale 处理

**Scale 共享**：
- 所有 token 共享同一个 scale
- Scale 只需要加载一次（在循环外）

**倒数优化**：
- 预先计算 `1.0 / scale`
- 使用乘法代替除法（性能提升约 2x）

### 2. 饱和截断（Saturation Clamping）

**作用**：
- 防止溢出：超出 FP8 范围的值被截断
- 保证数值安全

**实现**：
```cpp
val = val > max_fp8 ? max_fp8 : (min_fp8 > val ? min_fp8 : val);
```

### 3. 融合操作（Fused Operation）

**设计**：
- 同时处理 K 和 V
- 减少 kernel 启动开销
- 提高缓存利用率

### 4. 索引重映射

**灵活性**：
- 支持位置索引的重映射
- 通过 `mult` 和 `offset` 支持多种模式

---

## 📊 性能分析

### 复杂度

**时间复杂度**：
```
O(num_tokens × num_heads × head_dim)
```

**并行化后**：
```
每个 token: O(num_heads × head_dim) / blockDim.x
```

### 内存访问

**读取**：
- `cache_k`: 1 次
- `cache_v`: 1 次
- `k_scale`: 1 次（共享）
- `v_scale`: 1 次（共享）
- `loc`: 1 次（共享）

**写入**：
- `output_k`: 1 次
- `output_v`: 1 次

**总访问**：
- 每个元素：2 次读取（K 和 V）+ 2 次写入

### 量化开销

**计算开销**：
- 1 次乘法（缩放）
- 1 次比较（截断）
- 1 次类型转换（FP8）

**内存节省**：
- 输入：16 位（`half`/`bfloat16`）
- 输出：8 位（FP8）
- **节省 50% 内存**

---

## 📝 总结

### 核心概念

1. **量化**：将高精度转换为低精度
2. **Scale 缩放**：将输入映射到 FP8 范围
3. **饱和截断**：防止溢出
4. **索引重映射**：支持灵活的位置映射

### 关键优化

- ✅ **倒数优化**：预先计算 `1.0 / scale`
- ✅ **融合操作**：同时处理 K 和 V
- ✅ **循环展开**：消除循环开销
- ✅ **Grid-Stride Loop**：支持任意大小

### 学习价值

Cast (FP8 Downcast) 展示了：
- 量化技术的实现
- 类型转换的技巧
- 内存优化的方法
- 模板特化的使用

---

## 🔗 相关资源

- **FP8 格式**：NVIDIA FP8 (E4M3) 规格
- **下一个算子**：[10_Concat_MLA算子.md](./10_Concat_MLA算子.md)
- **量化技术**：Quantization 原理







