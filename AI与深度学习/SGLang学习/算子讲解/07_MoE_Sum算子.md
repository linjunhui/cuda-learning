# MoE Sum 算子详解

## 📖 算子概述

**MoE Sum** 是混合专家（Mixture of Experts）模型中的关键算子，用于将多个专家（TopK）的输出求和，得到最终的 MoE 输出。

**用途**：
- **MoE 模型**：混合专家模型的输出聚合
- **TopK Experts**：将 TopK 个专家的输出求和
- **路由聚合**：根据路由权重聚合专家输出

**特点**：
- **简单高效**：逐元素求和操作
- **编译时优化**：使用模板参数 `TOPK`，编译器可以展开循环
- **向量化友好**：使用 `SGLANG_LDG` 优化内存访问

---

## 🔢 公式与算法

### 数学公式

对于 MoE 模型，每个 token 选择 TopK 个专家，然后将它们的输出求和：

```
output[i] = Σ(input[i][k])  for k = 0 to TopK-1
```

**向量形式**：
```
output[t][d] = Σ(input[t][k][d])  for k = 0 to TopK-1
```

其中：
- `t`：token 索引
- `k`：expert 索引（TopK 个）
- `d`：隐藏维度索引

**完整公式**：
```
for each token t:
    for each dimension d:
        output[t][d] = input[t][0][d] + input[t][1][d] + ... + input[t][TopK-1][d]
```

### 算法步骤

```
1. 对于每个 token t：
   对于每个维度 d：
      2. 初始化 sum = 0
      3. 对于 k = 0 to TopK-1：
         sum += input[t][k][d]
      4. output[t][d] = sum
```

**复杂度**：
- **时间复杂度**：O(num_tokens × hidden_size × TopK)
- **空间复杂度**：O(1)
- **并行度**：num_tokens × hidden_size（每个位置独立计算）

---

## 🧠 算法原理

### 基本原理

MoE Sum 算子执行的是**简单的逐元素求和**：

**输入**：
- `input`: `[num_tokens, TopK, hidden_size]`
  - 每个 token 有 TopK 个专家的输出
  - 每个专家输出 hidden_size 维的向量

**输出**：
- `output`: `[num_tokens, hidden_size]`
  - 每个 token 的输出是 TopK 个专家输出的和

**示例**（TopK=2, hidden_size=4）：

```
输入 input[0] (token 0 的 TopK 专家输出):
  Expert 0: [1.0, 2.0, 3.0, 4.0]
  Expert 1: [5.0, 6.0, 7.0, 8.0]

输出 output[0]:
  [1.0+5.0, 2.0+6.0, 3.0+7.0, 4.0+8.0]
  = [6.0, 8.0, 10.0, 12.0]
```

### 为什么需要求和？

在 MoE 模型中：
1. **TopK 路由**：每个 token 选择 TopK 个专家
2. **专家计算**：每个专家独立计算输出
3. **输出聚合**：将 TopK 个专家的输出求和（或加权求和）

**简化假设**：
- 当前实现假设**所有专家的权重相同**（都是 1.0）
- 实际中可能需要加权求和（`sum += weight[k] * input[t][k][d]`）

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/moe/moe_sum.cu`

### 核心 Kernel 代码

```11:25:SGLang学习/sglang/sgl-kernel/csrc/moe/moe_sum.cu
template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += SGLANG_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}
```

### 代码逐行解析

#### 第 1 行：模板 Kernel 定义

```cpp
template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(...)
```

**关键点**：
- **`template <typename scalar_t>`**：支持多种数据类型（`float`、`half`、`bfloat16`）
- **`template <int TOPK>`**：TopK 作为模板参数，编译时确定
- **为什么 TOPK 是模板参数？**
  - 编译器可以**展开循环**（`#pragma unroll`）
  - 避免运行时检查，性能更好
  - 生成多个 kernel 实例化（TopK=2, 3, 4 等）

#### 第 2-4 行：参数声明

```cpp
scalar_t* __restrict__ out,          // [num_tokens, hidden_size]
const scalar_t* __restrict__ input,  // [num_tokens, TopK, hidden_size]
const int d)                         // hidden_size
```

**内存布局**：
- **输入**：`input[token_idx * TOPK * d + k * d + idx]`
  - 扁平化存储：`[token][expert][dim]`
- **输出**：`out[token_idx * d + idx]`
  - 扁平化存储：`[token][dim]`

**`__restrict__`**：
- 告诉编译器指针不会重叠，可以优化

#### 第 5 行：计算 Token 索引

```cpp
const int64_t token_idx = blockIdx.x;
```

**线程分配**：
- **每个 Block 处理一个 Token**：`grid(num_tokens)`
- **Block 内的线程**：并行处理一个 token 的不同维度

**设计模式**：
```
Block 0: 处理 token 0
Block 1: 处理 token 1
...
Block N-1: 处理 token N-1
```

#### 第 6 行：Grid-Stride Loop

```cpp
for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
```

**并行化**：
- **每个线程处理多个维度**（Grid-Stride Loop）
- **线程 idx**：处理维度 `idx, idx+blockDim.x, idx+2*blockDim.x, ...`

**示例**（`d=128`, `blockDim.x=32`）：
- 线程 0：处理维度 0, 32, 64, 96
- 线程 1：处理维度 1, 33, 65, 97
- ...
- 线程 31：处理维度 31, 63, 95, 127

#### 第 7 行：初始化累加器

```cpp
scalar_t x = 0.0;
```

**关键点**：
- 在寄存器中累加（非常快）
- 每个线程有独立的累加器

#### 第 8-11 行：求和循环（关键优化）

```cpp
#pragma unroll
for (int k = 0; k < TOPK; ++k) {
  x += SGLANG_LDG(&input[token_idx * TOPK * d + k * d + idx]);
}
```

**`#pragma unroll`**：
- **循环展开**：编译器将循环展开成 `TOPK` 条语句
- **性能提升**：消除循环开销，允许更多优化
- **代码膨胀**：每个 `TOPK` 值会生成不同的 kernel

**`SGLANG_LDG`**：
- **优化宏**：用于只读全局内存访问
- **作用**：使用 `__ldg()` 指令，通过纹理缓存访问
- **优势**：对于只读数据，纹理缓存比 L2 缓存更快

**展开后的伪代码**（TopK=2）：
```cpp
// 展开后的代码（编译器生成）
x += SGLANG_LDG(&input[token_idx * TOPK * d + 0 * d + idx]);  // k=0
x += SGLANG_LDG(&input[token_idx * TOPK * d + 1 * d + idx]);  // k=1
```

#### 第 12 行：写回结果

```cpp
out[token_idx * d + idx] = x;
```

**关键点**：
- 每个线程只写一次（无竞争）
- 连续写入，产生合并访问

### 主机端调用

```27:66:SGLang学习/sglang/sgl-kernel/csrc/moe/moe_sum.cu
void moe_sum(
    torch::Tensor& input,   // [num_tokens, topk, hidden_size]
    torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
    case 2:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 2>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 3:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 3>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 4:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 4>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}
```

#### 代码解析

**配置参数**：
```cpp
dim3 grid(num_tokens);        // 每个 token 一个 block
dim3 block(std::min(hidden_size, 1024));  // 每个 block 最多 1024 个线程
```

**Switch 分发**：
- **TopK=2, 3, 4**：使用优化的 kernel（循环展开）
- **其他 TopK**：使用 PyTorch 的通用求和（`at::sum_out`）

**为什么只支持 TopK=2,3,4？**
- 大多数 MoE 模型使用 TopK=2（如 Mixtral）
- 少数模型使用 TopK=3 或 4
- 如果 TopK 很大（如 8），循环展开会导致代码膨胀，得不偿失

---

## 🎯 性能优化技巧

### 1. 循环展开（`#pragma unroll`）

**优势**：
- 消除循环开销（跳转、条件检查）
- 编译器可以做更多优化（指令级并行）

**权衡**：
- 代码膨胀：每个 TopK 值生成不同的 kernel
- TopK 太大时，展开反而不好

### 2. 只读内存优化（`SGLANG_LDG`）

**`SGLANG_LDG` 宏**：
- 使用 `__ldg()` 指令
- 通过纹理缓存访问只读全局内存

**优势**：
- 纹理缓存通常比 L2 缓存更快
- 减少缓存缺失

**注意**：
- 只适用于**只读数据**
- 写入数据不能使用

### 3. Grid-Stride Loop

**为什么使用？**
- 支持任意大小的 `hidden_size`
- 即使 `hidden_size > 1024`，也能正确工作

**性能考虑**：
- 如果 `hidden_size` 很小（< 1024），每个线程只处理一个元素
- 如果 `hidden_size` 很大，每个线程处理多个元素

### 4. 模板特化

**设计**：
- 为每个 TopK 值生成专门的 kernel
- 编译器可以针对性地优化

**示例**：
```cpp
// 编译器会生成：
moe_sum_kernel<float, 2>  // TopK=2 的版本
moe_sum_kernel<float, 3>  // TopK=3 的版本
moe_sum_kernel<float, 4>  // TopK=4 的版本
```

---

## 📊 性能分析

### 复杂度

**时间复杂度**：
```
O(num_tokens × hidden_size × TopK)
```

**并行化后**：
```
每个 token: O(hidden_size × TopK) / blockDim.x
```

**实际执行时间**：
- 主要受内存带宽限制（读取 TopK 个专家的输出）
- TopK=2 时，每个元素需要读取 2 次，写入 1 次

### 内存访问模式

**读取**：
```
input[token_idx * TOPK * d + k * d + idx]  // TopK 次读取
```

**写入**：
```
out[token_idx * d + idx]  // 1 次写入
```

**内存访问比**：
- **读取/写入** = TopK / 1
- TopK=2 时：2:1（读两次，写一次）

### 优化空间

**潜在优化**：
1. **向量化加载**：一次加载多个元素（如 `float4`）
2. **共享内存**：如果 TopK 很大，可以先将数据载入共享内存
3. **融合操作**：与其他算子融合（如 MoE Sum + Activation）

---

## 💡 简化版本（理解核心逻辑）

如果你想理解核心逻辑，这里是简化版本：

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

template<typename T, int TOPK>
__global__ void moe_sum_simple(
    T* output,          // [num_tokens, hidden_size]
    const T* input,     // [num_tokens, TopK, hidden_size]
    int num_tokens,
    int hidden_size) {
    
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (token_idx >= num_tokens || dim_idx >= hidden_size) return;
    
    // 累加 TopK 个专家的输出
    T sum = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
        int input_idx = token_idx * TOPK * hidden_size + k * hidden_size + dim_idx;
        sum += input[input_idx];
    }
    
    int output_idx = token_idx * hidden_size + dim_idx;
    output[output_idx] = sum;
}

void moe_sum_host(
    float* d_input,
    float* d_output,
    int num_tokens,
    int hidden_size,
    int topk) {
    
    dim3 grid(num_tokens);
    dim3 block(hidden_size < 1024 ? hidden_size : 1024);
    
    if (topk == 2) {
        moe_sum_simple<float, 2><<<grid, block>>>(
            d_output, d_input, num_tokens, hidden_size);
    } else if (topk == 3) {
        moe_sum_simple<float, 3><<<grid, block>>>(
            d_output, d_input, num_tokens, hidden_size);
    } else {
        // 通用实现
        printf("Unsupported TopK: %d\n", topk);
    }
    
    cudaDeviceSynchronize();
}
```

---

## 📝 总结

### 核心概念

1. **逐元素求和**：对 TopK 个专家的输出求和
2. **模板特化**：TopK 作为模板参数，编译器展开循环
3. **Grid-Stride Loop**：支持任意大小的 hidden_size
4. **只读内存优化**：使用 `SGLANG_LDG` 优化只读访问

### 关键优化

- ✅ **循环展开**：`#pragma unroll` 消除循环开销
- ✅ **只读优化**：`SGLANG_LDG` 使用纹理缓存
- ✅ **模板特化**：为每个 TopK 生成专门的 kernel
- ✅ **Grid-Stride Loop**：支持任意大小

### 学习价值

MoE Sum 展示了：
- 简单的逐元素操作
- 模板参数的使用
- 循环展开优化
- 只读内存访问优化

---

## 🔗 相关资源

- **下一个算子**：[08_Merge_Attention_States算子.md](./08_Merge_Attention_States算子.md)
- **MoE 模型**：Mixture of Experts 架构
- **TopK 选择**：TopK 路由算法








