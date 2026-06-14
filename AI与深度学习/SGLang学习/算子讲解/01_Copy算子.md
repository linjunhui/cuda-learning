# Copy 算子详解

## 📖 算子概述

**Copy** 算子是最简单的算子之一，它的功能是将数据从 CPU 内存复制到 GPU 内存（或从 GPU 复制到 GPU）。

**用途**：
- 将 CPU 上的小数组复制到 GPU
- 设备间数据传输
- 常量数据初始化

**特点**：
- 最简单的 CUDA kernel
- 理解 CUDA 编程模型的基础
- 没有复杂的计算，只有数据移动

---

## 🔢 公式与算法

### 数学公式

```
output[i] = input[i],  ∀ i ∈ [0, N-1]
```

**含义**：将输入数组的每个元素原样复制到输出数组。

### 算法步骤

```
1. 获取当前线程的全局索引 idx
2. 检查边界：if (idx < N)
3. 复制：output[idx] = input[idx]
```

**复杂度**：
- **时间复杂度**：O(N)
- **空间复杂度**：O(1)
- **并行度**：N 个线程并行执行

---

## 🧠 算法原理

### 基本原理

Copy 算子是最简单的数据并行操作：
- **每个线程处理一个元素**：线程 `i` 负责复制 `input[i]` 到 `output[i]`
- **无依赖关系**：每个线程独立工作，不需要同步
- **内存访问模式简单**：连续的内存访问

### 线程分配

```
输入数组: [0][1][2][3][4][5][6][7][8][9]
          ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
线程分配: [T0][T1][T2][T3][T4][T5][T6][T7][T8][T9]
          ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
输出数组: [0][1][2][3][4][5][6][7][8][9]
```

**图示**：
- 每个线程处理一个元素
- 1:1 映射关系
- 并行执行，互不干扰

---

## 💻 代码实现

### 源码位置

`SGLang学习/sglang/sgl-kernel/csrc/elementwise/copy.cu`

### 完整代码

```12:18:SGLang学习/sglang/sgl-kernel/csrc/elementwise/copy.cu
template <int N>
__global__ void copy_to_gpu_no_ce_kernel(const InputArray<N> input_array, int* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    output[idx] = input_array.values[idx];
  }
}
```

### 代码逐行解析

#### 第 1 行：Kernel 定义

```cpp
template <int N>
__global__ void copy_to_gpu_no_ce_kernel(const InputArray<N> input_array, int* output)
```

**关键点**：
- **`template <int N>`**：模板参数，编译时确定数组大小
- **`__global__`**：CUDA kernel 函数，从主机端调用，在设备端执行
- **参数**：
  - `InputArray<N>`：输入的常量数组（在 GPU 常量内存中）
  - `int* output`：输出数组指针（在 GPU 全局内存中）

**为什么用模板？**
- 编译时确定大小，编译器可以优化
- 小数组可以使用常量内存（更快）
- 避免运行时检查

#### 第 2 行：计算线程索引

```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

**含义**：
- `threadIdx.x`：线程在 block 内的索引（0 到 blockDim.x-1）
- `blockIdx.x`：block 在整个 grid 中的索引
- `blockDim.x`：每个 block 的线程数
- **`idx`**：线程在整个 grid 中的全局索引

**示例**：
- Block 0, Thread 2：`idx = 0 × 256 + 2 = 2`
- Block 1, Thread 2：`idx = 1 × 256 + 2 = 258`

#### 第 3 行：边界检查

```cpp
if (idx < N) {
```

**为什么需要边界检查？**
- Grid 的线程数可能大于数组大小 `N`
- 例如：`N=10`，但启动了 256 个线程
- 边界检查防止越界访问

**示例**：
- `N=10`, `grid=1`, `block=256`
- 线程 0-9：会执行复制
- 线程 10-255：跳过（边界检查失败）

#### 第 4 行：执行复制

```cpp
output[idx] = input_array.values[idx];
```

**操作**：
- 从常量内存读取 `input_array.values[idx]`
- 写入全局内存 `output[idx]`

**内存访问**：
- **读取**：常量内存（`const InputArray`），只读，缓存友好
- **写入**：全局内存（`int*`），可写

### InputArray 结构体

```7:10:SGLang学习/sglang/sgl-kernel/csrc/elementwise/copy.cu
template <int N>
struct InputArray {
  int values[N];
};
```

**作用**：
- 封装固定大小的数组
- 可以在 kernel 启动时直接传递（拷贝到 GPU 常量内存）
- 避免额外的内存分配

### 主机端调用

```20:46:SGLang学习/sglang/sgl-kernel/csrc/elementwise/copy.cu
template <int N>
void copy_to_gpu_no_ce_impl(const at::Tensor& input, at::Tensor& output) {
  TORCH_CHECK(input.dim() == 1, "input must be 1-D");
  TORCH_CHECK(static_cast<int>(input.numel()) == N, "input numel must equal template N");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dtype() == torch::kInt32, "input dtype must be int32");

  TORCH_CHECK(output.dim() == 1, "output dim");
  TORCH_CHECK(static_cast<int>(output.numel()) == N, "output size");
  TORCH_CHECK(output.is_contiguous(), "output contiguous");
  TORCH_CHECK(output.dtype() == torch::kInt32, "output dtype");

  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

  InputArray<N> input_array;
  const int* input_ptr = input.data_ptr<int>();
  for (int i = 0; i < N; ++i)
    input_array.values[i] = input_ptr[i];

  // may use multi thread blocks if performance bottleneck
  dim3 grid(1);
  dim3 block(static_cast<int>(input.numel()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(input_array, output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

#### 关键步骤解析

**步骤 1：参数验证**
```cpp
TORCH_CHECK(input.dim() == 1, "input must be 1-D");
TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
```
- 检查输入是否是一维数组
- 检查是否连续（保证内存访问效率）

**步骤 2：准备数据**
```cpp
InputArray<N> input_array;
const int* input_ptr = input.data_ptr<int>();
for (int i = 0; i < N; ++i)
  input_array.values[i] = input_ptr[i];
```
- 在主机端创建 `InputArray` 对象
- 从 PyTorch Tensor 复制数据到结构体
- 启动 kernel 时，这个结构体会拷贝到 GPU 常量内存

**步骤 3：配置 Kernel 参数**
```cpp
dim3 grid(1);        // 1 个 block
dim3 block(N);       // N 个线程（N 是数组大小）
```
- **Grid 配置**：使用 1 个 block
- **Block 配置**：使用 `N` 个线程（每个元素一个线程）

**步骤 4：启动 Kernel**
```cpp
copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(
    input_array, 
    output.data_ptr<int>()
);
```
- **`<<<grid, block, shared_mem, stream>>>`**：CUDA kernel 启动语法
- `input_array`：作为参数传递（会拷贝到 GPU）
- `output.data_ptr<int>()`：GPU 内存中的输出指针

---

## 🎯 设计要点

### 1. 为什么使用常量内存？

**常量内存的特点**：
- **只读**：`const InputArray`
- **缓存**：有专门的常量缓存，访问更快
- **广播**：所有线程访问同一位置时，只需要一次内存访问

**适用场景**：
- 小数组（通常 < 100 个元素）
- 所有线程需要相同的数据
- 只读数据

### 2. 为什么用模板？

**模板的优势**：
- **编译时优化**：编译器知道数组大小，可以展开循环
- **类型安全**：编译时检查，避免运行时错误
- **性能**：避免动态分配和检查

**限制**：
- 数组大小必须在编译时确定
- 需要为每个大小创建模板实例化

### 3. 为什么只用一个 Block？

**当前实现**：
```cpp
dim3 grid(1);        // 只有 1 个 block
dim3 block(N);       // N 个线程
```

**原因**：
- 数组通常很小（如 64、72）
- 一个 block 足够处理
- 简化代码，不需要多 block 协调

**注意**：SGLang 的 Copy 算子**只适用于小数组**（64 和 72）。虽然 kernel 代码本身支持多 block（使用了 `blockIdx.x`），但实际调用时只配置了单 block（`grid(1)`），并且入口函数只支持这两个特定大小。如果数组很大（超过 1024），当前实现会失败。处理大数组的通用方案请参考下面的"扩展内容"章节。

### 4. 为什么是 64 和 72？

**这两个数字的来源**：

这两个数字来源于 **MoE（Mixture of Experts）模型**的实际使用场景：

1. **64 个 experts**：
   - 这是许多大型 MoE 模型的标准配置
   - 例如：**DeepSeek-V2**、**Qwen2-MoE** 等主流模型使用 64 个 experts
   - `num_recv_tokens_per_expert` 数组的长度等于 expert 数量

2. **72 个 experts**：
   - 可能是 **64 个基础 experts + 8 个冗余 experts** = 72
   - 在 Expert Parallel (EP) 模式下，为了容错和负载均衡，会添加冗余 experts
   - 代码中可以看到：`num_local_experts + ep_num_redundant_experts`

**实际使用场景**：

```python
# 在 deep_gemm.py 中的实际使用
def copy_list_to_gpu_no_ce(arr: List[int]):
    # arr 是 num_recv_tokens_per_expert，长度为 expert 数量
    # 每个元素表示该 expert 接收到的 token 数量
    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)  # 需要 arr 的长度为 64 或 72
    return tensor_gpu
```

**设计考虑**：
- ✅ **性能优化**：针对特定大小编译模板，编译器可以充分优化
- ✅ **常量内存**：小数组适合放在常量内存中，访问更快
- ✅ **实际需求**：覆盖了 SGLang 支持的主要 MoE 模型的 expert 数量
- ⚠️ **扩展性**：代码注释 `// Can use macro if there are more N needed` 表明，如果未来需要支持其他大小，可以通过宏或模板实例化添加

**常见 MoE 模型的 expert 数量**：
- Mixtral: 8 experts
- DeepSeek-V2: 64 experts
- Qwen2-MoE: 64 experts
- DBRX: 16 experts
- PhiMoE: 16 experts

---

## 📚 扩展内容：处理超大数组的通用方案

> **注意**：以下内容是通用的 CUDA 编程知识，**不是 SGLang 中 Copy 算子的实现**。
> 
> SGLang 的 Copy 算子设计用于**小数组**（只支持 64、72 两个大小），虽然 kernel 代码技术上支持多 block，但实际配置为单 Block。入口函数也只支持这两个特定大小，其他大小会直接报错。
> 
> 如果你需要处理超大数组，可以参考以下通用方案，但 SGLang 本身不提供这些实现。

### 🚨 问题：超过最大线程数的情况

### CUDA 线程限制

**关键限制**：
- **每个 Block 最大线程数**：1024（所有现代 GPU）
- **Grid 最大 Block 数**：一维网格最多 2³¹-1 个 blocks
- **当前实现的问题**：如果 `N > 1024`，使用 `dim3 block(N)` 会**启动失败**

**示例问题场景**：
```cpp
// 如果 N = 5000
dim3 grid(1);
dim3 block(5000);  // ❌ 错误！超过 1024 的限制
// 启动 kernel 时会报错：invalid configuration argument
```

### 解决方案 1：多 Block 并行（适合中等大小数组）

**适用场景**：数组大小在 1024 到几百万之间

**实现方法**：
```cpp
template <int N>
void copy_to_gpu_no_ce_impl_multi_block(const at::Tensor& input, at::Tensor& output) {
  // ... 前面的验证代码相同 ...
  
  InputArray<N> input_array;
  const int* input_ptr = input.data_ptr<int>();
  for (int i = 0; i < N; ++i)
    input_array.values[i] = input_ptr[i];

  // 使用多个 blocks
  const int threads_per_block = 256;  // 每个 block 256 个线程
  int num_blocks = (N + threads_per_block - 1) / threads_per_block;  // 向上取整
  
  dim3 grid(num_blocks);
  dim3 block(threads_per_block);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(
      input_array, output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

**工作原理**：
- **线程索引计算**：`idx = blockIdx.x * blockDim.x + threadIdx.x`
- **示例**：`N=5000`, `threads_per_block=256`
  - `num_blocks = (5000 + 256 - 1) / 256 = 20`
  - Block 0: 线程 0-255 处理元素 0-255
  - Block 1: 线程 0-255 处理元素 256-511
  - ...
  - Block 19: 线程 0-255 处理元素 4864-5119（边界检查确保不越界）

**优点**：
- ✅ 简单直接，易于理解
- ✅ 适合中等大小数组（几百万元素以内）
- ✅ 每个线程只处理一个元素，逻辑清晰

**限制**：
- ⚠️ 如果数组非常大（如 10 亿元素），需要创建大量 blocks
- ⚠️ Grid 大小有硬件限制（通常 65535 个 blocks）

### 解决方案 2：Grid-Stride Loop（适合超大数组）

**适用场景**：数组大小可能非常大（几百万到几十亿元素）

**核心思想**：每个线程处理多个元素，而不是 1:1 映射

**Kernel 实现**：
```cpp
template <int N>
__global__ void copy_to_gpu_no_ce_kernel_grid_stride(
    const InputArray<N> input_array, int* output) {
  // 计算总线程数
  int total_threads = blockDim.x * gridDim.x;
  
  // 计算当前线程的起始索引
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Grid-Stride Loop：每个线程处理多个元素
  for (int i = idx; i < N; i += total_threads) {
    output[i] = input_array.values[i];
  }
}
```

**主机端调用**：
```cpp
template <int N>
void copy_to_gpu_no_ce_impl_grid_stride(const at::Tensor& input, at::Tensor& output) {
  // ... 验证代码 ...
  
  InputArray<N> input_array;
  const int* input_ptr = input.data_ptr<int>();
  for (int i = 0; i < N; ++i)
    input_array.values[i] = input_ptr[i];

  // 固定数量的 blocks，不随数组大小变化
  const int threads_per_block = 256;
  const int num_blocks = 1024;  // 固定使用 1024 个 blocks
  
  dim3 grid(num_blocks);
  dim3 block(threads_per_block);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_no_ce_kernel_grid_stride<<<grid, block, 0, stream>>>(
      input_array, output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

**工作原理详解**：

假设 `N = 10000`，`threads_per_block = 256`，`num_blocks = 1024`：

```
总线程数 = 1024 × 256 = 262144
stride = 262144（每个线程的步长）

线程 0：处理元素 0, 262144, 524288, ...（但只有 0 < 10000，所以只处理元素 0）
线程 1：处理元素 1, 262145, 524289, ...（但只有 1 < 10000，所以只处理元素 1）
...
线程 9999：处理元素 9999, 272143, ...（但只有 9999 < 10000，所以只处理元素 9999）
线程 10000-262143：循环条件 i < 10000 不满足，不执行任何操作
```

**关键优势**：
- ✅ **固定 Grid 大小**：无论数组多大，都使用相同的 grid 配置
- ✅ **可扩展性强**：可以处理任意大小的数组（只要不超过 int64 范围）
- ✅ **负载均衡**：所有线程均匀分配工作
- ✅ **避免 Grid 限制**：不需要创建大量 blocks

**性能考虑**：
- 对于小数组（< 1000），Grid-Stride Loop 可能有轻微开销（循环检查）
- 对于大数组（> 100万），性能与多 Block 方案相当或更好
- 对于超大数组（> 1亿），Grid-Stride Loop 是唯一可行的方案

### 两种方案对比

| 特性 | 多 Block 方案 | Grid-Stride Loop 方案 |
|------|--------------|---------------------|
| **适用数组大小** | 1024 ~ 几百万 | 任意大小（几百万到几十亿） |
| **Grid 配置** | 动态（随 N 变化） | 固定（如 1024 blocks） |
| **每个线程处理** | 1 个元素 | 多个元素（循环） |
| **代码复杂度** | 简单 | 稍复杂（需要循环） |
| **硬件限制** | 受 Grid 大小限制 | 不受限制 |
| **小数组性能** | 更好（无循环开销） | 稍差（有循环检查） |
| **大数组性能** | 相当 | 相当或更好 |

### 实际应用建议

**选择策略**：

1. **小数组（N < 1024）**：
   - 使用当前实现（单 Block）
   - 最简单，性能最好

2. **中等数组（1024 ≤ N < 1000万）**：
   - 使用多 Block 方案
   - 代码简单，性能好

3. **超大数组（N ≥ 1000万）**：
   - 使用 Grid-Stride Loop 方案
   - 唯一可行的方案，避免硬件限制

**PyTorch 的实践**：
PyTorch 内部广泛使用 Grid-Stride Loop 模式，因为它：
- 可以处理任意大小的 Tensor
- 代码更通用，不需要为不同大小写不同版本
- 性能经过充分优化

**参考实现**：
```cpp
// PyTorch 风格的 Grid-Stride Loop
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void copy_kernel(const int* input, int* output, int N) {
  CUDA_KERNEL_LOOP(idx, N) {
    output[idx] = input[idx];
  }
}
```

### 💡 总结

**SGLang Copy 算子的设计哲学**：
- ✅ **专为小数组优化**：使用单 Block，简单高效
- ✅ **编译时确定大小**：通过模板参数，编译器可以充分优化
- ✅ **使用常量内存**：小数组适合放在常量内存中，访问更快

**如果你需要处理大数组**：
- 可以考虑使用 PyTorch 的通用 CUDA 操作（如 `torch.copy_`）
- 或者参考上述通用方案自行实现
- 但要注意，大数组不适合使用常量内存（`InputArray` 结构体），需要改用全局内存指针

---

## 📊 性能分析

### 内存访问模式

**读取（常量内存）**：
- 所有线程读取相同的数据（广播）
- 使用常量缓存，速度很快
- 如果缓存命中，几乎无延迟

**写入（全局内存）**：
- 每个线程写入不同的位置
- 连续写入，内存合并访问
- 带宽利用率高

### 性能优化建议

1. **使用常量内存**：对于小数组，已经实现 ✓
2. **内存合并访问**：连续访问，已实现 ✓
3. **向量化**：对于更大的数组，可以使用 `float4` 一次复制 4 个元素

**向量化版本示例**：
```cpp
__global__ void copy_vectorized(int* output, const int* input, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        // 一次复制 4 个元素
        *((float4*)&output[idx]) = *((float4*)&input[idx]);
    }
}
```

---

## 🔍 简化版本（不依赖 PyTorch）

如果你想理解核心逻辑，这里是纯 CUDA 版本：

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// 最简单的 Copy Kernel
__global__ void copy_kernel(const int* input, int* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx];
    }
}

int main() {
    const int N = 1000;
    size_t size = N * sizeof(int);
    
    // 主机端数据
    int* h_input = (int*)malloc(size);
    int* h_output = (int*)malloc(size);
    
    // 初始化输入
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }
    
    // 设备端数据
    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // 复制输入到设备
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // 配置 kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // 启动 kernel
    copy_kernel<<<blocks, threads_per_block>>>(
        d_input, d_output, N);
    
    // 同步等待
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_input[i] != h_output[i]) {
            printf("Error at index %d: %d != %d\n", i, h_input[i], h_output[i]);
            success = false;
        }
    }
    
    if (success) {
        printf("✓ Copy successful!\n");
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

## 📝 总结

### 核心概念

1. **CUDA Kernel**：`__global__` 函数，在 GPU 上执行
2. **线程索引**：`idx = blockIdx.x * blockDim.x + threadIdx.x`
3. **边界检查**：`if (idx < N)` 防止越界
4. **内存类型**：
   - 常量内存：`const` 参数，只读，有缓存
   - 全局内存：指针参数，可读写

### 关键点

- ✅ **最简单的并行模式**：每个线程处理一个元素
- ✅ **无依赖**：线程间不需要同步
- ✅ **内存合并**：连续访问，高效
- ✅ **常量内存优化**：小数组使用常量内存

### 学习价值

Copy 算子是学习 CUDA 的**最佳起点**，因为它：
- 展示了 CUDA kernel 的基本结构
- 说明了线程索引的计算
- 演示了内存访问模式
- 没有复杂的计算，容易理解

---

## 🔗 相关资源

- **CUDA 编程模型**：理解 thread、block、grid 的概念
- **内存层次**：常量内存 vs 全局内存
- **下一个算子**：[02_Activation算子.md](./02_Activation算子.md)（SiLU、GELU）

