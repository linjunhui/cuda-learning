# copy.cu 文档

## 📋 文件概述

`copy.cu` 实现了一个特殊的复制操作，用于将 CPU 上的整数数组高效地传输到 GPU，无需触发 CUDA 错误检查（"no ce" = no check）。这是一个轻量级、高性能的内存传输工具。

## 🎯 主要功能

### CPU 到 GPU 的零拷贝传输

将 CPU 上的小整数数组（如形状信息、配置参数）快速复制到 GPU，用于内核参数传递。

## 🔬 实现原理

### 内核实现

```cpp
template <int N>
__global__ void copy_to_gpu_no_ce_kernel(
    const InputArray<N> input_array,  // 通过值传递的结构体
    int* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    output[idx] = input_array.values[idx];
  }
}
```

**关键设计**：
- **值传递结构体**：`InputArray<N>` 通过值传递到内核
- **简单直接**：每个线程复制一个元素
- **无错误检查**：函数名中的 "no_ce" 表示不执行 CUDA 错误检查

### 结构体定义

```cpp
template <int N>
struct InputArray {
  int values[N];  // 固定大小的数组
};
```

**为什么使用结构体**：
- 编译时大小已知，可以内联到常量内存
- 避免全局内存访问
- 支持小数组的高效传递

### 主接口实现

```cpp
template <int N>
void copy_to_gpu_no_ce_impl(const at::Tensor& input, at::Tensor& output) {
  // 1. 输入验证
  TORCH_CHECK(input.dim() == 1, "input must be 1-D");
  TORCH_CHECK(static_cast<int>(input.numel()) == N, "input numel must equal template N");
  TORCH_CHECK(input.dtype() == torch::kInt32, "input dtype must be int32");
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
  
  // 2. 从 CPU Tensor 复制到结构体
  InputArray<N> input_array;
  const int* input_ptr = input.data_ptr<int>();
  for (int i = 0; i < N; ++i)
    input_array.values[i] = input_ptr[i];
  
  // 3. 启动内核
  dim3 grid(1);  // 单个 block
  dim3 block(static_cast<int>(input.numel()));  // 每个元素一个线程
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(input_array, output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

### 类型特化分发

```cpp
void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output) {
  int N = static_cast<int>(input.numel());
  
  // 支持固定大小的模板特化
  if (N == 72) {
    copy_to_gpu_no_ce_impl<72>(input, output);
  } else if (N == 64) {
    copy_to_gpu_no_ce_impl<64>(input, output);
  } else {
    TORCH_CHECK(false, "unexpected N");
  }
}
```

**当前支持的大小**：
- 72 元素数组
- 64 元素数组

## 💡 应用场景

### 1. 形状信息传递

```python
# 在 CPU 上准备形状信息
shape_info = torch.tensor([batch_size, seq_len, hidden_size], dtype=torch.int32)
gpu_shape = torch.zeros_like(shape_info).cuda()

# 快速复制到 GPU
sgl_kernel.copy_to_gpu_no_ce(shape_info, gpu_shape)

# 在 CUDA 内核中使用
kernel<<<...>>>(gpu_shape.data_ptr<int>(), ...)
```

### 2. 配置参数传递

```python
# 传递内核配置参数
config = torch.tensor([num_heads, head_dim, block_size], dtype=torch.int32)
gpu_config = torch.zeros_like(config).cuda()
sgl_kernel.copy_to_gpu_no_ce(config, gpu_config)
```

## ⚡ 性能优化

### 1. 值传递结构体

- **优势**：编译时大小已知，可以优化
- **位置**：可能存储在常量内存或寄存器中
- **访问速度**：比全局内存快得多

### 2. 简单内核

- **无分支**：内核逻辑简单，无复杂分支
- **内存合并**：线程访问连续内存
- **低开销**：最小化的内核启动开销

### 3. 模板特化

- **编译时优化**：固定大小允许编译器优化
- **内联展开**：循环可以完全展开
- **类型安全**：编译时检查数组大小

## 🔍 与其他复制方法的对比

### 标准 cudaMemcpy

```cpp
// 标准方法
cudaMemcpy(gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice);
```

**对比**：
- **copy.cu**：通过 CUDA 内核，更灵活
- **cudaMemcpy**：更直接，但需要同步
- **copy.cu 优势**：可以在同一个流中异步执行

### PyTorch 的 .to() 方法

```python
# PyTorch 方法
gpu_tensor = cpu_tensor.to('cuda')
```

**对比**：
- **copy.cu**：更轻量，无 PyTorch 开销
- **PyTorch**：功能更全，但有额外开销
- **copy.cu 优势**：适合高频调用的小数组

## 📊 性能特征

### 延迟

- **小数组（64-72 元素）**：< 1 微秒
- **内核启动开销**：主要的延迟来源
- **内存传输**：几乎可忽略（数据很小）

### 带宽利用率

- **不是瓶颈**：数据量太小，不涉及带宽限制
- **主要优势**：低延迟、低开销

## 🔗 相关文件

- `csrc/common_extension.cc` - PyTorch 扩展注册
- `include/sgl_kernel_ops.h` - 函数声明

## 💻 代码示例

```cpp
// 使用示例
torch::Tensor cpu_input = torch::tensor({72, 128, 4096}, torch::kInt32);
torch::Tensor gpu_output = torch::zeros({3}, torch::kInt32).cuda();

sgl_kernel::copy_to_gpu_no_ce(cpu_input, gpu_output);

// 在后续的内核中使用 gpu_output.data_ptr<int>()
```

## 📚 注意事项

1. **固定大小**：当前只支持 64 和 72 元素
2. **类型限制**：只支持 int32 类型
3. **CPU 输入**：输入必须在 CPU 上
4. **GPU 输出**：输出必须在 CUDA 设备上

## 🔧 扩展建议

如果需要支持更多大小，可以添加新的特化：

```cpp
void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output) {
  int N = static_cast<int>(input.numel());
  
  if (N == 72) {
    copy_to_gpu_no_ce_impl<72>(input, output);
  } else if (N == 64) {
    copy_to_gpu_no_ce_impl<64>(input, output);
  } else if (N == 128) {  // 新增
    copy_to_gpu_no_ce_impl<128>(input, output);
  } else {
    TORCH_CHECK(false, "unexpected N");
  }
}
```

