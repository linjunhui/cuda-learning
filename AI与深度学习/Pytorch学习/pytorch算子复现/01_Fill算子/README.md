# Fill 算子演示

## 📖 简介

这是一个简化版的 Fill 算子实现，展示了如何使用 CUDA 实现张量填充操作。

## 🎯 功能说明

Fill 算子使用指定的标量值填充整个张量：
- **输入**：张量（GPU 内存）和填充值
- **输出**：填充后的张量

## 🔧 编译

### 前置条件
- CUDA Toolkit（推荐 11.0 或更高版本）
- NVIDIA GPU
- GCC 或 Clang 编译器

### 编译命令

```bash
# 使用 nvcc 编译
nvcc -o fill_demo fill_demo.cu -arch=sm_75

# 或者指定更通用的架构
nvcc -o fill_demo fill_demo.cu -arch=sm_60

# 如果需要调试信息
nvcc -g -G -o fill_demo fill_demo.cu -arch=sm_75

# 优化版本
nvcc -O3 -o fill_demo fill_demo.cu -arch=sm_75
```

### 架构选择

根据你的 GPU 架构选择：
- **sm_60**: Pascal (GTX 10xx)
- **sm_75**: Turing (RTX 20xx, GTX 16xx)
- **sm_80**: Ampere (RTX 30xx, A100)
- **sm_86**: Ampere (RTX 30xx 移动版)
- **sm_89**: Ada Lovelace (RTX 40xx)

查看你的 GPU 架构：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## 🚀 运行

```bash
./fill_demo
```

## 📝 代码结构

### 1. FillFunctor（函数对象）

```cpp
template<typename T>
struct FillFunctor {
    T value;
    FillFunctor(T v): value(v) {}
    __device__ __forceinline__ T operator() () const {
        return value;
    }
};
```

**说明**：
- 这是 C++ 的 Functor 模式
- 重载 `operator()` 使对象可像函数一样调用
- `__device__` 表示可在 GPU 上执行

### 2. CUDA Kernel（两个版本）

#### 版本 1：使用 Functor
```cpp
template<typename T>
__global__ void fill_kernel(T* output, int64_t numel, FillFunctor<T> functor) {
    // Grid-Stride Loop 模式
    // 每个线程处理多个元素
}
```

#### 版本 2：直接传递值（简化版）
```cpp
template<typename T>
__global__ void fill_kernel_simple(T* output, int64_t numel, T value) {
    // 更简单，避免 functor 拷贝
}
```

**Grid-Stride Loop 模式**：
- 每个线程处理的元素数 = 总元素数 / (blocks × threads)
- 支持任意大小的张量
- 自动处理边界情况

### 3. 主机端封装函数

- `fill_cuda()`: 使用 Functor 版本
- `fill_cuda_simple()`: 使用简化版本

## 🧪 测试说明

程序包含三个测试：

1. **测试 1**: 使用 FillFunctor 填充 float 类型张量为 1.0
2. **测试 2**: 使用简化版本填充 float 类型张量为 2.5
3. **测试 3**: 测试 int 类型，填充值为 42

每个测试都会：
- 在 GPU 上分配内存
- 启动 kernel 执行填充
- 将结果复制回 CPU
- 验证结果是否正确

## 📊 性能优化说明

### Grid-Stride Loop 的优势

```cpp
int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
int64_t stride = blockDim.x * gridDim.x;
for (int64_t i = idx; i < numel; i += stride) {
    // 处理元素
}
```

**优点**：
- 支持任意大小的数组（不需要是线程数的倍数）
- 自动负载均衡
- 简化边界检查

### 参数选择

- **threads_per_block = 256**: 经验值，平衡占用率和寄存器使用
- **max_blocks = 1024**: 限制最大 block 数，避免过度并行

## 🔍 调试技巧

### 1. 添加 CUDA 错误检查

```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
}
```

### 2. 使用 CUDA-GDB 调试

```bash
# 编译调试版本
nvcc -g -G -o fill_demo fill_demo.cu

# 使用 cuda-gdb
cuda-gdb ./fill_demo
```

### 3. 使用 nsight-compute 分析性能

```bash
ncu --set full ./fill_demo
```

### 4. 打印调试信息（在 kernel 中）

```cpp
__global__ void fill_kernel(...) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Debug info\n");
    }
}
```

## 📚 扩展阅读

### 与 PyTorch 实现的对比

PyTorch 的 Fill 算子使用 `TensorIterator` 来处理：
- 自动广播
- 不同内存布局
- 类型转换

我们的简化版本只处理了简单的 contiguous 内存布局。

### 下一步学习

1. **支持非 contiguous 内存布局**：需要 stride 信息
2. **支持广播**：处理不同形状的张量
3. **支持多种数据类型**：自动类型转换
4. **向量化优化**：一次加载/存储多个元素

## ⚠️ 常见问题

### 问题 1: "no kernel image is available"

**原因**: GPU 架构不匹配

**解决**: 重新编译时指定正确的架构，或使用更通用的架构：
```bash
nvcc -arch=sm_60 -o fill_demo fill_demo.cu
```

### 问题 2: 结果不正确

**可能原因**:
- 内存未正确分配/释放
- Kernel 启动参数错误
- 没有同步等待 kernel 完成

**检查**:
- 添加 `cudaDeviceSynchronize()` 确保 kernel 完成
- 检查 CUDA 错误

### 问题 3: 性能不佳

**优化建议**:
- 增加每个线程处理的元素数（向量化）
- 使用共享内存（如果需要）
- 调整 block 大小

## 📖 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch CUDA 算子实现详解](../CUDA算子实现详解.md)
- [Grid-Stride Loop Pattern](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)

