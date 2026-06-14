# CUTLASS GEMM 使用教程

## 目录
1. [CUTLASS 简介](#cutlass-简介)
2. [环境准备](#环境准备)
3. [项目结构](#项目结构)
4. [核心实现解析](#核心实现解析)
5. [使用方法](#使用方法)
6. [性能测试](#性能测试)
7. [常见问题](#常见问题)

---

## CUTLASS 简介

CUTLASS (CUDA Templates for Linear Algebra Subroutines) 是 NVIDIA 开发的一个 CUDA C++ 模板库，专门用于高性能的线性代数运算，特别是矩阵乘法（GEMM）操作。

### 主要特点：
- **高性能**：针对不同 GPU 架构优化的 GEMM 实现
- **灵活配置**：通过模板参数可以自定义线程块大小、warp 形状等
- **易于集成**：可以与 PyTorch、TensorFlow 等框架集成
- **支持多种数据类型**：float32、float16、int8 等

---

## 环境准备

### 1. 系统要求
- CUDA Toolkit (>= 11.0)
- Python 3.7+
- PyTorch (支持 CUDA)
- CUTLASS 库

### 2. 安装 CUTLASS

```bash
# 克隆 CUTLASS 仓库
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# 或者下载特定版本
git clone --branch v2.11.0 https://github.com/NVIDIA/cutlass.git
```

### 3. 项目结构

```
00_cutlass/
├── cutlass/                    # CUTLASS 库目录
│   └── include/               # CUTLASS 头文件
├── cutlass_matrix_multiply.cu  # CUDA 实现文件
├── gemm_benchmark.py          # Python 测试脚本
└── CUTLASS_GEMM教程.md        # 本教程文档
```

---

## 核心实现解析

### 1. 头文件包含

```cpp
#include <torch/extension.h>      // PyTorch 扩展支持
#include <cuda_runtime.h>        // CUDA 运行时
#include "cutlass/cutlass.h"     // CUTLASS 核心
#include "cutlass/gemm/device/gemm.h"  // GEMM 设备端实现
```

### 2. 数据类型和布局定义

```cpp
// 数据类型：使用 float32
using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;

// 内存布局：行主序（Row Major）
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
```

**说明**：
- **行主序 (Row Major)**：数据按行存储，如 `A[i][j]` 在内存中连续
- **列主序 (Column Major)**：数据按列存储，如 `A[i][j]` 和 `A[i+1][j]` 在内存中连续

### 3. 线程块和 Warp 配置

```cpp
// 线程块形状：128 x 256 x 32
// 表示每个线程块处理 128 行 x 256 列 x 32 深度的矩阵块
using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;

// Warp 形状：64 x 64 x 32
// 每个 warp 处理 64 x 64 x 32 的矩阵块
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
```

**配置选择原则**：
- **大矩阵**：使用更大的 ThreadblockShape（如 256x128x32）
- **小矩阵**：使用较小的 ThreadblockShape（如 64x64x32）
- **内存带宽受限**：增加 K 维度（如 128x128x64）

### 4. GEMM 操作定义

```cpp
using GemmOperation = cutlass::gemm::device::GemmUniversalAdapter<
    ThreadblockShape,    // 线程块形状
    WarpShape,           // Warp 形状
    ElementA, LayoutA,   // A 矩阵类型和布局
    ElementB, LayoutB,   // B 矩阵类型和布局
    ElementC, LayoutC,   // C 矩阵类型和布局
    ElementAccumulator    // 累加器类型
>;
```

### 5. 参数设置

```cpp
// 计算 leading dimensions（行主序）
int lda = N;  // A 是 M x N，行主序，所以 lda = N
int ldb = K;  // B 是 N x K，行主序，所以 ldb = K
int ldc = K;  // C 是 M x K，行主序，所以 ldc = K

// GEMM 参数
typename GemmOperation::Arguments arguments{
    cutlass::gemm::GemmCoord(M, N, K),  // 问题规模：M x N x K
    {A_ptr, lda},                        // A 矩阵指针和 leading dimension
    {B_ptr, ldb},                        // B 矩阵指针和 leading dimension
    {C_ptr, ldc},                        // C 矩阵指针（输入）
    {C_ptr, ldc},                        // D 矩阵指针（输出，这里 C 和 D 相同）
    {1.0f, 0.0f}                         // alpha=1.0, beta=0.0
};
```

**GEMM 公式**：`D = alpha * A * B + beta * C`
- `alpha = 1.0`：A * B 的系数
- `beta = 0.0`：C 的系数（0 表示不累加，直接覆盖）

### 6. 执行 GEMM

```cpp
// 创建 GEMM 操作实例
GemmOperation gemm_op;

// 初始化
cutlass::Status status = gemm_op.initialize(arguments);
TORCH_CHECK(status == cutlass::Status::kSuccess, "初始化失败");

// 执行（异步）
status = gemm_op(stream);
TORCH_CHECK(status == cutlass::Status::kSuccess, "执行失败");

// 同步等待完成
cudaStreamSynchronize(stream);
```

---

## 使用方法

### 1. 基本使用

```python
import torch
from torch.utils.cpp_extension import load

# 加载 CUDA 模块
lib = load(
    name="gemm_cuda",
    sources=["cutlass_matrix_multiply.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    include_dirs=["./cutlass/include"],
    extra_cflags=["-std=c++17"],
)

# 创建矩阵
M, N, K = 1024, 1024, 1024
A = torch.randn(M, N, dtype=torch.float32, device='cuda')
B = torch.randn(N, K, dtype=torch.float32, device='cuda')
C = torch.zeros(M, K, dtype=torch.float32, device='cuda')

# 执行矩阵乘法：C = A * B
lib.cutlass_gemm(A, B, C)

# 验证结果
C_torch = torch.matmul(A, B)
print(f"最大误差: {(C - C_torch).abs().max().item()}")
```

### 2. 性能测试

运行完整的基准测试：

```bash
python gemm_benchmark.py
```

测试脚本会：
1. 测试不同大小的矩阵（32x32 到 1024x1024）
2. 对比 CUTLASS、PyTorch 和 NumPy 的性能
3. 验证计算精度
4. 生成性能对比图

### 3. 自定义测试

```python
import time
import torch

def benchmark_cutlass(M, N, K, iterations=100):
    """自定义性能测试"""
    A = torch.randn(M, N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, K, dtype=torch.float32, device='cuda')
    C = torch.zeros(M, K, dtype=torch.float32, device='cuda')
    
    # 预热
    for _ in range(10):
        lib.cutlass_gemm(A, B, C)
    torch.cuda.synchronize()
    
    # 计时
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.time()
        lib.cutlass_gemm(A, B, C)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    gflops = (2.0 * M * N * K) / avg_time / 1e9
    print(f"大小: {M}x{N}x{K}, 时间: {avg_time*1000:.3f} ms, 性能: {gflops:.2f} GFLOPS")
    
    return avg_time, gflops

# 测试不同大小
for size in [256, 512, 1024, 2048]:
    benchmark_cutlass(size, size, size)
```

---

## 性能测试

### 性能指标

**GFLOPS 计算**：
```
GFLOPS = (2 × M × N × K) / 时间(秒) / 10^9
```

其中 `2 × M × N × K` 是矩阵乘法的浮点运算次数（每个元素需要一次乘法和一次加法）。

### 典型性能

在 NVIDIA RTX 3090 上的典型性能：
- 512x512: ~500-800 GFLOPS
- 1024x1024: ~800-1200 GFLOPS
- 2048x2048: ~1000-1500 GFLOPS

**注意**：实际性能取决于：
- GPU 架构（Turing、Ampere、Ada 等）
- 矩阵大小和形状
- 内存带宽
- CUDA 版本

---

## 常见问题

### 1. 编译错误：找不到 CUTLASS 头文件

**问题**：
```
fatal error: cutlass/cutlass.h: No such file or directory
```

**解决方案**：
```python
# 确保 include_dirs 指向正确的路径
include_dirs=["./cutlass/include"]  # 相对路径
# 或
include_dirs=["/path/to/cutlass/include"]  # 绝对路径
```

### 2. 模板参数错误

**问题**：编译时出现模板参数不匹配错误

**解决方案**：
- 检查 CUTLASS 版本，不同版本的 API 可能不同
- 确保 `GemmUniversalAdapter` 的模板参数顺序正确
- 参考 CUTLASS 官方文档或示例代码

### 3. 结果精度问题

**问题**：CUTLASS 结果与 PyTorch 结果有较大差异

**可能原因**：
- 使用了 `--use_fast_math` 导致精度降低
- 累加顺序不同导致的舍入误差
- 数据类型不匹配

**解决方案**：
```cpp
// 移除 --use_fast_math 以获得更高精度
extra_cuda_cflags=["-O3"]  // 不使用 --use_fast_math
```

### 4. 性能不如预期

**优化建议**：
1. **调整线程块大小**：根据矩阵大小选择合适的 ThreadblockShape
2. **使用 Tensor Core**：对于支持的 GPU，使用 float16 或 int8
3. **批处理**：对于多个小矩阵，考虑批处理
4. **内存对齐**：确保矩阵数据对齐到 16 字节边界

### 5. PyTorch 绑定问题

**问题**：`PYBIND11_MODULE` 宏未定义

**解决方案**：
```cpp
// 确保包含正确的头文件
#include <torch/extension.h>

// 使用正确的模块名称
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_gemm", &cutlass_gemm, "CUTLASS GEMM");
}
```

---

## 进阶使用

### 1. 支持不同的数据类型

```cpp
// 使用 float16
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
```

### 2. 使用列主序布局

```cpp
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

// 注意：leading dimension 需要相应调整
int lda = M;  // 列主序时，lda = M
int ldb = N;  // 列主序时，ldb = N
int ldc = M;  // 列主序时，ldc = M
```

### 3. 使用 Tensor Core

```cpp
// 使用 Tensor Core 需要特定的配置
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;  // Tensor Core 指令形状
```

### 4. 批处理 GEMM

```cpp
// 使用 CUTLASS 的批处理 GEMM API
using BatchedGemmOperation = cutlass::gemm::device::BatchedGemmUniversalAdapter<...>;
```

---

## 参考资料

1. **CUTLASS 官方文档**：https://github.com/NVIDIA/cutlass
2. **CUTLASS 示例**：`cutlass/examples/` 目录
3. **PyTorch C++ 扩展**：https://pytorch.org/tutorials/advanced/cpp_extension.html
4. **CUDA 编程指南**：https://docs.nvidia.com/cuda/

---

## 总结

本教程介绍了如何使用 CUTLASS 实现高性能的矩阵乘法操作。关键点：

1. **正确配置**：选择合适的线程块大小和数据类型
2. **理解布局**：行主序和列主序的区别
3. **性能优化**：根据矩阵大小和 GPU 架构调整参数
4. **错误处理**：检查 CUTLASS 返回的状态码

通过本教程，你应该能够：
- 理解 CUTLASS GEMM 的基本原理
- 在自己的项目中使用 CUTLASS
- 进行性能测试和优化
- 解决常见问题

如有问题，请参考 CUTLASS 官方文档或提交 Issue。
