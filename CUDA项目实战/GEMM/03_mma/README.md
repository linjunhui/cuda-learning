# Tensor Core WMMA 矩阵乘法 Demo

本目录演示如何使用 CUDA Tensor Core 的 WMMA (Warp-level Matrix Multiply-Accumulate) API 实现高性能矩阵乘法。

## 文件说明

- `mma.cu`: 使用 WMMA API 的 CUDA 核函数实现
- `mma_benchmark.py`: 性能测试和基准测试脚本
- `README.md`: 本文件

## 前置要求

### 硬件要求
- **GPU 计算能力 >= 7.0** (Volta/Turing/Ampere/Ada/Hopper 架构)
- 支持的 GPU 示例：
  - NVIDIA V100 (Volta, sm_70)
  - NVIDIA RTX 20/30 系列 (Turing/Ampere, sm_75/sm_80)
  - NVIDIA A100 (Ampere, sm_80)
  - NVIDIA RTX 40 系列 (Ada, sm_89)
  - NVIDIA H100 (Hopper, sm_90)

### 软件要求
- CUDA Toolkit >= 11.0
- PyTorch (支持 CUDA)
- Python 3.x
- NumPy, Matplotlib

## WMMA API 简介

WMMA (Warp-level Matrix Multiply-Accumulate) 是 CUDA 提供的高级 API，用于访问 Tensor Core。

### 支持的矩阵大小
- **16×16×16**: 最常用的配置
- **8×32×16**: 适合某些特定场景
- **32×8×16**: 适合某些特定场景

### 数据类型
- **half precision (FP16)**: 本 demo 使用此类型
- **Tensor Core 专为 half precision 优化**

### 关键 API
```cpp
// 加载矩阵片段
wmma::load_matrix_sync(a_frag, &A[row * N + col], N);

// 执行矩阵乘法累加
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储结果
wmma::store_matrix_sync(&C[row * K + col], c_frag, K, wmma::mem_row_major);
```

## 使用方法

### 1. 运行基准测试

```bash
cd 03_mma
python mma_benchmark.py
```

### 2. 代码示例

```python
import torch
from torch.utils.cpp_extension import load

# 加载 CUDA 模块
lib = load(
    name="mma_cuda",
    sources=["mma.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_70"],
    extra_cflags=["-std=c++17"],
)

# 准备数据 (必须是 float16)
M, N, K = 1024, 1024, 1024
A = torch.randn(M, N, dtype=torch.float16, device='cuda')
B = torch.randn(N, K, dtype=torch.float16, device='cuda')
C = torch.zeros(M, K, dtype=torch.float16, device='cuda')

# 执行矩阵乘法
lib.mma_gemm(A, B, C)

# 验证结果
C_torch = torch.matmul(A, B)
print(f"最大误差: {(C.float() - C_torch.float()).abs().max().item()}")
```

## 实现细节

### 核函数设计

1. **Tile 大小**: 每个 warp (32 个线程) 处理一个 16×16 的输出 tile
2. **Block 配置**: `(32, 4)` = 4 个 warp，每个 warp 32 个线程
3. **Grid 配置**: 根据矩阵大小自动计算

### 矩阵布局
- **A 矩阵**: Row-major (行主序)
- **B 矩阵**: Column-major (列主序) - WMMA API 要求
- **C 矩阵**: Row-major (行主序)

### 性能优化要点

1. **使用 half precision**: Tensor Core 专为 FP16 优化
2. **Tile 大小对齐**: 矩阵维度最好是 16 的倍数
3. **内存访问模式**: WMMA API 自动处理内存访问优化

## 性能对比

运行 `mma_benchmark.py` 会输出：
- WMMA 实现的时间
- PyTorch `torch.matmul` 的时间
- GFLOPS 性能指标
- 性能对比图

### 预期性能

在支持的 GPU 上，WMMA 实现通常可以达到：
- **V100**: ~100-200 GFLOPS (FP16)
- **RTX 3090**: ~200-400 GFLOPS (FP16)
- **A100**: ~300-600 GFLOPS (FP16)

*注意：实际性能取决于矩阵大小、GPU 型号和系统配置*

## 常见问题

### Q: 为什么需要 half precision？
A: Tensor Core 专为 half precision (FP16) 优化，使用 FP32 无法利用 Tensor Core。

### Q: 如何检查 GPU 是否支持？
A: 运行 `mma_benchmark.py`，脚本会自动检查 GPU 计算能力。

### Q: 精度如何？
A: FP16 的精度较低，误差通常在 1e-2 到 1e-1 范围内，这是正常的。

### Q: 为什么性能不如预期？
A: 可能的原因：
- 矩阵大小不是 16 的倍数
- GPU 频率未达到最高
- 有其他程序占用 GPU
- 内存带宽成为瓶颈

## 扩展阅读

- [CUDA WMMA API 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-level-matrix-functions)
- [Tensor Core 编程指南](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)
- [PyTorch C++ 扩展文档](https://pytorch.org/tutorials/advanced/cpp_extension.html)

## 注意事项

1. **精度限制**: FP16 精度较低，不适合需要高精度的应用
2. **硬件要求**: 必须使用计算能力 >= 7.0 的 GPU
3. **矩阵大小**: 建议使用 16 的倍数以获得最佳性能
4. **编译标志**: 确保使用正确的 `-arch=sm_XX` 标志
