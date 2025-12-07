# CUTLASS 介绍

## 目录
1. [什么是 CUTLASS](#什么是-cutlass)
2. [CUTLASS 的核心特性](#cutlass-的核心特性)
3. [CUTLASS 架构](#cutlass-架构)
4. [CUTE 库](#cute-库)
5. [基本使用示例](#基本使用示例)
6. [CUTLASS 的优势](#cutlass-的优势)
7. [适用场景](#适用场景)
8. [参考资料](#参考资料)

---

## 什么是 CUTLASS

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) 是 NVIDIA 开发的一个高性能 CUDA C++ 模板库，专门用于实现高性能的线性代数运算，特别是矩阵乘法（GEMM）和相关的计算内核。

CUTLASS 提供了：
- **模块化设计**：通过模板元编程实现高度可配置的 CUDA 内核
- **高性能**：针对不同 GPU 架构优化的 GEMM 实现
- **灵活性**：支持多种数据类型、布局和计算模式
- **可扩展性**：易于定制和扩展以满足特定需求

---

## CUTLASS 的核心特性

### 1. 模板化设计
CUTLASS 使用 C++ 模板实现，允许在编译时进行优化，生成高度优化的 CUDA 内核代码。

### 2. 多精度支持
支持多种数据类型：
- **FP32** (单精度浮点数)
- **FP16** (半精度浮点数)
- **FP64** (双精度浮点数)
- **INT8/INT4** (整数类型)
- **BF16** (Bfloat16)
- **TF32** (TensorFloat-32)

### 3. 多种计算模式
- **GEMM** (General Matrix Multiply)
- **GEMM + Bias**
- **GEMM + Activation**
- **Convolution**
- **Batch GEMM**

### 4. 架构优化
针对不同 GPU 架构提供优化实现：
- **Volta** (SM 7.0)
- **Turing** (SM 7.5)
- **Ampere** (SM 8.0)
- **Ada Lovelace** (SM 8.9)
- **Hopper** (SM 9.0)

---

## CUTLASS 架构

CUTLASS 采用分层架构设计：

```
┌─────────────────────────────────────┐
│      CUTLASS API (高级接口)          │
├─────────────────────────────────────┤
│      GEMM 操作符 (Operators)         │
├─────────────────────────────────────┤
│      线程块级平铺 (Threadblock)      │
├─────────────────────────────────────┤
│      线程级平铺 (Thread)              │
├─────────────────────────────────────┤
│      Warp 级操作 (Warp-level)        │
├─────────────────────────────────────┤
│      Tensor Core / MMA 操作          │
└─────────────────────────────────────┘
```

### 关键组件

1. **Threadblock Tile**
   - 定义线程块处理的数据块大小
   - 管理共享内存的使用

2. **Warp Tile**
   - 定义 warp 处理的数据块
   - 协调 warp 间的数据共享

3. **Thread Tile**
   - 定义单个线程处理的数据
   - 管理寄存器使用

4. **MMA (Matrix Multiply-Accumulate)**
   - 利用 Tensor Core 进行矩阵运算
   - 支持不同的 MMA 指令变体

---

## CUTE 库

**CUTE** (CUDA Template for Efficient) 是 CUTLASS 3.x 引入的新库，提供了更现代、更灵活的抽象。

### CUTE 的核心概念

#### 1. Tensor 抽象
CUTE 提供了统一的 Tensor 抽象，可以表示不同内存层次的数据：

```cpp
#include "cute/tensor.hpp"

using namespace cute;

// 创建全局内存 Tensor
Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));

// 创建局部 Tile
Tensor tzr = local_tile(tz, make_shape(Int<8>{}), make_coord(idx));
```

#### 2. Layout 系统
CUTE 使用 Layout 来描述数据在内存中的排列方式，支持：
- **行主序 (Row-major)**
- **列主序 (Column-major)**
- **自定义布局**
- **分块布局 (Tiled Layout)**

#### 3. Copy 操作
CUTE 提供了高效的 Copy 操作，可以自动选择最优的内存访问模式：

```cpp
// 从全局内存复制到寄存器
copy(txr, txR);  // LDG.128 (128位加载)

// 从寄存器复制到全局内存
copy(tzRx, tzr); // STG.128 (128位存储)
```

#### 4. MMA 操作
CUTE 提供了灵活的 MMA 抽象：

```cpp
// 创建 Tiled MMA
auto tiled_mma = make_tiled_mma(
    SM80_16x8x16_F32F16F16F32_TN{},  // MMA 原子操作
    Layout<Shape<_4, _1, _1>>{},      // Atom 布局
    Layout<Shape<_1, _2, _1>>{}       // Value 布局
);
```

---

## 基本使用示例

### 示例 1: 向量加法

以下是一个使用 CUTE 实现的向量加法示例：

```cpp
#include "cute/tensor.hpp"

template <int kNumElemPerThread = 8>
__global__ void vector_add(
    half *z, int num, 
    const half *x, const half *y, 
    const half a, const half b, const half c) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / kNumElemPerThread) {
    return;
  }

  // 创建全局内存 Tensor
  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

  // 创建局部 Tile (每个线程处理 kNumElemPerThread 个元素)
  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

  // 创建寄存器 Tensor
  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // 从全局内存加载到寄存器 (LDG.128)
  copy(txr, txR);
  copy(tyr, tyR);

  // 计算: z = a*x + b*y + c
  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  auto tzR2 = recast<half2>(tzR);
  auto txR2 = recast<half2>(txR);
  auto tyR2 = recast<half2>(tyR);

  #pragma unroll
  for (int i = 0; i < size(tzR2); ++i) {
    tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
  }

  // 写回全局内存 (STG.128)
  auto tzRx = recast<half>(tzR2);
  copy(tzRx, tzr);
}
```

**关键点：**
- 使用 `local_tile` 将全局内存划分为线程级别的块
- 使用 `copy` 操作进行高效的内存传输
- 使用 `recast` 进行类型转换以利用向量化指令

### 示例 2: GEMM 操作

CUTLASS 提供了高级的 GEMM API：

```cpp
#include "cutlass/gemm/device/gemm.h"

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                    // ElementA
    cutlass::layout::RowMajor,          // LayoutA
    cutlass::half_t,                    // ElementB
    cutlass::layout::ColumnMajor,       // LayoutB
    cutlass::half_t,                    // ElementC
    cutlass::layout::RowMajor           // LayoutC
>;

Gemm::Arguments arguments{
    {M, N, K},                          // Problem size
    {A, lda},                           // Tensor A
    {B, ldb},                           // Tensor B
    {C, ldc},                           // Tensor C
    {D, ldd},                           // Tensor D
    {alpha, beta}                       // Epilogue parameters
};

Gemm gemm_op;
gemm_op(arguments);
```

---

## CUTLASS 的优势

### 1. 性能优势
- **接近硬件峰值性能**：针对不同 GPU 架构深度优化
- **自动选择最优配置**：根据问题规模和硬件特性自动选择最佳实现
- **充分利用 Tensor Core**：高效利用现代 GPU 的 Tensor Core

### 2. 开发优势
- **模块化设计**：易于理解和定制
- **类型安全**：编译时类型检查
- **丰富的示例**：提供大量示例代码

### 3. 灵活性
- **可定制**：可以针对特定需求进行定制
- **可扩展**：易于添加新的数据类型和操作
- **跨架构**：支持多种 GPU 架构

---

## 适用场景

CUTLASS 特别适用于：

1. **深度学习框架**
   - 实现高效的 GEMM 操作
   - 优化卷积神经网络
   - 加速 Transformer 模型

2. **高性能计算**
   - 科学计算中的矩阵运算
   - 线性代数库的实现

3. **自定义内核开发**
   - 需要高性能矩阵运算的应用
   - 需要精细控制内存访问模式的应用

4. **研究和开发**
   - 研究新的矩阵运算算法
   - 开发新的 GPU 优化技术

---

## 参考资料

### 官方资源
- **GitHub**: https://github.com/NVIDIA/cutlass
- **文档**: https://github.com/NVIDIA/cutlass/tree/main/media/docs
- **示例**: https://github.com/NVIDIA/cutlass/tree/main/examples

### 关键概念
- **CUTE 库**: CUTLASS 3.x 的核心抽象库
- **MMA (Matrix Multiply-Accumulate)**: Tensor Core 的矩阵运算指令
- **Tiled Layout**: 分块布局，用于优化内存访问
- **Epilogue**: GEMM 操作的后处理阶段（如激活函数、偏置等）

### 学习路径
1. 从简单的向量操作开始（如向量加法）
2. 理解 CUTE 的 Tensor 和 Layout 概念
3. 学习基本的 GEMM 操作
4. 深入理解 MMA 和 Tiled MMA
5. 探索高级特性（Epilogue、Fusion 等）

---

## 总结

CUTLASS 是一个强大的 CUDA 模板库，为高性能线性代数运算提供了灵活且高效的解决方案。通过 CUTE 库提供的现代抽象，开发者可以更容易地编写高性能的 CUDA 内核，同时充分利用 GPU 的硬件特性。

无论是开发深度学习框架、高性能计算应用，还是进行 GPU 编程研究，CUTLASS 都是一个值得深入学习和使用的工具。
