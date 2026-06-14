# CUTLASS 和 CuTe 入门

## 📚 学习目标

1. 理解 CUTLASS 的基本概念
2. 理解 CuTe 张量抽象
3. 掌握基本的 CUTLASS/CuTe 使用
4. 理解 Flash-Attention 中的 CUTLASS/CuTe 使用

---

## 🎯 CUTLASS 简介

### 什么是 CUTLASS

**CUTLASS**（CUDA Templates for Linear Algebra Subroutines）：
- NVIDIA 的 CUDA C++ 模板库
- 提供高性能的 GEMM（矩阵乘法）实现
- 支持 Tensor Core
- 使用模板元编程实现高度优化

### CUTLASS 的核心概念

#### 1. Tile（瓦片）

**定义**：计算的基本单位，将大矩阵分成小块

**层次结构**：
```
Thread Block Tile (线程块瓦片)
  └── Warp Tile (Warp 瓦片)
        └── Thread Tile (线程瓦片)
```

**示例**：
```cuda
// Thread Block Tile: 128×128
// Warp Tile: 64×64
// Thread Tile: 8×8
```

#### 2. Layout（布局）

**定义**：描述数据在内存中的排列方式

**常见布局**：
- **Row Major**：按行存储
- **Column Major**：按列存储
- **Swizzle**：交错排列（避免 bank conflict）

#### 3. MMA（Matrix Multiply-Accumulate）

**定义**：矩阵乘累加操作，Tensor Core 的核心操作

**Tensor Core**：
- 专门用于矩阵乘法
- 支持 FP16、BF16、INT8、INT4 等数据类型
- 性能比普通 CUDA Core 高很多

---

## 🧩 CuTe 张量抽象

### 什么是 CuTe

**CuTe**（CUDA Templates）：
- CUTLASS 的一部分
- 提供张量抽象和布局描述
- 简化内存访问模式的定义
- 使用现代 C++ 模板技术

### CuTe 核心概念

#### 1. Tensor（张量）

**定义**：多维数组的抽象

**创建张量**：
```cuda
#include <cute/tensor.hpp>

using namespace cute;

// 从指针创建张量
float* data_ptr = ...;
auto tensor = make_tensor(
    make_gmem_ptr(data_ptr),      // 全局内存指针
    make_shape(M, N),             // 形状 (M, N)
    make_stride(stride_M, stride_N) // 步长
);
```

#### 2. Shape（形状）

**定义**：张量的维度

**示例**：
```cuda
// 2D 张量
auto shape_2d = make_shape(128, 64);  // 128×64

// 3D 张量
auto shape_3d = make_shape(32, 128, 64);  // 32×128×64

// 动态形状
int M = 128, N = 64;
auto shape_dynamic = make_shape(M, N);
```

#### 3. Stride（步长）

**定义**：每个维度在内存中的步长

**示例**：
```cuda
// Row Major 布局
auto stride_row_major = make_stride(N, 1);  // (stride_M, stride_N)

// Column Major 布局
auto stride_col_major = make_stride(1, M);  // (stride_M, stride_N)

// 自定义步长
auto stride_custom = make_stride(64, 1);  // 每行 64 个元素
```

#### 4. Layout（布局）

**定义**：Shape 和 Stride 的组合

**创建布局**：
```cuda
auto layout = make_layout(
    make_shape(M, N),
    make_stride(stride_M, stride_N)
);
```

### CuTe 操作

#### 1. 张量访问

```cuda
auto tensor = make_tensor(...);

// 访问单个元素
float val = tensor(0, 0);

// 访问一行
auto row = tensor(0, _);  // 第 0 行

// 访问一列
auto col = tensor(_, 0);  // 第 0 列

// 切片
auto slice = tensor(0, make_range(0, 64));  // 第 0 行，前 64 列
```

#### 2. 张量操作

```cuda
// 转置
auto transposed = make_tensor(tensor.data(), 
                              make_shape(N, M),
                              make_stride(stride_N, stride_M));

// 重塑（Reshape）
auto reshaped = make_tensor(tensor.data(),
                            make_shape(M * N),
                            make_stride(1));

// 局部切片（Local Tile）
auto local_tile = local_tile(tensor, 
                             make_shape(64, 64),  // Tile 大小
                             make_coord(0, 0));   // Tile 位置
```

---

## 🔧 Flash-Attention 中的 CUTLASS/CuTe 使用

### Q、K、V 张量的定义

**代码示例**（简化）：
```cuda
#include <cute/tensor.hpp>
using namespace cute;

// 定义 Q 张量
auto q_tensor = make_tensor(
    make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + 
                   binfo.q_offset(params.q_batch_stride, 
                                 params.q_row_stride, 
                                 bidb)),
    make_shape(binfo.actual_seqlen_q, params.h, params.d),
    make_stride(params.q_row_stride, 
                params.q_head_stride, 
                _1{})
);

// 获取 Q 块
auto q_block = local_tile(q_tensor(_, bidh, _), 
                          Shape<Int<kBlockM>, Int<kHeadDim>>{},
                          make_coord(m_block, 0));
```

### 内存布局定义

**Flash-Attention 中的布局**（简化）：
```cuda
// Q 的共享内存布局
using SmemLayoutQ = decltype(
    composition(Swizzle<kSwizzle, 3, 3>{},
                Layout<Shape<Int<kBlockM>, Int<kHeadDim>>,
                       Stride<Int<kHeadDim>, _1>>{})
);

// K 的共享内存布局
using SmemLayoutK = decltype(
    composition(Swizzle<kSwizzle, 3, 3>{},
                Layout<Shape<Int<kBlockN>, Int<kHeadDim>>,
                       Stride<Int<kHeadDim>, _1>>{})
);
```

**Swizzle**：
- 交错排列，避免 bank conflict
- `kSwizzle` 通常是 2 或 3

### GEMM 操作

**使用 CUTLASS GEMM**（简化）：
```cuda
// QK^T 计算
using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,  // Tensor Core 配置
    Layout<Shape<Int<kNWarps>, _1, _1>>,     // Warp 布局
    Tile<Int<16 * kNWarps>, _16, _16>>       // Tile 大小
>;

// 执行 GEMM
cute::gemm(TiledMma{}, 
           q_tile, k_tile, 
           s_tile);  // s_tile = QK^T
```

---

## 📝 实际示例

### 示例 1：简单的矩阵乘法

```cuda
#include <cute/tensor.hpp>
using namespace cute;

__global__ void simple_gemm(float* A, float* B, float* C, 
                             int M, int N, int K) {
    // 创建张量
    auto tensor_A = make_tensor(
        make_gmem_ptr(A),
        make_shape(M, K),
        make_stride(K, 1)  // Row Major
    );
    
    auto tensor_B = make_tensor(
        make_gmem_ptr(B),
        make_shape(K, N),
        make_stride(N, 1)  // Row Major
    );
    
    auto tensor_C = make_tensor(
        make_gmem_ptr(C),
        make_shape(M, N),
        make_stride(N, 1)  // Row Major
    );
    
    // 获取当前线程处理的块
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_M = 64, tile_N = 64;
    int m_tile = tid / (N / tile_N);
    int n_tile = tid % (N / tile_N);
    
    // 获取局部块
    auto A_tile = local_tile(tensor_A, 
                            make_shape(tile_M, K),
                            make_coord(m_tile, 0));
    auto B_tile = local_tile(tensor_B,
                            make_shape(K, tile_N),
                            make_coord(0, n_tile));
    auto C_tile = local_tile(tensor_C,
                            make_shape(tile_M, tile_N),
                            make_coord(m_tile, n_tile));
    
    // 计算（简化，实际需要使用 CUTLASS GEMM）
    for (int i = 0; i < tile_M; i++) {
        for (int j = 0; j < tile_N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A_tile(i, k) * B_tile(k, j);
            }
            C_tile(i, j) = sum;
        }
    }
}
```

### 示例 2：共享内存布局

```cuda
__global__ void shared_memory_layout() {
    __shared__ float smem[128][128];
    
    // 创建共享内存张量
    auto smem_tensor = make_tensor(
        make_smem_ptr(smem),
        make_shape(128, 128),
        make_stride(128, 1)  // Row Major
    );
    
    // 使用 Swizzle 避免 bank conflict
    auto swizzled_layout = composition(
        Swizzle<2, 3, 3>{},  // Swizzle 配置
        make_layout(make_shape(128, 128),
                   make_stride(128, 1))
    );
    
    auto swizzled_tensor = make_tensor(
        make_smem_ptr(smem),
        swizzled_layout
    );
    
    // 访问（自动应用 Swizzle）
    int tid = threadIdx.x;
    float val = swizzled_tensor(tid / 32, tid % 32);
}
```

---

## 🎯 Flash-Attention 中的关键使用

### 1. 张量创建和布局

**Q、K、V 张量**：
```cuda
// 从全局内存指针创建张量
auto q_tensor = make_tensor(
    make_gmem_ptr(q_ptr),
    make_shape(seqlen_q, h, d),
    make_stride(q_row_stride, q_head_stride, 1)
);
```

### 2. 局部切片（Local Tile）

**获取块**：
```cuda
// 获取 Q 块
auto q_block = local_tile(
    q_tensor(_, bidh, _),           // 选择特定的 head
    Shape<Int<kBlockM>, Int<kHeadDim>>{},  // Tile 形状
    make_coord(m_block, 0)          // Tile 坐标
);
```

### 3. 共享内存布局

**Swizzle 布局**：
```cuda
using SmemLayoutQ = decltype(
    composition(
        Swizzle<kSwizzle, 3, 3>{},  // Swizzle
        Layout<Shape<Int<kBlockM>, Int<kHeadDim>>,
               Stride<Int<kHeadDim>, _1>>{}
    )
);
```

### 4. GEMM 操作

**使用 CUTLASS GEMM**：
```cuda
// QK^T
cute::gemm(TiledMma{}, q_tile, k_tile, s_tile);

// PV
cute::gemm(TiledMma{}, p_tile, v_tile, o_tile);
```

---

## 📊 性能优势

### CUTLASS 的优势

1. **高度优化**：
   - 针对不同硬件架构优化
   - 使用 Tensor Core
   - 优化的内存访问模式

2. **灵活性**：
   - 模板元编程
   - 编译时优化
   - 支持多种数据类型

3. **易用性**：
   - CuTe 简化了张量操作
   - 清晰的抽象
   - 易于理解和维护

### Flash-Attention 中的优势

1. **内存访问优化**：
   - Swizzle 避免 bank conflict
   - 合并访问提高带宽

2. **计算优化**：
   - Tensor Core 加速
   - 高效的 GEMM 实现

3. **代码简洁**：
   - CuTe 简化了代码
   - 易于理解和维护

---

## 🎯 关键要点总结

### CUTLASS 要点

1. **Tile 层次结构**：Thread Block → Warp → Thread
2. **Layout 重要性**：影响内存访问性能
3. **Tensor Core**：高性能矩阵乘法

### CuTe 要点

1. **张量抽象**：简化多维数组操作
2. **布局描述**：Shape 和 Stride 的组合
3. **局部切片**：方便处理块数据

### Flash-Attention 中的应用

1. **张量创建**：从指针创建张量
2. **局部切片**：获取 Q、K、V 块
3. **共享内存布局**：使用 Swizzle 优化
4. **GEMM 操作**：使用 CUTLASS 加速

---

## 📝 学习检查点

- [ ] 理解 CUTLASS 的基本概念
- [ ] 理解 CuTe 张量抽象
- [ ] 能够创建和使用张量
- [ ] 理解布局的作用
- [ ] 理解 Flash-Attention 中的使用

---

## 📚 参考资源

### 官方文档
- CUTLASS GitHub: https://github.com/NVIDIA/cutlass
- CuTe 文档: https://github.com/NVIDIA/cutlass/tree/main/cute

### 教程
- CUTLASS 教程: https://github.com/NVIDIA/cutlass/tree/main/media/docs
- CuTe 教程: https://github.com/NVIDIA/cutlass/tree/main/cute/doc

### Flash-Attention 源码
- `csrc/flash_attn/src/kernel_traits.h` - 布局定义
- `csrc/flash_attn/src/flash_fwd_kernel.h` - 实际使用

---

**学习时间**：2-3 天  
**难度**：⭐⭐⭐⭐☆
