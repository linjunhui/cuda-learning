# CUDA 向量化数据类型详解

## 目录

1. [概述](#概述)
2. [FP32 相关类型](#fp32-相关类型)
3. [FP16 相关类型](#fp16-相关类型)
4. [向量化类型的使用](#向量化类型的使用)
5. [性能优势](#性能优势)
6. [实际应用示例](#实际应用示例)
7. [注意事项](#注意事项)

---

## 概述

CUDA 提供了多种向量化数据类型（vectorized data types），用于优化内存访问和计算性能。这些类型允许一次加载/存储多个标量值，从而提高内存带宽利用率。

### 为什么需要向量化类型？

1. **内存带宽优化**：GPU 内存访问以 128 位（16 字节）为单位最有效
2. **减少内存事务**：一次访问多个元素，减少内存事务数量
3. **提高计算吞吐量**：利用 GPU 的 SIMD（单指令多数据）能力

---

## FP32 相关类型

### `float`（标量类型）

**定义**：32 位单精度浮点数

**特点**：
- 标准 IEEE 754 单精度格式
- 精度：约 7 位有效数字
- 范围：±3.4×10³⁸

**使用示例**：
```cuda
float a = 1.5f;
float b = 2.3f;
float c = a + b;  // 标量加法
```

### `float2`（2 元素向量）

**定义**：包含 2 个 `float` 的结构体，共 64 位（8 字节）

**结构**：
```cuda
struct float2 {
    float x, y;
};
```

**使用示例**：
```cuda
float2 vec;
vec.x = 1.0f;
vec.y = 2.0f;

// 向量加法
float2 a = make_float2(1.0f, 2.0f);
float2 b = make_float2(3.0f, 4.0f);
float2 c;
c.x = a.x + b.x;
c.y = a.y + b.y;
```

### `float4`（4 元素向量）⭐

**定义**：包含 4 个 `float` 的结构体，共 128 位（16 字节）

**结构**：
```cuda
struct float4 {
    float x, y, z, w;
};
```

**特点**：
- **128 位对齐**：正好匹配 GPU 内存访问的最优大小
- **合并访问**：一次加载/存储 4 个 float，提高内存带宽利用率
- **常用优化**：是 FP32 向量化最常用的类型

**使用示例**：
```cuda
// 方式一：使用 make_float4 构造函数
float4 vec = make_float4(1.0f, 2.0f, 3.0f, 4.0f);

// 方式二：直接赋值
float4 vec;
vec.x = 1.0f;
vec.y = 2.0f;
vec.z = 3.0f;
vec.w = 4.0f;

// 向量化加载（从内存）
float *data = ...;
float4 reg = *reinterpret_cast<float4*>(&data[idx]);

// 向量化存储（到内存）
float4 result;
*reinterpret_cast<float4*>(&data[idx]) = result;
```

**在项目中的使用**：
```cuda
// elementwise.cu 中的宏定义
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// 使用示例
float4 reg_a = FLOAT4(a[idx]);  // 一次性加载 4 个 float（128 位）
float4 reg_b = FLOAT4(b[idx]);
float4 reg_c;
reg_c.x = reg_a.x + reg_b.x;
reg_c.y = reg_a.y + reg_b.y;
reg_c.z = reg_a.z + reg_b.z;
reg_c.w = reg_a.w + reg_b.w;
FLOAT4(c[idx]) = reg_c;  // 一次性存储 4 个 float
```

---

## FP16 相关类型

### `__half` / `half`（标量类型）

**定义**：16 位半精度浮点数

**特点**：
- IEEE 754 半精度格式
- 精度：约 3-4 位有效数字
- 范围：±65504
- **内存占用减半**：相比 FP32，内存占用减少 50%
- **性能提升**：在现代 GPU（如 Volta+）上，FP16 计算速度更快

**使用示例**：
```cuda
#include <cuda_fp16.h>

half a = __float2half(1.5f);  // 从 float 转换
half b = __float2half(2.3f);
half c = __hadd(a, b);  // FP16 加法（必须使用内置函数）

// 转换回 float
float result = __half2float(c);
```

**重要内置函数**：
```cuda
// 类型转换
__half __float2half(float);      // float -> half
float __half2float(__half);      // half -> float

// 算术运算（必须使用内置函数，不能直接用 +、-、*、/）
__half __hadd(__half a, __half b);    // 加法
__half __hsub(__half a, __half b);    // 减法
__half __hmul(__half a, __half b);    // 乘法
__half __hdiv(__half a, __half b);    // 除法

// 比较运算
bool __heq(__half a, __half b);       // 等于
bool __hne(__half a, __half b);       // 不等于
bool __hlt(__half a, __half b);       // 小于
bool __hle(__half a, __half b);       // 小于等于
bool __hgt(__half a, __half b);       // 大于
bool __hge(__half a, __half b);       // 大于等于
```

### `half2`（2 元素向量）⭐

**定义**：包含 2 个 `half` 的结构体，共 32 位（4 字节）

**结构**：
```cuda
struct half2 {
    half x, y;
};
```

**特点**：
- **向量化运算**：可以使用 `__hadd2`、`__hmul2` 等内置函数
- **性能优势**：一次处理 2 个 FP16 值，提高吞吐量
- **内存对齐**：32 位对齐，适合向量化访问

**使用示例**：
```cuda
#include <cuda_fp16.h>

// 创建 half2
half2 vec = make_half2(__float2half(1.0f), __float2half(2.0f));

// 向量化加载
half *data = ...;
half2 reg = *reinterpret_cast<half2*>(&data[idx]);

// 向量化加法（一次计算 2 个元素）
half2 a = make_half2(__float2half(1.0f), __float2half(2.0f));
half2 b = make_half2(__float2half(3.0f), __float2half(4.0f));
half2 c = __hadd2(a, b);  // 同时计算 a.x+b.x 和 a.y+b.y

// 向量化存储
*reinterpret_cast<half2*>(&data[idx]) = c;
```

**重要内置函数**：
```cuda
// 向量化算术运算
half2 __hadd2(half2 a, half2 b);    // 向量加法
half2 __hsub2(half2 a, half2 b);     // 向量减法
half2 __hmul2(half2 a, half2 b);     // 向量乘法
half2 __hdiv2(half2 a, half2 b);     // 向量除法

// 类型转换
half2 __floats2half2_rn(float a, float b);  // 两个 float -> half2
```

**在项目中的使用**：
```cuda
// elementwise.cu 中的宏定义
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

// 使用示例
half2 reg_a = HALF2(a[idx]);  // 一次性加载 2 个 half（32 位）
half2 reg_b = HALF2(b[idx]);
half2 reg_c;
reg_c.x = __hadd(reg_a.x, reg_b.x);
reg_c.y = __hadd(reg_a.y, reg_b.y);
HALF2(c[idx]) = reg_c;  // 一次性存储 2 个 half

// 或者使用向量化函数（更高效）
half2 reg_c = __hadd2(reg_a, reg_b);
```

### `__nv_bfloat16` / `__nv_bfloat162`（BF16 类型）

**定义**：Brain Float 16，另一种 16 位浮点格式

**特点**：
- 与 FP32 相同的指数位（8 位），但尾数位更少（7 位）
- 范围与 FP32 相同，但精度更低
- 主要用于深度学习训练，避免下溢问题

**使用**：
```cuda
#include <cuda_bf16.h>

__nv_bfloat16 a = __float2bfloat16(1.5f);
__nv_bfloat162 vec = make_bfloat162(a, a);
```

---

## 向量化类型的使用

### 1. 类型转换宏

在项目中定义了便捷的宏来简化类型转换：

```cuda
// 将指针转换为向量类型（用于加载）
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

// 128 位加载/存储（用于打包优化）
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
```

**工作原理**：
- `reinterpret_cast`：C++ 类型转换，不改变底层数据
- `&(value)[0]`：获取数组/指针的地址，然后转换为向量类型指针

### 2. 内存对齐要求

**重要**：向量化访问要求数据内存对齐！

- `float4`：需要 16 字节对齐（128 位）
- `half2`：需要 4 字节对齐（32 位）

**对齐检查**：
```cuda
// 检查指针是否对齐
bool is_aligned(void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

// 使用
float* data = ...;
assert(is_aligned(data, 16));  // float4 需要 16 字节对齐
```

### 3. 向量化加载模式

#### 模式一：直接转换指针
```cuda
float *data = ...;
int idx = ...;

// 加载 4 个 float
float4 reg = *reinterpret_cast<float4*>(&data[idx]);
// 等价于：
// reg.x = data[idx + 0];
// reg.y = data[idx + 1];
// reg.z = data[idx + 2];
// reg.w = data[idx + 3];
```

#### 模式二：使用宏（推荐）
```cuda
float *data = ...;
int idx = ...;

float4 reg = FLOAT4(data[idx]);  // 更简洁
```

#### 模式三：打包数组（用于复杂场景）
```cuda
half pack[8];  // 8 个 half = 128 位

// 一次性加载 128 位（8 个 half）
LDST128BITS(pack[0]) = LDST128BITS(data[idx]);

// 现在可以逐个处理
half2 vec0 = HALF2(pack[0]);
half2 vec1 = HALF2(pack[2]);
// ...
```

---

## 性能优势

### 1. 内存带宽提升

**标量访问**（每个线程 1 个元素）：
```
线程 0: 读取 data[0]   (32 位)
线程 1: 读取 data[1]   (32 位)
线程 2: 读取 data[2]   (32 位)
线程 3: 读取 data[3]   (32 位)
→ 4 次内存事务，每次 32 位
```

**向量化访问**（每个线程 4 个元素）：
```
线程 0: 读取 data[0-3] (128 位，一次事务)
→ 1 次内存事务，128 位
```

**性能提升**：理论上可提升 4 倍内存带宽利用率

### 2. 实际性能对比

根据项目测试结果：

| 实现方式 | 相对性能 | 说明 |
|---------|---------|------|
| `f32` (标量) | 1.0x | 基准 |
| `f32x4` (向量化) | 1.2-1.3x | 提升 20-30% |
| `f16` (标量) | 1.0x | FP16 基准 |
| `f16x2` (向量化) | 1.25-1.3x | 提升 25-30% |
| `f16x8` (8 元素) | 1.3-1.4x | 提升 30-40% |
| `f16x8_pack` (打包) | 1.4-1.5x | 提升 40-50% |

### 3. 为什么打包版本更快？

`f16x8_pack` 版本的优势：

1. **减少内存事务**：使用 `float4` 一次性加载 128 位（8 个 half）
2. **使用向量化函数**：`__hadd2` 一次计算 2 个元素
3. **寄存器优化**：使用局部数组，编译器可以更好地优化寄存器使用

```cuda
// 打包版本的关键代码
half pack_a[8], pack_b[8], pack_c[8];

// 一次性加载 128 位（8 个 half）
LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

// 使用向量化函数计算
#pragma unroll
for (int i = 0; i < 8; i += 2) {
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
}

// 一次性存储 128 位
LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
```

---

## 实际应用示例

### 示例 1：FP32 向量化加法

```cuda
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N) {
    // 每个线程处理 4 个元素
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx < N) {
        // 向量化加载
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        
        // 向量化计算
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        
        // 向量化存储
        FLOAT4(c[idx]) = reg_c;
    }
}
```

**关键点**：
- 线程索引乘以 4（每个线程处理 4 个元素）
- 使用 `FLOAT4` 宏进行类型转换
- 边界检查：`idx < N`（检查起始位置）

### 示例 2：FP16 向量化加法（half2）

```cuda
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
    // 每个线程处理 2 个元素
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx < N) {
        // 向量化加载
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        
        // 方式一：逐个元素计算
        half2 reg_c;
        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        
        // 方式二：使用向量化函数（更高效）
        // half2 reg_c = __hadd2(reg_a, reg_b);
        
        // 向量化存储
        HALF2(c[idx]) = reg_c;
    }
}
```

### 示例 3：FP16 打包优化版本

```cuda
__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    // 局部数组（存储在寄存器中）
    half pack_a[8], pack_b[8], pack_c[8];
    
    // 一次性加载 128 位（8 个 half）
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
    
    // 向量化计算（使用 __hadd2）
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }
    
    // 边界处理
    if ((idx + 7) < N) {
        // 一次性存储 128 位
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    } else {
        // 边界情况：逐个处理
        for (int i = 0; idx + i < N; i++) {
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
}
```

**关键优化点**：
1. **打包加载**：使用 `float4` 一次性加载 128 位
2. **向量化函数**：使用 `__hadd2` 提高计算效率
3. **循环展开**：`#pragma unroll` 提示编译器展开循环
4. **边界处理**：高效处理边界情况

---

## 注意事项

### 1. 内存对齐

**问题**：未对齐的访问可能导致性能下降或错误

**解决**：
```cuda
// 确保数据对齐
float* data = ...;
assert((reinterpret_cast<uintptr_t>(data) % 16) == 0);  // float4 需要 16 字节对齐

// 或者使用对齐分配
float* data;
cudaMalloc(&data, size * sizeof(float));  // CUDA 分配的内存默认对齐
```

### 2. 边界检查

**问题**：向量化访问可能越界

**解决**：
```cuda
int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

// 检查起始位置
if (idx < N) {
    // 检查是否有足够的元素
    if (idx + 3 < N) {
        // 安全：可以向量化访问
        float4 reg = FLOAT4(a[idx]);
    } else {
        // 边界情况：逐个处理
        for (int i = 0; idx + i < N; i++) {
            c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}
```

### 3. FP16 运算限制

**问题**：FP16 不能直接使用 `+`、`-`、`*`、`/` 运算符

**解决**：必须使用内置函数
```cuda
// ❌ 错误
half c = a + b;

// ✅ 正确
half c = __hadd(a, b);
half2 c = __hadd2(a, b);
```

### 4. 类型转换开销

**问题**：频繁的类型转换可能影响性能

**解决**：尽量在寄存器中操作，减少转换次数
```cuda
// ❌ 不好：频繁转换
for (int i = 0; i < 100; i++) {
    float4 reg = FLOAT4(data[i * 4]);
    // ...
}

// ✅ 更好：一次性加载，在寄存器中操作
float4 reg = FLOAT4(data[idx]);
reg.x += 1.0f;
reg.y += 1.0f;
reg.z += 1.0f;
reg.w += 1.0f;
FLOAT4(data[idx]) = reg;
```

### 5. 编译器优化

**提示**：使用 `#pragma unroll` 帮助编译器优化
```cuda
#pragma unroll
for (int i = 0; i < 8; i += 2) {
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
}
```

---

## 总结

### 类型选择指南

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| FP32 高性能 | `float4` | 128 位对齐，最优内存访问 |
| FP16 基础 | `half2` | 32 位向量，平衡性能和复杂度 |
| FP16 极致优化 | `half2` + 打包 | 128 位打包，最高性能 |
| 内存受限 | `half` / `half2` | 内存占用减半 |
| 简单场景 | `float` / `half` | 代码简单，易于理解 |

### 最佳实践

1. **优先使用向量化类型**：`float4`、`half2` 等
2. **确保内存对齐**：检查数据对齐
3. **使用内置函数**：FP16 必须使用 `__hadd`、`__hadd2` 等
4. **合理边界处理**：避免越界访问
5. **性能测试**：实际测试不同实现的性能

### 性能提升预期

- **FP32 向量化**：提升 20-30%
- **FP16 向量化**：提升 25-40%
- **FP16 打包优化**：提升 40-50%
- **FP16 vs FP32**：在大规模数据上，FP16 快 5-6 倍

---

## 参考资源

- [CUDA C++ Programming Guide - Built-in Vector Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types)
- [CUDA Math API - Half Precision](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__HALF.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
