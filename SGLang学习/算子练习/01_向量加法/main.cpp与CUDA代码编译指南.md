# main.cpp 与 CUDA 代码一起编译指南

## 概述

在 CUDA 项目中，有两种方式组织代码：
1. **方案1**：使用 `main.cu`（所有代码由 nvcc 编译）
2. **方案2**：使用 `main.cpp` + 分开编译（`.cpp` 由 g++ 编译，`.cu` 由 nvcc 编译，最后链接）

本文档详细说明如何实现方案2，让 `main.cpp` 与 CUDA 代码一起编译。

---

## 核心问题

### 为什么不能直接在 main.cpp 中包含 CUDA 内核实现？

**问题**：如果 `main.cpp` 包含的头文件中有 CUDA 内核函数的实现（包含 `threadIdx`, `blockDim` 等 CUDA 内置变量），g++ 编译器无法理解这些 CUDA 语法。

**原因**：
- `#include` 是文本替换，头文件内容会被插入到源文件中
- g++ 编译 `main.cpp` 时，会看到并尝试编译头文件中的所有代码
- g++ 不认识 CUDA 语法（`__global__`, `threadIdx` 等）→ 编译错误

---

## 解决方案：分开编译

### 文件结构

```
项目目录/
├── src/
│   ├── vector_add.cuh    # 头文件：只包含声明（无 CUDA 代码）
│   ├── vector_add.cu     # CUDA 实现：包含实现 + 显式实例化
│   └── main.cpp          # 主程序：只包含头文件（看到声明）
└── CMakeLists.txt        # 构建配置
```

### 关键原则

1. **头文件（.cuh）只包含声明**：不能有 CUDA 内核函数的实现
2. **实现放在 .cu 文件中**：由 nvcc 编译
3. **显式实例化**：在 .cu 文件中显式实例化模板函数
4. **main.cpp 只看到声明**：由 g++ 编译，不会遇到 CUDA 代码

---

## 详细实现步骤

### 步骤1：创建头文件（vector_add.cuh）

**只包含声明，不包含实现**：

```cpp
#ifndef VECTOR_ADD_H_
#define VECTOR_ADD_H_

#include <cuda_runtime.h>
#include <cstdint>

// 只声明，不实现
template <typename T>
__global__ void vectorAdd(T *input1, T *input2, T *output, int64_t N);

#endif
```

**关键点**：
- ✅ 只有函数声明
- ✅ 没有函数体（没有 `{ ... }`）
- ✅ g++ 可以理解这个声明

### 步骤2：创建 CUDA 实现文件（vector_add.cu）

**包含实现和显式实例化**：

```cpp
#include "vector_add.cuh"

// 模板函数实现
template <typename T>
__global__ void vectorAdd(T *input1, T *input2, T *output, int64_t N) {
    // 计算全局索引
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        output[tid] = input1[tid] + input2[tid];
    }
}

// 显式实例化常用的类型
// 这样链接器才能找到这些函数的实现
template __global__ void vectorAdd<float>(float*, float*, float*, int64_t);
template __global__ void vectorAdd<double>(double*, double*, double*, int64_t);
template __global__ void vectorAdd<int>(int*, int*, int*, int64_t);
```

**关键点**：
- ✅ 包含头文件（获取声明）
- ✅ 提供模板函数实现
- ✅ **显式实例化**：告诉编译器为哪些类型生成代码
- ✅ 由 nvcc 编译，理解所有 CUDA 语法

### 步骤3：创建 main.cpp

**只包含头文件，调用 CUDA 函数**：

```cpp
#include <cstdio>
#include <cstdlib>
#include "vector_add.cuh"  // 只看到声明

template<typename T>
void launch_kernel(T *h_input1, T *h_input2, T *h_output, int64_t N, int64_t M_SIZE) {
    T *d_input1, *d_input2, *d_output;

    // 分配设备内存
    cudaMalloc((void **)&d_input1, M_SIZE);
    cudaMalloc((void **)&d_input2, M_SIZE);
    cudaMalloc((void **)&d_output, M_SIZE);

    // 复制数据到设备
    cudaMemcpy(d_input1, h_input1, M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, M_SIZE, cudaMemcpyHostToDevice);

    // 配置内核启动参数
    const int BLOCK_SIZE = 256;
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 启动内核（注意：需要显式实例化的类型）
    vectorAdd<<<gridSize, blockSize>>>(d_input1, d_input2, d_output, N);
    
    // 等待内核完成
    cudaDeviceSynchronize();

    // 复制结果回主机
    cudaMemcpy(h_output, d_output, M_SIZE, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

int main() {
    float *h_input1, *h_input2, *h_output;
    int64_t N = 1000;
    int64_t M_SIZE = sizeof(float) * N;  // 注意：使用正确的类型大小

    // 分配主机内存
    h_input1 = (float *)malloc(M_SIZE);
    h_input2 = (float *)malloc(M_SIZE);
    h_output = (float *)malloc(M_SIZE);

    // 初始化数据
    for(int i = 0; i < N; i++) {
        h_input1[i] = 1.0f * i;
        h_input2[i] = 2.0f * i;
    }

    // 启动内核
    launch_kernel(h_input1, h_input2, h_output, N, M_SIZE);

    // 验证结果
    for(int i = 0; i < 10; i++) {
        printf("h_output[%ld] = %f\n", i, h_output[i]);
    }

    // 释放主机内存
    free(h_input1);
    free(h_input2);
    free(h_output);

    return 0;
}
```

**关键点**：
- ✅ 包含头文件（只看到声明）
- ✅ 可以调用 CUDA 运行时 API（`cudaMalloc`, `cudaMemcpy` 等）
- ✅ 可以启动内核函数（`<<<gridSize, blockSize>>>`）
- ✅ 由 g++ 编译，但只看到声明，不会遇到 CUDA 内核实现

### 步骤4：配置 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)

project(VectorAdd LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 源文件
set(SOURCES
    src/vector_add.cu   # CUDA 源文件，由 nvcc 编译
    src/main.cpp        # C++ 源文件，由 g++ 编译
)

# 头文件
set(HEADERS
    src/vector_add.cuh
)

# 创建可执行文件
add_executable(vector_add ${SOURCES} ${HEADERS})

# 启用 CUDA 分离编译
set_property(TARGET vector_add PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 设置文件语言属性，确保正确的编译器
set_source_files_properties(src/vector_add.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CXX)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA architectures")
```

**关键点**：
- ✅ 两个源文件都添加到 `SOURCES`
- ✅ 设置 `CUDA_SEPARABLE_COMPILATION ON`（允许分开编译）
- ✅ 明确指定文件语言（`.cu` → CUDA，`.cpp` → CXX）

---

## 编译流程

### 使用 CMake

```bash
mkdir build && cd build
cmake ..
make
./bin/vector_add
```

### 编译过程详解

```
1. nvcc 编译 vector_add.cu
   ├─ 看到头文件声明
   ├─ 看到模板实现
   ├─ 看到显式实例化
   ├─ 生成 vectorAdd<float> 等函数的符号
   └─ 输出：vector_add.cu.o

2. g++ 编译 main.cpp
   ├─ 包含 vector_add.cuh（只看到声明）
   ├─ 看到 vectorAdd 的声明（知道函数存在）
   ├─ 生成对 vectorAdd<float> 的调用
   └─ 输出：main.cpp.o

3. 链接器
   ├─ 读取 vector_add.cu.o（找到 vectorAdd<float> 的实现）
   ├─ 读取 main.cpp.o（找到对 vectorAdd<float> 的调用）
   ├─ 将调用和实现连接起来
   └─ 输出：vector_add（可执行文件）
```

---

## 常见问题

### Q1: 为什么需要显式实例化？

**A**: 模板函数在编译时不会自动生成代码，只有在使用时才会实例化。如果实现放在 `.cu` 文件中，而调用在 `.cpp` 文件中，链接器需要找到具体的函数符号。显式实例化告诉编译器："请为这些类型生成代码"。

### Q2: 如果使用新类型怎么办？

**A**: 在 `vector_add.cu` 中添加新的显式实例化：
```cpp
template __global__ void vectorAdd<long>(long*, long*, long*, int64_t);
```

### Q3: 可以直接用 nvcc 编译 main.cpp 吗？

**A**: 可以，但需要将 `main.cpp` 重命名为 `main.cu`。这样所有代码都由 nvcc 编译，不需要分开编译。

### Q4: 为什么头文件中不能有实现？

**A**: 因为 `main.cpp` 由 g++ 编译，如果头文件中有 CUDA 代码（如 `threadIdx`），g++ 无法理解这些语法。

### Q5: 可以混合使用 .cu 和 .cpp 吗？

**A**: 可以，这正是分开编译的目的。`.cu` 文件由 nvcc 编译，`.cpp` 文件由 g++ 编译，最后链接。

---

## 方案对比

| 特性 | 方案1: main.cu | 方案2: main.cpp + 分开编译 |
|------|---------------|---------------------------|
| **文件扩展名** | `.cu` | `.cpp` |
| **编译器** | nvcc | g++ |
| **头文件** | 可以包含实现 | 只能包含声明 |
| **实现位置** | 头文件或 .cu | 必须在 .cu 文件 |
| **显式实例化** | 不需要 | 需要 |
| **编译复杂度** | 简单 | 稍复杂 |
| **适用场景** | 简单项目 | 大型项目，需要分离关注点 |

---

## 最佳实践

1. **头文件只包含声明**：避免 g++ 看到 CUDA 代码
2. **实现放在 .cu 文件**：由 nvcc 编译
3. **显式实例化常用类型**：确保链接时能找到符号
4. **使用 CMake**：自动处理编译器和链接
5. **明确文件语言**：在 CMakeLists.txt 中设置 `LANGUAGE` 属性

---

## 总结

要让 `main.cpp` 与 CUDA 代码一起编译，关键是：

1. ✅ **头文件只有声明**（无 CUDA 代码）
2. ✅ **实现放在 .cu 文件**（由 nvcc 编译）
3. ✅ **显式实例化模板函数**（让链接器找到符号）
4. ✅ **CMake 配置分开编译**（`.cu` 和 `.cpp` 分别编译后链接）

这样，`main.cpp` 可以保持为纯 C++ 代码，由 g++ 编译，同时能够调用 CUDA 内核函数。
