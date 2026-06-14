/*
 * CUDA 向量加法内核实现
 * ======================
 * 
 * 本文件实现了两个版本的向量加法 CUDA 内核：
 * 1. vector_add_f32_kernel:  基础版本，逐个元素相加
 * 2. vector_add_f32x4_kernel: 优化版本，使用 float4 向量化，每次处理4个元素
 * 
 * 并通过 PyTorch 的 pybind11 绑定机制将 CUDA 函数导出到 Python。
 */

// ============================================================================
// 头文件包含
// ============================================================================
#include <algorithm>
#include <cuda_bf16.h>        // CUDA bfloat16 类型支持
#include <cuda_fp16.h>        // CUDA half (float16) 类型支持
#include <cuda_fp8.h>         // CUDA float8 类型支持
#include <cuda_runtime.h>     // CUDA 运行时 API
#include <float.h>            // 浮点数相关常量
#include <stdio.h>            // 标准输入输出
#include <stdlib.h>           // 标准库函数
#include <torch/extension.h>  // PyTorch 扩展头文件（包含 pybind11 和 Tensor 类型）
#include <torch/types.h>      // PyTorch 类型定义
#include <vector>             // C++ 标准库向量容器

// ============================================================================
// 宏定义
// ============================================================================
#define WARP_SIZE 32          // CUDA warp 大小（32个线程）
// FLOAT4 宏：将 float 指针转换为 float4 类型进行向量化访问
// float4 是 CUDA 内置的向量类型，包含4个 float 成员（x, y, z, w）
// reinterpret_cast 用于类型转换，不改变底层内存布局
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])


// ============================================================================
// CUDA 内核函数 1: 基础向量加法（逐个元素处理）
// ============================================================================
/*
 * 实现思路：
 * - 最朴素的实现方法，直接对位相加
 * - 每个线程处理一个元素
 * - float 类型：4字节，32位
 * 
 * 线程索引计算：
 * - blockIdx.x: 当前线程块在网格中的 x 维度索引
 * - blockDim.x: 每个线程块中 x 维度的线程数
 * - threadIdx.x: 当前线程在线程块中的 x 维度索引
 * - 全局线程索引 = blockIdx.x * blockDim.x + threadIdx.x
 */
__global__ void vector_add_f32_kernel(float *a, float *b, float *c, int N) {
    // 计算当前线程的全局索引
    // 这是 CUDA 中计算线程全局 ID 的标准公式
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查：确保索引不越界
    // 由于网格大小可能向上取整，需要检查 idx < N
    if(idx < N) {
        // 执行向量加法：c[i] = a[i] + b[i]
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================================================
// CUDA 内核函数 2: 向量化向量加法（使用 float4，每次处理4个元素）
// ============================================================================
/*
 * 实现思路：
 * - 使用 float4 向量类型进行向量化操作
 * - 每个线程处理4个连续的元素（向量化）
 * - 可以减少内存访问次数和提高内存带宽利用率
 * - 逻辑上相当于将数据拆分成4列，每行4个元素一起处理
 * 
 * 优势：
 * - 减少线程数量（每个线程处理4个元素）
 * - 提高内存访问效率（合并访问）
 * - 减少指令数量（向量化操作）
 */
__global__ void vector_add_f32x4_kernel(float *a, float *b, float *c, int N) {
    // 计算当前线程的起始索引
    // 由于每个线程处理4个元素，所以起始索引要乘以4
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    // 边界检查：确保索引不越界（注意这里检查的是起始索引）
    if(idx < N) {
        // 使用 FLOAT4 宏将 float 指针转换为 float4 类型
        // 这样可以一次性读取/写入4个连续的 float 值
        float4 reg_a = FLOAT4(a[idx]);  // 从内存读取4个 float 到寄存器
        float4 reg_b = FLOAT4(b[idx]);  // 从内存读取4个 float 到寄存器
        float4 reg_c;                    // 结果寄存器

        // 执行4个元素的向量加法
        reg_c.x = reg_a.x + reg_b.x;     // 第1个元素
        reg_c.y = reg_a.y + reg_b.y;     // 第2个元素
        reg_c.z = reg_a.z + reg_b.z;     // 第3个元素
        reg_c.w = reg_a.w + reg_b.w;     // 第4个元素
        
        // 将结果写回内存（一次性写入4个 float）
        FLOAT4(c[idx]) = reg_c;
    }
}

// ============================================================================
// PyTorch 绑定辅助宏
// ============================================================================

// STRINGFY 宏：将宏参数转换为字符串字面量
// 例如：STRINGFY(vector_add_f32) 会被展开为 "vector_add_f32"
// # 是 C 预处理器中的字符串化运算符
#define STRINGFY(str) #str

// TORCH_BINDING_COMMON_EXTENSION 宏：通用的函数绑定宏
// 用于将 C++ 函数绑定到 Python 模块
// m: pybind11 模块对象（在 PYBIND11_MODULE 中定义）
// m.def() 的参数：
//   - 第一个参数：Python 中的函数名（字符串）
//   - 第二个参数：C++ 函数指针
//   - 第三个参数：函数文档字符串（可选）
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// ============================================================================
// 张量数据类型检查宏
// ============================================================================
// 用于在运行时检查 PyTorch 张量的数据类型是否符合预期
// T: 要检查的 torch::Tensor 对象
// th_type: 期望的数据类型（如 torch::kFloat32）
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
if (((T).options().dtype() != (th_type))) {                                  \
std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
throw std::runtime_error("values must be " #th_type);                      \
}


// ============================================================================
// 向量加法函数绑定宏（使用模板宏生成多个版本的绑定函数）
// ============================================================================
/*
 * 这个宏用于生成完整的 Python 绑定函数，包括：
 * 1. 数据类型检查
 * 2. 线程块和网格大小计算
 * 3. CUDA 内核启动
 * 
 * 参数说明：
 * - packed_type: 函数名后缀（如 f32, f32x4）
 * - th_type: PyTorch 数据类型（如 torch::kFloat32）
 * - element_type: C++ 元素类型（如 float）
 * - n_elements: 每个线程处理的元素数量（1 或 4）
 * 
 * ## 是 C 预处理器中的连接运算符，用于连接两个标记
 * 例如：vector_add_##packed_type 如果 packed_type 是 f32，则展开为 vector_add_f32
 * TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
 */
#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
void vector_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                    torch::Tensor c) {                        \
/* 检查输入和输出张量的数据类型 */                                          \
CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
/* 获取张量的维度数 */                                                      \
const int ndim = a.dim();                                                  \
/* 如果张量不是2维，则按1维处理（展平） */                                  \
if (ndim != 2) {                                                           \
    /* 计算总元素数量 */                                                    \
    int N = 1;                                                               \
    for (int i = 0; i < ndim; ++i) {                                         \
    N *= a.size(i);                                                        \
    }                                                                        \
    /* 计算线程块大小：256 / n_elements（确保每个线程块有足够的线程） */     \
    dim3 block(256 / (n_elements));                                          \
    /* 计算网格大小：向上取整 (N + 256 - 1) / 256 */                        \
    dim3 grid((N + 256 - 1) / 256);                                          \
    /* 启动 CUDA 内核 */                                                    \
    vector_add_##packed_type##_kernel<<<grid, block>>>(                 \
        reinterpret_cast<element_type *>(a.data_ptr()),                      \
        reinterpret_cast<element_type *>(b.data_ptr()),                      \
        reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
} else {                                                                   \
    /* 2维张量的优化处理 */                                                 \
    const int S = a.size(0);                                                 \
    const int K = a.size(1);                                                 \
    const int N = S * K;                                                     \
    /* 如果 K / n_elements <= 1024，使用优化的线程块配置 */                \
    if ((K / (n_elements)) <= 1024) {                                        \
    /* 每个线程块处理一行（或部分行） */                                    \
    dim3 block(K / (n_elements));                                          \
    dim3 grid(S);                                                          \
    /* 启动 CUDA 内核 */                                                    \
    vector_add_##packed_type##_kernel<<<grid, block>>>(               \
        reinterpret_cast<element_type *>(a.data_ptr()),                    \
        reinterpret_cast<element_type *>(b.data_ptr()),                    \
        reinterpret_cast<element_type *>(c.data_ptr()), N);                \
    } else {                                                                 \
    /* K 太大，回退到1维处理方式 */                                        \
    int N = 1;                                                             \
    for (int i = 0; i < ndim; ++i) {                                       \
        N *= a.size(i);                                                      \
    }                                                                      \
    dim3 block(256 / (n_elements));                                        \
    dim3 grid((N + 256 - 1) / 256);                                        \
    /* 启动 CUDA 内核 */                                                    \
    vector_add_##packed_type##_kernel<<<grid, block>>>(               \
        reinterpret_cast<element_type *>(a.data_ptr()),                    \
        reinterpret_cast<element_type *>(b.data_ptr()),                    \
        reinterpret_cast<element_type *>(c.data_ptr()), N);                \
    }                                                                        \
}                                                                          \
}

// ============================================================================
// 使用宏生成绑定函数
// ============================================================================
// 生成 vector_add_f32 函数：
// - 处理 float32 类型
// - 每个线程处理1个元素
TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)

// 生成 vector_add_f32x4 函数：
// - 处理 float32 类型
// - 每个线程处理4个元素（使用 float4 向量化）
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)

// ============================================================================
// PyTorch 扩展模块定义（使用 pybind11）
// ============================================================================
/*
 * PYBIND11_MODULE 是 pybind11 提供的宏，用于定义 Python 扩展模块
 * 
 * TORCH_EXTENSION_NAME 是 PyTorch 自动定义的宏，值为模块名称（"vector_add"）
 * m 是 pybind11::module 对象，用于向模块中添加函数、类等
 * 
 * 这个模块会在 Python 中通过 torch.utils.cpp_extension.load() 加载
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 将 vector_add_f32 函数绑定到 Python
    // 在 Python 中可以通过 lib.vector_add_f32() 调用
    TORCH_BINDING_COMMON_EXTENSION(vector_add_f32)
    
    // 将 vector_add_f32x4 函数绑定到 Python
    // 在 Python 中可以通过 lib.vector_add_f32x4() 调用
    TORCH_BINDING_COMMON_EXTENSION(vector_add_f32x4)
}