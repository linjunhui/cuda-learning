"""
Python 与 CUDA 函数绑定示例 - 向量加法
===========================================

本示例演示如何使用 PyTorch 的 cpp_extension 模块将 CUDA 内核函数绑定到 Python，
实现高效的 GPU 向量加法运算。

主要步骤：
1. 使用 torch.utils.cpp_extension.load() 动态编译和加载 CUDA 扩展
2. 调用绑定的 CUDA 函数进行向量加法运算
3. 性能基准测试和对比
"""

import time
from functools import partial
from typing import Optional

import torch
# torch.utils.cpp_extension.load 是 PyTorch 提供的动态编译工具
# 它可以在运行时编译 C++/CUDA 代码并加载为 Python 模块
from torch.utils.cpp_extension import load

# ============================================================================
# 动态加载 CUDA 扩展模块
# ============================================================================
# load() 函数会在首次调用时：
# 1. 编译 vector_add.cu 文件（使用 nvcc 编译器）
# 2. 编译生成的 C++ 包装代码（使用 g++/clang++）
# 3. 将编译好的共享库加载到 Python 中
# 4. 返回一个模块对象，其中包含所有通过 PYBIND11_MODULE 导出的函数
lib = load(
    name="vector_add",                    # 扩展模块的名称，用于生成临时文件
    sources=["vector_add.cu"],            # CUDA 源文件列表
    extra_cuda_cflags=[                   # 传递给 nvcc 的额外编译选项
        "-O3",                            # 最高级别的优化
        "-U__CUDA_NO_HALF_OPERATORS__",   # 启用 half 类型的运算符
        "-U__CUDA_NO_HALF_CONVERSIONS__", # 启用 half 类型的转换
        "-U__CUDA_NO_HALF2_OPERATORS__",  # 启用 half2 类型的运算符
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", # 启用 bfloat16 类型的转换
        "--expt-relaxed-constexpr",       # 放宽 constexpr 的限制
        "--expt-extended-lambda",         # 允许在 lambda 中使用设备代码
        "--use_fast_math",                # 使用快速数学库（可能降低精度但提高速度）
    ],
    extra_cflags=["-std=c++17"],         # 传递给 C++ 编译器的额外选项（C++17 标准）
    verbose=True                          # 显示详细的编译信息
)
# 编译完成后，lib 对象包含：
# - lib.vector_add_f32:  单精度浮点数向量加法（逐个元素处理）
# - lib.vector_add_f32x4: 单精度浮点数向量加法（使用 float4 向量化，每次处理4个元素）

# ============================================================================
# 性能基准测试函数
# ============================================================================
def run_benchmark(
    perf_func: callable,                  # 要测试的性能函数（CUDA 内核的 Python 绑定）
    a: torch.Tensor,                      # 输入张量 A（GPU 上的连续内存）
    b: torch.Tensor,                      # 输入张量 B（GPU 上的连续内存）
    tag: str,                             # 测试标签（用于输出标识）
    out: Optional[torch.Tensor] = None,  # 输出张量（可选，如果为 None 则函数内部分配）
    warmup: int = 10,                     # 预热迭代次数（消除首次运行的初始化开销）
    iters: int = 1000,                    # 实际测试迭代次数
    show_all: bool = False                # 是否显示完整的输出张量
):
    """
    运行性能基准测试
    
    测试流程：
    1. 预热阶段：运行 warmup 次，让 GPU 完成初始化、缓存预热等
    2. 同步 GPU：确保所有预热操作完成
    3. 计时阶段：运行 iters 次，记录总时间
    4. 计算平均时间并输出结果
    """
    # 如果提供了输出张量，先清零（确保测试的一致性）
    if out is not None:
        out.fill_(0)

    # ========================================================================
    # 预热阶段（Warmup）
    # ========================================================================
    # 预热的目的：
    # - 让 GPU 完成 CUDA 上下文的初始化
    # - 预热 GPU 缓存（L1/L2 cache）
    # - 让驱动完成 JIT 编译（如果有）
    # - 消除首次运行的开销
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)          # 调用 CUDA 函数（原地操作）
    else:
        for i in range(warmup):
            _ = perf_func(a, b)           # 调用 CUDA 函数（返回新张量）

    # ========================================================================
    # 同步 GPU，确保所有预热操作完成
    # ========================================================================
    # torch.cuda.synchronize() 会阻塞 CPU 直到 GPU 完成所有待处理的操作
    # 这对于准确计时非常重要，因为 CUDA 操作默认是异步的
    torch.cuda.synchronize()
    start = time.time()
    
    # ========================================================================
    # 实际测试阶段
    # ========================================================================
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)          # 执行 iters 次 CUDA 操作
    else:
        for i in range(iters):
            out = perf_func(a, b)         # 执行 iters 次 CUDA 操作

    # ========================================================================
    # 再次同步，确保所有测试操作完成，然后记录结束时间
    # ========================================================================
    torch.cuda.synchronize()
    end = time.time()

    # 计算总时间和平均时间（转换为毫秒）
    total_time = (end - start) * 1000     # 总时间（毫秒）
    mean_time = total_time / iters        # 平均每次操作的时间（毫秒）

    # ========================================================================
    # 输出结果
    # ========================================================================
    out_info = f"out_{tag}"
    # 获取输出张量的前两个元素用于验证结果正确性
    # flatten(): 展平张量
    # detach(): 从计算图中分离（不需要梯度）
    # cpu(): 将数据从 GPU 转移到 CPU
    # numpy(): 转换为 numpy 数组
    # tolist()[:2]: 转换为列表并取前两个元素
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]  # 四舍五入到8位小数
    print(f"{out_info:>18}: {out_val}, time: {mean_time:.8f} ms")

    if show_all:
        print(out)
    return out, mean_time


# ============================================================================
# 主测试程序
# ============================================================================
# 定义测试的矩阵维度
# S: 行数（第一个维度）
# K: 列数（第二个维度）
Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
# 生成所有 (S, K) 的组合
SKs = [(S, K) for S in Ss for K in Ks]

# 对每个 (S, K) 组合进行测试
for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    
    # ========================================================================
    # 创建测试数据
    # ========================================================================
    # torch.randn(): 生成标准正态分布的随机数
    # .cuda(): 将张量移动到 GPU
    # .float(): 确保数据类型为 float32
    # .contiguous(): 确保内存布局是连续的（CUDA 内核要求）
    a = torch.randn((S, K)).cuda().float().contiguous()  # 输入张量 A
    b = torch.randn((S, K)).cuda().float().contiguous()  # 输入张量 B
    c = torch.zeros_like(a).cuda().float().contiguous()  # 输出张量 C（初始化为0）
    
    # ========================================================================
    # 运行基准测试
    # ========================================================================
    # 测试两种不同的实现：
    # 1. vector_add_f32:  逐个元素处理（基础版本）
    # 2. vector_add_f32x4: 使用 float4 向量化，每次处理4个元素（优化版本）
    run_benchmark(lib.vector_add_f32, a, b, "f32", c)      # 基础版本
    run_benchmark(lib.vector_add_f32x4, a, b, "f32x4", c)  # 向量化版本
    print("-" * 85)
