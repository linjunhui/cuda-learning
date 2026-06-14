import numpy as np
import time
import sys
import os
import torch
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

def setup_chinese_font():
    # 设置中文字体
    FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    try:
        # 1. 创建字体属性对象，指定文件路径
        chinese_font = font_manager.FontProperties(fname=FONT_PATH)
        print(chinese_font)      
        # 2. 设置 Matplotlib 默认字体为该字体 (使用其内部名称)
        matplotlib.rcParams['font.sans-serif'] = [chinese_font.get_name()]       
        # 3. 修正负号显示
        matplotlib.rcParams['axes.unicode_minus'] = False       
    except Exception as e:
        # 简单的错误处理
        print(f"Warning: Failed to load font {FONT_PATH}. Error: {e}")

setup_chinese_font()

lib = load(
    name="elementwise_cuda",
    sources=["elementwise.cu"],
    extra_cuda_cflags=[
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
    verbose=True,
)

def test_elementwise(warmup: int = 10, iters: int = 1000):
    """测试逐元素加法"""
    print("\n" + "="*60)
    print("测试逐元素加法")
    print("="*60)
    
    # 测试不同大小的矩阵
    test_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    test_sizes = [x * 100 for x in range(10, 100)]

    for size in test_sizes:
        size = size * 1024
        print(f"测试大小: {size}")
        a = torch.randn(size, dtype=torch.float32, device="cuda")
        b = torch.randn(size, dtype=torch.float32, device="cuda")
        c = torch.zeros(size, dtype=torch.float32, device="cuda")
        for i in range(warmup):
            #lib.elementwise_add_f32(a, b, c)
            lib.elementwise_add_f32x4(a, b, c)
        # 同步GPU, 等待GPU的计算完成
        torch.cuda.synchronize()
        start = time.time()
        for i in range(iters):
            #lib.elementwise_add_f32(a, b, c)
            lib.elementwise_add_f32x4(a, b, c)
        torch.cuda.synchronize()
        end = time.time()
        # 计算 算力(GFLOPS)
        gflops = iters * size / (end - start) / 1e9
        print(f"时间: {end - start} 秒, 算力: {gflops:.2f} GFLOPS")



test_elementwise()