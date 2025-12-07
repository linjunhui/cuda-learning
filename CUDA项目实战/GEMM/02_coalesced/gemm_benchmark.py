import numpy as np
import time
import sys
import os
import torch
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
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

# 初始化字体
setup_chinese_font()

torch.set_grad_enabled(False)
# 使用 PyTorch 的 cpp_extension.load 加载 CUDA 模块（共享内存 / 合并访存版本）
try:
    lib = load(
        name="gemm_coalesced_cuda",
        sources=["coalesced_matrix_multiply.cu"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ],
        extra_cflags=["-std=c++17"],
        verbose=True,
    )
    CUDA_AVAILABLE = True
    print("✓ CUDA 模块加载成功")
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"✗ CUDA 模块加载失败: {e}")
    sys.exit(1)


def test_coalesced_gemm():
    """测试共享内存/合并访存矩阵乘法"""
    print("\n" + "="*60)
    print("测试共享内存/合并访存矩阵乘法 (Coalesced GEMM)")
    print("="*60)
    
    # 测试不同大小的矩阵
    test_sizes = [
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    for M, N, K in test_sizes:
        print(f"\n测试矩阵大小: A[{M}x{N}] × B[{N}x{K}] = C[{M}x{K}]")
        
        # 生成随机矩阵 (PyTorch 张量)
        torch.manual_seed(42)
        A = torch.randn(M, N, dtype=torch.float32, device='cuda')
        B = torch.randn(N, K, dtype=torch.float32, device='cuda')
        C = torch.zeros(M, K, dtype=torch.float32, device='cuda')
        
        # CUDA 计算
        torch.cuda.synchronize()
        start_time = time.time()
        lib.coalesced_gemm(A, B, C)
        torch.cuda.synchronize()
        cuda_time = time.time() - start_time
        
        # PyTorch 参考实现
        torch.cuda.synchronize()
        start_time = time.time()
        C_torch = torch.matmul(A, B)
        torch.cuda.synchronize()
        torch_time = time.time() - start_time
        
        # NumPy 参考实现 (CPU)
        A_np = A.cpu().numpy()
        B_np = B.cpu().numpy()
        start_time = time.time()
        C_numpy = np.dot(A_np, B_np)
        numpy_time = time.time() - start_time
        
        # 验证结果
        C_cuda_np = C.cpu().numpy()
        max_error = np.max(np.abs(C_cuda_np - C_numpy))
        mean_error = np.mean(np.abs(C_cuda_np - C_numpy))
        
        print(f"  CUDA 时间: {cuda_time*1000:.3f} ms")
        print(f"  PyTorch 时间: {torch_time*1000:.3f} ms")
        print(f"  NumPy 时间: {numpy_time*1000:.3f} ms")
        print(f"  相对 PyTorch 加速比: {torch_time/cuda_time:.2f}x")
        print(f"  相对 NumPy 加速比: {numpy_time/cuda_time:.2f}x")
        print(f"  最大误差: {max_error:.6f}")
        print(f"  平均误差: {mean_error:.6f}")
        
        # 检查精度
        if max_error < 1e-3:
            print(f"  ✓ 结果验证通过")
        else:
            print(f"  ✗ 结果验证失败，误差过大")
            print(f"    前5x5结果对比:")
            print(f"    CUDA结果:\n{C_cuda_np[:5, :5]}")
            print(f"    NumPy结果:\n{C_numpy[:5, :5]}")


def benchmark_single_size(M, N, K, num_iterations=10, warmup=5):
    """对单个矩阵大小进行基准测试（只测试 coalesced GEMM）"""
    # 准备 GPU 数据
    A_gpu = torch.randn(M, N, dtype=torch.float32, device='cuda')
    B_gpu = torch.randn(N, K, dtype=torch.float32, device='cuda')
    C_gpu = torch.zeros(M, K, dtype=torch.float32, device='cuda')
    
    # 预热
    for _ in range(warmup):
        lib.coalesced_gemm(A_gpu, B_gpu, C_gpu)
    torch.cuda.synchronize()
    
    # 计时
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        lib.coalesced_gemm(A_gpu, B_gpu, C_gpu)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    return np.mean(times)


def benchmark_coalesced_gemm():
    """性能基准测试"""
    print("\n" + "="*60)
    print("性能基准测试")
    print("="*60)
    
    sizes = [256, 512, 1024, 2048]
    num_iterations = 10
    
    for size in sizes:
        M, N, K = size, size, size
        A = torch.randn(M, N, dtype=torch.float32, device='cuda')
        B = torch.randn(N, K, dtype=torch.float32, device='cuda')
        C = torch.zeros(M, K, dtype=torch.float32, device='cuda')
        
        # 预热
        for _ in range(5):
            lib.coalesced_gemm(A, B, C)
        torch.cuda.synchronize()
        
        # 计时
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            lib.coalesced_gemm(A, B, C)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K) / (avg_time / 1000.0) / 1e9  # GFLOPS
        
        print(f"大小 {size}x{size}: {avg_time:.3f} ± {std_time:.3f} ms, {gflops:.2f} GFLOPS")


def plot_performance_comparison():
    """绘制性能图（只绘制 coalesced GEMM）"""
    print("\n" + "="*60)
    print("生成性能图...")
    print("="*60)
    
    # 测试不同的问题大小
    problem_sizes = [128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072]
    problem_sizes = [3072]
    num_iterations = 10
    warmup = 5
    
    my_times = []
    sizes_list = []
    
    for size in problem_sizes:
        print(f"测试大小: {size}x{size}...")
        avg_time = benchmark_single_size(size, size, size, num_iterations, warmup)
        
        my_times.append(avg_time)
        sizes_list.append(size)
        
        print(f"  Coalesced GEMM: {avg_time*1000:.3f} ms")
        gflops = (2.0 * size * size * size) / avg_time / 1e9
        print(f"  性能: {gflops:.2f} GFLOPS")
        print()
    
    # 转换为 numpy 数组
    sizes_array = np.array(sizes_list)
    my_times = np.array(my_times)
    
    # 绘制图表
    plt.figure(figsize=(12, 8))
    
    # 尝试使用中文，如果失败则使用英文
    try:
        label = '我的共享内存矩阵乘法'
        xlabel = '问题大小 (矩阵维度)'
        ylabel = '时间 (单位: 秒)'
        title = 'Coalesced GEMM 性能测试'
    except:
        label = 'My Shared-memory Matrix Multiplication'
        xlabel = 'Problem Size (Matrix Dimension)'
        ylabel = 'Time (seconds)'
        title = 'Coalesced GEMM Performance Test'
    
    plt.semilogy(sizes_array, my_times, 'r-s', label=label, linewidth=2, markersize=8)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = 'gemm_performance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 性能图已保存到: {output_file}")
    
    # 显示图片
    plt.show()
    
    return sizes_array, my_times


if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        sys.exit(1)
    
    # 运行测试
    test_coalesced_gemm()
    
    # 运行基准测试
    benchmark_coalesced_gemm()
    
    # 生成性能对比图
    plot_performance_comparison()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)

