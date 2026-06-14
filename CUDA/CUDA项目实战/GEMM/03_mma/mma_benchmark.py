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

# 检查 CUDA 和计算能力
if not torch.cuda.is_available():
    print("✗ CUDA 不可用")
    sys.exit(1)

# 检查计算能力 (WMMA 需要计算能力 >= 7.0)
compute_capability = torch.cuda.get_device_capability()
print(f"GPU 计算能力: {compute_capability[0]}.{compute_capability[1]}")
if compute_capability[0] < 7:
    print("✗ WMMA API 需要计算能力 >= 7.0 (Volta/Turing/Ampere/Ada/Hopper)")
    sys.exit(1)

# 使用 PyTorch 的 cpp_extension.load 加载 CUDA 模块
# 根据 GPU 计算能力设置架构
compute_cap = compute_capability[0] * 10 + compute_capability[1]
arch_flag = f"-arch=sm_{compute_cap}"

try:
    lib = load(
        name="mma_cuda",
        sources=["mma.cu"],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            arch_flag,  # 使用当前 GPU 的架构
        ],
        extra_cflags=["-std=c++17"],
        verbose=True,
    )
    CUDA_AVAILABLE = True
    print("✓ CUDA 模块加载成功")
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"✗ CUDA 模块加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def test_mma_gemm():
    """测试使用 Tensor Core 的矩阵乘法"""
    print("\n" + "="*60)
    print("测试 Tensor Core WMMA 矩阵乘法")
    print("="*60)
    
    # 测试不同大小的矩阵 (需要是 16 的倍数以获得最佳性能)
    test_sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    for M, N, K in test_sizes:
        print(f"\n测试矩阵大小: A[{M}x{N}] × B[{N}x{K}] = C[{M}x{K}]")
        
        # 生成随机矩阵 (PyTorch 张量，使用 half precision)
        torch.manual_seed(42)
        A = torch.randn(M, N, dtype=torch.float16, device='cuda')
        B = torch.randn(N, K, dtype=torch.float16, device='cuda')
        C = torch.zeros(M, K, dtype=torch.float16, device='cuda')
        
        # CUDA WMMA 计算
        torch.cuda.synchronize()
        start_time = time.time()
        lib.mma_gemm(A, B, C)
        torch.cuda.synchronize()
        mma_time = time.time() - start_time
        
        # PyTorch 参考实现 (half precision)
        torch.cuda.synchronize()
        start_time = time.time()
        C_torch = torch.matmul(A, B)
        torch.cuda.synchronize()
        torch_time = time.time() - start_time
        
        # 转换为 float32 进行精度验证
        A_fp32 = A.float()
        B_fp32 = B.float()
        C_fp32 = C.float()
        C_torch_fp32 = C_torch.float()
        
        # NumPy 参考实现 (CPU, float32)
        A_np = A_fp32.cpu().numpy()
        B_np = B_fp32.cpu().numpy()
        start_time = time.time()
        C_numpy = np.dot(A_np, B_np)
        numpy_time = time.time() - start_time
        
        # 验证结果
        C_mma_np = C_fp32.cpu().numpy()
        C_torch_np = C_torch_fp32.cpu().numpy()
        
        # 与 PyTorch 对比
        max_error_vs_torch = np.max(np.abs(C_mma_np - C_torch_np))
        mean_error_vs_torch = np.mean(np.abs(C_mma_np - C_torch_np))
        
        # 与 NumPy 对比
        max_error_vs_numpy = np.max(np.abs(C_mma_np - C_numpy))
        mean_error_vs_numpy = np.mean(np.abs(C_mma_np - C_numpy))
        
        print(f"  WMMA 时间: {mma_time*1000:.3f} ms")
        print(f"  PyTorch 时间: {torch_time*1000:.3f} ms")
        print(f"  NumPy 时间: {numpy_time*1000:.3f} ms")
        print(f"  相对 PyTorch 加速比: {torch_time/mma_time:.2f}x")
        print(f"  相对 NumPy 加速比: {numpy_time/mma_time:.2f}x")
        print(f"  与 PyTorch 最大误差: {max_error_vs_torch:.6f}")
        print(f"  与 PyTorch 平均误差: {mean_error_vs_torch:.6f}")
        print(f"  与 NumPy 最大误差: {max_error_vs_numpy:.6f}")
        print(f"  与 NumPy 平均误差: {mean_error_vs_numpy:.6f}")
        
        # 计算 GFLOPS
        gflops_mma = (2.0 * M * N * K) / (mma_time) / 1e9
        gflops_torch = (2.0 * M * N * K) / (torch_time) / 1e9
        print(f"  WMMA 性能: {gflops_mma:.2f} GFLOPS")
        print(f"  PyTorch 性能: {gflops_torch:.2f} GFLOPS")
        
        # 检查精度 (half precision 的误差会比较大)
        if max_error_vs_torch < 1.0:  # half precision 的误差容忍度较大
            print(f"  ✓ 结果验证通过")
        else:
            print(f"  ⚠ 结果验证警告，误差较大（half precision 精度限制）")
            print(f"    前5x5结果对比:")
            print(f"    WMMA结果:\n{C_mma_np[:5, :5]}")
            print(f"    PyTorch结果:\n{C_torch_np[:5, :5]}")


def benchmark_single_size(M, N, K, num_iterations=10, warmup=5):
    """对单个矩阵大小进行基准测试"""
    # 准备 GPU 数据
    A_gpu = torch.randn(M, N, dtype=torch.float16, device='cuda')
    B_gpu = torch.randn(N, K, dtype=torch.float16, device='cuda')
    C_gpu = torch.zeros(M, K, dtype=torch.float16, device='cuda')
    
    # 预热
    for _ in range(warmup):
        lib.mma_gemm(A_gpu, B_gpu, C_gpu)
    torch.cuda.synchronize()
    
    # 计时
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        lib.mma_gemm(A_gpu, B_gpu, C_gpu)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    return np.mean(times)


def benchmark_mma_gemm():
    """性能基准测试"""
    print("\n" + "="*60)
    print("性能基准测试")
    print("="*60)
    
    sizes = [128, 256, 512, 1024, 2048]
    num_iterations = 10
    
    for size in sizes:
        M, N, K = size, size, size
        A = torch.randn(M, N, dtype=torch.float16, device='cuda')
        B = torch.randn(N, K, dtype=torch.float16, device='cuda')
        C = torch.zeros(M, K, dtype=torch.float16, device='cuda')
        
        # 预热
        for _ in range(5):
            lib.mma_gemm(A, B, C)
        torch.cuda.synchronize()
        
        # 计时
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            lib.mma_gemm(A, B, C)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000
        gflops = (2.0 * M * N * K) / (avg_time / 1000.0) / 1e9  # GFLOPS
        
        print(f"大小 {size}x{size}: {avg_time:.3f} ± {std_time:.3f} ms, {gflops:.2f} GFLOPS")


def plot_performance_comparison():
    """绘制性能图"""
    print("\n" + "="*60)
    print("生成性能图...")
    print("="*60)
    
    # 测试不同的问题大小 (16 的倍数)
    problem_sizes = [128, 256, 512, 768, 1024, 1536, 2048]
    num_iterations = 10
    warmup = 5
    
    mma_times = []
    torch_times = []
    sizes_list = []
    
    for size in problem_sizes:
        print(f"测试大小: {size}x{size}...")
        M, N, K = size, size, size
        
        # 准备数据
        A = torch.randn(M, N, dtype=torch.float16, device='cuda')
        B = torch.randn(N, K, dtype=torch.float16, device='cuda')
        C_mma = torch.zeros(M, K, dtype=torch.float16, device='cuda')
        C_torch = torch.zeros(M, K, dtype=torch.float16, device='cuda')
        
        # 测试 WMMA
        for _ in range(warmup):
            lib.mma_gemm(A, B, C_mma)
        torch.cuda.synchronize()
        
        times_mma = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            lib.mma_gemm(A, B, C_mma)
            torch.cuda.synchronize()
            times_mma.append(time.time() - start)
        
        # 测试 PyTorch
        for _ in range(warmup):
            torch.matmul(A, B, out=C_torch)
        torch.cuda.synchronize()
        
        times_torch = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            torch.matmul(A, B, out=C_torch)
            torch.cuda.synchronize()
            times_torch.append(time.time() - start)
        
        avg_time_mma = np.mean(times_mma)
        avg_time_torch = np.mean(times_torch)
        
        mma_times.append(avg_time_mma)
        torch_times.append(avg_time_torch)
        sizes_list.append(size)
        
        gflops_mma = (2.0 * size * size * size) / avg_time_mma / 1e9
        gflops_torch = (2.0 * size * size * size) / avg_time_torch / 1e9
        
        print(f"  WMMA: {avg_time_mma*1000:.3f} ms, {gflops_mma:.2f} GFLOPS")
        print(f"  PyTorch: {avg_time_torch*1000:.3f} ms, {gflops_torch:.2f} GFLOPS")
        print()
    
    # 转换为 numpy 数组
    sizes_array = np.array(sizes_list)
    mma_times = np.array(mma_times)
    torch_times = np.array(torch_times)
    
    # 绘制图表
    plt.figure(figsize=(12, 8))
    
    try:
        label_mma = 'WMMA (Tensor Core)'
        label_torch = 'PyTorch matmul'
        xlabel = '问题大小 (矩阵维度)'
        ylabel = '时间 (单位: 秒)'
        title = 'Tensor Core WMMA 性能对比'
    except:
        label_mma = 'WMMA (Tensor Core)'
        label_torch = 'PyTorch matmul'
        xlabel = 'Problem Size (Matrix Dimension)'
        ylabel = 'Time (seconds)'
        title = 'Tensor Core WMMA Performance Comparison'
    
    plt.semilogy(sizes_array, mma_times, 'r-s', label=label_mma, linewidth=2, markersize=8)
    plt.semilogy(sizes_array, torch_times, 'b-o', label=label_torch, linewidth=2, markersize=8)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlim(left=0)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = 'mma_performance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 性能图已保存到: {output_file}")
    
    # 显示图片
    plt.show()
    
    return sizes_array, mma_times, torch_times


if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        sys.exit(1)
    
    # 运行测试
    test_mma_gemm()
    
    # 运行基准测试
    benchmark_mma_gemm()
    
    # 生成性能对比图
    plot_performance_comparison()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
