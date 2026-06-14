import torch
from torch.utils.cpp_extension import load
import sys

sys.stdout.flush()

lib = load(
        name="gemm",
        sources=["coalesced_gemm.cu"],
        extra_cuda_cflags = [
            "-O3",
            "-gencode","arch=compute_75,code=sm_75"
        ],
        extra_cflags = [
            "-std=C++17"
        ],
        verbose = True
    )

def benchmark_coalesced_gemm(M, K, N, iters=10):
    input_a = torch.randn(M, K, device="cuda")
    input_b = torch.randn(K, N, device="cuda")

    output = torch.zeros(M, N).to("cuda")
    torch_ouput = input_a @ input_b

    lib.gemm(input_a, input_b, output)
    torch.cuda.synchronize()
    print(output)
    print(torch_ouput)
    sys.stdout.flush()
    # 自实现 GEMM 与 PyTorch 内部 GEMM 的浮点累加顺序不同，会产生非常小的数值误差
    # 使用稍微宽松的容差判断数值正确性
    torch.testing.assert_close(output, torch_ouput, rtol=1e-3, atol=1e-4)

    print("---")

if __name__ == "__main__":
    # 矩阵尺寸
    matrix_size = (
        (2, 4, 2),
        (32, 33, 32),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    )

    for M, K, N in matrix_size:
        benchmark_coalesced_gemm(M, K, N, iters=10)