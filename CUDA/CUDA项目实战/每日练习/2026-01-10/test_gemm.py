import torch
from torch.utils.cpp_extension import load

gemm_lib = load(
    name = "gemm",
    sources = ["gemm.cu"],
    extra_cuda_cflags = [
        "-O3",
        "-gencode", "arch=compute_75,code=sm_75"
    ],
    extra_cflags = [
        "-std=C++17"
    ],
    verbose = True
)

def benchmark_gemm(M, K, N, iters = 10):
    input_a = torch.randn(M, K, device="cuda")
    input_b = torch.randn(K, N, device="cuda")

    output = torch.randn(M, N, device="cuda")

    torch_output = input_a @ input_b

    gemm_lib.gemm(input_a, input_b, output)
    print(output)
    print(torch_output)
    torch.testing.assert_close(output, torch_output, rtol=1e-6, atol=1e-6)



if __name__ == "__main__":
    benchmark_gemm(2, 2, 2)
    benchmark_gemm(1024, 1024, 1024)