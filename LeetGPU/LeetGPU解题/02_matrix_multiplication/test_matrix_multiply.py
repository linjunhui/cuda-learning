import torch
from torch.utils.cpp_extension import load

matrix_multiply_lib = load(
    name="matrix_multiply",
    sources = [
        "matrix_multiply.cu"
    ],
    extra_cuda_cflags = [
        "-gencode", "arch=compute_75,code=sm_75"
    ]
)


def  test_benchmark(M, K, N, iters=100):

    matrix_a = torch.randn(M, K, device="cuda")
    matrix_b = torch.randn(K, N, device="cuda")
    matrix_ouput = torch.randn(M, N, device="cuda")

    print(matrix_a @ matrix_b)

    matrix_multiply_lib.matrix_multiply(matrix_a, matrix_b, matrix_ouput)

if __name__ == '__main__':
    test_benchmark(2, 4, 2)