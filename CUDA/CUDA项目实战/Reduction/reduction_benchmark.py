from torch.utils.cpp_extension import load
import torch

lib = load(
    name="reduction_lib",
    sources=["reduction_naive.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
)

def test_reduction_func():
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float32, device="cuda")
    output = lib.reduce_naive_add(input)
    print(output)

    input = input.to("cpu")
    output_cpu = lib.reduce_cpu(input)
    print(output_cpu)

    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float32, device="cuda")
    output = lib.reduce_block_add(input)
    print(output)
    print(input)

test_reduction_func()
