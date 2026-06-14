import torch
from torch.utils.cpp_extension import load

vector_add_lib = load(
    name="vector_add",
    sources=[
        "vector_addition.cu"
    ],
    extra_cuda_cflags = [
        "-gencode", "arch=compute_75,code=sm_75"
    ]
)

def test_benchmark(N: int = 4, iters=1000):
    vector_a = torch.randn(N, device='cuda')
    vector_b = torch.randn(N, device='cuda')
    vector_output = torch.empty_like(vector_a, device='cuda')

    vector_c = vector_a + vector_b
    #print(vector_c)

    vector_add_lib.vector_add(vector_a, vector_b, vector_output, N)
    #print(vector_output)

    torch.testing.assert_close(
        vector_c,
        vector_output,
        rtol=1e-6, # 相对误差
        atol=1e-6, # 绝对误差
        msg="自定义向量加法与PyTorch原生加法不一致"
    )

    # warpup
    for i  in  range(10):
        vector_add_lib.vector_add(vector_a, vector_b, vector_output, N)


    # 开始 
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(iters):
        vector_add_lib.vector_add(vector_a, vector_b, vector_output, N)
    end_event.record()
    torch.cuda.synchronize()

    time_elapsed = start_event.elapsed_time(end_event)
    average_time = time_elapsed / iters
        
    # 吞吐量计算 GB/s
    through_output = 3 * 4 * N / (average_time * 1e-3) / 1e9

    print(f"Size = {N}, 迭代次数 {iters} 平均耗时：{average_time:.4f} ms  吞吐量: {through_output:.4f} GB/s")

if __name__ == '__main__':
    for i in range(100000, 9000000, 100000):
        test_benchmark(i)