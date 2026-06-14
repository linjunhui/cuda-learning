import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F
import os

# 获取当前路径
abs_path = os.path.dirname(os.path.abspath(__file__))
sources = [
    os.path.join(abs_path, 'csrc/bindings.cpp'),
    os.path.join(abs_path, 'csrc/softmax_kernel.cu'),
    os.path.join(abs_path, 'csrc/softmax_kernel_vec.cu')
]

# 1. 【即时编译】直接加载
# 这种方式不会受 pip 配置和清华源的影响
print("正在编译 CUDA 算子，请稍候...")
softmax_lib = load(
    name="online_softmax",
    sources=sources,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-gencode", "arch=compute_75,code=sm_75", # 1660 Super 核心配置
    ],
    verbose=True
)
print("✅ 编译并加载成功！\n")

def benchmark_softmax(N, D, dtype=torch.float32, iters=10, impl_name="online_softmax", use_vec=False):
    input_d = torch.randn(N, D, device='cuda', dtype=dtype)
    
    # 向量化实现需要 D 是 4 的倍数
    if use_vec and D % 4 != 0:
        print(f"⚠️  跳过 {impl_name}: D={D} 不是4的倍数（向量化实现要求）\n")
        return

    # 2. 正确性验证
    # 官方结果
    ref_out = F.softmax(input_d, dim=-1)
    # 你的算子结果
    if use_vec:
        out = softmax_lib.online_softmax_vec(input_d)
    else:
        out = softmax_lib.online_softmax(input_d)

    try:
        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)
        print(f"✅ {impl_name} 正确性验证 PASS: (N={N}, D={D})")
    except Exception as e:
        print(f"❌ {impl_name} 正确性验证 FAIL: (N={N}, D={D})")
        print(e)
        return

    # 3. 性能测试
    # Warmup
    for _ in range(20):
        if use_vec:
            softmax_lib.online_softmax_vec(input_d)
        else:
            softmax_lib.online_softmax(input_d)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        if use_vec:
            softmax_lib.online_softmax_vec(input_d)
        else:
            softmax_lib.online_softmax(input_d)
    end_event.record()

    torch.cuda.synchronize()
    avg_ms = start_event.elapsed_time(end_event) / iters

    # 计算吞吐量 (GB/s): 读取 + 写入
    elem_size = 4 if dtype == torch.float32 else 2
    gbps = (N * D * elem_size * 2) / (avg_ms / 1000) / 1e9

    print(f"📊 {impl_name}: {avg_ms:.4f} ms | {gbps:.2f} GB/s\n")

if __name__ == "__main__":
    test_cases = [
        (1024, 1024),
        (1024, 4096),
        (1024, 8192),
        (2048, 1024),
        (2048, 4096),
        (2048, 8192)
    ]
    
    print("=" * 60)
    print("测试基础实现 (online_softmax)")
    print("=" * 60)
    for N, D in test_cases:
        benchmark_softmax(N, D, impl_name="基础实现", use_vec=True)
    
    print("\n" + "=" * 60)
    print("测试向量化实现 (online_softmax_vec)")
    print("=" * 60)
    for N, D in test_cases:
        benchmark_softmax(N, D, impl_name="向量化实现", use_vec=True)