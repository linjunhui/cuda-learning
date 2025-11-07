#!/usr/bin/env python3
"""
Flash-Attention 测试脚本
用于验证 Flash-Attention 安装和基本功能
"""

import torch
import time
import sys

def test_flash_attn_import():
    """测试 Flash-Attention 导入"""
    try:
        from flash_attn import flash_attn_func
        print("✅ Flash-Attention 导入成功")
        return flash_attn_func
    except ImportError as e:
        print(f"❌ Flash-Attention 导入失败: {e}")
        print("请检查安装是否正确")
        sys.exit(1)

def test_basic_functionality(flash_attn_func):
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    # 配置参数
    batch_size = 2
    seq_len = 1024
    num_heads = 12
    head_dim = 64
    
    print(f"配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    
    # 创建测试数据
    print("\n创建测试数据...")
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    device='cuda', dtype=torch.float16)
    
    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")
    
    # 运行 Flash-Attention
    print("\n运行 Flash-Attention...")
    try:
        out = flash_attn_func(q, k, v)
        print(f"✅ 成功！输出形状: {out.shape}")
        print(f"  输出数据类型: {out.dtype}")
        print(f"  输出设备: {out.device}")
        return True
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance(flash_attn_func):
    """测试性能"""
    print("\n=== 性能测试 ===")
    
    # 配置参数
    batch_size = 2
    seq_len = 2048
    num_heads = 12
    head_dim = 64
    num_iterations = 100
    
    print(f"配置:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Iterations: {num_iterations}")
    
    # 创建测试数据
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                    device='cuda', dtype=torch.float16)
    
    # 预热
    print("\n预热...")
    for _ in range(10):
        _ = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    
    # 性能测试
    print("开始性能测试...")
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        _ = flash_attn_func(q, k, v)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_iterations * 1000  # ms
    throughput = (batch_size * seq_len) / (avg_time / 1000)  # tokens/s
    
    print(f"✅ 性能测试完成")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  吞吐量: {throughput:.0f} tokens/s")

def test_different_seq_lengths(flash_attn_func):
    """测试不同序列长度"""
    print("\n=== 不同序列长度测试 ===")
    
    batch_size = 2
    num_heads = 12
    head_dim = 64
    seq_lengths = [512, 1024, 2048, 4096]
    
    for seq_len in seq_lengths:
        print(f"\n序列长度: {seq_len}")
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                        device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                        device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                        device='cuda', dtype=torch.float16)
        
        try:
            torch.cuda.synchronize()
            start = time.time()
            out = flash_attn_func(q, k, v)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            
            print(f"  ✅ 成功 - 时间: {elapsed:.2f} ms")
            print(f"  输出形状: {out.shape}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("Flash-Attention 测试脚本")
    print("=" * 60)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，请检查 GPU 和驱动")
        sys.exit(1)
    
    print(f"\n✅ CUDA 可用")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  PyTorch 版本: {torch.__version__}")
    
    # 测试导入
    flash_attn_func = test_flash_attn_import()
    
    # 测试基本功能
    if not test_basic_functionality(flash_attn_func):
        print("\n❌ 基本功能测试失败，停止测试")
        sys.exit(1)
    
    # 性能测试
    test_performance(flash_attn_func)
    
    # 不同序列长度测试
    test_different_seq_lengths(flash_attn_func)
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
