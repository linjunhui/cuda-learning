# SGLang Softmax实现分析报告

## 概述

SGLang项目中有多种softmax实现方式，主要包括自定义CUDA kernel、Triton kernel以及部分场景下使用PyTorch的torch.softmax。本文档详细分析了各种实现及其使用场景。

## Softmax实现类型统计

### 1. 自定义CUDA Kernel实现

#### 1.1 MoE TopK Softmax (GPU)
**位置**: `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu`

**实现方式**:
- **`moeSoftmax`**: 通用softmax kernel，使用三遍扫描算法
  - 第一遍：应用变换（tanh softcapping、correction bias），找到最大值
  - 第二遍：计算exp(x - max)的和
  - 第三遍：计算最终的softmax值
  - 使用CUB库进行block级别的reduce操作
  
- **`topkGatingSoftmax`**: 融合了softmax和top-k选择的优化kernel
  - 专门针对MoE路由场景优化
  - 当expert数量是2的幂次时，使用warp级别的reduce（无需shared memory）
  - 融合了softmax计算和top-k选择，减少内存访问
  - 支持renormalize选项

**特点**:
- 支持float32、float16、bfloat16数据类型
- 支持moe_softcapping（tanh软限制）
- 支持correction_bias
- 针对2的幂次expert数量有特殊优化路径

**调用接口**: `torch.ops.sgl_kernel.topk_softmax.default()`

#### 1.2 CPU版本TopK Softmax
**位置**: `sgl-kernel/csrc/cpu/topk.cpp`

**实现方式**:
- `topk_softmax_kernel_impl`: CPU版本的topk softmax实现
- 使用模板特化针对不同expert数量（1, 2, 4, 8, 16, 32, 64, 128, 160, 256）
- 使用`std::partial_sort`进行top-k选择

**调用接口**: `torch.ops.sgl_kernel.topk_softmax_cpu()`

#### 1.3 Flash Attention中的Softmax
**位置**: Flash Attention相关kernel（通过`torch.ops.sgl_kernel.fwd.default`等调用）

**实现方式**:
- Flash Attention kernel内部实现了softmax的在线计算
- 使用logsumexp (LSE) 方式避免数值不稳定
- 不显式存储完整的attention矩阵，而是在线计算softmax
- 返回`softmax_lse`（logsumexp值）用于后续计算

**特点**:
- 内存高效（不存储完整attention矩阵）
- 数值稳定（使用logsumexp）
- 融合在attention计算中

**相关接口**:
- `torch.ops.sgl_kernel.fwd.default()` - Flash Attention forward
- `torch.ops.sgl_kernel.fwd_sparse.default()` - Sparse Flash Attention
- `torch.ops.sgl_kernel.fwd_kvcache_mla.default()` - MLA (Multi-head Latent Attention)
- `torch.ops.sgl_kernel.fwd_kvcache_mla_fp8.default()` - FP8版本的MLA

### 2. Triton Kernel实现

#### 2.1 Log Softmax (Triton)
**位置**: `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`

**实现方式**:
- `_log_softmax_kernel`: 使用Triton JIT编译的log_softmax kernel
- 三遍扫描：
  1. 找到最大值
  2. 计算exp(x - max)的和，然后取log
  3. 计算log_softmax = x - max - log_sum_exp

**特点**:
- 支持任意形状的2D tensor（沿最后一个维度计算）
- 使用Triton的向量化操作
- 针对batch invariant场景优化

**调用接口**: `log_softmax(input, dim=-1)`

#### 2.2 Decode Attention中的Softmax
**位置**: `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`

**实现方式**:
- `_decode_softmax_reducev_fwd`: 在decode attention中使用的softmax
- 融合了softmax和reduce操作

### 3. PyTorch torch.softmax使用场景

虽然SGLang有大量自定义实现，但在以下场景中仍使用`torch.softmax`：

#### 3.1 测试代码中的参考实现
- `sgl-kernel/tests/test_moe_topk_softmax.py`: 用于验证自定义kernel的正确性
- `sgl-kernel/tests/test_flash_attention*.py`: 用于对比测试
- `sgl-kernel/tests/test_sampling.py`: 采样相关的测试

#### 3.2 模型特定实现
- **Sampler层** (`python/sglang/srt/layers/sampler.py`):
  - 在某些条件下使用`torch.softmax`进行概率计算
  - 当不需要top-k/top-p采样时，直接使用torch.softmax
  - 用于计算原始logprob（未经过temperature scaling）

- **特定模型**:
  - `phi4mm_utils.py`: `masked_softmax`函数使用torch.softmax
  - `mixtral_quant.py`: MoE路由权重计算
  - `deepseek_janus_pro.py`: 某些注意力计算
  - `gemma3n_audio.py`: 概率计算

#### 3.3 Sampling相关
- `sgl-kernel/python/sgl_kernel/sampling.py`:
  - `top_k_top_p_sampling`: 在某些路径中使用`torch.softmax`
  - 主要用于top-k/top-p采样前的概率归一化

#### 3.4 Speculative Decoding
- `eagle_worker.py`, `eagle_worker_v2.py`: 用于计算target probabilities
- `eagle_info.py`, `eagle_info_v2.py`: 用于概率计算

## 实现统计总结

| 实现类型 | 位置 | 用途 | 是否调用torch.softmax |
|---------|------|------|---------------------|
| MoE TopK Softmax (CUDA) | `moe_topk_softmax_kernels.cu` | MoE路由 | ❌ 否 |
| MoE TopK Softmax (CPU) | `csrc/cpu/topk.cpp` | MoE路由(CPU) | ❌ 否 |
| Flash Attention Softmax | Flash Attention kernels | Attention计算 | ❌ 否（内部实现） |
| Log Softmax (Triton) | `batch_invariant_ops.py` | Batch invariant场景 | ❌ 否 |
| Decode Attention Softmax | `triton_ops/decode_attention.py` | Decode attention | ❌ 否 |
| Sampler层 | `layers/sampler.py` | 采样 | ✅ 是（部分场景） |
| 测试代码 | `tests/` | 参考实现 | ✅ 是 |
| 模型特定实现 | 各模型文件 | 特定计算 | ✅ 是（部分场景） |

## 核心结论

1. **SGLang主要使用自定义softmax实现**，而不是依赖torch.softmax
2. **自定义实现的原因**:
   - **性能优化**: MoE场景下需要融合softmax和top-k选择
   - **内存效率**: Flash Attention中在线计算softmax，不存储完整矩阵
   - **数值稳定性**: 使用logsumexp避免数值溢出
   - **特定优化**: 针对2的幂次expert数量等场景的特殊优化

3. **torch.softmax的使用场景**:
   - 主要用于测试代码中的参考实现
   - 部分模型特定场景（如masked_softmax）
   - 采样层中的某些路径（当不需要特殊优化时）
   - Speculative decoding中的概率计算

4. **实现数量统计**:
   - **CUDA Kernel**: 2个主要实现（moeSoftmax, topkGatingSoftmax）
   - **CPU实现**: 1个（topk_softmax_kernel_impl）
   - **Triton Kernel**: 2个（log_softmax, decode_softmax）
   - **Flash Attention**: 多个变体（fwd, fwd_sparse, fwd_kvcache_mla等）
   - **总计**: 约**6-8种**不同的softmax实现方式

## 代码示例

### MoE TopK Softmax使用示例
```python
from sgl_kernel import topk_softmax

# GPU版本
topk_weights, topk_indices = topk_softmax(
    gating_output,  # [num_tokens, num_experts]
    topk=2,
    renormalize=True
)
```

### Flash Attention中的Softmax
```python
from sgl_kernel import flash_attn_func

out, softmax_lse = flash_attn_func(
    q, k, v,
    softmax_scale=1.0 / math.sqrt(head_dim),
    return_softmax_lse=True
)
```

### Triton Log Softmax
```python
from sglang.srt.batch_invariant_ops.batch_invariant_ops import log_softmax

log_probs = log_softmax(logits, dim=-1)
```

## 总结

SGLang项目中有**多种softmax实现**（约6-8种），主要采用**自定义CUDA/Triton kernel**而非直接调用torch.softmax。自定义实现主要用于：
- MoE路由（性能关键路径）
- Flash Attention（内存和性能优化）
- Batch invariant场景（Triton优化）

torch.softmax主要用于测试参考实现和部分非性能关键路径。
