# SGLang 模型推理流程与算子详解

## 目录

1. [推理流程总览](#1-推理流程总览)
2. [Prefill 阶段详解](#2-prefill-阶段详解)
3. [Decode 阶段详解](#3-decode-阶段详解)
4. [Transformer 层内部流程](#4-transformer-层内部流程)
5. [核心算子详解](#5-核心算子详解)
6. [算子融合优化](#6-算子融合优化)
7. [性能优化技术](#7-性能优化技术)

---

## 1. 推理流程总览

### 1.1 整体架构

SGLang 的推理流程分为两个主要阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                    SGLang 推理流程                          │
└─────────────────────────────────────────────────────────────┘

输入: Token IDs [batch_size, seq_len]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: Prefill (预填充阶段)                                │
├─────────────────────────────────────────────────────────────┤
│ 处理整个 prompt，计算所有 token 的隐藏状态                  │
│ 建立 KV Cache，为后续解码做准备                             │
│ 计算复杂度: O(seq_len²)                                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段 2: Decode (解码阶段)                                   │
├─────────────────────────────────────────────────────────────┤
│ 逐个生成新 token，每次只处理一个新 token                    │
│ 利用 KV Cache，避免重复计算                                 │
│ 计算复杂度: O(seq_len) per token                           │
└─────────────────────────────────────────────────────────────┘
    ↓
输出: 生成的 Token 序列
```

### 1.2 关键组件

- **ModelRunner**: 负责协调整个推理流程
- **ForwardBatch**: 包含当前批次的所有 token 和元数据
- **Attention Backend**: 处理注意力计算（FlashAttention、FlashInfer、Triton 等）
- **KV Cache**: 存储历史 token 的 Key 和 Value
- **CUDA Kernels**: 高性能的 GPU 算子实现

---

## 2. Prefill 阶段详解

### 2.1 Prefill 阶段的作用

Prefill 阶段处理输入的 prompt，一次性计算所有 token 的隐藏状态，并建立 KV Cache。这是生成过程的初始化阶段。

### 2.2 完整流程

```
输入: Token IDs [batch_size, seq_len]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: Token Embedding                                     │
├─────────────────────────────────────────────────────────────┤
│ 算子: Embedding Lookup (PyTorch)                           │
│ 实现: torch.nn.Embedding                                    │
│ 输入: [batch_size, seq_len]                                │
│ 输出: [batch_size, seq_len, hidden_size]                   │
│                                                              │
│ 操作:                                                       │
│   hidden_states = embedding_layer(input_ids)                │
│   将每个 token ID 映射为 dense 向量                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: 位置编码 (RoPE)                                     │
├─────────────────────────────────────────────────────────────┤
│ 算子: RoPE Kernel                                           │
│ 实现: sgl-kernel/csrc/elementwise/rope.cu                  │
│ 输入: Q, K [batch_size, seq_len, num_heads, head_dim]      │
│ 操作: 对每个位置应用旋转位置编码                            │
│ 输出: Q_rotated, K_rotated                                  │
│                                                              │
│ 数学公式:                                                   │
│   R_θ = [cos(θ)  -sin(θ)]                                   │
│         [sin(θ)   cos(θ)]                                   │
│                                                              │
│   Q_rotated = Q @ R_θ                                       │
│   K_rotated = K @ R_θ                                       │
│                                                              │
│ CUDA Kernel: rotary_embedding_kernel()                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: Transformer Layers (N 层)                          │
├─────────────────────────────────────────────────────────────┤
│ 对每一层执行以下操作:                                       │
│                                                              │
│  3.1 Attention (Prefill)                                    │
│      ├─ QKV 投影 (GEMM)                                     │
│      ├─ RoPE 位置编码                                       │
│      ├─ FlashAttention 计算                                 │
│      └─ 存储 K, V 到 KV Cache                               │
│                                                              │
│  3.2 残差连接 + RMSNorm                                     │
│      ├─ Fused Add RMSNorm                                   │
│      └─ 融合操作减少内存访问                                 │
│                                                              │
│  3.3 MLP (前馈网络)                                         │
│      ├─ Gate 投影 (GEMM)                                    │
│      ├─ SiLU 激活                                           │
│      ├─ Up 投影 (GEMM)                                      │
│      ├─ SiLU and Mul (融合)                                 │
│      └─ Down 投影 (GEMM)                                    │
│                                                              │
│  3.4 残差连接 + RMSNorm                                     │
│      └─ Fused Add RMSNorm                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 4: 最终归一化                                          │
├─────────────────────────────────────────────────────────────┤
│ 算子: RMSNorm                                               │
│ 输入: 最后一层的隐藏状态                                    │
│ 输出: 归一化后的隐藏状态                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 5: LM Head (输出层)                                    │
├─────────────────────────────────────────────────────────────┤
│ 算子: GEMM (矩阵乘法)                                       │
│ 实现: gemm/*.cu                                             │
│ 输入: [batch_size, seq_len, hidden_size]                   │
│ 操作: hidden_states @ lm_head_weight^T                      │
│ 输出: Logits [batch_size, seq_len, vocab_size]             │
│                                                              │
│ 注意: 通常只取最后一个 token 的 logits                      │
│   logits = logits[:, -1, :]  # [batch_size, vocab_size]    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 6: 采样                                                │
├─────────────────────────────────────────────────────────────┤
│ 算子: TopK Kernel                                           │
│ 实现: elementwise/topk.cu                                   │
│ 输入: Logits [vocab_size]                                   │
│ 操作:                                                       │
│   1. TopK 采样: 找到 TopK 个最大值的索引                    │
│   2. 计算概率分布 (可选): softmax(logits / temperature)    │
│   3. 根据概率采样: TopK/TopP 采样                           │
│ 输出: 下一个 token ID                                       │
│                                                              │
│ CUDA Kernel: fast_topk_cuda_tl()                           │
└─────────────────────────────────────────────────────────────┘
    ↓
输出: 下一个 token ID
```

### 2.3 Prefill 阶段的 Attention 计算

Prefill 阶段使用 FlashAttention 或类似的优化注意力实现：

```python
# 伪代码
def prefill_attention(q, k, v, causal_mask=True):
    """
    Prefill 阶段的注意力计算
    
    输入:
        q: [batch_size, seq_len, num_heads, head_dim]
        k: [batch_size, seq_len, num_heads, head_dim]
        v: [batch_size, seq_len, num_heads, head_dim]
    
    输出:
        attn_output: [batch_size, seq_len, num_heads, head_dim]
    """
    # 1. 计算注意力分数
    scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
    # scores: [batch_size, seq_len, seq_len]
    
    # 2. 应用 Causal Mask (确保只能看到之前的 token)
    if causal_mask:
        mask = torch.tril(torch.ones(seq_len, seq_len))
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Softmax
    attn_weights = softmax(scores, dim=-1)
    # attn_weights: [batch_size, seq_len, seq_len]
    
    # 4. 加权求和
    attn_output = attn_weights @ v
    # attn_output: [batch_size, seq_len, num_heads, head_dim]
    
    # 5. 存储 K, V 到 KV Cache
    kv_cache.store_kv(k, v, positions=position_ids)
    
    return attn_output
```

**关键优化**：
- **分块计算**: FlashAttention 将大矩阵分成小块，避免 O(n²) 内存占用
- **在线 Softmax**: 在计算过程中逐步归一化，不需要存储完整的注意力矩阵
- **KV Cache 存储**: 同时将 K, V 存储到 KV Cache，供后续 Decode 阶段使用

---

## 3. Decode 阶段详解

### 3.1 Decode 阶段的作用

Decode 阶段逐个生成新 token。每次只处理一个新 token，利用 KV Cache 避免重复计算历史 token 的 Key 和 Value。

### 3.2 完整流程（单步解码）

```
输入: 新 token ID [batch_size, 1]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: Token Embedding                                     │
├─────────────────────────────────────────────────────────────┤
│ 算子: Embedding Lookup (PyTorch)                           │
│ 输入: [batch_size, 1]                                       │
│ 输出: [batch_size, 1, hidden_size]                         │
│                                                              │
│ 注意: 只处理一个新 token，形状是 [1] 而不是 [seq_len]      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: 位置编码 (RoPE)                                     │
├─────────────────────────────────────────────────────────────┤
│ 算子: RoPE Kernel                                           │
│ 输入: Q, K [batch_size, 1, num_heads, head_dim]            │
│ 位置: current_pos = prompt_len + decode_step                │
│ 操作: 对新 token 的位置应用旋转矩阵                         │
│ 输出: Q_rotated, K_rotated                                  │
│                                                              │
│ 注意: 只需要计算一个新位置的旋转                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: Transformer Layers (N 层)                          │
├─────────────────────────────────────────────────────────────┤
│ 对每一层执行以下操作:                                       │
│                                                              │
│  3.1 Attention (Decode) ⭐ 核心优化                         │
│      ├─ QKV 投影 (GEMM)                                     │
│      ├─ RoPE 位置编码                                       │
│      ├─ 从 KV Cache 读取历史的 K, V                        │
│      ├─ Lightning Attention Decode 计算                     │
│      └─ 更新 KV Cache (追加新的 K, V)                       │
│                                                              │
│  3.2 残差连接 + RMSNorm                                     │
│      └─ Fused Add RMSNorm                                   │
│                                                              │
│  3.3 MLP (前馈网络)                                         │
│      └─ 同 Prefill 阶段                                     │
│                                                              │
│  3.4 残差连接 + RMSNorm                                     │
│      └─ Fused Add RMSNorm                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 4: 最终归一化                                          │
├─────────────────────────────────────────────────────────────┤
│ 算子: RMSNorm                                               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 5: LM Head (输出层)                                    │
├─────────────────────────────────────────────────────────────┤
│ 算子: GEMM                                                  │
│ 输入: [batch_size, 1, hidden_size]                          │
│ 输出: Logits [batch_size, 1, vocab_size]                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 6: 采样                                                │
├─────────────────────────────────────────────────────────────┤
│ 算子: TopK Kernel                                           │
│ 输出: 下一个 token ID                                       │
└─────────────────────────────────────────────────────────────┘
    ↓
输出: 下一个 token ID
```

### 3.3 Decode 阶段的 Attention 计算

Decode 阶段使用 Lightning Attention Decode 或类似的优化实现：

```python
# 伪代码
def decode_attention(q_new, k_new, v_new, kv_cache, current_pos):
    """
    Decode 阶段的注意力计算
    
    输入:
        q_new: [batch_size, 1, num_heads, head_dim]  # 新 token 的 Q
        k_new: [batch_size, 1, num_heads, head_dim]  # 新 token 的 K
        v_new: [batch_size, 1, num_heads, head_dim]  # 新 token 的 V
        kv_cache: 存储历史 token 的 K, V
        current_pos: 当前 token 的位置
    
    输出:
        attn_output: [batch_size, 1, num_heads, head_dim]
    """
    # 1. 从 KV Cache 读取历史的 K, V
    past_k = kv_cache.get_k()  # [batch_size, num_heads, past_len, head_dim]
    past_v = kv_cache.get_v()  # [batch_size, num_heads, past_len, head_dim]
    
    # 2. 计算新 token 与所有历史 token 的注意力分数
    # 只需要计算 q_new @ past_k^T，而不是完整的注意力矩阵
    scores = q_new @ past_k.transpose(-2, -1) / sqrt(head_dim)
    # scores: [batch_size, 1, past_len]
    
    # 3. Softmax
    attn_weights = softmax(scores, dim=-1)
    # attn_weights: [batch_size, 1, past_len]
    
    # 4. 加权求和
    attn_output = attn_weights @ past_v
    # attn_output: [batch_size, 1, num_heads, head_dim]
    
    # 5. 更新 KV Cache (追加新的 K, V)
    kv_cache.append_kv(k_new, v_new, position=current_pos)
    
    return attn_output
```

**关键优化**：
- **增量计算**: 只计算新 token 与历史 token 的注意力，复杂度从 O(n²) 降到 O(n)
- **KV Cache 复用**: 历史 token 的 K, V 直接从缓存读取，无需重新计算
- **Lightning Attention**: 专门优化的解码阶段注意力 kernel，减少内存访问

---

## 4. Transformer 层内部流程

### 4.1 单层 Transformer 的完整流程

每个 Transformer 层包含以下组件：

```
输入: hidden_states [batch_size, seq_len, hidden_size]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 子层 1: Self-Attention                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1.1 QKV 投影                                               │
│     算子: GEMM                                              │
│     输入: hidden_states [batch_size, seq_len, hidden_size] │
│     操作:                                                   │
│       Q = hidden_states @ W_q^T  # [batch, seq, num_heads, head_dim]│
│       K = hidden_states @ W_k^T                            │
│       V = hidden_states @ W_v^T                            │
│                                                              │
│  1.2 RoPE 位置编码                                          │
│     算子: RoPE Kernel                                       │
│     输入: Q, K                                              │
│     输出: Q_rotated, K_rotated                              │
│                                                              │
│  1.3 Attention 计算                                         │
│     算子: FlashAttention (Prefill) / Lightning Attention (Decode)│
│     输入: Q_rotated, K_rotated, V                           │
│     输出: attn_output [batch, seq, num_heads, head_dim]     │
│                                                              │
│  1.4 Output 投影                                            │
│     算子: GEMM                                              │
│     输入: attn_output                                       │
│     输出: attn_output [batch, seq, hidden_size]            │
│                                                              │
│  1.5 残差连接 + RMSNorm                                     │
│     算子: Fused Add RMSNorm                                 │
│     操作:                                                   │
│       hidden_states = RMSNorm(hidden_states + attn_output)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 子层 2: MLP (前馈网络)                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  2.1 Gate 投影                                              │
│     算子: GEMM                                              │
│     输入: hidden_states [batch, seq, hidden_size]          │
│     输出: gate [batch, seq, intermediate_size]              │
│                                                              │
│  2.2 SiLU 激活                                              │
│     算子: Activation Kernel                                 │
│     输入: gate                                              │
│     操作: silu(x) = x / (1 + exp(-x))                      │
│     输出: gate_activated                                    │
│                                                              │
│  2.3 Up 投影                                                │
│     算子: GEMM                                              │
│     输入: hidden_states [batch, seq, hidden_size]          │
│     输出: up [batch, seq, intermediate_size]               │
│                                                              │
│  2.4 SiLU and Mul (融合操作)                                │
│     算子: Fused Activation Kernel                           │
│     操作: output = SiLU(gate) * up                          │
│     输出: mlp_intermediate                                  │
│                                                              │
│  2.5 Down 投影                                              │
│     算子: GEMM                                              │
│     输入: mlp_intermediate                                  │
│     输出: mlp_output [batch, seq, hidden_size]             │
│                                                              │
│  2.6 残差连接 + RMSNorm                                     │
│     算子: Fused Add RMSNorm                                 │
│     操作:                                                   │
│       hidden_states = RMSNorm(hidden_states + mlp_output)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    ↓
输出: hidden_states [batch_size, seq_len, hidden_size]
```

### 4.2 代码层面的实现

在 SGLang 的模型实现中，Transformer 层的 forward 方法大致如下：

```python
# 伪代码（基于 SGLang 的实际实现）
class TransformerDecoderLayer(nn.Module):
    def forward(self, hidden_states, positions, forward_batch, residual=None):
        # 1. Self-Attention
        # 1.1 QKV 投影
        q, k, v = self.attention.qkv_proj(hidden_states)
        
        # 1.2 RoPE 位置编码
        q, k = self.rotary_emb(positions, q, k)
        
        # 1.3 Attention 计算
        if forward_batch.is_prefill:
            # Prefill 阶段：使用 FlashAttention
            attn_output = self.attention.forward_prefill(
                q, k, v, forward_batch
            )
        else:
            # Decode 阶段：使用 Lightning Attention
            attn_output = self.attention.forward_decode(
                q, k, v, forward_batch
            )
        
        # 1.4 Output 投影
        attn_output = self.attention.o_proj(attn_output)
        
        # 1.5 残差连接 + RMSNorm
        if residual is None:
            residual = hidden_states
        hidden_states, residual = self.norm1(
            hidden_states + attn_output, residual
        )
        
        # 2. MLP
        # 2.1 Gate 投影
        gate = self.mlp.gate_proj(hidden_states)
        
        # 2.2 Up 投影
        up = self.mlp.up_proj(hidden_states)
        
        # 2.3 SiLU and Mul (融合)
        mlp_intermediate = silu(gate) * up
        
        # 2.4 Down 投影
        mlp_output = self.mlp.down_proj(mlp_intermediate)
        
        # 2.5 残差连接 + RMSNorm
        hidden_states, residual = self.norm2(
            hidden_states + mlp_output, residual
        )
        
        return hidden_states, residual
```

---

## 5. 核心算子详解

### 5.1 算子分类

SGLang 中的算子可以分为以下几类：

1. **基础算子**: Embedding、Copy 等
2. **线性变换算子**: GEMM（矩阵乘法）
3. **注意力算子**: FlashAttention、Lightning Attention 等
4. **激活函数算子**: SiLU、GELU 等
5. **归一化算子**: RMSNorm、LayerNorm 等
6. **位置编码算子**: RoPE
7. **采样算子**: TopK、TopP 等
8. **融合算子**: Fused Add RMSNorm、SiLU and Mul 等

### 5.2 关键算子详解

#### 5.2.1 GEMM (矩阵乘法)

**作用**: 执行线性变换，是 Transformer 中最常用的算子

**实现位置**: `sgl-kernel/csrc/gemm/*.cu`

**主要用途**:
- QKV 投影: `Q = hidden_states @ W_q^T`
- Output 投影: `output = attn_output @ W_o^T`
- MLP 投影: `gate = hidden_states @ W_gate^T`
- LM Head: `logits = hidden_states @ W_lm^T`

**优化技术**:
- 使用 CUTLASS 库进行高性能矩阵乘法
- 支持 INT8/FP8 量化
- 支持 Tensor Parallelism

#### 5.2.2 RoPE (旋转位置编码)

**作用**: 为 Query 和 Key 添加位置信息

**实现位置**: `sgl-kernel/csrc/elementwise/rope.cu`

**数学原理**:
```
对于位置 pos 和维度 i:
  θ_i = 10000^(-2i/d)
  
  R_θ = [cos(θ)  -sin(θ)]
        [sin(θ)   cos(θ)]
  
  Q_rotated = Q @ R_θ
  K_rotated = K @ R_θ
```

**CUDA Kernel**: `rotary_embedding_kernel()`

**优化技术**:
- 预计算旋转矩阵并缓存
- 向量化实现
- 支持批量处理

#### 5.2.3 FlashAttention (Prefill)

**作用**: Prefill 阶段的优化注意力计算

**实现位置**: `sgl-kernel/csrc/attention/*.cu`

**关键特性**:
- **分块计算**: 将大矩阵分成小块，避免 O(n²) 内存
- **在线 Softmax**: 逐步归一化，不需要存储完整注意力矩阵
- **Causal Mask**: 自动处理因果掩码

**内存复杂度**: O(n) 而不是 O(n²)

#### 5.2.4 Lightning Attention Decode

**作用**: Decode 阶段的优化注意力计算

**实现位置**: `sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu`

**关键特性**:
- **增量计算**: 只计算新 token 与历史的注意力
- **KV Cache 复用**: 直接从缓存读取历史 K, V
- **共享内存优化**: 使用共享内存减少全局内存访问

**计算复杂度**: O(n) per token

#### 5.2.5 Fused Add RMSNorm

**作用**: 融合残差连接和归一化操作

**实现位置**: `sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu`

**操作**:
```python
# 融合前（两次内存访问）
x = x + residual
x = RMSNorm(x)

# 融合后（一次内存访问）
x = FusedAddRMSNorm(x, residual)
```

**优势**:
- 减少内存访问次数
- 减少 kernel 启动开销
- 提高缓存利用率

#### 5.2.6 SiLU 激活函数

**作用**: MLP 中的激活函数

**实现位置**: `sgl-kernel/csrc/elementwise/activation.cu`

**数学公式**:
```
SiLU(x) = x / (1 + exp(-x))
```

**融合版本**: SiLU and Mul
```python
# 融合操作
output = SiLU(gate) * up
```

#### 5.2.7 TopK 采样

**作用**: 从 logits 中选择下一个 token

**实现位置**: `sgl-kernel/csrc/elementwise/topk.cu`

**算法**:
- 使用基数排序优化
- 支持 TopK 和 TopP 采样
- 支持温度缩放

**CUDA Kernel**: `fast_topk_cuda_tl()`

---

## 6. 算子融合优化

### 6.1 融合的目的

算子融合将多个连续的操作合并到一个 CUDA kernel 中执行，主要目的是：

1. **减少内存访问**: 中间结果不需要写回全局内存
2. **减少 kernel 启动开销**: 多个操作合并为一个 kernel
3. **提高缓存利用率**: 数据在寄存器/共享内存中流动

### 6.2 SGLang 中的融合算子

#### 6.2.1 Fused Add RMSNorm

**融合前**:
```python
# 需要两次 kernel 启动
x = x + residual  # Kernel 1
x = RMSNorm(x)    # Kernel 2
```

**融合后**:
```python
# 一次 kernel 启动
x = FusedAddRMSNorm(x, residual)
```

**实现**: `fused_add_rms_norm_kernel.cu`

#### 6.2.2 SiLU and Mul

**融合前**:
```python
# 需要两次 kernel 启动
gate_activated = SiLU(gate)  # Kernel 1
output = gate_activated * up # Kernel 2
```

**融合后**:
```python
# 一次 kernel 启动
output = FusedSiLUMul(gate, up)
```

**实现**: `activation.cu` 中的融合版本

#### 6.2.3 Attention and KV Update

在 Prefill 阶段，注意力和 KV Cache 更新可以融合：

```python
# 融合操作
attn_output, k_cache, v_cache = FusedAttentionKVUpdate(
    q, k, v, positions
)
```

### 6.3 融合的性能收益

- **内存带宽**: 减少 30-50% 的内存访问
- **延迟**: 减少 20-40% 的 kernel 启动开销
- **吞吐量**: 整体提升 15-30%

---

## 7. 性能优化技术

### 7.1 内存优化

#### 7.1.1 KV Cache 优化

- **PagedAttention**: 分页管理 KV Cache，提高内存利用率
- **RadixAttention**: 共享前缀缓存，减少重复计算
- **压缩存储**: 使用 INT8/FP8 压缩 KV Cache

#### 7.1.2 内存池管理

- **预分配内存池**: 减少动态内存分配开销
- **内存复用**: 在不同请求间复用内存

### 7.2 计算优化

#### 7.2.1 量化

- **INT8 GEMM**: 矩阵乘法使用 INT8，减少计算量和内存
- **FP8 GEMM**: 使用 FP8 精度，在精度和性能间平衡
- **AWQ**: 激活感知权重量化

#### 7.2.2 批处理优化

- **动态批处理**: 根据请求动态调整批次大小
- **连续内存布局**: 优化内存访问模式

### 7.3 并行优化

#### 7.3.1 Tensor Parallelism

- **模型并行**: 将模型参数分布到多个 GPU
- **通信优化**: 使用高效的 AllReduce 通信

#### 7.3.2 Pipeline Parallelism

- **流水线并行**: 将模型层分布到多个 GPU
- **微批次**: 使用微批次提高流水线效率

### 7.4 调度优化

#### 7.4.1 CUDA Graph

- **图捕获**: 捕获完整的计算图
- **图重放**: 减少 kernel 启动开销

#### 7.4.2 推测解码

- **小模型生成**: 使用小模型生成候选 token
- **大模型验证**: 大模型验证候选 token

---

## 8. 总结

### 8.1 推理流程总结

SGLang 的推理流程可以总结为：

1. **Prefill 阶段**: 处理整个 prompt，建立 KV Cache
2. **Decode 阶段**: 逐个生成 token，利用 KV Cache 加速

### 8.2 关键算子总结

| 算子 | 用途 | 实现位置 |
|------|------|----------|
| Embedding | Token 嵌入 | PyTorch |
| RoPE | 位置编码 | `elementwise/rope.cu` |
| FlashAttention | Prefill 注意力 | `attention/*.cu` |
| Lightning Attention | Decode 注意力 | `attention/lightning_attention_decode_kernel.cu` |
| GEMM | 矩阵乘法 | `gemm/*.cu` |
| Fused Add RMSNorm | 融合归一化 | `elementwise/fused_add_rms_norm_kernel.cu` |
| SiLU | 激活函数 | `elementwise/activation.cu` |
| TopK | 采样 | `elementwise/topk.cu` |

### 8.3 优化技术总结

- **算子融合**: 减少内存访问和 kernel 启动开销
- **量化**: 使用低精度减少计算量和内存
- **并行**: Tensor Parallelism 和 Pipeline Parallelism
- **缓存**: KV Cache、RadixAttention 等优化
- **调度**: CUDA Graph、推测解码等

---

## 参考文献

- SGLang 官方文档
- FlashAttention 论文
- Lightning Attention 实现
- CUTLASS 库文档










