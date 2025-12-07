# Decode-Only 架构详解

## 📖 概述

**Decode-Only（仅解码器）架构**是现代大语言模型（LLM）的主流架构，被 GPT、LLaMA、PaLM 等模型广泛采用。

**核心特点**：
- **只使用解码器**：移除了编码器部分
- **掩码自注意力**：使用因果掩码（Causal Mask）防止看到未来 token
- **自回归生成**：逐个生成 token，形成序列

**代表模型**：
- **GPT 系列**：GPT-1, GPT-2, GPT-3, GPT-4
- **LLaMA 系列**：LLaMA, LLaMA 2, LLaMA 3
- **其他**：PaLM, Chinchilla, Mistral, Qwen

---

## 🏗️ 架构对比

### Decode-Only vs 完整 Transformer

| 特性 | 完整 Transformer | Decode-Only |
|------|-----------------|-------------|
| **编码器** | ✅ 有 | ❌ 无 |
| **解码器** | ✅ 有 | ✅ 有（简化版） |
| **交叉注意力** | ✅ 有 | ❌ 无 |
| **掩码注意力** | ✅ 有（解码器） | ✅ 有（所有层） |
| **应用场景** | 序列到序列（翻译） | 文本生成（对话、续写） |

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│              Decode-Only Transformer                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  输入: [token₁, token₂, ..., tokenₙ]                    │
│         ↓                                                 │
│  ┌─────────────────────────────────────┐               │
│  │  词嵌入 + 位置编码                    │               │
│  └─────────────────────────────────────┘               │
│         ↓                                                 │
│  ┌─────────────────────────────────────┐               │
│  │  解码器层 × N                        │               │
│  │  ┌───────────────────────────────┐  │               │
│  │  │ 掩码自注意力 (Causal Mask)     │  │               │
│  │  └───────────────────────────────┘  │               │
│  │         ↓                            │               │
│  │  ┌───────────────────────────────┐  │               │
│  │  │ 前馈网络 (FFN)                │  │               │
│  │  └───────────────────────────────┘  │               │
│  └─────────────────────────────────────┘               │
│         ↓                                                 │
│  ┌─────────────────────────────────────┐               │
│  │  Layer Norm                         │               │
│  └─────────────────────────────────────┘               │
│         ↓                                                 │
│  ┌─────────────────────────────────────┐               │
│  │  LM Head (输出层)                    │               │
│  └─────────────────────────────────────┘               │
│         ↓                                                 │
│  输出: Logits [vocab_size]                               │
│         ↓                                                 │
│  采样 → 下一个 token                                      │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 核心机制

### 1. 掩码自注意力（Masked Self-Attention）

#### 1.1 为什么需要掩码？

**问题**：在生成任务中，模型不应该看到"未来"的 token

**示例**：
```
输入序列: "The cat sat on"
生成过程:
  t=0: "The" → 预测 "cat"
  t=1: "The cat" → 预测 "sat"
  t=2: "The cat sat" → 预测 "on"
```

在 `t=1` 时，模型只能看到 `["The", "cat"]`，不能看到 `"sat"` 和 `"on"`。

#### 1.2 因果掩码（Causal Mask）

**定义**：下三角矩阵，上三角部分设为 `-∞`（或 0）

**形状**：`[seq_len, seq_len]`

**示例**（seq_len=4）：

```
Causal Mask:
┌─────────────────┐
│ 1  0  0  0 │  ← token 0 只能看到自己
│ 1  1  0  0 │  ← token 1 可以看到 0, 1
│ 1  1  1  0 │  ← token 2 可以看到 0, 1, 2
│ 1  1  1  1 │  ← token 3 可以看到 0, 1, 2, 3
└─────────────────┘
```

**代码实现**：

```python
def generate_causal_mask(seq_len, device='cuda'):
    """
    生成因果掩码（下三角矩阵）
    
    Args:
        seq_len: 序列长度
    
    Returns:
        mask: [seq_len, seq_len]，1 表示允许，0 表示禁止
    """
    # 创建下三角矩阵
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask

# 使用示例
mask = generate_causal_mask(seq_len=5)
print(mask)
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

**在注意力中的应用**：

```python
def masked_self_attention(Q, K, V, causal_mask=None):
    """
    Args:
        Q, K, V: [batch_size, seq_len, d_model]
        causal_mask: [seq_len, seq_len]
    """
    d_k = Q.size(-1)
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    # scores: [batch_size, seq_len, seq_len]
    
    # 应用因果掩码
    if causal_mask is not None:
        # 将掩码应用到 scores（0 的位置设为 -inf）
        mask = causal_mask.unsqueeze(0)  # [1, seq_len, seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax（-inf 会被转换为 0）
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

#### 1.3 掩码的效果

**无掩码**（错误）：
```
Token 0 的注意力: [0.3, 0.2, 0.3, 0.2]  ← 可以看到所有位置
Token 1 的注意力: [0.2, 0.3, 0.2, 0.3]  ← 可以看到所有位置（包括未来）
```

**有掩码**（正确）：
```
Token 0 的注意力: [1.0, 0.0, 0.0, 0.0]  ← 只能看到自己
Token 1 的注意力: [0.4, 0.6, 0.0, 0.0]  ← 只能看到 0, 1
Token 2 的注意力: [0.2, 0.3, 0.5, 0.0]  ← 只能看到 0, 1, 2
```

### 2. 自回归生成（Autoregressive Generation）

#### 2.1 生成过程

**自回归**：每次生成一个 token，然后将其作为输入继续生成下一个

**流程**：

```
步骤 0: 输入 "The" → 输出 logits → 采样 → "cat"
步骤 1: 输入 "The cat" → 输出 logits → 采样 → "sat"
步骤 2: 输入 "The cat sat" → 输出 logits → 采样 → "on"
步骤 3: 输入 "The cat sat on" → 输出 logits → 采样 → "the"
...
```

**代码实现**：

```python
def autoregressive_generate(model, tokenizer, prompt, max_length=100):
    """
    自回归生成文本
    
    Args:
        model: Decode-only 模型
        tokenizer: 分词器
        prompt: 输入提示
        max_length: 最大生成长度
    """
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated = input_ids.clone()
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]  # 只取最后一个位置的 logits
            
            # 采样（这里使用贪婪采样）
            next_token_id = torch.argmax(logits, dim=-1)
            
            # 检查是否结束
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # 追加到序列
            generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)
    
    # 解码输出
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text
```

#### 2.2 训练 vs 推理

**训练阶段（Teacher Forcing）**：

```
输入:  [<bos>, "The", "cat", "sat", "on"]
目标:  ["The", "cat", "sat", "on", "<eos>"]

并行计算所有位置的 logits：
  - 位置 0: 预测 "The"（基于 <bos>）
  - 位置 1: 预测 "cat"（基于 <bos>, "The"）
  - 位置 2: 预测 "sat"（基于 <bos>, "The", "cat"）
  - 位置 3: 预测 "on"（基于 <bos>, "The", "cat", "sat"）
```

**推理阶段（自回归）**：

```
步骤 0: 输入 [<bos>] → 输出 logits → 采样 → "The"
步骤 1: 输入 [<bos>, "The"] → 输出 logits → 采样 → "cat"
步骤 2: 输入 [<bos>, "The", "cat"] → 输出 logits → 采样 → "sat"
...
```

**关键区别**：
- **训练**：并行处理所有位置，使用真实标签（Teacher Forcing）
- **推理**：串行生成，每次只处理一个新 token

### 3. KV Cache 机制

#### 3.1 为什么需要 KV Cache？

**问题**：在推理时，每次生成新 token 都需要重新计算所有历史 token 的 K 和 V

**传统方式**（低效）：
```
生成 token 3 时：
  - 重新计算 token 0, 1, 2, 3 的 K, V
  - 计算 Q₃ 与所有 K 的注意力
  - 时间复杂度：O(n²)
```

**KV Cache 方式**（高效）：
```
生成 token 3 时：
  - 从 Cache 读取 token 0, 1, 2 的 K, V（已计算）
  - 只计算 token 3 的 K, V
  - 计算 Q₃ 与所有 K 的注意力
  - 时间复杂度：O(n)
```

#### 3.2 KV Cache 的实现

**数据结构**：

```python
class KVCache:
    """KV Cache 管理"""
    
    def __init__(self, num_layers, num_heads, head_dim, max_len):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_len = max_len
        
        # 为每一层创建 Cache
        self.cache = {}
        for layer_id in range(num_layers):
            self.cache[layer_id] = {
                'k': torch.zeros(1, num_heads, max_len, head_dim),
                'v': torch.zeros(1, num_heads, max_len, head_dim),
                'length': 0
            }
    
    def append(self, layer_id, k, v):
        """追加新的 K, V"""
        pos = self.cache[layer_id]['length']
        self.cache[layer_id]['k'][:, :, pos, :] = k
        self.cache[layer_id]['v'][:, :, pos, :] = v
        self.cache[layer_id]['length'] += 1
    
    def get(self, layer_id):
        """获取当前层的 K, V"""
        length = self.cache[layer_id]['length']
        k = self.cache[layer_id]['k'][:, :, :length, :]
        v = self.cache[layer_id]['v'][:, :, :length, :]
        return k, v
```

**在解码中的使用**：

```python
def decode_step(model, new_token_id, kv_cache, position):
    """
    单步解码
    
    Args:
        model: Decode-only 模型
        new_token_id: 新生成的 token ID
        kv_cache: KV Cache
        position: 当前位置
    """
    # 1. 词嵌入（只处理一个新 token）
    hidden_states = model.embed_tokens(new_token_id)  # [batch, 1, hidden_size]
    
    # 2. 逐层处理
    for layer_id, layer in enumerate(model.layers):
        # 2.1 计算当前 token 的 Q, K, V
        qkv = layer.attention.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
        
        # 2.2 位置编码（RoPE）
        q, k = layer.rotary_emb(position, q, k)
        
        # 2.3 从 Cache 获取历史的 K, V
        past_k, past_v = kv_cache.get(layer_id)
        
        # 2.4 拼接新的 K, V
        k_full = torch.cat([past_k, k], dim=2)  # [batch, heads, past_len+1, dim]
        v_full = torch.cat([past_v, v], dim=2)
        
        # 2.5 计算注意力（只计算新 token 的注意力）
        attn_output = layer.attention.forward_decode(q, k_full, v_full)
        
        # 2.6 更新 Cache
        kv_cache.append(layer_id, k, v)
        
        # 2.7 残差连接和归一化
        hidden_states = hidden_states + attn_output
        hidden_states = layer.norm1(hidden_states)
        
        # 2.8 前馈网络
        ff_output = layer.mlp(hidden_states)
        hidden_states = hidden_states + ff_output
        hidden_states = layer.norm2(hidden_states)
    
    # 3. 输出 logits
    logits = model.lm_head(hidden_states)  # [batch, 1, vocab_size]
    
    return logits
```

#### 3.3 KV Cache 的内存占用

**计算**：

假设：
- 层数：`L = 32`
- 头数：`H = 32`
- 头维度：`d = 128`
- 最大长度：`max_len = 2048`
- 数据类型：`float16`（2 bytes）

**每层的 Cache 大小**：
```
K Cache: 1 × 32 × 2048 × 128 × 2 bytes = 16 MB
V Cache: 1 × 32 × 2048 × 128 × 2 bytes = 16 MB
每层总计: 32 MB
```

**所有层的 Cache 大小**：
```
总大小: 32 × 32 MB = 1024 MB = 1 GB
```

**优化策略**：
- **量化**：使用 FP8 或 INT8 减少内存
- **分页**：动态分配，只分配实际使用的长度
- **压缩**：使用更高效的存储格式

---

## 💻 完整实现

### Decode-Only Transformer 模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderOnlyTransformer(nn.Module):
    """Decode-Only Transformer 模型（类似 GPT）"""
    
    def __init__(
        self,
        vocab_size=50000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码（可学习）
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # 权重共享：LM Head 和 Embedding
        self.lm_head.weight = self.embedding.weight
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, use_cache=False, past_key_values=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            use_cache: 是否使用 KV Cache
            past_key_values: 过去的 KV Cache
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            past_key_values: 更新后的 KV Cache（如果 use_cache=True）
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. 词嵌入
        token_embeds = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # 2. 位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)  # [batch, seq_len, d_model]
        
        # 3. 相加
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.dropout(hidden_states)
        
        # 4. 生成因果掩码
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # 5. 逐层处理
        new_past_key_values = [] if use_cache else None
        
        for layer_id, layer in enumerate(self.layers):
            if use_cache and past_key_values is not None:
                # 使用 KV Cache（推理阶段）
                past_k, past_v = past_key_values[layer_id]
                hidden_states, new_k, new_v = layer(
                    hidden_states,
                    causal_mask,
                    past_k=past_k,
                    past_v=past_v,
                )
                new_past_key_values.append((new_k, new_v))
            else:
                # 不使用 Cache（训练阶段）
                hidden_states = layer(hidden_states, causal_mask)
        
        # 6. 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 7. 输出 logits
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        if use_cache:
            return logits, new_past_key_values
        else:
            return logits
    
    def _generate_causal_mask(self, seq_len, device):
        """生成因果掩码"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


class DecoderLayer(nn.Module):
    """解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask, past_k=None, past_v=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [1, 1, seq_len, seq_len]
            past_k, past_v: 过去的 K, V（推理时使用）
        """
        # 1. 掩码自注意力
        residual = x
        x = self.norm1(x)
        
        if past_k is not None and past_v is not None:
            # 推理阶段：使用 KV Cache
            attn_output, new_k, new_v = self.self_attn(
                x, x, x, mask, past_k=past_k, past_v=past_v
            )
        else:
            # 训练阶段：不使用 Cache
            attn_output = self.self_attn(x, x, x, mask)[0]
            new_k, new_v = None, None
        
        x = residual + self.dropout(attn_output)
        
        # 2. 前馈网络
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)
        
        if new_k is not None:
            return x, new_k, new_v
        else:
            return x


class MultiHeadAttention(nn.Module):
    """多头注意力（支持 KV Cache）"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None, past_k=None, past_v=None):
        """
        Args:
            Q, K, V: [batch_size, seq_len, d_model]
            mask: [1, 1, seq_len, seq_len]
            past_k, past_v: [batch, heads, past_len, d_k]（可选）
        """
        batch_size, seq_len = Q.size(0), Q.size(1)
        
        # 1. 线性投影
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 2. 重塑为多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch, heads, seq_len, d_k]
        
        # 3. 如果有 past_k, past_v，拼接
        if past_k is not None and past_v is not None:
            K = torch.cat([past_k, K], dim=2)  # [batch, heads, past_len+seq_len, d_k]
            V = torch.cat([past_v, V], dim=2)
            new_k, new_v = K, V
        else:
            new_k, new_v = K, V
        
        # 4. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            # 调整掩码大小（如果使用了 past_k）
            if past_k is not None:
                past_len = past_k.size(2)
                # 扩展掩码：前 past_len 列全为 1（可以看到所有历史）
                extended_mask = torch.ones(
                    batch_size, self.num_heads, seq_len, past_len,
                    device=mask.device
                )
                mask = torch.cat([extended_mask, mask], dim=-1)
            
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # 5. 拼接所有头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        # 6. 输出投影
        output = self.W_o(output)
        
        return output, new_k, new_v


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model, d_ff, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
```

### 训练代码

```python
def train_step(model, batch, optimizer, criterion):
    """训练一步"""
    input_ids, labels = batch
    
    # 前向传播（不使用 Cache）
    logits = model(input_ids, use_cache=False)
    
    # 计算损失（只计算非填充位置）
    # labels 是 input_ids 向右移动一位
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss = criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 推理代码（带 KV Cache）

```python
def generate(model, tokenizer, prompt, max_new_tokens=100):
    """使用 KV Cache 生成文本"""
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated = input_ids.clone()
    
    # 初始化 KV Cache
    past_key_values = None
    
    model.eval()
    with torch.no_grad():
        # 1. 预填充阶段：处理输入 prompt
        logits, past_key_values = model(
            input_ids,
            use_cache=True,
            past_key_values=None
        )
        
        # 采样第一个 token
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)
        
        # 2. 解码阶段：逐个生成新 token
        for _ in range(max_new_tokens - 1):
            # 只处理最后一个 token
            new_token = next_token_id.unsqueeze(0).unsqueeze(0)  # [1, 1]
            
            # 前向传播（使用 KV Cache）
            logits, past_key_values = model(
                new_token,
                use_cache=True,
                past_key_values=past_key_values
            )
            
            # 采样
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            
            # 检查结束
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # 追加到序列
            generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)
    
    # 解码输出
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text
```

---

## 🎯 关键设计要点

### 1. Pre-Norm vs Post-Norm

**Post-Norm**（原始 Transformer）：
```python
x = x + Sublayer(LayerNorm(x))
```

**Pre-Norm**（现代 LLM 常用，如 LLaMA）：
```python
x = x + Sublayer(LayerNorm(x))
```

**优势**：
- **训练更稳定**：梯度流动更好
- **允许更深网络**：可以训练更深的模型
- **现代 LLM 标准**：GPT-3, LLaMA, PaLM 都使用 Pre-Norm

### 2. 权重共享（Weight Tying）

**概念**：LM Head 的权重与 Embedding 的权重共享

**优势**：
- **减少参数**：节省一半的权重矩阵
- **正则化效果**：有助于训练
- **常见实践**：GPT、LLaMA 都使用

**实现**：
```python
self.lm_head.weight = self.embedding.weight
```

### 3. 位置编码的选择

**可学习位置编码**（GPT-1, GPT-2）：
- 简单，但受限于最大长度

**RoPE**（LLaMA, GPT-3）：
- 相对位置编码
- 更好的外推能力
- 现代 LLM 的标准选择

### 4. 激活函数

**ReLU**（早期）：
- 简单但可能死神经元

**GELU**（GPT, LLaMA）：
- 更平滑，性能更好

**SiLU/Swish**（LLaMA 2+）：
- `SiLU(x) = x * sigmoid(x)`
- 性能略好于 GELU

---

## 📊 训练和推理的区别

### 训练阶段

**特点**：
- **并行处理**：所有位置同时计算
- **Teacher Forcing**：使用真实标签
- **无 KV Cache**：每次都重新计算
- **批量处理**：处理多个序列

**流程**：
```
输入: [<bos>, "The", "cat", "sat", "on"]
目标: ["The", "cat", "sat", "on", "<eos>"]

并行计算：
  - 位置 0: 基于 [<bos>] 预测 "The"
  - 位置 1: 基于 [<bos>, "The"] 预测 "cat"
  - 位置 2: 基于 [<bos>, "The", "cat"] 预测 "sat"
  - 位置 3: 基于 [<bos>, "The", "cat", "sat"] 预测 "on"
```

### 推理阶段

**特点**：
- **串行生成**：逐个生成 token
- **自回归**：使用生成的 token 作为输入
- **KV Cache**：缓存历史的 K, V
- **单序列**：通常一次处理一个序列

**流程**：
```
步骤 0: 输入 [<bos>] → 输出 logits → 采样 → "The"
步骤 1: 输入 [<bos>, "The"] → 输出 logits → 采样 → "cat"
步骤 2: 输入 [<bos>, "The", "cat"] → 输出 logits → 采样 → "sat"
...
```

---

## 🔍 实际模型示例

### GPT-2 架构

```python
# GPT-2 Small
config = {
    'vocab_size': 50257,
    'd_model': 768,
    'num_heads': 12,
    'num_layers': 12,
    'd_ff': 3072,
    'max_len': 1024,
    'activation': 'gelu',
}

# GPT-2 Large
config = {
    'vocab_size': 50257,
    'd_model': 1280,
    'num_heads': 20,
    'num_layers': 36,
    'd_ff': 5120,
    'max_len': 1024,
    'activation': 'gelu',
}
```

### LLaMA 架构

```python
# LLaMA-7B
config = {
    'vocab_size': 32000,
    'd_model': 4096,
    'num_heads': 32,
    'num_kv_heads': 32,  # 注意：GQA（Grouped Query Attention）
    'num_layers': 32,
    'd_ff': 11008,
    'max_len': 2048,
    'activation': 'silu',
    'rope_theta': 10000,
}

# LLaMA-2-70B
config = {
    'vocab_size': 32000,
    'd_model': 8192,
    'num_heads': 64,
    'num_kv_heads': 8,  # GQA：8 个 KV 头共享
    'num_layers': 80,
    'd_ff': 28672,
    'max_len': 4096,
    'activation': 'silu',
    'rope_theta': 10000,
}
```

---

## 📝 总结

### 核心特点

1. **掩码自注意力**：使用因果掩码防止看到未来 token
2. **自回归生成**：逐个生成 token，形成序列
3. **KV Cache**：推理时缓存 K, V，加速生成
4. **权重共享**：Embedding 和 LM Head 共享权重

### 优势

- ✅ **简单**：架构比编码器-解码器更简单
- ✅ **高效**：训练和推理都很快
- ✅ **通用**：适合各种生成任务
- ✅ **可扩展**：可以轻松扩展到更大规模

### 应用场景

- **文本生成**：续写、创作
- **对话系统**：聊天机器人
- **代码生成**：GitHub Copilot
- **翻译**：虽然不如编码器-解码器，但也可以使用

---

## 🔗 相关资源

- **原始论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **GPT 论文**：[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **LLaMA 论文**：[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **相关文档**：
  - [Transformer 架构详解](./Transformer架构详解.md)
  - [词嵌入详解](./词嵌入详解.md)
  - [Lightning Attention Decode](../SGLang学习/算子讲解/03_Lightning_Attention_Decode.md)

---

## 📚 扩展阅读

### 1. 生成策略

- **贪婪解码**：总是选择概率最高的 token
- **采样**：根据概率分布随机采样
- **Top-k 采样**：只从 top-k 候选中采样
- **Top-p 采样**：从累积概率达到 p 的候选中采样
- **Beam Search**：维护多个候选序列（但现代 LLM 很少使用）

### 2. 优化技术

- **Flash Attention**：分块计算注意力，减少内存
- **Paged Attention**：分页管理 KV Cache
- **量化**：FP8/INT8 减少内存和计算
- **推测解码**：使用小模型生成，大模型验证

### 3. 架构变体

- **GQA（Grouped Query Attention）**：多个 Q 头共享一个 K, V 头
- **MQA（Multi-Query Attention）**：所有 Q 头共享一个 K, V 头
- **MoE（Mixture of Experts）**：使用多个专家网络

