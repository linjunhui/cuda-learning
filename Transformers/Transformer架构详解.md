# Transformer 架构详解

## 📖 概述

**Transformer** 是 2017 年由 Vaswani 等人在论文 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) 中提出的深度学习架构，彻底改变了自然语言处理领域。

**核心创新**：
- **完全基于注意力机制**：摒弃了 RNN 和 CNN，完全依赖注意力
- **并行化训练**：所有位置可以并行处理，训练速度快
- **长距离依赖**：注意力机制能够直接建模任意距离的依赖关系

**应用**：
- **编码器架构**：BERT、RoBERTa（用于理解任务）
- **解码器架构**：GPT、LLaMA（用于生成任务）
- **编码器-解码器架构**：T5、BART（用于序列到序列任务）

---

## 🏗️ 整体架构

### 编码器-解码器架构（原始 Transformer）

```
输入序列 → [编码器] → 编码表示 → [解码器] → 输出序列
```

**完整结构**：

```
┌─────────────────────────────────────────────────────────┐
│                      Transformer                         │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │   编码器 (N层)    │         │   解码器 (N层)    │    │
│  │                  │         │                  │    │
│  │  ┌────────────┐  │         │  ┌────────────┐  │    │
│  │  │ Self-Attn  │  │         │  │ Masked     │  │    │
│  │  │            │  │         │  │ Self-Attn  │  │    │
│  │  └────────────┘  │         │  └────────────┘  │    │
│  │        ↓         │         │        ↓         │    │
│  │  ┌────────────┐  │         │  ┌────────────┐  │    │
│  │  │   FFN      │  │         │  │ Enc-Dec    │  │    │
│  │  │            │  │         │  │ Attention  │  │    │
│  │  └────────────┘  │         │  └────────────┘  │    │
│  │                  │         │        ↓         │    │
│  └──────────────────┘         │  ┌────────────┐  │    │
│                                │  │   FFN      │  │    │
│                                │  └────────────┘  │    │
│                                └──────────────────┘    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 仅解码器架构（GPT 系列）

```
输入序列 → [解码器层 × N] → 输出 logits → 采样 → 下一个 token
```

**特点**：
- 只使用解码器部分
- 使用掩码自注意力（Masked Self-Attention）
- 自回归生成：逐个生成 token

### 仅编码器架构（BERT 系列）

```
输入序列 → [编码器层 × N] → 上下文表示 → 任务特定层
```

**特点**：
- 只使用编码器部分
- 双向注意力（可以看到整个序列）
- 用于理解任务（分类、NER 等）

---

## 🧠 核心组件详解

### 1. 注意力机制（Attention Mechanism）

#### 1.1 自注意力（Self-Attention）

**核心思想**：每个位置可以关注序列中的所有位置（包括自己）

**数学公式**：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- `Q`（Query）：查询矩阵，形状 `[seq_len, d_k]`
- `K`（Key）：键矩阵，形状 `[seq_len, d_k]`
- `V`（Value）：值矩阵，形状 `[seq_len, d_v]`
- `d_k`：键的维度（通常等于 `d_v`）

**计算步骤**：

```
步骤 1：计算注意力分数
scores = Q @ K^T  # [seq_len, seq_len]

步骤 2：缩放（防止梯度消失）
scores = scores / √d_k

步骤 3：应用 softmax（归一化为概率分布）
attention_weights = softmax(scores)  # [seq_len, seq_len]

步骤 4：加权求和
output = attention_weights @ V  # [seq_len, d_v]
```

**代码实现**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def self_attention(Q, K, V, mask=None):
    """
    Args:
        Q: [batch_size, seq_len, d_k]
        K: [batch_size, seq_len, d_k]
        V: [batch_size, seq_len, d_v]
        mask: [batch_size, seq_len, seq_len] (可选)
    
    Returns:
        output: [batch_size, seq_len, d_v]
        attention_weights: [batch_size, seq_len, seq_len]
    """
    d_k = Q.size(-1)
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]
    
    # 缩放
    scores = scores / (d_k ** 0.5)
    
    # 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]
    
    # 加权求和
    output = torch.matmul(attention_weights, V)  # [batch, seq_len, d_v]
    
    return output, attention_weights
```

**为什么除以 √d_k？**

- **防止梯度消失**：当 `d_k` 很大时，`QK^T` 的值会很大
- **稳定 softmax**：大值会导致 softmax 梯度接近 0
- **数学证明**：假设 Q 和 K 的元素独立且均值为 0、方差为 1，则 `QK^T` 的方差为 `d_k`

#### 1.2 多头注意力（Multi-Head Attention）

**核心思想**：使用多个注意力头，从不同角度捕捉信息

**数学公式**：

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) W^O

其中：
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**参数**：
- `h`：注意力头的数量（通常 8、12、16、32）
- `d_model`：模型维度（例如 768、1024、4096）
- `d_k = d_v = d_model / h`：每个头的维度

**代码实现**：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # 1. 线性投影并重塑为多头
        Q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch, num_heads, seq_len, d_k]
        
        # 2. 计算注意力（每个头独立计算）
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        # attention_output: [batch, num_heads, seq_len, d_k]
        
        # 3. 拼接所有头
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        # attention_output: [batch, seq_len, d_model]
        
        # 4. 输出投影
        output = self.W_o(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
```

**多头注意力的优势**：
- **多角度理解**：不同头关注不同的语义关系
- **并行计算**：所有头可以并行计算
- **表达能力**：比单头注意力更强大

#### 1.3 掩码注意力（Masked Attention）

**用途**：在解码器中，防止看到未来的 token

**掩码类型**：

1. **因果掩码（Causal Mask）**：下三角矩阵
```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

2. **填充掩码（Padding Mask）**：标记填充位置
```python
# 假设序列长度为 5，实际长度为 3
mask = [[1, 1, 1, 0, 0]]  # 1 表示有效，0 表示填充
```

**代码实现**：

```python
def generate_causal_mask(seq_len, device):
    """生成因果掩码（下三角矩阵）"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    return mask.to(device)

# 使用
causal_mask = generate_causal_mask(seq_len=10, device='cuda')
# 在注意力计算中应用
scores = scores + causal_mask  # 未来位置设为 -inf
```

### 2. 位置编码（Positional Encoding）

**问题**：注意力机制是位置无关的，需要显式编码位置信息

#### 2.1 正弦位置编码（Sinusoidal Positional Encoding）

**公式**：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中：
- `pos`：位置索引（0, 1, 2, ...）
- `i`：维度索引（0, 1, 2, ..., d_model/2-1）
- `d_model`：模型维度

**代码实现**：

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
```

**特点**：
- **固定编码**：不需要学习，计算简单
- **外推能力**：可以处理比训练时更长的序列
- **相对位置**：编码了相对位置关系

#### 2.2 可学习位置编码（Learned Positional Embedding）

**实现**：

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb
```

**特点**：
- **可学习**：通过训练学习最优位置表示
- **灵活性**：可以学习任务特定的位置模式
- **限制**：不能处理超过 `max_len` 的序列

#### 2.3 RoPE（旋转位置编码）

**现代 LLM 常用**：LLaMA、GPT-3、PaLM 等

**原理**：通过旋转矩阵编码位置信息

**优势**：
- **相对位置**：直接编码相对位置关系
- **外推能力**：可以处理更长的序列
- **计算高效**：可以与注意力计算融合

（详见 [RoPE 算子详解](../SGLang学习/算子讲解/04_RoPE算子.md)）

### 3. 前馈网络（Feed Forward Network, FFN）

**结构**：两层全连接网络，中间有激活函数

**公式**：

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

或者使用 GELU：

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

**代码实现**：

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.linear1(x)  # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.linear2(x)  # [batch_size, seq_len, d_model]
        return x
```

**参数设置**：
- `d_ff`：通常是 `d_model` 的 4 倍（例如 d_model=768，d_ff=3072）
- **激活函数**：ReLU（BERT）、GELU（GPT、LLaMA）、SiLU（LLaMA 2+）

### 4. 残差连接和层归一化

#### 4.1 残差连接（Residual Connection）

**公式**：

```
output = LayerNorm(x + Sublayer(x))
```

**作用**：
- **梯度流动**：缓解梯度消失问题
- **身份映射**：允许网络学习残差
- **深层网络**：使深层网络更容易训练

#### 4.2 层归一化（Layer Normalization）

**公式**：

```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

其中：
- `μ`：均值（在最后一个维度上计算）
- `σ²`：方差
- `γ, β`：可学习参数
- `ε`：小常数（防止除零）

**代码实现**：

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

**位置**：
- **Pre-Norm**：`x + Sublayer(LayerNorm(x))`（现代 LLM 常用）
- **Post-Norm**：`LayerNorm(x + Sublayer(x))`（原始 Transformer）

---

## 🔄 完整前向传播流程

### 编码器层（Encoder Layer）

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 1. 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 解码器层（Decoder Layer）

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. 掩码自注意力（只关注已生成的 token）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 编码器-解码器注意力
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

### 完整 Transformer 模型

```python
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        src_emb = self.embedding(src)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        # 解码器
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        # 输出 logits
        output = self.output_proj(decoder_output)
        
        return output
```

---

## 📊 模型变体

### 1. GPT（仅解码器）

**特点**：
- 只使用解码器（带掩码自注意力）
- 自回归生成
- 预训练 + 微调

**架构**：
```
输入 → 词嵌入 → [解码器层 × N] → LM Head → Logits
```

### 2. BERT（仅编码器）

**特点**：
- 只使用编码器
- 双向注意力
- 预训练任务：MLM + NSP

**架构**：
```
输入 → 词嵌入 → [编码器层 × N] → [CLS] 表示 → 任务层
```

### 3. T5（编码器-解码器）

**特点**：
- 完整的编码器-解码器架构
- 文本到文本的转换
- 统一的任务格式

---

## 🎯 关键设计决策

### 1. 为什么使用注意力而不是 RNN？

**RNN 的问题**：
- ❌ **顺序处理**：无法并行化
- ❌ **长距离依赖**：梯度消失/爆炸
- ❌ **计算慢**：O(seq_len) 的串行计算

**注意力的优势**：
- ✅ **并行计算**：所有位置同时计算
- ✅ **长距离依赖**：直接建模任意距离
- ✅ **计算快**：O(seq_len²) 但可并行

### 2. 为什么使用多头注意力？

**单头注意力的限制**：
- 只能学习一种类型的依赖关系
- 表达能力有限

**多头注意力的优势**：
- 不同头关注不同的语义关系
- 更丰富的表示能力
- 计算复杂度相同（并行计算）

### 3. 为什么使用残差连接？

**作用**：
- **梯度流动**：允许梯度直接流过
- **身份映射**：网络可以学习残差
- **深层网络**：使训练深层网络成为可能

### 4. 为什么使用层归一化？

**作用**：
- **稳定训练**：归一化激活值
- **加速收敛**：减少内部协变量偏移
- **允许更大的学习率**

---

## 📈 复杂度分析

### 时间复杂度

**自注意力**：
- **计算复杂度**：O(seq_len² × d_model)
- **并行度**：O(seq_len²)（所有注意力分数可以并行计算）

**前馈网络**：
- **计算复杂度**：O(seq_len × d_model × d_ff)
- **通常**：d_ff = 4 × d_model，所以是 O(seq_len × d_model²)

**总体**：
- **每层**：O(seq_len² × d_model + seq_len × d_model²)
- **N 层**：O(N × (seq_len² × d_model + seq_len × d_model²))

### 空间复杂度

**注意力矩阵**：
- **存储**：O(seq_len²)（每个位置的注意力权重）
- **这是主要瓶颈**：长序列时内存消耗巨大

**优化方法**：
- **Flash Attention**：分块计算，不存储完整注意力矩阵
- **稀疏注意力**：只计算部分位置的注意力
- **线性注意力**：使用线性复杂度的方法

---

## 🔍 实际应用示例

### 使用 Hugging Face Transformers

```python
from transformers import AutoModel, AutoTokenizer

# 加载模型
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 编码输入
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播
outputs = model(**inputs)
hidden_states = outputs.last_hidden_state
# hidden_states: [batch_size=1, seq_len=7, hidden_size=768]
```

### 自定义 Transformer

```python
# 创建模型
transformer = Transformer(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    num_encoder_layers=12,
    num_decoder_layers=12,
    d_ff=3072,
    max_len=512,
    dropout=0.1
)

# 训练
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for batch in dataloader:
    src, tgt = batch
    output = transformer(src, tgt[:, :-1])  # 输入：去掉最后一个 token
    loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 📝 总结

### 核心创新

1. **注意力机制**：完全基于注意力，摒弃 RNN/CNN
2. **并行化**：所有位置并行处理，训练速度快
3. **长距离依赖**：直接建模任意距离的关系

### 关键组件

- ✅ **多头自注意力**：捕捉不同角度的依赖关系
- ✅ **位置编码**：编码位置信息
- ✅ **前馈网络**：非线性变换
- ✅ **残差连接**：缓解梯度消失
- ✅ **层归一化**：稳定训练

### 影响

Transformer 架构：
- 成为现代 NLP 的基础
- 催生了 GPT、BERT、T5 等模型
- 扩展到视觉、音频等多模态领域
- 推动了 LLM 的发展

---

## 🔗 相关资源

- **原始论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Hugging Face 文档**：[Transformers 文档](https://huggingface.co/docs/transformers)
- **可视化**：[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- **相关文档**：
  - [词嵌入详解](./词嵌入详解.md)
  - [RoPE 算子详解](../SGLang学习/算子讲解/04_RoPE算子.md)
  - [Lightning Attention Decode](../SGLang学习/算子讲解/03_Lightning_Attention_Decode.md)

---

## 📚 扩展阅读

### 1. 注意力机制的变体

- **线性注意力**：O(n) 复杂度
- **稀疏注意力**：只计算部分位置
- **局部注意力**：只关注局部窗口
- **Longformer**：滑动窗口注意力

### 2. 位置编码的变体

- **RoPE**：旋转位置编码（现代 LLM 常用）
- **ALiBi**：相对位置偏置
- **T5 Bias**：可学习的相对位置偏置

### 3. 架构优化

- **Pre-Norm vs Post-Norm**：归一化位置的影响
- **深度缩放**：深层网络的初始化策略
- **激活函数选择**：GELU vs ReLU vs SiLU

### 4. 训练技巧

- **学习率调度**：Warmup + 衰减
- **梯度裁剪**：防止梯度爆炸
- **混合精度训练**：FP16/BF16 加速

