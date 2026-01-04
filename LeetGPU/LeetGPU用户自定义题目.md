# LeetGPU 用户自定义题目

来源：基于 SGLang 模型推理流程中的核心算子

## 统计信息

- 总题目数：15题
- 难度分布：包含 Easy、Medium、Hard 三个难度级别
- 题目类型：SGLang 推理流程中的核心 CUDA 算子实现

---

## 题目列表

### Easy 难度题目

#### 1. RoPE (旋转位置编码)
- **ID**: rope-rotary-position-encoding
- **URL**: /challenges/rope-rotary-position-encoding
- **难度**: Easy
- **描述**: 实现旋转位置编码（Rotary Position Embedding）算子，为 Query 和 Key 添加位置信息

**输入输出规格**:
- **输入**:
  - `q`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: float16/float32
    - `batch_size`: 批次大小（同时处理的请求数）
    - `seq_len`: 序列长度（Prefill 阶段为 prompt 长度，Decode 阶段为 1）
    - `num_heads`: 注意力头数（如 32、64）
    - `head_dim`: 每个头的维度（如 64、128），必须是偶数（因为使用复数对）
  - `k`: 形状同 `q`
  - `positions`: `[batch_size, seq_len]`, dtype: int32
    - 每个 token 在序列中的位置索引（从 0 开始）
  - `cos_cache`: `[max_seq_len, head_dim // 2]`, dtype: float16/float32
    - 预计算的余弦值缓存
  - `sin_cache`: `[max_seq_len, head_dim // 2]`, dtype: float16/float32
    - 预计算的正弦值缓存
- **输出**:
  - `q_rotated`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - 应用旋转位置编码后的 Query 向量
  - `k_rotated`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - 应用旋转位置编码后的 Key 向量

**形状含义说明**:
- `head_dim` 必须是偶数，因为 RoPE 将维度分成 `head_dim // 2` 对，每对应用旋转
- 例如：`head_dim = 128`，则分成 64 对，每对 `(x_i, x_{i+64})` 应用旋转矩阵
- `positions` 中的值表示每个 token 在序列中的绝对位置，用于查找对应的旋转角度

**在推理流程中的位置**:
- **Prefill 阶段**: 在 QKV 投影之后、FlashAttention 之前
  - 输入 `seq_len` = prompt 长度（如 512、1024）
  - 所有 token 同时处理
- **Decode 阶段**: 在 QKV 投影之后、Lightning Attention 之前
  - 输入 `seq_len` = 1（只处理新生成的 token）
  - `positions` 包含当前 token 的位置（如 prompt_len + decode_step）

**典型数值示例**:
```python
# Prefill 阶段示例
batch_size = 4
seq_len = 512
num_heads = 32
head_dim = 128
# q, k: [4, 512, 32, 128]
# positions: [4, 512]  # 每行都是 [0, 1, 2, ..., 511]

# Decode 阶段示例
batch_size = 4
seq_len = 1
num_heads = 32
head_dim = 128
# q, k: [4, 1, 32, 128]
# positions: [4, 1]  # 每行都是 [512]（假设 prompt 长度为 512）
```

**题目背景**: 
  - RoPE 是现代大语言模型（如 LLaMA、GPT-NeoX）中广泛使用的位置编码方法
  - 在 SGLang 推理流程中，RoPE 应用于每个 Transformer 层的 Attention 计算之前
  - 相比传统的位置编码，RoPE 通过复数旋转的方式将位置信息编码到 Q、K 向量中，能够更好地处理长序列
  - 需要实现高效的 CUDA kernel，支持批量处理和向量化优化
- **核心要求**:
  - 实现旋转矩阵的计算和应用
  - 支持不同位置和维度的旋转角度计算
  - 优化内存访问模式，支持向量化加载
- **相关知识点**: 复数旋转、位置编码、CUDA 向量化

#### 2. SiLU 激活函数
- **ID**: silu-activation
- **URL**: /challenges/silu-activation
- **难度**: Easy
- **描述**: 实现 SiLU (Sigmoid Linear Unit) 激活函数，用于 MLP 前馈网络

**输入输出规格**:
- **输入**:
  - `x`: `[batch_size, seq_len, intermediate_size]`, dtype: float16/float32
    - `batch_size`: 批次大小
    - `seq_len`: 序列长度（Prefill 阶段为 prompt 长度，Decode 阶段为 1）
    - `intermediate_size`: MLP 中间层维度（通常是 `hidden_size` 的 2-4 倍，如 11008、14336）
- **输出**:
  - `output`: `[batch_size, seq_len, intermediate_size]`, dtype: 同输入
    - 应用 SiLU 激活后的输出，形状与输入相同

**形状含义说明**:
- `intermediate_size` 是 MLP 的扩展维度，通常比 `hidden_size` 大
- 例如：LLaMA-7B 中 `hidden_size = 4096`, `intermediate_size = 11008`
- 这是逐元素操作，每个元素独立计算：`output[i,j,k] = SiLU(x[i,j,k])`

**在推理流程中的位置**:
- **位置**: Transformer 层的 MLP 子层中，Gate 投影之后
- **流程**: `hidden_states` → Gate 投影 → **SiLU** → 与 Up 投影结果相乘
- **典型使用**: 
  ```python
  gate = gate_proj(hidden_states)  # [B, L, intermediate_size]
  gate_activated = SiLU(gate)        # [B, L, intermediate_size] ← 这里
  up = up_proj(hidden_states)       # [B, L, intermediate_size]
  output = gate_activated * up      # [B, L, intermediate_size]
  ```

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
seq_len = 512
intermediate_size = 11008
# x: [4, 512, 11008]

# Decode 阶段
batch_size = 4
seq_len = 1
intermediate_size = 11008
# x: [4, 1, 11008]
```

**题目背景**:
  - SiLU 是 Transformer 模型中 MLP 层常用的激活函数
  - 在 SGLang 中，SiLU 用于 MLP 的 Gate 投影后的激活
  - 公式：`SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`
  - 需要实现高效的逐元素计算 kernel，支持批量处理
- **核心要求**:
  - 实现 SiLU 的数学公式
  - 处理数值稳定性（避免溢出，特别是 `exp(-x)` 在 x 很大时）
  - 优化内存访问，支持向量化
- **相关知识点**: 激活函数、数值稳定性、逐元素操作

#### 3. RMSNorm (Root Mean Square Normalization)
- **ID**: rms-normalization
- **URL**: /challenges/rms-normalization
- **难度**: Easy
- **描述**: 实现 RMS 归一化算子，用于 Transformer 层的归一化操作

**输入输出规格**:
- **输入**:
  - `x`: `[batch_size, seq_len, hidden_size]`, dtype: float16/float32
    - `batch_size`: 批次大小
    - `seq_len`: 序列长度
    - `hidden_size`: 隐藏层维度（如 4096、8192）
  - `weight`: `[hidden_size]`, dtype: 同输入
    - 可学习的缩放参数（gamma）
  - `eps`: float, 默认 1e-6
    - 数值稳定性参数，防止除零
- **输出**:
  - `output`: `[batch_size, seq_len, hidden_size]`, dtype: 同输入
    - 归一化后的输出，形状与输入相同

**形状含义说明**:
- 归一化在最后一个维度（`hidden_size`）上进行
- 对于每个 `(batch_idx, seq_idx)`，计算该 `hidden_size` 维向量的 RMS
- 公式：`output[i,j,:] = x[i,j,:] * weight / sqrt(mean(x[i,j,:]^2) + eps)`

**在推理流程中的位置**:
- **位置 1**: Attention 子层之后
  - 输入：`hidden_states + attn_output` → RMSNorm → 输出到 MLP
- **位置 2**: MLP 子层之后
  - 输入：`hidden_states + mlp_output` → RMSNorm → 输出到下一层
- **注意**: 实际使用中通常与残差连接融合（Fused Add RMSNorm）

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
seq_len = 512
hidden_size = 4096
# x: [4, 512, 4096]
# weight: [4096]

# Decode 阶段
batch_size = 4
seq_len = 1
hidden_size = 4096
# x: [4, 1, 4096]
# weight: [4096]
```

**题目背景**:
  - RMSNorm 是 LayerNorm 的简化版本，在现代 LLM 中广泛使用（如 LLaMA）
  - 在 SGLang 中，RMSNorm 用于 Attention 和 MLP 后的归一化
  - 相比 LayerNorm，RMSNorm 不需要计算均值，计算更高效
  - 公式：`RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)`
- **核心要求**:
  - 实现并行归约计算均方根（对 `hidden_size` 维度归约）
  - 处理数值稳定性（eps 参数）
  - 支持批量归一化
- **相关知识点**: 归一化、并行归约、数值稳定性

---

### Medium 难度题目

#### 4. Fused Add RMSNorm
- **ID**: fused-add-rmsnorm
- **URL**: /challenges/fused-add-rmsnorm
- **难度**: Medium
- **描述**: 实现融合的残差连接和 RMSNorm 算子，将两个操作合并到一个 kernel 中

**输入输出规格**:
- **输入**:
  - `x`: `[batch_size, seq_len, hidden_size]`, dtype: float16/float32
    - 当前层的输出（Attention 或 MLP 的输出）
  - `residual`: `[batch_size, seq_len, hidden_size]`, dtype: 同输入
    - 残差连接的输入（来自上一层的 hidden_states）
  - `weight`: `[hidden_size]`, dtype: 同输入
    - RMSNorm 的可学习缩放参数
  - `eps`: float, 默认 1e-6
- **输出**:
  - `output`: `[batch_size, seq_len, hidden_size]`, dtype: 同输入
    - 融合操作的结果：`RMSNorm(x + residual)`
  - `residual_out` (可选): `[batch_size, seq_len, hidden_size]`, dtype: 同输入
    - 更新后的残差，用于下一层（通常就是 `output`）

**形状含义说明**:
- 融合操作：`output = RMSNorm(x + residual)`
- 在同一个 kernel 中完成：
  1. 逐元素加法：`sum = x + residual`
  2. 计算 RMS：`rms = sqrt(mean(sum^2) + eps)`
  3. 归一化：`output = sum * weight / rms`
- 避免将 `x + residual` 写回全局内存，直接在寄存器/共享内存中完成归一化

**在推理流程中的位置**:
- **位置 1**: Attention 子层之后
  ```python
  attn_output = attention(q, k, v)  # [B, L, H]
  hidden_states, residual = FusedAddRMSNorm(
      attn_output,      # x
      hidden_states,    # residual (输入到 Attention 的 hidden_states)
      weight, eps
  )
  ```
- **位置 2**: MLP 子层之后
  ```python
  mlp_output = mlp(hidden_states)  # [B, L, H]
  hidden_states, residual = FusedAddRMSNorm(
      mlp_output,      # x
      hidden_states,    # residual (输入到 MLP 的 hidden_states)
      weight, eps
  )
  ```

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
seq_len = 512
hidden_size = 4096
# x: [4, 512, 4096] (Attention 或 MLP 输出)
# residual: [4, 512, 4096] (上一层的 hidden_states)
# output: [4, 512, 4096]

# Decode 阶段
batch_size = 4
seq_len = 1
hidden_size = 4096
# x: [4, 1, 4096]
# residual: [4, 1, 4096]
# output: [4, 1, 4096]
```

**题目背景**:
  - 在 Transformer 层中，残差连接和归一化经常连续执行：`x = RMSNorm(x + residual)`
  - 融合这两个操作可以减少一次内存访问，提高性能
  - 在 SGLang 中，这是关键的优化技术，可以减少 30-50% 的内存带宽使用
  - 需要在一个 kernel 中同时完成加法和归一化计算
- **核心要求**:
  - 融合残差连接和 RMSNorm 计算
  - 减少中间结果的全局内存写入（`x + residual` 不写回）
  - 优化共享内存使用（用于归约计算）
- **相关知识点**: 算子融合、内存优化、CUDA 共享内存

#### 5. SiLU and Mul (融合激活)
- **ID**: fused-silu-mul
- **URL**: /challenges/fused-silu-mul
- **难度**: Medium
- **描述**: 实现融合的 SiLU 激活和乘法操作，用于 MLP 的 Gate 和 Up 投影融合

**输入输出规格**:
- **输入**:
  - `gate`: `[batch_size, seq_len, intermediate_size]`, dtype: float16/float32
    - Gate 投影的输出
  - `up`: `[batch_size, seq_len, intermediate_size]`, dtype: 同输入
    - Up 投影的输出
- **输出**:
  - `output`: `[batch_size, seq_len, intermediate_size]`, dtype: 同输入
    - 融合操作的结果：`SiLU(gate) * up`
    - 逐元素计算：`output[i,j,k] = SiLU(gate[i,j,k]) * up[i,j,k]`

**形状含义说明**:
- `gate` 和 `up` 形状完全相同，都是 MLP 中间层的输出
- 融合操作避免先计算 `SiLU(gate)` 写回内存，再读取进行乘法
- 在同一个 kernel 中：计算 `SiLU(gate[i,j,k])` 后立即与 `up[i,j,k]` 相乘

**在推理流程中的位置**:
- **位置**: MLP 子层中，Gate 和 Up 投影之后
  ```python
  # MLP 前馈网络流程
  gate = gate_proj(hidden_states)  # [B, L, intermediate_size]
  up = up_proj(hidden_states)      # [B, L, intermediate_size]
  
  # 融合操作（避免两次 kernel 启动）
  mlp_intermediate = FusedSiLUMul(gate, up)  # [B, L, intermediate_size] ← 这里
  
  mlp_output = down_proj(mlp_intermediate)   # [B, L, hidden_size]
  ```

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
seq_len = 512
intermediate_size = 11008
# gate: [4, 512, 11008]
# up: [4, 512, 11008]
# output: [4, 512, 11008]

# Decode 阶段
batch_size = 4
seq_len = 1
intermediate_size = 11008
# gate: [4, 1, 11008]
# up: [4, 1, 11008]
# output: [4, 1, 11008]
```

**题目背景**:
  - 在 MLP 中，常见模式是：`output = SiLU(gate) * up`
  - 将 SiLU 和乘法融合可以减少一次 kernel 启动和内存访问
  - 在 SGLang 的 MLP 实现中，这是标准的优化模式
  - 需要在一个 kernel 中完成 SiLU 计算和逐元素乘法
- **核心要求**:
  - 融合 SiLU 和逐元素乘法
  - 优化寄存器使用（避免中间结果写回内存）
  - 减少内存访问次数
- **相关知识点**: 算子融合、逐元素操作、寄存器优化

#### 6. TopK 采样
- **ID**: topk-sampling
- **URL**: /challenges/topk-sampling
- **难度**: Medium
- **描述**: 实现 TopK 采样算子，从 logits 中选择概率最高的 K 个 token

**输入输出规格**:
- **输入**:
  - `logits`: `[batch_size, vocab_size]`, dtype: float16/float32
    - `batch_size`: 批次大小（同时生成的请求数）
    - `vocab_size`: 词汇表大小（通常 32000、50000、128000 等）
    - 每个元素表示对应 token 的未归一化分数（logit）
  - `k`: int, TopK 参数（如 50、100）
    - 选择概率最高的 k 个 token
  - `temperature`: float, 默认 1.0
    - 温度缩放参数：`logits = logits / temperature`
    - temperature < 1.0 使分布更尖锐，> 1.0 使分布更平滑
- **输出**:
  - `topk_values`: `[batch_size, k]`, dtype: 同输入
    - TopK 个最大 logit 值
  - `topk_indices`: `[batch_size, k]`, dtype: int32
    - TopK 个最大 logit 对应的 token ID
  - `probs` (可选): `[batch_size, k]`, dtype: float32
    - 归一化后的概率：`softmax(topk_values)`

**形状含义说明**:
- `logits` 是 LM Head 的输出，表示每个 token 的得分
- 对于每个 batch，从 `vocab_size` 个 logits 中选择最大的 k 个
- `topk_indices` 中的值就是下一个 token 的候选 ID（0 到 vocab_size-1）

**在推理流程中的位置**:
- **位置**: 推理流程的最后一步，LM Head 之后
  ```python
  # 完整流程
  hidden_states = transformer_layers(input_ids)  # [B, L, H]
  logits = lm_head(hidden_states[:, -1, :])      # [B, vocab_size] ← 只取最后一个 token
  topk_values, topk_indices = TopK(logits, k=50)  # ← 这里
  next_token_id = sample_from_topk(topk_indices)  # 从 TopK 中采样一个
  ```

**典型数值示例**:
```python
# 单个请求
batch_size = 1
vocab_size = 32000
k = 50
# logits: [1, 32000]  # 32000 个 token 的得分
# topk_values: [1, 50]  # 最高的 50 个得分
# topk_indices: [1, 50]  # 对应的 token ID（如 [1234, 5678, ...]）

# 批量请求
batch_size = 4
vocab_size = 50000
k = 100
# logits: [4, 50000]
# topk_values: [4, 100]
# topk_indices: [4, 100]
```

**题目背景**:
  - TopK 采样是 LLM 生成过程中的关键步骤
  - 在 SGLang 中，TopK 用于从词汇表大小的 logits 中选择下一个 token
  - 需要高效处理大规模 logits（通常 vocab_size > 50K）
  - 支持温度缩放和 TopP 采样（可选）
- **核心要求**:
  - 实现高效的 TopK 选择算法（如基数排序、堆排序）
  - 支持温度缩放：`logits = logits / temperature`
  - 处理大规模数组（vocab_size 可能很大，需要优化）
- **相关知识点**: 排序算法、采样、数值稳定性

#### 7. Softmax (数值稳定版本)
- **ID**: stable-softmax
- **URL**: /challenges/stable-softmax
- **难度**: Medium
- **描述**: 实现数值稳定的 Softmax 函数，用于注意力机制

**输入输出规格**:
- **输入**:
  - `x`: `[batch_size, num_heads, seq_len, seq_len]` (Prefill) 或 `[batch_size, num_heads, 1, seq_len]` (Decode), dtype: float16/float32
    - 注意力分数矩阵（未归一化）
    - Prefill: `seq_len` = prompt 长度，计算所有 token 之间的注意力
    - Decode: 第一个 `seq_len` = 1（新 token），第二个 `seq_len` = 历史长度
  - `scale`: float, 默认 `1.0 / sqrt(head_dim)`
    - 缩放因子，用于 `x = x * scale`
- **输出**:
  - `output`: 形状同输入, dtype: 同输入
    - 归一化后的注意力权重，每行和为 1
    - 对于 Prefill: `output[b, h, i, :]` 的和为 1（第 i 个 token 对所有 token 的注意力权重）
    - 对于 Decode: `output[b, h, 0, :]` 的和为 1（新 token 对所有历史 token 的注意力权重）

**形状含义说明**:
- Prefill 阶段：`[B, H, L, L]`，计算完整的注意力矩阵
- Decode 阶段：`[B, H, 1, L_past]`，只计算新 token 与历史的注意力
- 数值稳定技巧：`softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`
  - 先找到每行的最大值，然后所有元素减去该最大值，再计算 exp

**在推理流程中的位置**:
- **位置**: Attention 计算中，QK^T 之后、与 V 相乘之前
  ```python
  # Attention 计算流程
  scores = q @ k.transpose(-2, -1)  # [B, H, L, L] 或 [B, H, 1, L_past]
  scores = scores / sqrt(head_dim)  # 缩放
  attn_weights = StableSoftmax(scores)  # ← 这里，归一化
  attn_output = attn_weights @ v        # 加权求和
  ```

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
num_heads = 32
seq_len = 512
# x: [4, 32, 512, 512]  # 每个 token 对所有 token 的注意力分数
# output: [4, 32, 512, 512]  # 归一化后，每行和为 1

# Decode 阶段
batch_size = 4
num_heads = 32
past_len = 512
# x: [4, 32, 1, 512]  # 新 token 对所有历史 token 的注意力分数
# output: [4, 32, 1, 512]  # 归一化后，和为 1
```

**题目背景**:
  - Softmax 是注意力机制的核心组件
  - 标准 Softmax 容易数值溢出，需要使用最大值归一化
  - 在 SGLang 的注意力计算中，Softmax 需要处理大规模分数矩阵
  - 公式：`softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`
- **核心要求**:
  - 实现数值稳定的 Softmax（使用最大值归一化）
  - 支持批量处理（对每个 head、每个 token 独立归一化）
  - 优化内存访问模式（分块计算）
- **相关知识点**: Softmax、数值稳定性、并行归约

#### 8. QKV 投影 (融合版本)
- **ID**: fused-qkv-projection
- **URL**: /challenges/fused-qkv-projection
- **难度**: Medium
- **描述**: 实现融合的 QKV 投影，将三个独立的矩阵乘法融合为一个操作

**输入输出规格**:
- **输入**:
  - `hidden_states`: `[batch_size, seq_len, hidden_size]`, dtype: float16/float32
    - Transformer 层的输入隐藏状态
  - `qkv_weight`: `[3 * hidden_size, hidden_size]`, dtype: 同输入
    - 融合的 QKV 权重矩阵，按 `[W_q; W_k; W_v]` 拼接
    - 形状：`[3 * hidden_size, hidden_size]`，其中 `hidden_size = num_heads * head_dim`
- **输出**:
  - `q`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - Query 投影结果
  - `k`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - Key 投影结果
  - `v`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - Value 投影结果

**形状含义说明**:
- 融合操作：`[Q; K; V] = hidden_states @ qkv_weight^T`
- 输出形状转换：
  - 中间结果：`[B, L, 3 * hidden_size]`
  - 分割为 Q、K、V：每个 `[B, L, hidden_size]`
  - Reshape 为多头：`[B, L, num_heads, head_dim]`
- 例如：`hidden_size = 4096`, `num_heads = 32`, `head_dim = 128`
  - `qkv_weight`: `[12288, 4096]` (3 * 4096 = 12288)
  - Q/K/V 各：`[B, L, 32, 128]`

**在推理流程中的位置**:
- **位置**: Transformer 层的 Attention 子层开始
  ```python
  # Attention 子层流程
  q, k, v = FusedQKVProjection(hidden_states, qkv_weight)  # ← 这里
  q, k = RoPE(q, k, positions)  # 位置编码
  attn_output = FlashAttention(q, k, v)  # 注意力计算
  ```

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
seq_len = 512
hidden_size = 4096
num_heads = 32
head_dim = 128
# hidden_states: [4, 512, 4096]
# qkv_weight: [12288, 4096]  # 3 * 4096
# q: [4, 512, 32, 128]
# k: [4, 512, 32, 128]
# v: [4, 512, 32, 128]

# Decode 阶段
batch_size = 4
seq_len = 1
# hidden_states: [4, 1, 4096]
# q: [4, 1, 32, 128]
# k: [4, 1, 32, 128]
# v: [4, 1, 32, 128]
```

**题目背景**:
  - 在 Attention 中，需要计算 Q、K、V 三个投影：`Q = X @ W_q^T, K = X @ W_k^T, V = X @ W_v^T`
  - 融合这三个操作可以减少 kernel 启动开销和内存访问
  - 在 SGLang 中，这是常见的优化模式
  - 需要在一个 kernel 中完成三个 GEMM 操作，共享输入矩阵
- **核心要求**:
  - 融合三个矩阵乘法操作（共享输入 `hidden_states`）
  - 优化内存访问（一次读取输入，三次计算）
  - 支持批量处理
- **相关知识点**: 矩阵乘法、算子融合、内存优化

#### 9. KV Cache 存储
- **ID**: kv-cache-store
- **URL**: /challenges/kv-cache-store
- **难度**: Medium
- **描述**: 实现 KV Cache 的存储和更新操作，用于缓存历史 token 的 Key 和 Value

**输入输出规格**:
- **输入**:
  - `k_new`: `[batch_size, num_heads, seq_len, head_dim]`, dtype: float16/float32
    - 新计算的 Key 向量
    - Prefill: `seq_len` = prompt 长度，存储所有 token 的 K
    - Decode: `seq_len` = 1，只存储新 token 的 K
  - `v_new`: `[batch_size, num_heads, seq_len, head_dim]`, dtype: 同输入
    - 新计算的 Value 向量，形状同 `k_new`
  - `kv_cache`: `[batch_size, num_heads, max_seq_len, head_dim]`, dtype: 同输入
    - KV Cache 存储缓冲区（预分配）
    - `max_seq_len`: 最大序列长度（如 2048、4096）
  - `cache_positions`: `[batch_size, seq_len]`, dtype: int32
    - 每个 token 在缓存中的位置索引
- **输出**:
  - `kv_cache` (in-place 更新): 形状同输入
    - 更新后的 KV Cache，新 K、V 已写入指定位置

**形状含义说明**:
- KV Cache 按 `[batch, head, position, dim]` 布局存储
- Prefill 阶段：一次性存储所有 token 的 K、V
  - `cache_positions[i] = [0, 1, 2, ..., seq_len-1]`
- Decode 阶段：增量追加新 token 的 K、V
  - `cache_positions[i] = [current_pos]` (如 [512], [513], ...)

**在推理流程中的位置**:
- **位置 1**: Prefill 阶段，Attention 计算之后
  ```python
  q, k, v = qkv_proj(hidden_states)  # [B, L, H, D]
  attn_output = flash_attention(q, k, v)
  KVCacheStore(k, v, kv_cache, positions)  # ← 这里，存储所有 K、V
  ```
- **位置 2**: Decode 阶段，Attention 计算之后
  ```python
  q_new, k_new, v_new = qkv_proj(hidden_states)  # [B, 1, H, D]
  attn_output = lightning_attention(q_new, kv_cache)  # 使用历史 K、V
  KVCacheStore(k_new, v_new, kv_cache, [current_pos])  # ← 这里，追加新 K、V
  ```

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
num_heads = 32
seq_len = 512
head_dim = 128
max_seq_len = 2048
# k_new: [4, 32, 512, 128]
# v_new: [4, 32, 512, 128]
# kv_cache: [4, 32, 2048, 128]  # 预分配
# cache_positions: [4, 512]  # 每行 [0, 1, 2, ..., 511]

# Decode 阶段（第 1 步）
# k_new: [4, 32, 1, 128]
# v_new: [4, 32, 1, 128]
# cache_positions: [4, 1]  # 每行 [512]（假设 prompt 长度为 512）

# Decode 阶段（第 2 步）
# cache_positions: [4, 1]  # 每行 [513]
```

**题目背景**:
  - KV Cache 是 LLM 推理中的关键优化技术
  - 在 Prefill 阶段，需要将计算出的 K、V 存储到缓存中
  - 在 Decode 阶段，需要增量更新 KV Cache
  - 在 SGLang 中，KV Cache 使用分页管理（PagedAttention）提高内存利用率
- **核心要求**:
  - 实现高效的 KV Cache 存储操作（合并内存访问）
  - 支持增量更新（追加新 token 的 K、V）
  - 优化内存布局和访问模式（连续内存访问）
- **相关知识点**: 缓存管理、内存布局、增量更新

---

### Hard 难度题目

#### 10. FlashAttention (Prefill 阶段)
- **ID**: flash-attention-prefill
- **URL**: /challenges/flash-attention-prefill
- **难度**: Hard
- **描述**: 实现 FlashAttention 算法，用于 Prefill 阶段的优化注意力计算

**输入输出规格**:
- **输入**:
  - `q`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: float16/float32
    - Query 向量，已应用 RoPE
  - `k`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - Key 向量，已应用 RoPE
  - `v`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - Value 向量
  - `causal_mask`: bool, 默认 True
    - 是否应用因果掩码（确保 token i 只能看到 token 0..i）
- **输出**:
  - `attn_output`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - 注意力输出
  - `k_cache` (可选): `[batch_size, num_heads, seq_len, head_dim]`, dtype: 同输入
    - 存储的 Key，用于后续 Decode 阶段
  - `v_cache` (可选): `[batch_size, num_heads, seq_len, head_dim]`, dtype: 同输入
    - 存储的 Value，用于后续 Decode 阶段

**形状含义说明**:
- 标准注意力计算：`attn_output = softmax(Q @ K^T / sqrt(d)) @ V`
- FlashAttention 优化：
  - 不存储完整的注意力矩阵 `[seq_len, seq_len]`（O(n²) 内存）
  - 分块计算，在线 Softmax，只存储最终输出（O(n) 内存）
- Causal Mask：确保 `attn[i, j] = 0` 当 `j > i`（只能看到之前的 token）

**在推理流程中的位置**:
- **位置**: Prefill 阶段，RoPE 之后
  ```python
  # Prefill 阶段 Attention 流程
  q, k, v = qkv_proj(hidden_states)  # [B, L, H, D]
  q, k = RoPE(q, k, positions)       # 位置编码
  attn_output = FlashAttention(q, k, v, causal_mask=True)  # ← 这里
  # 同时存储 k, v 到 KV Cache
  ```

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
seq_len = 512
num_heads = 32
head_dim = 128
# q: [4, 512, 32, 128]
# k: [4, 512, 32, 128]
# v: [4, 512, 32, 128]
# attn_output: [4, 512, 32, 128]

# 内存优化：
# 标准 Attention: 需要存储 [4, 32, 512, 512] 的注意力矩阵 ≈ 134MB (FP16)
# FlashAttention: 分块计算，不存储完整矩阵，内存 ≈ 2MB (FP16)
```

**题目背景**:
  - FlashAttention 是注意力计算的重要优化，将内存复杂度从 O(n²) 降到 O(n)
  - 在 SGLang 的 Prefill 阶段，使用 FlashAttention 处理整个 prompt
  - 核心思想是分块计算和在线 Softmax，避免存储完整的注意力矩阵
  - 需要处理 Causal Mask（因果掩码），确保只能看到之前的 token
- **核心要求**:
  - 实现分块注意力计算（将 Q、K、V 分成多个块）
  - 实现在线 Softmax（逐步归一化，不需要存储完整矩阵）
  - 处理 Causal Mask（在分块计算中正确处理掩码）
  - 优化共享内存使用（减少全局内存访问）
- **相关知识点**: 注意力机制、分块计算、在线 Softmax、Causal Mask

#### 11. Lightning Attention Decode
- **ID**: lightning-attention-decode
- **URL**: /challenges/lightning-attention-decode
- **难度**: Hard
- **描述**: 实现 Lightning Attention Decode，用于 Decode 阶段的优化注意力计算

**输入输出规格**:
- **输入**:
  - `q_new`: `[batch_size, 1, num_heads, head_dim]`, dtype: float16/float32
    - 新 token 的 Query 向量，已应用 RoPE
    - `seq_len = 1`（只处理一个新 token）
  - `kv_cache`: `[batch_size, num_heads, past_len, head_dim]`, dtype: 同输入
    - 历史 token 的 K、V 缓存
    - `past_len`: 历史序列长度（prompt_len + 已生成的 token 数）
  - `k_new` (可选): `[batch_size, 1, num_heads, head_dim]`, dtype: 同输入
    - 新 token 的 Key（如果需要在 kernel 内更新 KV Cache）
  - `v_new` (可选): `[batch_size, 1, num_heads, head_dim]`, dtype: 同输入
    - 新 token 的 Value
- **输出**:
  - `attn_output`: `[batch_size, 1, num_heads, head_dim]`, dtype: 同输入
    - 注意力输出（新 token 的表示）
  - `kv_cache` (in-place 更新, 可选): 形状同输入
    - 更新后的 KV Cache（追加新 token 的 K、V）

**形状含义说明**:
- 增量注意力计算：
  - 标准方式：计算 `[1, past_len+1]` 的注意力矩阵，然后与 `[past_len+1, head_dim]` 的 V 相乘
  - 优化方式：只计算 `q_new @ past_k^T` → `[1, past_len]`，然后与 `[past_len, head_dim]` 的 V 相乘
- 复杂度：从 O(past_len²) 降到 O(past_len)

**在推理流程中的位置**:
- **位置**: Decode 阶段，RoPE 之后
  ```python
  # Decode 阶段 Attention 流程
  q_new, k_new, v_new = qkv_proj(hidden_states)  # [B, 1, H, D]
  q_new, k_new = RoPE(q_new, k_new, [current_pos])  # 位置编码
  attn_output = LightningAttentionDecode(
      q_new,           # 新 token 的 Q
      kv_cache,        # 历史 K、V
      k_new, v_new     # 新 token 的 K、V（用于更新缓存）
  )  # ← 这里
  ```

**典型数值示例**:
```python
# Decode 阶段（第 1 步，假设 prompt 长度为 512）
batch_size = 4
num_heads = 32
head_dim = 128
past_len = 512
# q_new: [4, 1, 32, 128]  # 新 token 的 Q
# kv_cache: [4, 32, 512, 128]  # 历史 K、V
# attn_output: [4, 1, 32, 128]

# Decode 阶段（第 10 步）
past_len = 522  # 512 + 10
# kv_cache: [4, 32, 522, 128]  # 已追加 10 个新 token 的 K、V
```

**题目背景**:
  - Decode 阶段每次只处理一个新 token，需要与历史 token 计算注意力
  - Lightning Attention 是专门为 Decode 阶段优化的注意力实现
  - 核心优化：增量计算、KV Cache 复用、共享内存优化
  - 在 SGLang 中，这是 Decode 阶段的核心算子，直接影响生成速度
- **核心要求**:
  - 实现增量注意力计算（只计算新 token 与历史的注意力）
  - 从 KV Cache 高效读取历史 K、V（合并内存访问）
  - 优化共享内存使用，减少全局内存访问
  - 支持批量解码（同时处理多个请求）
- **相关知识点**: 增量计算、KV Cache、共享内存优化、矩阵向量乘法

#### 12. Multi-Head Attention (优化版本)
- **ID**: optimized-multi-head-attention
- **URL**: /challenges/optimized-multi-head-attention
- **难度**: Hard
- **描述**: 实现优化的多头注意力机制，支持 Prefill 和 Decode 两种模式

**输入输出规格**:
- **输入**:
  - `hidden_states`: `[batch_size, seq_len, hidden_size]`, dtype: float16/float32
    - Transformer 层的输入
  - `qkv_weight`: `[3 * hidden_size, hidden_size]`, dtype: 同输入
    - 融合的 QKV 权重
  - `o_weight`: `[hidden_size, hidden_size]`, dtype: 同输入
    - Output 投影权重
  - `positions`: `[batch_size, seq_len]`, dtype: int32
    - 位置索引
  - `kv_cache` (Decode 模式): `[batch_size, num_heads, past_len, head_dim]`, dtype: 同输入
    - 历史 K、V 缓存
  - `mode`: str, "prefill" 或 "decode"
    - 运行模式
- **输出**:
  - `attn_output`: `[batch_size, seq_len, hidden_size]`, dtype: 同输入
    - 多头注意力的输出
  - `kv_cache` (in-place 更新, Decode 模式): 形状同输入
    - 更新后的 KV Cache

**形状含义说明**:
- 多头分割：`hidden_size = num_heads * head_dim`
  - Q/K/V: `[B, L, H, D]` → 分割为 `num_heads` 个头，每个 `[B, L, 1, D]`
- 多头合并：所有头的输出拼接为 `[B, L, H]`
- Prefill: 使用 FlashAttention，处理完整序列
- Decode: 使用 Lightning Attention，只处理新 token

**在推理流程中的位置**:
- **位置**: Transformer 层的 Attention 子层
  ```python
  # 完整的 Multi-Head Attention 流程
  attn_output = MultiHeadAttention(
      hidden_states,    # [B, L, H]
      qkv_weight,       # [3*H, H]
      o_weight,         # [H, H]
      positions,        # [B, L]
      kv_cache,         # [B, num_heads, past_len, D] (Decode)
      mode="prefill"    # 或 "decode"
  )  # ← 这里
  # 输出: [B, L, H]
  ```

**典型数值示例**:
```python
# Prefill 模式
batch_size = 4
seq_len = 512
hidden_size = 4096
num_heads = 32
head_dim = 128
# hidden_states: [4, 512, 4096]
# attn_output: [4, 512, 4096]

# Decode 模式
seq_len = 1
past_len = 512
# hidden_states: [4, 1, 4096]
# kv_cache: [4, 32, 512, 128]
# attn_output: [4, 1, 4096]
```

**题目背景**:
  - Multi-Head Attention 是 Transformer 的核心组件
  - 需要同时处理多个注意力头，并合并结果
  - 在 SGLang 中，需要支持 Prefill（FlashAttention）和 Decode（Lightning Attention）两种模式
  - 需要高效处理 QKV 投影、多头分割、注意力计算、输出投影等步骤
- **核心要求**:
  - 实现多头注意力的完整流程（QKV 投影 → 多头分割 → 注意力计算 → 多头合并 → Output 投影）
  - 支持 Prefill 和 Decode 两种模式
  - 优化内存布局（支持连续内存访问，减少转置操作）
  - 支持批量处理
- **相关知识点**: 多头注意力、内存布局优化、批量处理

#### 13. Fused Attention and KV Update
- **ID**: fused-attention-kv-update
- **URL**: /challenges/fused-attention-kv-update
- **难度**: Hard
- **描述**: 实现融合的注意力计算和 KV Cache 更新，将两个操作合并到一个 kernel

**输入输出规格**:
- **输入**:
  - `q`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: float16/float32
    - Query 向量，已应用 RoPE
  - `k`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - Key 向量，已应用 RoPE
  - `v`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - Value 向量
  - `kv_cache`: `[batch_size, num_heads, max_seq_len, head_dim]`, dtype: 同输入
    - KV Cache 存储缓冲区
  - `positions`: `[batch_size, seq_len]`, dtype: int32
    - 每个 token 在缓存中的位置
- **输出**:
  - `attn_output`: `[batch_size, seq_len, num_heads, head_dim]`, dtype: 同输入
    - 注意力输出
  - `kv_cache` (in-place 更新): 形状同输入
    - 更新后的 KV Cache（K、V 已存储）

**形状含义说明**:
- 融合操作：在计算注意力的同时，将 K、V 写入 KV Cache
- 避免先计算注意力，再单独存储 K、V（两次内存访问）
- 在同一个 kernel 中：
  1. 计算注意力：`attn_output = softmax(Q @ K^T) @ V`
  2. 存储 K、V：`kv_cache[:, :, positions, :] = k, v`

**在推理流程中的位置**:
- **位置**: Prefill 阶段，RoPE 之后
  ```python
  # Prefill 阶段
  q, k, v = qkv_proj(hidden_states)  # [B, L, H, D]
  q, k = RoPE(q, k, positions)       # 位置编码
  attn_output, kv_cache = FusedAttentionKVUpdate(
      q, k, v, kv_cache, positions
  )  # ← 这里，同时计算注意力和存储 K、V
  ```

**典型数值示例**:
```python
# Prefill 阶段
batch_size = 4
seq_len = 512
num_heads = 32
head_dim = 128
max_seq_len = 2048
# q, k, v: [4, 512, 32, 128]
# kv_cache: [4, 32, 2048, 128]
# positions: [4, 512]  # [0, 1, 2, ..., 511]
# attn_output: [4, 512, 32, 128]
```

**题目背景**:
  - 在 Prefill 阶段，注意力计算和 KV Cache 存储经常连续执行
  - 融合这两个操作可以减少内存访问和 kernel 启动开销
  - 在 SGLang 中，这是重要的优化技术
  - 需要在一个 kernel 中完成注意力计算和 KV Cache 的存储
- **核心要求**:
  - 融合注意力计算和 KV Cache 更新（在计算过程中同时写入缓存）
  - 优化内存访问模式（合并写入操作）
  - 减少中间结果的全局内存写入（K、V 直接写入缓存，不经过中间变量）
  - 支持批量处理
- **相关知识点**: 算子融合、注意力机制、KV Cache、内存优化

#### 14. GEMM with INT8 Quantization
- **ID**: int8-quantized-gemm
- **URL**: /challenges/int8-quantized-gemm
- **难度**: Hard
- **描述**: 实现 INT8 量化的矩阵乘法，用于模型量化加速

**输入输出规格**:
- **输入**:
  - `A`: `[M, K]`, dtype: float16/float32
    - 输入矩阵 A（激活值）
  - `B`: `[K, N]`, dtype: int8
    - 权重矩阵 B（已量化为 INT8）
  - `scale_a`: `[M]` 或 scalar, dtype: float32
    - A 的量化缩放因子
  - `scale_b`: `[N]` 或 scalar, dtype: float32
    - B 的量化缩放因子
  - `zero_point_a` (可选): `[M]` 或 scalar, dtype: int8
    - A 的零点偏移（对称量化可省略）
  - `zero_point_b` (可选): `[N]` 或 scalar, dtype: int8
    - B 的零点偏移
- **输出**:
  - `C`: `[M, N]`, dtype: float16/float32
    - 矩阵乘法结果（已反量化）
  - 计算过程：`C = (A_int8 @ B_int8) * scale_a * scale_b`

**形状含义说明**:
- 量化公式：`A_int8 = round(A / scale_a) + zero_point_a`
- 矩阵乘法：`C_int32 = A_int8 @ B_int8`（INT8 乘法，累加为 INT32）
- 反量化：`C = C_int32 * scale_a * scale_b`
- 典型应用：
  - QKV 投影：`A = hidden_states [B*L, H]`, `B = weight [H, 3*H]`
  - MLP 投影：`A = hidden_states [B*L, H]`, `B = weight [H, intermediate_size]`

**在推理流程中的位置**:
- **位置**: 所有矩阵乘法操作（QKV 投影、MLP 投影、LM Head）
  ```python
  # 量化推理流程
  hidden_states_fp16 = ...  # [B, L, H]
  hidden_states_int8, scale_a = quantize(hidden_states_fp16)
  weight_int8, scale_b = quantize(weight_fp16)  # 预量化
  output_int32 = INT8GEMM(hidden_states_int8, weight_int8)  # ← 这里
  output_fp16 = dequantize(output_int32, scale_a, scale_b)
  ```

**典型数值示例**:
```python
# QKV 投影示例
batch_size = 4
seq_len = 512
hidden_size = 4096
# A (hidden_states): [2048, 4096]  # B*L, H
# B (qkv_weight): [4096, 12288]    # H, 3*H (INT8)
# C (output): [2048, 12288]       # B*L, 3*H

# 内存节省：
# FP16: 2048 * 4096 * 2 + 4096 * 12288 * 2 ≈ 100MB
# INT8: 2048 * 4096 * 2 + 4096 * 12288 * 1 ≈ 60MB (节省 40%)
```

**题目背景**:
  - INT8 量化是模型加速的重要技术，可以减少内存占用和计算量
  - 在 SGLang 中，支持 INT8 GEMM 用于量化推理
  - 需要实现量化（FP32 → INT8）和反量化（INT8 → FP32）操作
  - 需要处理量化误差和数值精度问题
- **核心要求**:
  - 实现 INT8 矩阵乘法（使用 Tensor Core 的 INT8 模式）
  - 处理量化和反量化操作（正确处理缩放因子）
  - 优化 INT8 计算性能（充分利用 Tensor Core）
  - 处理量化误差（选择合适的量化参数）
- **相关知识点**: 量化、矩阵乘法、Tensor Core、数值精度

#### 15. FP16 GEMM (高性能版本)
- **ID**: fp16-high-performance-gemm
- **URL**: /challenges/fp16-high-performance-gemm
- **难度**: Hard
- **描述**: 实现高性能的 FP16 矩阵乘法，使用 Tensor Core 加速

**输入输出规格**:
- **输入**:
  - `A`: `[M, K]`, dtype: float16
    - 输入矩阵 A
  - `B`: `[K, N]`, dtype: float16
    - 权重矩阵 B
  - `bias` (可选): `[N]`, dtype: float16
    - 偏置向量（可选）
- **输出**:
  - `C`: `[M, N]`, dtype: float16
    - 矩阵乘法结果：`C = A @ B + bias`

**形状含义说明**:
- 典型应用场景：
  - QKV 投影：`A = [B*L, H]`, `B = [H, 3*H]`, `C = [B*L, 3*H]`
  - MLP Gate/Up：`A = [B*L, H]`, `B = [H, I]`, `C = [B*L, I]`
  - MLP Down：`A = [B*L, I]`, `B = [I, H]`, `C = [B*L, H]`
  - LM Head：`A = [B, H]`, `B = [H, V]`, `C = [B, V]`
- Tensor Core 要求：
  - 矩阵维度必须是 16 的倍数（或使用填充）
  - 使用 WMMA (Warp Matrix Multiply Accumulate) API

**在推理流程中的位置**:
- **位置**: 所有矩阵乘法操作
  ```python
  # QKV 投影
  qkv = FP16GEMM(hidden_states, qkv_weight)  # [B*L, H] @ [H, 3*H] → [B*L, 3*H]
  
  # MLP 投影
  gate = FP16GEMM(hidden_states, gate_weight)  # [B*L, H] @ [H, I] → [B*L, I]
  mlp_output = FP16GEMM(mlp_intermediate, down_weight)  # [B*L, I] @ [I, H] → [B*L, H]
  
  # LM Head
  logits = FP16GEMM(hidden_states, lm_weight)  # [B, H] @ [H, V] → [B, V]
  ```

**典型数值示例**:
```python
# QKV 投影（Prefill）
batch_size = 4
seq_len = 512
hidden_size = 4096
# A (hidden_states): [2048, 4096]  # B*L, H
# B (qkv_weight): [4096, 12288]    # H, 3*H
# C (qkv): [2048, 12288]           # B*L, 3*H

# MLP Gate 投影
intermediate_size = 11008
# A (hidden_states): [2048, 4096]  # B*L, H
# B (gate_weight): [4096, 11008]   # H, I
# C (gate): [2048, 11008]           # B*L, I

# LM Head
vocab_size = 32000
# A (hidden_states): [4, 4096]      # B, H (只取最后一个 token)
# B (lm_weight): [4096, 32000]      # H, V
# C (logits): [4, 32000]            # B, V
```

**题目背景**:
  - FP16 GEMM 是现代 GPU 推理的标准操作
  - 在 SGLang 中，大部分矩阵乘法使用 FP16 精度
  - 需要充分利用 Tensor Core（如 A100、H100）的硬件加速
  - 需要优化内存访问、寄存器使用、共享内存布局等
- **核心要求**:
  - 实现高性能 FP16 矩阵乘法（使用 Tensor Core WMMA API）
  - 优化内存访问模式（合并访问、预取）
  - 优化共享内存布局（bank conflict 最小化）
  - 支持批量矩阵乘法（Batched GEMM）
- **相关知识点**: FP16 精度、Tensor Core、内存优化、高性能计算

---

## 题目分类

### 基础算子类
- 激活函数（SiLU）
- 归一化（RMSNorm）
- 位置编码（RoPE）

### 融合算子类
- Fused Add RMSNorm
- Fused SiLU and Mul
- Fused QKV Projection
- Fused Attention and KV Update

### 注意力机制类
- FlashAttention (Prefill)
- Lightning Attention Decode
- Multi-Head Attention
- Softmax (数值稳定版本)

### 优化技术类
- TopK 采样
- KV Cache 存储
- INT8 量化 GEMM
- FP16 高性能 GEMM

---

## 学习路径建议

### 入门路径（Easy → Medium）
1. **SiLU 激活函数** - 理解逐元素操作和 CUDA kernel 基础
2. **RMSNorm** - 学习并行归约和归一化
3. **RoPE** - 理解位置编码和复数旋转
4. **Fused Add RMSNorm** - 学习算子融合技术
5. **TopK 采样** - 学习排序和采样算法

### 进阶路径（Medium → Hard）
1. **Softmax (数值稳定版本)** - 理解注意力机制的基础
2. **KV Cache 存储** - 理解推理优化的关键
3. **FlashAttention (Prefill)** - 学习注意力优化算法
4. **Lightning Attention Decode** - 学习解码阶段优化
5. **Multi-Head Attention** - 综合应用所学知识

### 高级路径（Hard）
1. **FP16 高性能 GEMM** - 学习高性能矩阵乘法
2. **INT8 量化 GEMM** - 学习模型量化技术
3. **Fused Attention and KV Update** - 学习高级融合技术

---

## 与 SGLang 推理流程的对应关系

| 推理步骤 | 对应题目 | 难度 |
|---------|---------|------|
| Token Embedding | (PyTorch 标准操作) | - |
| RoPE 位置编码 | RoPE (旋转位置编码) | Easy |
| Attention (Prefill) | FlashAttention (Prefill 阶段) | Hard |
| Attention (Decode) | Lightning Attention Decode | Hard |
| KV Cache 更新 | KV Cache 存储 | Medium |
| 残差连接 + RMSNorm | Fused Add RMSNorm | Medium |
| MLP Gate/Up 投影 | QKV 投影 (融合版本) | Medium |
| SiLU 激活 | SiLU 激活函数 | Easy |
| SiLU and Mul | SiLU and Mul (融合激活) | Medium |
| MLP Down 投影 | GEMM 相关题目 | Medium/Hard |
| LM Head | GEMM 相关题目 | Medium/Hard |
| TopK 采样 | TopK 采样 | Medium |

---

## 参考资源

- SGLang 官方文档
- FlashAttention 论文
- Lightning Attention 实现
- CUDA 编程最佳实践
- CUTLASS 库文档

---

**最后更新时间**: 2025-01-XX  
**数据来源**: 基于 SGLang 模型推理流程与算子详解文档

