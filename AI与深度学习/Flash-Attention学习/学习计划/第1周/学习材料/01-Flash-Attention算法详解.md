# Flash-Attention 算法详解

## 📚 学习目标

1. 理解标准 Attention 的瓶颈和问题
2. 掌握 Flash-Attention 的核心思想
3. 理解 Online Softmax 的数学原理
4. 理解 Tiling 策略的作用

---

## 🔴 标准 Attention 的瓶颈

### 标准 Attention 算法

**输入**：
- Query: Q ∈ R^(N×d)
- Key: K ∈ R^(N×d)
- Value: V ∈ R^(N×d)

**计算流程**：
```python
# 步骤1：计算注意力分数
S = Q @ K^T  # Shape: (N, N)

# 步骤2：应用 Softmax
P = softmax(S)  # Shape: (N, N)
# P[i,j] = exp(S[i,j] - max(S[i])) / sum(exp(S[i,:] - max(S[i])))

# 步骤3：计算输出
O = P @ V  # Shape: (N, d)
```

### 内存复杂度分析

**标准实现的内存占用**：
```
Q, K, V: 3 × N × d × sizeof(float16) = 3 × N × d × 2 bytes
S: N × N × sizeof(float32) = N × N × 4 bytes
P: N × N × sizeof(float32) = N × N × 4 bytes
O: N × d × sizeof(float16) = N × d × 2 bytes

总内存 ≈ 8 × N² bytes (当 N >> d 时)
```

**问题**：
- ❌ **O(N²) 的内存复杂度**：当序列长度 N 很大时（如 32K），内存占用巨大
- ❌ **大量 HBM 访问**：需要频繁读写全局内存
- ❌ **内存带宽成为瓶颈**：计算速度受限于内存带宽

### 实际例子

**GPT-3 规模**：
- 序列长度：N = 2048
- Head 维度：d = 128
- Head 数量：h = 96

**内存占用**：
```
S 矩阵：2048 × 2048 × 4 bytes = 16 MB (每个 head)
P 矩阵：2048 × 2048 × 4 bytes = 16 MB (每个 head)
总计：32 MB × 96 heads = 3 GB (仅注意力矩阵)
```

**问题**：
- GPU 显存有限（如 A100 40GB）
- 需要存储多个注意力矩阵
- 内存访问成为性能瓶颈

---

## ✅ Flash-Attention 核心思想

### 核心创新

Flash-Attention 通过三个关键技术解决标准 Attention 的问题：

1. **Tiling（分块）**：将 Q、K、V 分成小块，避免存储完整的注意力矩阵
2. **Online Softmax**：在线计算 softmax，避免存储完整的 S 和 P 矩阵
3. **内存重计算**：反向传播时重新计算中间结果，不存储中间激活值

### Flash-Attention 算法流程

**伪代码**：
```python
def flash_attention(Q, K, V, block_size_M, block_size_N):
    """
    Q: (N, d)
    K: (N, d)
    V: (N, d)
    block_size_M: Q 的块大小（如 64）
    block_size_N: K/V 的块大小（如 64）
    """
    N, d = Q.shape
    T_r = ceil(N / block_size_M)  # Q 的块数
    T_c = ceil(N / block_size_N)  # K/V 的块数
    
    O = zeros_like(Q)  # 输出
    m = full(N, -inf)   # 每行的最大值
    l = zeros(N)        # 每行的归一化因子
    
    for i in range(T_r):
        # 加载 Q 的第 i 块
        q_i = Q[i * block_size_M:(i+1) * block_size_M]  # (block_size_M, d)
        
        # 初始化当前块的输出
        o_i = zeros(block_size_M, d)
        
        for j in range(T_c):
            # 加载 K、V 的第 j 块
            k_j = K[j * block_size_N:(j+1) * block_size_N]  # (block_size_N, d)
            v_j = V[j * block_size_N:(j+1) * block_size_N]  # (block_size_N, d)
            
            # 计算注意力分数
            s_ij = q_i @ k_j^T  # (block_size_M, block_size_N)
            
            # Online Softmax 更新
            m_ij = max(m[i*block_size_M:(i+1)*block_size_M], rowmax(s_ij))
            p_ij = exp(s_ij - m_ij)
            l_ij = sum(p_ij, dim=1)
            
            # 更新输出
            alpha = exp(m_old - m_ij)
            o_i = alpha * o_i + p_ij @ v_j
            l = alpha * l + l_ij
            m = m_ij
        
        # 归一化输出
        O[i*block_size_M:(i+1)*block_size_M] = o_i / l
    
    return O
```

### 内存复杂度分析

**Flash-Attention 的内存占用**：
```
Q, K, V 块：3 × block_size × d × sizeof(float16)
中间结果：
  - s_ij: block_size_M × block_size_N × sizeof(float32)
  - p_ij: block_size_M × block_size_N × sizeof(float32)
  - o_i: block_size_M × d × sizeof(float16)
  - m, l: block_size_M × sizeof(float32)

总内存 ≈ O(block_size × d) = O(1) (相对于 N)
```

**优势**：
- ✅ **O(N) 的内存复杂度**：只存储块大小的中间结果
- ✅ **减少 HBM 访问**：数据在共享内存中处理
- ✅ **内存带宽不再是瓶颈**：计算速度受限于计算能力

---

## 🧮 Online Softmax 数学原理

### 标准 Softmax

对于向量 **s** = [s₁, s₂, ..., sₙ]，标准 softmax 定义为：

```
P[i] = exp(s[i] - m) / Σⱼ exp(s[j] - m)

其中 m = max(s)
```

### Online Softmax 问题

**问题**：如何在线计算 softmax？

假设我们已经处理了前 k 个元素，现在要添加第 k+1 个元素：

```
s_old = [s₁, s₂, ..., sₖ]
s_new = [s₁, s₂, ..., sₖ, sₖ₊₁]
```

**挑战**：需要重新归一化所有元素。

### Online Softmax 算法

**关键思想**：维护三个统计量：
- **m**：当前最大值
- **l**：归一化因子的分子（sum of exp）
- **o**：加权和（用于计算输出）

**更新规则**：

```python
def online_softmax_update(m_old, l_old, o_old, s_new_block, v_new_block):
    """
    m_old: 旧的最大值
    l_old: 旧的归一化因子
    o_old: 旧的加权和
    s_new_block: 新块的注意力分数
    v_new_block: 新块的 Value
    """
    # 计算新块的最大值
    m_new = max(m_old, max(s_new_block))
    
    # 计算缩放因子
    alpha = exp(m_old - m_new)
    
    # 更新归一化因子
    p_new = exp(s_new_block - m_new)
    l_new = alpha * l_old + sum(p_new)
    
    # 更新输出
    o_new = alpha * o_old + p_new @ v_new_block
    
    return m_new, l_new, o_new
```

### 数学推导

**引理 1**：对于两个块 s₁ 和 s₂，设 m₁ = max(s₁)，m₂ = max(s₂)，m = max(m₁, m₂)

则：
```
exp(s₁ - m) = exp(s₁ - m₁) × exp(m₁ - m)
exp(s₂ - m) = exp(s₂ - m₂) × exp(m₂ - m)
```

**引理 2**：对于输出计算，我们需要：
```
O = Σⱼ P[j] × V[j] = Σⱼ (exp(s[j] - m) / l) × V[j]
```

其中 l = Σⱼ exp(s[j] - m)

**在线更新**：

假设我们已经处理了 s₁，现在要添加 s₂：

```
m_old = max(s₁)
l_old = Σⱼ exp(s₁[j] - m_old)
o_old = Σⱼ (exp(s₁[j] - m_old) / l_old) × V₁[j]

m_new = max(m_old, max(s₂))
alpha = exp(m_old - m_new)

l_new = alpha × l_old + Σⱼ exp(s₂[j] - m_new)
o_new = alpha × o_old + Σⱼ (exp(s₂[j] - m_new) / l_new) × V₂[j]
```

**最终归一化**：
```
O_final = o_new / l_new
```

### 数值稳定性

**问题**：exp 函数可能溢出。

**解决方案**：
- 减去最大值：`exp(s - m)` 而不是 `exp(s)`
- 使用 log-sum-exp trick

**验证**：
```
exp(s - m) = exp(s) / exp(m)
当 s = m 时，exp(s - m) = 1
当 s << m 时，exp(s - m) ≈ 0
```

---

## 🧩 Tiling 策略

### Tiling 的作用

**目标**：将大矩阵分成小块，使得：
1. 每个块可以放入共享内存
2. 减少全局内存访问
3. 提高内存访问效率

### Tiling 策略

**Q 的 Tiling**：
- 将 Q 分成 T_r 块，每块大小为 block_size_M
- 每个线程块处理一个 Q 块

**K、V 的 Tiling**：
- 将 K、V 分成 T_c 块，每块大小为 block_size_N
- 对每个 Q 块，遍历所有 K、V 块

### 内存访问模式

**标准实现**：
```
Q: HBM → 寄存器 → 计算
K: HBM → 寄存器 → 计算
V: HBM → 寄存器 → 计算
S: 寄存器 → HBM (写入)
P: HBM → 寄存器 → 计算
O: 寄存器 → HBM (写入)
```

**Flash-Attention**：
```
Q 块: HBM → 共享内存 → 寄存器 → 计算
K 块: HBM → 共享内存 → 寄存器 → 计算
V 块: HBM → 共享内存 → 寄存器 → 计算
中间结果: 寄存器 → 寄存器 (不写回 HBM)
O: 寄存器 → HBM (只写一次)
```

### 块大小选择

**考虑因素**：
1. **共享内存大小**：每个 SM 有 48KB 或 164KB 共享内存
2. **寄存器数量**：每个线程的寄存器限制
3. **Warp 数量**：影响占用率
4. **内存带宽**：需要足够的计算掩盖内存延迟

**常见配置**：
- `block_size_M = 64` 或 `128`
- `block_size_N = 64` 或 `128`
- `head_dim = 64, 128, 192, 256`

---

## 📊 性能对比

### 内存占用对比

| 序列长度 | 标准 Attention | Flash-Attention | 节省 |
|---------|---------------|----------------|------|
| 1K | 4 MB | 0.5 MB | 8× |
| 2K | 16 MB | 0.5 MB | 32× |
| 4K | 64 MB | 0.5 MB | 128× |
| 8K | 256 MB | 0.5 MB | 512× |
| 16K | 1 GB | 0.5 MB | 2048× |

### 速度对比

| GPU | 序列长度 | 标准 Attention | Flash-Attention | 加速比 |
|-----|---------|---------------|----------------|--------|
| A100 | 2K | 100 ms | 25 ms | 4× |
| A100 | 4K | 400 ms | 50 ms | 8× |
| A100 | 8K | 1600 ms | 100 ms | 16× |

---

## 🎯 关键要点总结

### Flash-Attention 的核心优势

1. **内存效率**：
   - 从 O(N²) 降到 O(N)
   - 可以处理更长的序列

2. **计算效率**：
   - 减少内存访问
   - 提高计算密度

3. **数值稳定性**：
   - Online Softmax 保证数值稳定
   - 避免溢出和下溢

### 学习检查点

- [ ] 能够解释为什么标准 Attention 需要 O(N²) 内存
- [ ] 能够解释 Flash-Attention 如何将内存降到 O(N)
- [ ] 能够推导 Online Softmax 的更新公式
- [ ] 理解 Tiling 策略的作用
- [ ] 理解内存访问模式的优化

---

## 📚 参考资源

### 论文
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://tridao.me/publications/flash2/flash2.pdf

### 代码
- Flash-Attention GitHub: https://github.com/Dao-AILab/flash-attention

---

**学习时间**：2-3 天  
**难度**：⭐⭐⭐☆☆
