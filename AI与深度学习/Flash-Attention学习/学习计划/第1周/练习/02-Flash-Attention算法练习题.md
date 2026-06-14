# Flash-Attention 算法练习题

## 📝 说明

本练习包含 Flash-Attention 算法相关的配套题目，每个题目对应一个知识点，帮助理解算法的核心思想。

---

## 第一部分：标准 Attention 的瓶颈

### 题目 1：内存复杂度分析

**知识点**：标准 Attention 的内存复杂度

**题目**：
给定标准 Attention 计算：
- 序列长度：N = 2048
- Head 维度：d = 128
- 数据类型：float32（4 字节）

请计算以下矩阵的内存占用（以 MB 为单位）：
1. Q 矩阵
2. S 矩阵（注意力分数矩阵）
3. P 矩阵（Softmax 后的概率矩阵）

**答案**：

1. **Q 矩阵**：2048 × 128 × 4 bytes = **1 MB**
2. **S 矩阵**：2048 × 2048 × 4 bytes = **16 MB**
3. **P 矩阵**：2048 × 2048 × 4 bytes = **16 MB**

**总内存**：Q + K + V + S + P = 3 + 16 + 16 = **35 MB**（每个 head）

---

### 题目 2：内存复杂度增长

**知识点**：O(N²) 内存复杂度

**题目**：
当序列长度 N 从 1K 增加到 32K 时，标准 Attention 的 S 矩阵内存占用增长了多少倍？

**答案**：

- N = 1K：S 矩阵 = 1024 × 1024 × 4 bytes = 4 MB
- N = 32K：S 矩阵 = 32768 × 32768 × 4 bytes = 4096 MB = 4 GB

**增长倍数**：(32K / 1K)² = **1024 倍**

**结论**：序列长度增加 32 倍，内存占用增加 1024 倍（平方关系）。

---

### 题目 3：内存瓶颈识别

**知识点**：识别内存瓶颈

**题目**：
在标准 Attention 计算中，以下哪个步骤是内存瓶颈？

```python
# 步骤1：计算注意力分数
S = Q @ K^T  # Shape: (N, N)

# 步骤2：应用 Softmax
P = softmax(S)  # Shape: (N, N)

# 步骤3：计算输出
O = P @ V  # Shape: (N, d)
```

**答案**：
- **步骤1 和步骤2** 是内存瓶颈
- **原因**：需要存储 N×N 的 S 和 P 矩阵，内存复杂度为 O(N²)
- **步骤3** 不是瓶颈：只需要 O(N×d) 内存

---

## 第二部分：Flash-Attention 核心思想

### 题目 4：Tiling 策略理解

**知识点**：Tiling（分块）策略

**题目**：
Flash-Attention 将 Q、K、V 分成小块处理。如果：
- 序列长度：N = 2048
- Q 块大小：block_size_M = 64
- K/V 块大小：block_size_N = 64

请计算：
1. Q 需要分成多少块？
2. K/V 需要分成多少块？
3. 总共需要多少次块对（Q 块 × K/V 块）的计算？

**答案**：

1. **Q 块数**：⌈2048 / 64⌉ = **32 块**
2. **K/V 块数**：⌈2048 / 64⌉ = **32 块**
3. **总计算次数**：32 × 32 = **1024 次块对计算**

**对比**：标准 Attention 需要一次计算完整的 N×N 矩阵，Flash-Attention 需要 1024 次小块计算。

---

### 题目 5：内存复杂度对比

**知识点**：Flash-Attention 的内存复杂度

**题目**：
Flash-Attention 使用 Tiling 策略后，内存复杂度从 O(N²) 降到了多少？

**答案**：
**O(N)** 或更准确地说 **O(block_size × d)**

**解释**：
- 标准 Attention：需要存储 N×N 的 S 和 P 矩阵 → O(N²)
- Flash-Attention：只需要存储块大小的中间结果（block_size × block_size）→ O(block_size²) = O(1)（相对于 N）

**实际内存占用**（block_size = 64, d = 128）：
- 块大小固定，不随 N 增长
- 内存占用：64 × 64 × 4 bytes ≈ 16 KB（每个块）

---

### 题目 6：Online Softmax 必要性

**知识点**：为什么需要 Online Softmax

**题目**：
为什么 Flash-Attention 不能先计算完整的 S 矩阵，然后再应用 Softmax？

**答案**：
- **内存限制**：完整的 S 矩阵需要 O(N²) 内存，当 N 很大时（如 32K），内存占用巨大
- **Online Softmax 的优势**：可以逐块处理，只需要 O(block_size²) 内存
- **数学等价性**：Online Softmax 在数学上等价于标准 Softmax，但内存效率高得多

---

## 第三部分：Online Softmax 数学原理

### 题目 7：Online Softmax 更新公式

**知识点**：Online Softmax 的更新规则

**题目**：
在 Online Softmax 中，当处理新的数据块时，需要更新统计量。请填写以下更新公式：

```
m_new = max(?, ?)
alpha = exp(? - ?)
l_new = ? * l_old + sum(exp(? - ?))
o_new = ? * o_old + exp(? - ?) @ v_new
```

**答案**：
```
m_new = max(m_old, max(s_new_block))
alpha = exp(m_old - m_new)
l_new = alpha * l_old + sum(exp(s_new_block - m_new))
o_new = alpha * o_old + exp(s_new_block - m_new) @ v_new_block
```

---

### 题目 8：缩放因子 alpha 的作用

**知识点**：alpha 缩放因子的作用

**题目**：
在 Online Softmax 更新中，为什么需要 alpha = exp(m_old - m_new) 这个缩放因子？

**答案**：
- **原因**：当新的最大值 m_new 出现时，旧的 exp 值需要重新缩放
- **数学原理**：exp(s - m_new) = exp(s - m_old) × exp(m_old - m_new)
- **作用**：alpha 用于将旧的统计量（基于 m_old）缩放到新的尺度（基于 m_new）

**示例**：
- 假设 m_old = 5, m_new = 7
- 旧的 exp(s - 5) 需要乘以 exp(5 - 7) = exp(-2) 才能得到 exp(s - 7)

---

### 题目 9：数值稳定性

**知识点**：Online Softmax 的数值稳定性

**题目**：
为什么 Online Softmax 使用 `exp(s - m)` 而不是 `exp(s)`？

**答案**：
- **避免溢出**：exp(s) 可能非常大，导致数值溢出
- **数值稳定性**：exp(s - m) 的最大值为 1（当 s = m 时），避免大数值
- **数学等价性**：softmax 只关心相对大小，exp(s - m) / sum(exp(s - m)) 等价于 exp(s) / sum(exp(s))

**示例**：
- 如果 s = 100，exp(100) 会溢出
- 但如果 m = 100，exp(100 - 100) = exp(0) = 1，不会溢出

---

### 题目 10：Online Softmax 计算步骤

**知识点**：Online Softmax 的完整计算流程

**题目**：
给定以下数据：
- m_old = 3.0
- l_old = 2.5
- o_old = [1.0, 2.0]
- s_new_block = [4.0, 3.5]
- v_new_block = [[0.5], [1.0]]

请计算 m_new, l_new, o_new。

**答案**：

1. **计算 m_new**：
   ```
   m_new = max(m_old, max(s_new_block))
        = max(3.0, max(4.0, 3.5))
        = max(3.0, 4.0)
        = 4.0
   ```

2. **计算 alpha**：
   ```
   alpha = exp(m_old - m_new)
        = exp(3.0 - 4.0)
        = exp(-1.0)
        ≈ 0.368
   ```

3. **计算 l_new**：
   ```
   l_new = alpha * l_old + sum(exp(s_new_block - m_new))
        = 0.368 * 2.5 + (exp(4.0 - 4.0) + exp(3.5 - 4.0))
        = 0.92 + (1.0 + exp(-0.5))
        = 0.92 + (1.0 + 0.607)
        ≈ 2.527
   ```

4. **计算 o_new**：
   ```
   p_new = exp(s_new_block - m_new) = [exp(0), exp(-0.5)] = [1.0, 0.607]
   o_new = alpha * o_old + p_new @ v_new_block
        = 0.368 * [1.0, 2.0] + [1.0, 0.607] @ [[0.5], [1.0]]
        = [0.368, 0.736] + [0.5, 0.607]
        = [0.868, 1.343]
   ```

---

## 第四部分：Tiling 策略

### 题目 11：块大小选择

**知识点**：Tiling 块大小的选择

**题目**：
选择 Tiling 块大小时，需要考虑哪些因素？

**答案**：
1. **共享内存大小**：每个 SM 有 48KB 或 164KB 共享内存
2. **寄存器数量**：每个线程的寄存器限制
3. **Warp 数量**：影响占用率
4. **内存带宽**：需要足够的计算掩盖内存延迟
5. **计算密度**：平衡内存访问和计算

**常见配置**：
- block_size_M = 64 或 128
- block_size_N = 64 或 128
- head_dim = 64, 128, 192, 256

---

### 题目 12：内存访问模式优化

**知识点**：Tiling 对内存访问的优化

**题目**：
对比标准 Attention 和 Flash-Attention 的内存访问模式，Flash-Attention 如何减少内存访问？

**答案**：

**标准 Attention**：
```
Q: HBM → 寄存器 → 计算
K: HBM → 寄存器 → 计算
S: 寄存器 → HBM (写入) ❌ 需要写回
P: HBM → 寄存器 → 计算
O: 寄存器 → HBM (写入)
```

**Flash-Attention**：
```
Q 块: HBM → 共享内存 → 寄存器 → 计算
K 块: HBM → 共享内存 → 寄存器 → 计算
中间结果: 寄存器 → 寄存器 (不写回 HBM) ✅
O: 寄存器 → HBM (只写一次) ✅
```

**优化点**：
1. ✅ 使用共享内存缓存数据
2. ✅ 中间结果不写回全局内存
3. ✅ 输出只写一次
4. ✅ 减少全局内存访问次数

---

### 题目 13：Tiling 计算流程

**知识点**：Tiling 的完整计算流程

**题目**：
请描述 Flash-Attention 使用 Tiling 策略的完整计算流程（伪代码）。

**答案**：
```python
def flash_attention_tiling(Q, K, V, block_size_M, block_size_N):
    N, d = Q.shape
    T_r = ceil(N / block_size_M)  # Q 的块数
    T_c = ceil(N / block_size_N)  # K/V 的块数
    
    O = zeros_like(Q)
    
    for i in range(T_r):
        q_i = Q[i * block_size_M:(i+1) * block_size_M]
        m_i = -inf
        l_i = 0
        o_i = 0
        
        for j in range(T_c):
            k_j = K[j * block_size_N:(j+1) * block_size_N]
            v_j = V[j * block_size_N:(j+1) * block_size_N]
            
            # 计算注意力分数
            s_ij = q_i @ k_j^T
            
            # Online Softmax 更新
            m_ij = max(m_i, rowmax(s_ij))
            p_ij = exp(s_ij - m_ij)
            l_ij = sum(p_ij, dim=1)
            alpha = exp(m_i - m_ij)
            o_i = alpha * o_i + p_ij @ v_j
            l_i = alpha * l_i + l_ij
            m_i = m_ij
        
        # 归一化输出
        O[i*block_size_M:(i+1)*block_size_M] = o_i / l_i
    
    return O
```

---

## 第五部分：综合理解

### 题目 14：算法优势总结

**知识点**：Flash-Attention 的核心优势

**题目**：
请总结 Flash-Attention 相比标准 Attention 的三个核心优势。

**答案**：
1. **内存效率**：
   - 从 O(N²) 降到 O(N)
   - 可以处理更长的序列（如 32K、64K）

2. **计算效率**：
   - 减少内存访问
   - 提高计算密度
   - 更好地利用 GPU 资源

3. **数值稳定性**：
   - Online Softmax 保证数值稳定
   - 避免溢出和下溢

---

### 题目 15：性能对比分析

**知识点**：Flash-Attention 的性能提升

**题目**：
根据以下数据，分析 Flash-Attention 的性能优势：

| 序列长度 | 标准 Attention 时间 | Flash-Attention 时间 | 加速比 |
|---------|-------------------|---------------------|--------|
| 2K | 100 ms | 25 ms | ? |
| 4K | 400 ms | 50 ms | ? |
| 8K | 1600 ms | 100 ms | ? |

**答案**：

| 序列长度 | 标准 Attention 时间 | Flash-Attention 时间 | 加速比 |
|---------|-------------------|---------------------|--------|
| 2K | 100 ms | 25 ms | **4×** |
| 4K | 400 ms | 50 ms | **8×** |
| 8K | 1600 ms | 100 ms | **16×** |

**分析**：
- 序列长度增加，Flash-Attention 的优势更明显
- 标准 Attention 的时间复杂度接近 O(N²)
- Flash-Attention 的时间复杂度接近 O(N)

---

## 📊 练习总结

### 知识点覆盖

- ✅ 标准 Attention 的瓶颈
- ✅ Flash-Attention 核心思想
- ✅ Tiling 策略
- ✅ Online Softmax 数学原理
- ✅ 内存复杂度分析
- ✅ 性能对比分析

### 建议

1. **理解数学原理**：深入理解 Online Softmax 的数学推导
2. **动手计算**：尝试手动计算 Online Softmax 的更新过程
3. **对比分析**：理解标准 Attention 和 Flash-Attention 的区别
4. **阅读论文**：参考 FlashAttention 论文深入理解

---

**完成日期**：________  
**正确率**：____ / 15
