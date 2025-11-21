# 第1周：算法和基础

## 📋 本周学习目标

1. ✅ 理解 Flash-Attention 的核心算法思想
2. ✅ 回顾 CUDA 编程基础知识
3. ✅ 学习 CUTLASS 和 CuTe 库的基本使用
4. ✅ 搭建开发环境并运行第一个示例

---

## 📚 学习内容

### Day 1-2: Flash-Attention 算法理解
- [学习材料](./学习材料/01-Flash-Attention算法详解.md)
- [笔记](./笔记/01-算法理解笔记.md)
- [练习题](./练习/02-Flash-Attention算法练习题.md)（15题）

### Day 3: CUDA 基础回顾
- [学习材料](./学习材料/02-CUDA基础回顾.md)
- [笔记](./笔记/02-CUDA基础笔记.md)
- [练习题](./练习/01-CUDA基础练习题.md)（15题）

### Day 4-5: CUTLASS/CuTe 入门
- [学习材料](./学习材料/03-CUTLASS和CuTe入门.md)
- [笔记](./笔记/03-CUTLASS笔记.md)
- [练习题](./练习/03-CUTLASS和CuTe练习题.md)（15题）

### Day 6-7: 开发环境搭建
- [学习材料](./学习材料/04-开发环境搭建.md)
- [环境配置记录](./笔记/04-环境配置记录.md)
- [测试代码](./代码/test_flash_attn.py)

---

## 📁 目录结构

```
第1周/
├── README.md                 # 本文件
├── 学习材料/                 # 学习材料
│   ├── 01-Flash-Attention算法详解.md
│   ├── 02-CUDA基础回顾.md
│   ├── 03-CUTLASS和CuTe入门.md
│   └── 04-开发环境搭建.md
├── 资料/                     # 学习资料
│   ├── FlashAttention论文.pdf
│   ├── FlashAttention2论文.pdf
│   └── 相关资源链接.md
├── 笔记/                     # 学习笔记
│   ├── 01-算法理解笔记.md
│   ├── 02-CUDA基础笔记.md
│   ├── 03-CUTLASS笔记.md
│   └── 04-环境配置记录.md
├── 代码/                     # 代码示例
│   ├── test_flash_attn.py
│   ├── simple_attention.py
│   └── cuda_basics/
└── 练习/                     # 练习题（共45题）
    ├── 01-CUDA基础练习题.md（15题）
    ├── 02-Flash-Attention算法练习题.md（15题）
    └── 03-CUTLASS和CuTe练习题.md（15题）
```

---

## ✅ 学习检查点

### 算法理解
- [ ] 能够解释为什么标准 Attention 需要 O(N²) 内存
- [ ] 能够解释 Flash-Attention 如何将内存降到 O(N)
- [ ] 能够推导 Online Softmax 的更新公式
- [ ] 理解 Tiling 策略的作用

### CUDA 基础
- [ ] 理解 CUDA 编程模型
- [ ] 理解 GPU 内存层次结构
- [ ] 理解内存合并访问
- [ ] 理解 Warp 和线程块的概念

### CUTLASS/CuTe
- [ ] 理解 CUTLASS 的基本概念
- [ ] 理解 CuTe 张量抽象
- [ ] 能够创建和使用张量
- [ ] 理解布局的作用

### 环境搭建
- [ ] CUDA Toolkit 已安装
- [ ] PyTorch 已安装并可以检测 GPU
- [ ] Flash-Attention 已安装
- [ ] 可以运行测试代码

---

## 📝 学习进度跟踪

### 第1天（算法理解 - 上）
- [ ] 阅读 FlashAttention 论文
- [ ] 理解标准 Attention 的瓶颈
- [ ] 理解 Flash-Attention 核心思想
- [ ] 完成算法理解笔记

### 第2天（算法理解 - 下）
- [ ] 推导 Online Softmax 公式
- [ ] 理解 Tiling 策略
- [ ] 完成算法推导练习
- [ ] 总结算法要点

### 第3天（CUDA 基础）
- [ ] 回顾 CUDA 编程模型
- [ ] 理解内存层次结构
- [ ] 理解内存访问优化
- [ ] 完成 CUDA 基础练习

### 第4天（CUTLASS 入门）
- [ ] 学习 CUTLASS 基本概念
- [ ] 理解 Tile 和 Layout
- [ ] 学习基本使用
- [ ] 完成 CUTLASS 笔记

### 第5天（CuTe 入门）
- [ ] 学习 CuTe 张量抽象
- [ ] 理解 Shape 和 Stride
- [ ] 学习张量操作
- [ ] 完成 CuTe 练习

### 第6天（环境搭建）
- [ ] 安装 CUDA Toolkit
- [ ] 安装 PyTorch
- [ ] 安装 Flash-Attention
- [ ] 验证安装

### 第7天（测试和总结）
- [ ] 运行测试代码
- [ ] 性能测试
- [ ] 总结本周学习
- [ ] 准备下周学习

---

## 🎯 本周重点

### 核心概念
1. **Flash-Attention 算法**：Tiling + Online Softmax
2. **CUDA 内存优化**：合并访问 + 共享内存
3. **CUTLASS/CuTe**：张量抽象 + 布局优化

### 关键公式
1. **Online Softmax 更新**：
   ```
   m_new = max(m_old, m_new_block)
   alpha = exp(m_old - m_new)
   l_new = alpha * l_old + sum(exp(s_new - m_new))
   o_new = alpha * o_old + exp(s_new - m_new) @ v_new
   ```

2. **全局线程 ID**：
   ```
   global_id = blockIdx.x * blockDim.x + threadIdx.x
   ```

---

## 📚 参考资源

### 论文
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://tridao.me/publications/flash2/flash2.pdf

### 文档
- CUDA C++ Programming Guide
- CUTLASS 文档
- CuTe 文档

### 代码
- Flash-Attention GitHub: https://github.com/Dao-AILab/flash-attention

---

## 🚀 下一步

完成第1周学习后，进入第2周：
- 代码结构分析
- 参数结构理解
- 内核启动机制
- 内核特性系统

---

**开始日期**：________  
**完成日期**：________  
**预计用时**：20-25 小时  
**难度**：⭐⭐⭐☆☆
