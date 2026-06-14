# Flash-Attention 学习计划目录

## 📚 学习计划文档

### 总体计划
- [README.md](../README.md) - 总体学习计划（10-12周）

### 详细周计划
- [第1周-算法和基础.md](./第1周-算法和基础.md) - 算法理解和基础准备

### 源码分析
- [源码结构梳理.md](./源码结构梳理.md) - 源码结构详细分析

---

## 🗺️ 学习路线图

```
第1周：算法和基础
  ├─ Flash-Attention 算法理解
  ├─ CUDA 基础回顾
  ├─ CUTLASS/CuTe 入门
  └─ 开发环境搭建

第2周：代码结构分析
  ├─ Python 接口层
  ├─ 参数结构
  ├─ 内核启动机制
  └─ 内核特性系统

第3-4周：前向传播内核
  ├─ compute_attn_1rowblock 分析
  ├─ Q、K、V 加载
  ├─ Online Softmax
  └─ 输出计算

第5-6周：高级优化技术
  ├─ 内存访问优化
  ├─ CUTLASS GEMM 集成
  ├─ 共享内存优化
  └─ 性能分析

第7-8周：反向传播内核
  ├─ 反向传播算法
  ├─ 梯度计算
  ├─ 内存重计算
  └─ 预处理和后处理

第9-10周：Flash-Attention 3
  ├─ SM90 特性
  ├─ FP8 数据类型
  ├─ Paged Attention
  └─ Split-KV 优化
```

---

## 📝 学习笔记目录

### 算法理解
- [ ] Flash-Attention 算法笔记
- [ ] Online Softmax 数学推导
- [ ] Tiling 策略分析

### 代码分析
- [ ] 前向传播内核分析
- [ ] 反向传播内核分析
- [ ] Softmax 实现分析
- [ ] 内存访问模式分析

### 优化技术
- [ ] 内存优化技术总结
- [ ] CUTLASS 使用总结
- [ ] 性能分析报告

---

## 🎯 学习检查点

### 阶段一检查点
- [ ] 能够解释 Flash-Attention 的核心思想
- [ ] 理解 Tiling 策略的作用
- [ ] 理解 Online Softmax 的数学原理

### 阶段二检查点
- [ ] 能够追踪代码调用流程
- [ ] 理解内核启动机制
- [ ] 理解参数结构的作用

### 阶段三检查点
- [ ] 能够解释前向传播的完整流程
- [ ] 理解 Online Softmax 的实现
- [ ] 理解内存访问模式

---

## 📖 参考资源

### 论文
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://tridao.me/publications/flash2/flash2.pdf
- Flash-Attention-3: https://tridao.me/publications/flash3/flash3.pdf

### 代码仓库
- Flash-Attention GitHub: https://github.com/Dao-AILab/flash-attention

### 文档
- CUDA C++ Programming Guide
- CUTLASS 文档
- CuTe 文档

---

**最后更新**：2024年
