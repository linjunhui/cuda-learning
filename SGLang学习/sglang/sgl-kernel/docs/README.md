# SGL Kernel 模块学习文档

本目录包含了 SGL Kernel 各个模块的详细学习文档，按照从简单到困难的顺序组织，帮助您循序渐进地理解每个模块的实现细节。

## 📚 文档结构

### Level 1: 基础模块 (Basics)
适合初学者，包含最基础的 GPU 操作

- [1.1 Elementwise 逐元素操作](./level1-basics/01-elementwise.md) ⭐
- [1.2 Memory 内存管理](./level1-basics/02-memory.md)
- [1.3 Grammar 语法约束](./level1-basics/03-grammar.md)

### Level 2: 中级模块 (Intermediate)
需要理解矩阵运算和量化原理

- [2.1 Quantization 量化](./level2-intermediate/01-quantization.md)
- [2.2 GEMM 矩阵乘法](./level2-intermediate/02-gemm.md)

### Level 3: 高级模块 (Advanced)
需要深入理解 Transformer 架构和序列模型

- [3.1 Attention 注意力机制](./level3-advanced/01-attention.md)
- [3.2 Mamba 状态空间模型](./level3-advanced/02-mamba.md)

### Level 4: 专家级模块 (Expert)
最复杂的模块，涉及分布式系统和高级优化

- [4.1 MOE 专家混合](./level4-expert/01-moe.md)
- [4.2 Speculative 推测性解码](./level4-expert/02-speculative.md)
- [4.3 KVCacheIO KV缓存I/O](./level4-expert/03-kvcacheio.md)
- [4.4 AllReduce 分布式通信](./level4-expert/04-allreduce.md)

## 🎯 学习路径建议

### 初学者路径
1. 从 Level 1 开始，理解基础的 GPU 操作
2. 掌握 Elementwise 操作，理解 CUDA 内核编写基础
3. 学习 Memory 管理，了解 GPU 内存布局
4. 理解 Grammar 约束，掌握基础的应用场景

### 进阶路径
5. 学习 Quantization，理解量化原理和实现
6. 深入 GEMM，掌握矩阵乘法优化技巧
7. 学习 Attention，理解 Transformer 核心机制
8. 研究 Mamba，了解现代序列模型

### 专家路径
9. 深入 MOE，掌握复杂的分支调度
10. 研究 Speculative Decoding，理解高级解码策略
11. 学习 KVCacheIO，掌握高效的内存管理
12. 掌握 AllReduce，理解分布式通信

## 📖 每个文档包含的内容

每个模块的文档都包含以下部分：

1. **模块概述** - 模块的作用和重要性
2. **算子来源** - 算法的来源和理论基础
3. **算法原理** - 详细的数学原理和算法描述
4. **应用场景** - 在 LLM 推理中的用处
5. **代码实现** - 关键代码片段和实现细节
6. **性能优化** - 优化技巧和最佳实践
7. **参考资料** - 相关论文和资源

## 🔍 如何阅读代码

每个文档都会引用实际的代码文件，建议：

1. 打开对应的代码文件（在 `csrc/` 目录下）
2. 结合文档理解代码逻辑
3. 运行相关测试理解行为
4. 查看基准测试了解性能特征

## 💡 学习建议

- **循序渐进**：按照难度顺序学习，不要跳过基础内容
- **动手实践**：阅读代码的同时，尝试运行和理解测试用例
- **深入理解**：每个模块都有其数学原理，不要只停留在表面
- **对比学习**：对比不同实现方式，理解设计选择的原因

## 🛠️ 环境准备

在阅读文档和代码前，请确保：

- 熟悉 CUDA 编程基础
- 了解 PyTorch 的 C++ 扩展机制
- 理解 Transformer 架构
- 熟悉 C++ 模板编程

---

**开始学习**：[Level 1: Elementwise 逐元素操作](./level1-basics/01-elementwise.md)

