# Elementwise 目录文档

本目录包含了所有逐元素操作的 CUDA 内核实现。这些操作是 SGL Kernel 中最基础的部分，但在大语言模型推理中频繁使用。

## 📂 文件列表

### 核心实现文件

1. **[activation.cu](./activation.cu文档.md)** - 激活函数实现
   - SiLU (Swish) 激活
   - GELU 及其变体
   - 融合激活+乘法操作

2. **[cast.cu](./cast.cu文档.md)** - 类型转换操作
   - FP8 下转换（downcast）
   - 类型转换模板特化

3. **[copy.cu](./copy.cu文档.md)** - 复制操作
   - CPU 到 GPU 的零拷贝传输
   - 优化的内存复制内核

4. **[concat_mla.cu](./concat_mla.cu文档.md)** - MLA 连接操作
   - Multi-head Latent Attention 相关的数据连接
   - 高效的向量化内存操作

5. **[fused_add_rms_norm_kernel.cu](./fused_add_rms_norm_kernel.cu文档.md)** - 融合 RMS 归一化
   - 加法 + RMSNorm 的融合实现
   - 性能优化的归一化操作

6. **[rope.cu](./rope.cu文档.md)** - 旋转位置编码
   - RoPE (Rotary Position Embedding) 实现
   - 基于 FlashInfer 的高性能实现

7. **[topk.cu](./topk.cu文档.md)** - Top-K 选择
   - 高效的 Top-K 元素选择算法
   - Radix Sort 优化实现

### 头文件

8. **[pos_enc.cuh](./pos_enc.cuh文档.md)** - 位置编码头文件
   - 位置编码相关的辅助函数和模板

9. **[utils.cuh](./utils.cuh文档.md)** - 工具函数头文件
   - 内存访问优化工具
   - Warp 级操作的辅助函数

## 🎯 模块功能概述

Elementwise 模块提供了以下类型的操作：

### 1. 激活函数
- 支持 SiLU、GELU 等多种激活函数
- 融合操作：激活+乘法，减少内存访问

### 2. 归一化操作
- RMSNorm 实现
- 融合的残差连接+归一化

### 3. 位置编码
- RoPE 旋转位置编码
- 高效的 cos/sin 缓存机制

### 4. 选择操作
- Top-K 选择
- 高效的排序算法

### 5. 类型转换
- FP8 量化支持
- 类型转换优化

### 6. 数据操作
- 内存复制
- 数据连接和重组

## 📊 性能特征

所有操作都针对以下方面进行了优化：

- ✅ **向量化内存访问**：使用 128 位对齐的向量类型
- ✅ **融合操作**：减少内存访问次数
- ✅ **Warp 级协作**：充分利用 GPU 的 SIMT 特性
- ✅ **共享内存优化**：减少全局内存访问
- ✅ **类型特化**：针对不同数据类型优化

## 🔗 相关文档

- [Level 1.1 Elementwise 详细文档](../../docs/level1-basics/01-elementwise.md)
- [csrc 目录分析文档](../../csrc目录分析文档.md)

## 📚 参考资料

- FlashInfer 项目：https://github.com/flashinfer-ai/flashinfer
- TileLang 项目：https://github.com/tile-ai/tilelang
- DeepEP 项目：https://github.com/deepseek-ai/DeepEP

---

**详细文档请查看每个文件对应的文档**

