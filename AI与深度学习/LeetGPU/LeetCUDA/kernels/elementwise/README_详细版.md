# Elementwise 目录文档索引

## 📚 文档列表

### 核心文档

1. **[核函数详解文档.md](./核函数详解文档.md)** ⭐
   - 详细讲解所有 6 个核函数的实现
   - 从基础到高级的优化路径
   - 性能分析和对比
   - **推荐先阅读此文档**

2. **[README.md](./README.md)**
   - 项目总览
   - 快速开始指南
   - 性能测试结果

3. **[CUDA_DATA_TYPES.md](./CUDA_DATA_TYPES.md)**
   - CUDA 向量化数据类型详解
   - float4、half2 等类型的使用

4. **[COMPILE.md](./COMPILE.md)**
   - 编译详细说明
   - 依赖安装指南
   - 故障排除

## 🎯 快速导航

### 按学习路径

**初学者路径**：
1. 阅读 [README.md](./README.md) 了解项目
2. 查看 [核函数详解文档.md](./核函数详解文档.md) 的第 1 节（FP32 标量版本）
3. 理解基础的 CUDA 编程概念
4. 逐步学习向量化优化

**进阶路径**：
1. 深入学习 [核函数详解文档.md](./核函数详解文档.md) 的所有章节
2. 理解每种优化技术的原理
3. 查看 [CUDA_DATA_TYPES.md](./CUDA_DATA_TYPES.md) 了解数据类型
4. 尝试修改代码并测试性能

**专家路径**：
1. 理解打包优化版本的每个细节
2. 探索进一步的优化空间
3. 实现其他 elementwise 操作
4. 应用到实际项目中

### 按功能分类

**核函数实现**：
- [核函数详解文档.md](./核函数详解文档.md) - 所有核函数详解

**数据类型**：
- [CUDA_DATA_TYPES.md](./CUDA_DATA_TYPES.md) - 向量化类型详解

**编译和使用**：
- [README.md](./README.md) - 快速开始
- [COMPILE.md](./COMPILE.md) - 编译指南

## 📖 核函数列表

### FP32 版本

1. **elementwise_add_f32_kernel**
   - 基础标量版本
   - 每个线程处理 1 个元素
   - 详见：[核函数详解文档.md#1](./核函数详解文档.md#1-elementwise_add_f32_kernel---fp32-标量版本)

2. **elementwise_add_f32x4_kernel**
   - 向量化版本（float4）
   - 每个线程处理 4 个元素
   - 详见：[核函数详解文档.md#2](./核函数详解文档.md#2-elementwise_add_f32x4_kernel---fp32-向量化版本)

### FP16 版本

3. **elementwise_add_f16_kernel**
   - 基础标量版本
   - 每个线程处理 1 个元素
   - 详见：[核函数详解文档.md#3](./核函数详解文档.md#3-elementwise_add_f16_kernel---fp16-标量版本)

4. **elementwise_add_f16x2_kernel**
   - 向量化版本（half2）
   - 每个线程处理 2 个元素
   - 详见：[核函数详解文档.md#4](./核函数详解文档.md#4-elementwise_add_f16x2_kernel---fp16-向量化版本2-元素)

5. **elementwise_add_f16x8_kernel**
   - 大规模向量化版本
   - 每个线程处理 8 个元素
   - 详见：[核函数详解文档.md#5](./核函数详解文档.md#5-elementwise_add_f16x8_kernel---fp16-向量化版本8-元素)

6. **elementwise_add_f16x8_pack_kernel** ⭐
   - 打包优化版本（最优）
   - 每个线程处理 8 个元素，128 位对齐
   - 详见：[核函数详解文档.md#6](./核函数详解文档.md#6-elementwise_add_f16x8_pack_kernel---fp16-打包优化版本-)

## 💡 学习建议

### 1. 循序渐进

- **不要跳过基础**：理解标量版本是理解优化的基础
- **逐步深入**：按顺序学习每个优化版本
- **对比学习**：对比不同版本的差异

### 2. 动手实践

- **运行代码**：实际运行测试脚本
- **修改实验**：尝试修改代码观察性能变化
- **测量性能**：使用性能分析工具（nsight）

### 3. 深入理解

- **阅读代码**：仔细阅读每个内核的实现
- **理解原理**：理解为什么这样优化
- **分析性能**：分析性能瓶颈和优化空间

## 🔗 相关资源

- [CUDA 官方文档](https://docs.nvidia.com/cuda/)
- [PyTorch C++ 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**开始学习**：[核函数详解文档.md](./核函数详解文档.md)



