# CUDA/C++ 算子练习题目集

本目录包含一系列简单但完整的 CUDA 和 C++ 练习题目，旨在帮助理解 CUDA 编程和 C++ 面向对象编程的基础知识。

## 题目列表

### 基础题目

1. **[向量加法](./01_向量加法.md)** ⭐
   - CUDA 内核函数基础
   - 内存管理
   - 主机-设备数据传输
   - **难度**：★☆☆☆☆

2. **[数组元素平方](./02_数组元素平方.md)** ⭐
   - CUDA 内核函数
   - 错误处理
   - 边界检查
   - **难度**：★☆☆☆☆

### 进阶题目

3. **[数组最大值（归约）](./03_数组最大值.md)** ⭐⭐
   - 归约算法
   - 共享内存
   - 线程同步
   - **难度**：★★☆☆☆

4. **[矩阵转置](./04_矩阵转置.md)** ⭐⭐
   - 二维线程索引
   - 共享内存优化
   - 内存访问模式
   - **难度**：★★☆☆☆

### C++ 封装题目

5. **[C++ 类封装](./05_C++类封装.md)** ⭐⭐
   - C++ 类和对象
   - RAII 原则
   - 资源管理
   - **难度**：★★☆☆☆

## 学习路径建议

### 初学者路径
1. 从题目 01（向量加法）开始
2. 完成题目 02（数组元素平方）
3. 理解基本的 CUDA 编程模式

### 进阶路径
1. 完成基础题目后
2. 学习题目 03（归约算法）
3. 学习题目 04（矩阵转置）
4. 理解性能优化技巧

### 高级路径
1. 完成所有基础题目
2. 学习题目 05（C++ 封装）
3. 尝试扩展练习
4. 实现自己的算子

## 通用要求

### 编译环境
- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit（建议 11.0+）
- GCC 或 Clang 编译器

### 编译命令
```bash
# 基本编译
nvcc -o program_name program_name.cu

# 带调试信息
nvcc -g -G -o program_name program_name.cu

# 优化编译
nvcc -O3 -o program_name program_name.cu

# 指定计算能力
nvcc -arch=sm_75 -o program_name program_name.cu
```

### 运行和调试
```bash
# 运行程序
./program_name

# 使用 cuda-gdb 调试
cuda-gdb ./program_name

# 检查 CUDA 错误
# 在代码中使用 cudaGetLastError()
```

## 知识点检查清单

完成每个题目后，检查以下知识点：

### CUDA 基础
- [ ] 理解 `__global__`、`__device__`、`__host__` 关键字
- [ ] 掌握线程索引计算（`threadIdx`, `blockIdx`, `blockDim`, `gridDim`）
- [ ] 理解网格和线程块的概念
- [ ] 掌握内核函数调用语法 `<<<gridSize, blockSize>>>`

### 内存管理
- [ ] 理解主机内存和设备内存的区别
- [ ] 掌握 `cudaMalloc`、`cudaFree`、`cudaMemcpy` 的使用
- [ ] 理解不同的内存传输方向
- [ ] 掌握内存对齐和合并访问的概念

### 共享内存
- [ ] 理解共享内存的作用和限制
- [ ] 掌握 `__shared__` 关键字
- [ ] 理解 `__syncthreads()` 的作用
- [ ] 了解 bank conflict 和如何避免

### 性能优化
- [ ] 理解合并访问的重要性
- [ ] 掌握如何使用共享内存优化性能
- [ ] 了解归约算法的优化技巧
- [ ] 理解线程块大小的选择

### C++ 基础
- [ ] 理解类和对象的概念
- [ ] 掌握构造函数和析构函数
- [ ] 理解 RAII 原则
- [ ] 了解拷贝构造和移动语义

## 常见问题

### Q1: 编译错误 "undefined reference to cudaMalloc"
**A**: 确保链接 CUDA 运行时库，使用 `nvcc` 编译器会自动处理。

### Q2: 运行时错误 "invalid device function"
**A**: 检查 GPU 的计算能力，使用 `-arch=sm_XX` 指定正确的架构。

### Q3: 结果不正确
**A**: 
- 检查边界条件（`if (idx < n)`）
- 检查内存传输方向
- 使用 `cudaDeviceSynchronize()` 确保内核完成
- 检查线程索引计算

### Q4: 性能不佳
**A**:
- 检查内存访问模式（是否合并访问）
- 考虑使用共享内存
- 调整线程块大小
- 使用 `nsight compute` 分析性能

## 扩展资源

### 官方文档
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 工具
- **Nsight Compute**: 性能分析工具
- **Nsight Systems**: 系统级性能分析
- **cuda-gdb**: CUDA 调试器
- **nvprof**: 性能分析器（已弃用，推荐 Nsight）

### 学习资源
- CUDA 官方示例代码
- SGLang 源码中的 CUDA 实现
- 其他开源 CUDA 项目

## 提交和反馈

完成题目后，建议：
1. 验证所有测试用例通过
2. 检查代码风格和注释
3. 尝试扩展练习
4. 记录遇到的问题和解决方案

## 更新日志

- 2024-XX-XX: 初始版本，包含 5 个基础题目







