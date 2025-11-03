# CUDA基础概念学习

## 学习目标

理解CUDA编程模型，掌握GPU并行计算的基本概念和编程方法。

## 学习时间

1个月（30天）

## 学习内容

### 第1周：GPU架构基础
- GPU与CPU的区别
- GPU硬件架构
- 流多处理器（SM）
- 内存层次结构
- 计算能力

### 第2周：CUDA编程模型
- 主机（Host）和设备（Device）
- 线程、块、网格概念
- 线程索引计算
- 内核函数（Kernel）
- 内存空间

### 第3周：CUDA内存模型
- 全局内存
- 共享内存
- 寄存器内存
- 常量内存
- 纹理内存

### 第4周：基本CUDA语法
- CUDA C/C++语法
- 内核函数调用
- 内存分配和释放
- 错误处理
- 编译和运行

## 实践项目

### 项目1：Hello CUDA（第1周）
编写第一个CUDA程序：
- 在GPU上打印Hello World
- 理解主机和设备交互
- 掌握基本的CUDA程序结构

### 项目2：向量加法（第2周）
实现GPU并行向量加法：
- 理解线程索引计算
- 掌握内核函数编写
- 比较CPU和GPU性能

### 项目3：矩阵运算（第3-4周）
实现简单的矩阵运算：
- 矩阵加法
- 矩阵乘法
- 性能分析和优化

## 学习资源

### 书籍
- 《CUDA编程指南》
- 《GPU高性能编程CUDA实战》

### 在线资源
- NVIDIA CUDA官方文档
- CUDA在线教程
- NVIDIA开发者博客

### 开发工具
- CUDA Toolkit
- NVIDIA Nsight Compute
- Visual Studio Code

## 每日学习计划

### 工作日（2-3小时）
- 理论学习：1小时
- 编程实践：1-2小时

### 周末（4-6小时）
- 理论学习：2小时
- 编程实践：2-4小时

## 检查点

### 第1周检查点
- [ ] 理解GPU架构基础
- [ ] 掌握CUDA编程模型
- [ ] 完成Hello CUDA程序

### 第2周检查点
- [ ] 掌握线程、块、网格概念
- [ ] 能够编写简单的内核函数
- [ ] 完成向量加法项目

### 第3周检查点
- [ ] 理解CUDA内存模型
- [ ] 掌握内存分配和释放
- [ ] 完成矩阵加法项目

### 第4周检查点
- [ ] 掌握基本CUDA语法
- [ ] 能够编写和调试CUDA程序
- [ ] 完成矩阵乘法项目

## 代码示例

### Hello CUDA示例
```cuda
#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU! Thread ID: %d\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    
    // 启动内核函数
    helloFromGPU<<<1, 5>>>();
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    return 0;
}
```

### 向量加法示例
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(float);
    
    // 主机内存分配
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // 设备内存分配
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 数据传输
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 启动内核函数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // 结果传输回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            printf("Error at index %d: %f\n", i, h_c[i]);
            return 1;
        }
    }
    printf("Vector addition completed successfully!\n");
    
    // 清理内存
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

## 硬件要求

### 推荐配置
- NVIDIA GPU（支持CUDA 11.0+）
- 4GB以上显存
- 8GB以上系统内存
- Linux/Windows操作系统

### 开发环境
- CUDA Toolkit 11.0+
- GCC/G++编译器
- CMake构建系统

## 常见问题

### Q: 如何确定线程块大小？
A: 线程块大小通常是32的倍数（warp大小），常见的选择是128、256、512。

### Q: 如何计算网格大小？
A: 网格大小 = (总线程数 + 线程块大小 - 1) / 线程块大小

### Q: 如何调试CUDA程序？
A: 使用cuda-gdb调试器，或者在内核函数中使用printf输出调试信息。

### Q: 内存分配失败怎么办？
A: 检查显存是否足够，使用cudaGetLastError()检查错误，确保正确释放内存。

---

**学习开始时间**：2024-01-15  
**预计完成时间**：2024-02-15
