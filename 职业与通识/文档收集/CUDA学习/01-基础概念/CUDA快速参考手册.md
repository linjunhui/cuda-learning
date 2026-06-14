# CUDA编程快速参考手册

## 基本语法

### 内核函数定义
```cuda
__global__ void kernelFunction(参数列表) {
    // 内核函数代码
}
```

### 内核函数调用
```cuda
kernelFunction<<<gridSize, blockSize>>>(参数列表);
```

### 内存管理
```cuda
// 分配设备内存
cudaMalloc(&device_ptr, size);

// 释放设备内存
cudaFree(device_ptr);

// 数据传输
cudaMemcpy(dst, src, size, direction);
```

### 同步操作
```cuda
// 等待GPU完成
cudaDeviceSynchronize();

// 线程块内同步
__syncthreads();
```

## 内置变量

### 线程索引
- `threadIdx.x`：线程块内线程索引
- `blockIdx.x`：网格内线程块索引
- `blockDim.x`：线程块大小
- `gridDim.x`：网格大小

### 全局索引计算
```cuda
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
```

## 内存类型

### 全局内存
- 所有线程可访问
- 访问速度较慢
- 容量大

### 共享内存
- 线程块内共享
- 访问速度快
- 容量小（通常48KB）

### 寄存器内存
- 线程私有
- 访问速度最快
- 容量很小

## 常用函数

### 错误检查
```cuda
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
}
```

### 设备信息
```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, deviceId);
```

### 内存复制方向
- `cudaMemcpyHostToDevice`：主机到设备
- `cudaMemcpyDeviceToHost`：设备到主机
- `cudaMemcpyDeviceToDevice`：设备到设备

## 性能优化技巧

### 内存合并访问
```cuda
// 好的访问模式（连续访问）
for (int i = 0; i < n; i++) {
    result[i] = input[i] * 2;
}

// 避免的访问模式（随机访问）
for (int i = 0; i < n; i++) {
    result[i] = input[random_index[i]] * 2;
}
```

### 共享内存使用
```cuda
__global__ void optimizedKernel(float *data) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = data[i];
    __syncthreads();
    
    // 使用共享内存进行计算
    // ...
}
```

### 线程块大小选择
- 通常是32的倍数（warp大小）
- 常见选择：128、256、512
- 考虑共享内存使用量

## 调试技巧

### 使用printf调试
```cuda
__global__ void debugKernel() {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (tid == 0 && bid == 0) {
        printf("Block %d, Thread %d\n", bid, tid);
    }
}
```

### 错误处理宏
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 使用示例
CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

## 编译和运行

### 编译命令
```bash
# 基本编译
nvcc -o program program.cu

# 指定计算能力
nvcc -arch=sm_75 -o program program.cu

# 调试版本
nvcc -g -G -o program program.cu

# 优化版本
nvcc -O3 -o program program.cu
```

### 运行命令
```bash
# 基本运行
./program

# 使用CUDA调试器
cuda-gdb ./program

# 性能分析
nsight compute ./program
```

## 常见错误和解决方案

### 内存分配失败
```cuda
// 检查错误
cudaError_t error = cudaMalloc(&ptr, size);
if (error != cudaSuccess) {
    printf("Memory allocation failed: %s\n", cudaGetErrorString(error));
}
```

### 内核启动失败
```cuda
// 检查内核启动
kernel<<<gridSize, blockSize>>>(args);
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
}
```

### 数据传输错误
```cuda
// 检查数据传输
cudaError_t error = cudaMemcpy(dst, src, size, direction);
if (error != cudaSuccess) {
    printf("Memory copy failed: %s\n", cudaGetErrorString(error));
}
```

## 性能分析工具

### nvcc编译选项
```bash
# 生成性能分析信息
nvcc -lineinfo -o program program.cu

# 生成调试信息
nvcc -g -G -o program program.cu
```

### 性能测量
```cuda
#include <time.h>

clock_t start = clock();
// CUDA代码
cudaDeviceSynchronize();
clock_t end = clock();

double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
printf("Execution time: %f seconds\n", time_spent);
```

## 最佳实践

### 代码组织
1. 将内核函数和主机代码分开
2. 使用有意义的变量名
3. 添加适当的注释
4. 检查所有CUDA函数调用

### 内存管理
1. 及时释放分配的内存
2. 检查内存分配是否成功
3. 使用适当的内存类型
4. 优化内存访问模式

### 性能优化
1. 使用共享内存减少全局内存访问
2. 确保内存合并访问
3. 选择合适的线程块大小
4. 使用异步执行提高并发性

---

**更新日期**：2024-01-15  
**版本**：1.0  
**适用CUDA版本**：11.0+
