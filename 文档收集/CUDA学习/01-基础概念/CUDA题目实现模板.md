# CUDA训练题目详细实现

## 第1周题目实现

### 题目1：第一个CUDA程序

#### 实现模板
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    // TODO: 在这里实现GPU上的Hello World程序
    // 提示：使用printf输出信息，使用threadIdx.x等内置变量
}

int main() {
    printf("Hello from CPU!\n");
    
    // TODO: 启动内核函数
    // 提示：使用<<<gridSize, blockSize>>>语法
    
    // TODO: 等待GPU完成
    // 提示：使用cudaDeviceSynchronize()
    
    return 0;
}
```

#### 测试用例
```cuda
// 期望输出：
// Hello from CPU!
// Hello from GPU! Thread ID: 0
// Hello from GPU! Thread ID: 1
// Hello from GPU! Thread ID: 2
// Hello from GPU! Thread ID: 3
// Hello from GPU! Thread ID: 4
```

#### 知识点检查
- [ ] 理解 `__global__` 关键字的作用
- [ ] 掌握内核函数调用语法 `<<<gridSize, blockSize>>>`
- [ ] 了解 `threadIdx.x`、`blockIdx.x` 等内置变量
- [ ] 理解 `cudaDeviceSynchronize()` 的作用

---

### 题目2：线程索引计算

#### 实现模板
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void calculateGlobalIndex() {
    // TODO: 计算全局索引
    // 提示：globalIdx = blockIdx.x * blockDim.x + threadIdx.x
    
    // TODO: 只让前10个线程打印索引信息
    // 提示：使用条件判断 if (globalIdx < 10)
}

int main() {
    // TODO: 配置网格和线程块
    // 提示：总共1000个线程，每个线程块256个线程
    
    // TODO: 启动内核函数
    
    // TODO: 等待GPU完成
    
    return 0;
}
```

#### 测试用例
```cuda
// 期望输出：
// Global Index: 0
// Global Index: 1
// Global Index: 2
// Global Index: 3
// Global Index: 4
// Global Index: 5
// Global Index: 6
// Global Index: 7
// Global Index: 8
// Global Index: 9
```

#### 知识点检查
- [ ] 掌握线程索引计算公式
- [ ] 理解网格和线程块的关系
- [ ] 了解条件执行的概念
- [ ] 能够计算网格大小

---

### 题目3：GPU信息查询

#### 实现模板
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    
    // TODO: 查询GPU设备数量
    // 提示：使用cudaGetDeviceCount()
    
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    // TODO: 遍历每个设备，显示其信息
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        
        // TODO: 获取设备属性
        // 提示：使用cudaGetDeviceProperties()
        
        // TODO: 显示设备信息
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Dimension: %d\n", prop.maxThreadsDim[0]);
        printf("  Max Grid Size: %d\n", prop.maxGridSize[0]);
        printf("\n");
    }
    
    return 0;
}
```

#### 测试用例
```cuda
// 期望输出（示例）：
// Number of CUDA devices: 1
// Device 0: NVIDIA GeForce RTX 3080
//   Compute Capability: 8.6
//   Total Memory: 10.00 GB
//   Multiprocessors: 68
//   Max Threads per Block: 1024
//   Max Threads per Dimension: 1024
//   Max Grid Size: 2147483647
```

#### 知识点检查
- [ ] 了解CUDA设备管理函数
- [ ] 理解设备属性的含义
- [ ] 掌握计算能力的概念
- [ ] 了解GPU硬件限制

---

## 第2周题目实现

### 题目4：向量加法

#### 实现模板
```cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // TODO: 计算线程索引
    // 提示：int i = blockIdx.x * blockDim.x + threadIdx.x
    
    // TODO: 检查边界条件
    // 提示：if (i < n)
    
    // TODO: 执行向量加法
    // 提示：c[i] = a[i] + b[i]
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    // TODO: 分配主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // TODO: 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // TODO: 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // TODO: 传输数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // TODO: 配置内核参数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // TODO: 启动内核函数
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // TODO: 等待GPU完成
    cudaDeviceSynchronize();
    
    // TODO: 传输结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // TODO: 验证结果
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            printf("Error at index %d: %f\n", i, h_c[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Vector addition completed successfully!\n");
    }
    
    // TODO: 清理内存
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

#### 测试用例
```cuda
// 期望输出：
// Vector addition completed successfully!
```

#### 知识点检查
- [ ] 掌握CUDA内存管理
- [ ] 理解数据传输方向
- [ ] 了解内核参数配置
- [ ] 掌握错误检查方法

---

### 题目5：向量点积

#### 实现模板
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dotProduct(float *a, float *b, float *result, int n) {
    // TODO: 使用共享内存存储部分结果
    __shared__ float sdata[256];
    
    // TODO: 计算线程索引
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: 计算部分点积
    sdata[tid] = (i < n) ? a[i] * b[i] : 0.0f;
    
    // TODO: 同步线程
    __syncthreads();
    
    // TODO: 在共享内存中进行归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // TODO: 将结果写入全局内存
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    // TODO: 分配和初始化内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_result = (float*)malloc(sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, sizeof(float));
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // TODO: 配置内核参数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // TODO: 启动内核函数
    dotProduct<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);
    
    // TODO: 等待GPU完成
    cudaDeviceSynchronize();
    
    // TODO: 传输结果回主机
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // TODO: 验证结果
    float expected = n * 2.0f;  // 1.0 * 2.0 * n
    printf("Dot product result: %f\n", *h_result);
    printf("Expected result: %f\n", expected);
    
    // TODO: 清理内存
    free(h_a); free(h_b); free(h_result);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
    
    return 0;
}
```

#### 测试用例
```cuda
// 期望输出：
// Dot product result: 20000.000000
// Expected result: 20000.000000
```

#### 知识点检查
- [ ] 理解归约算法
- [ ] 掌握共享内存使用
- [ ] 了解线程同步机制
- [ ] 理解点积计算原理

---

## 编译和运行说明

### 编译命令
```bash
# 编译CUDA程序
nvcc -o program_name program_name.cu

# 运行程序
./program_name
```

### 调试技巧
1. 使用 `printf` 在内核函数中输出调试信息
2. 使用 `cudaGetLastError()` 检查CUDA错误
3. 使用 `cuda-gdb` 进行调试
4. 逐步验证程序的正确性

### 性能测试
1. 使用 `clock()` 函数测量执行时间
2. 比较CPU和GPU版本的性能
3. 使用 `nsight compute` 分析性能瓶颈
4. 优化内存访问模式

---

**注意**：这些题目按照难度递增设计，建议按顺序完成。每个题目都有详细的实现模板和测试用例，帮助你更好地理解CUDA编程概念。
