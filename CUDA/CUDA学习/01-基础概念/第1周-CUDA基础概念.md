# 第1周：CUDA基础概念

## 学习目标

理解CUDA编程模型，掌握GPU并行计算的基本概念和编程方法。

## 学习内容

### 1. GPU架构基础

#### 1.1 GPU与CPU的区别

**CPU（中央处理器）的特点：**
- **核心数量少**：通常有2-16个核心
- **每个核心功能强大**：支持复杂指令集，有分支预测、乱序执行等高级功能
- **大容量缓存**：L1、L2、L3缓存容量大，访问延迟低
- **适合串行处理**：专门优化单线程性能
- **低延迟**：单个任务响应时间短
- **通用性强**：可以处理各种类型的计算任务

**GPU（图形处理器）的特点：**
- **核心数量多**：有数百到数千个核心
- **每个核心功能简单**：专门为并行计算设计，指令集相对简单
- **小容量缓存**：缓存容量小，但带宽高
- **适合并行处理**：专门优化大规模并行计算
- **高吞吐量**：单位时间内处理的数据量大
- **专用性强**：特别适合数据并行的计算任务

**为什么GPU适合并行计算？**
1. **SIMD架构**：Single Instruction, Multiple Data，一条指令同时处理多个数据
2. **内存带宽高**：GPU有更高的内存带宽，适合大量数据的读写
3. **线程切换开销小**：GPU的线程切换比CPU快得多
4. **计算密度高**：在相同面积内提供更多的计算单元

```cuda
// CPU vs GPU 架构对比示例
/*
CPU特点：
- 少量核心，每个核心功能强大
- 大容量缓存
- 适合串行处理
- 低延迟

GPU特点：
- 大量核心，每个核心功能简单
- 小容量缓存
- 适合并行处理
- 高吞吐量
*/
```

#### 1.2 GPU硬件架构

**GPU的层次化架构：**

GPU采用层次化的架构设计，从大到小分为以下几个层次：

**1. GPU芯片级别：**
- **全局内存（Global Memory）**：GPU的主要存储空间，容量大但访问延迟高
- **L2缓存**：连接全局内存和各个SM的高速缓存
- **常量内存（Constant Memory）**：只读内存，有专门的缓存
- **纹理内存（Texture Memory）**：专门用于图像处理的内存类型

**2. 流多处理器（SM - Streaming Multiprocessor）级别：**
- **CUDA核心**：GPU的基本计算单元，执行浮点运算和整数运算
- **共享内存（Shared Memory）**：SM内所有线程共享的高速内存
- **寄存器文件**：每个线程私有的存储空间
- **调度器**：负责线程调度和指令分发
- **特殊功能单元**：如Tensor Core（用于深度学习加速）

**3. 线程级别：**
- **线程**：GPU执行的最小单位
- **Warp**：32个线程组成一个warp，是GPU调度的基本单位
- **寄存器**：每个线程私有的存储空间

**关键概念解释：**

**流多处理器（SM）：**
- SM是GPU的核心计算单元
- 每个SM包含多个CUDA核心（通常16-128个）
- SM内部有共享内存和寄存器文件
- 多个SM可以并行执行不同的任务

**CUDA核心：**
- GPU的基本计算单元
- 可以执行浮点运算、整数运算等
- 现代GPU的CUDA核心还支持特殊运算（如Tensor Core）

**共享内存：**
- SM内所有线程共享的内存
- 访问速度比全局内存快得多
- 容量有限（通常16-128KB）
- 需要程序员显式管理

```cuda
// GPU硬件层次结构示例
/*
GPU
├── 流多处理器 (SM - Streaming Multiprocessor)
│   ├── CUDA核心 (CUDA Cores)
│   ├── 共享内存 (Shared Memory)
│   ├── 寄存器文件 (Register File)
│   └── 调度器 (Scheduler)
├── 全局内存 (Global Memory)
├── 常量内存 (Constant Memory)
└── 纹理内存 (Texture Memory)
*/
```

#### 1.3 计算能力

**什么是计算能力（Compute Capability）？**

计算能力是NVIDIA用来标识GPU架构和功能的版本号，它决定了GPU支持哪些CUDA功能。

**计算能力的作用：**
1. **功能支持**：不同计算能力支持不同的CUDA功能
2. **性能特征**：影响GPU的性能表现和优化策略
3. **兼容性**：决定CUDA程序能否在特定GPU上运行

**主要GPU架构和计算能力：**

**Kepler架构（计算能力 3.0-3.7）：**
- 代表产品：GTX 680, GTX 780, Tesla K20
- 特点：引入了动态并行、Hyper-Q等新功能
- 适用：基础CUDA编程学习

**Maxwell架构（计算能力 5.0-5.2）：**
- 代表产品：GTX 980, GTX 1080
- 特点：功耗效率大幅提升，支持更多CUDA功能
- 适用：性能优化和高级特性学习

**Pascal架构（计算能力 6.0-6.1）：**
- 代表产品：GTX 1080, GTX 1080 Ti, Tesla P100
- 特点：引入统一内存、Pascal架构优化
- 适用：现代CUDA开发

**Volta架构（计算能力 7.0-7.2）：**
- 代表产品：Tesla V100, Titan V
- 特点：引入Tensor Core，支持混合精度计算
- 适用：深度学习和AI应用

**Turing架构（计算能力 7.5）：**
- 代表产品：RTX 2080, RTX 3080
- 特点：支持光线追踪，RT Core
- 适用：图形渲染和游戏开发

**Ampere架构（计算能力 8.0-8.6）：**
- 代表产品：RTX 3080, RTX 4080, A100
- 特点：第三代Tensor Core，支持更多AI功能
- 适用：现代AI和深度学习

**Hopper架构（计算能力 9.0）：**
- 代表产品：H100
- 特点：第四代Tensor Core，Transformer Engine
- 适用：大规模AI训练

**如何查看GPU计算能力：**
```cuda
// 查看GPU计算能力的代码示例
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("---\n");
    }
    
    return 0;
}
```

**选择GPU的建议：**
- **学习阶段**：计算能力6.0以上的GPU即可
- **开发阶段**：建议使用计算能力7.0以上的GPU
- **生产环境**：根据具体应用需求选择

### 2. CUDA编程模型

#### 2.1 主机和设备

**CUDA编程模型的核心概念：**

CUDA编程模型基于**主机-设备**（Host-Device）架构，这是理解CUDA编程的基础。

**主机（Host）：**
- **定义**：运行CPU的计算机系统
- **作用**：负责程序控制、数据准备、结果处理
- **特点**：执行串行代码，管理GPU资源
- **内存**：主机内存（Host Memory），CPU可以直接访问

**设备（Device）：**
- **定义**：GPU及其相关硬件
- **作用**：执行并行计算任务
- **特点**：执行CUDA内核函数，处理大量并行数据
- **内存**：设备内存（Device Memory），GPU可以直接访问

**主机和设备的关系：**
1. **分工明确**：主机负责控制，设备负责计算
2. **内存分离**：主机内存和设备内存是独立的
3. **数据传输**：需要显式地在主机和设备间传输数据
4. **异步执行**：主机和设备可以并行工作

**为什么需要主机-设备分离？**
- **性能优化**：避免CPU和GPU之间的频繁交互
- **内存管理**：GPU内存访问模式与CPU不同
- **并行执行**：CPU和GPU可以同时工作
- **资源管理**：更好地控制GPU资源的使用

**实际应用场景：**
```cuda
// 主机-设备编程示例
#include <stdio.h>
#include <cuda_runtime.h>

// 设备函数（在GPU上执行）
__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

// 主机函数（在CPU上执行）
int main() {
    // 主机代码
    printf("Hello from CPU!\n");
    
    // 启动设备代码
    helloFromGPU<<<1, 1>>>();
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    return 0;
}
```

**关键概念解释：**

**`__global__`关键字：**
- 表示这是一个CUDA内核函数
- 可以在主机上调用，在设备上执行
- 是连接主机和设备的桥梁

**`<<<1, 1>>>`语法：**
- 这是CUDA内核启动语法
- 第一个参数：网格大小（Grid Size）
- 第二个参数：线程块大小（Block Size）
- 这里表示启动1个线程块，每个线程块包含1个线程

**`cudaDeviceSynchronize()`：**
- 同步函数，等待GPU完成所有操作
- 确保主机和设备之间的同步
- 避免主机在GPU完成之前继续执行

#### 2.2 线程层次结构
```cuda
// CUDA线程层次结构
/*
Grid (网格)
├── Block 0
│   ├── Thread (0,0) Thread (0,1) Thread (0,2)
│   ├── Thread (1,0) Thread (1,1) Thread (1,2)
│   └── Thread (2,0) Thread (2,1) Thread (2,2)
├── Block 1
│   ├── Thread (0,0) Thread (0,1) Thread (0,2)
│   ├── Thread (1,0) Thread (1,1) Thread (1,2)
│   └── Thread (2,0) Thread (2,1) Thread (2,2)
└── Block 2
    ├── Thread (0,0) Thread (0,1) Thread (0,2)
    ├── Thread (1,0) Thread (1,1) Thread (1,2)
    └── Thread (2,0) Thread (2,1) Thread (2,2)
*/
```

#### 2.2 线程层次结构

**CUDA线程层次结构的重要性：**

CUDA的线程层次结构是理解GPU并行计算的核心概念。与CPU的单线程模型不同，GPU使用大规模并行线程模型。

**线程层次结构的三级模型：**

**1. 网格（Grid）：**
- **定义**：整个CUDA内核启动的所有线程块的集合
- **特点**：一个网格包含多个线程块
- **作用**：定义整个并行任务的规模
- **限制**：网格大小受GPU硬件限制

**2. 线程块（Block）：**
- **定义**：一组协作的线程
- **特点**：线程块内的线程可以共享内存和同步
- **作用**：线程协作的基本单位
- **限制**：每个线程块最多1024个线程（取决于GPU）

**3. 线程（Thread）：**
- **定义**：GPU执行的最小单位
- **特点**：每个线程有唯一的标识符
- **作用**：执行具体的计算任务
- **限制**：线程数量受硬件资源限制

**线程层次结构的优势：**

**1. 可扩展性：**
- 可以根据问题规模调整线程数量
- 支持从小规模到大规模的计算任务

**2. 协作性：**
- 线程块内的线程可以协作
- 支持共享内存和同步操作

**3. 硬件映射：**
- 线程块映射到SM（流多处理器）
- 线程映射到CUDA核心

**线程标识符系统：**

CUDA提供了内置变量来标识线程：
- `threadIdx`：线程在块内的索引
- `blockIdx`：线程块在网格内的索引
- `blockDim`：线程块的维度
- `gridDim`：网格的维度

**实际应用示例：**
```cuda
// CUDA线程层次结构示例
__global__ void threadIndexDemo() {
    // 线程索引
    int tid = threadIdx.x;                    // 块内线程ID
    int bid = blockIdx.x;                     // 块ID
    int gid = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程ID
    
    // 多维索引
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    printf("Thread (%d,%d) in Block (%d,%d) has global ID (%d,%d)\n",
           tid_x, tid_y, bid_x, bid_y, gid_x, gid_y);
}

// 启动内核
int main() {
    // 启动2D网格，每个维度2个块，每个块3个线程
    dim3 blockSize(3, 3);
    dim3 gridSize(2, 2);
    
    threadIndexDemo<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
```

**线程层次结构的可视化：**
```
Grid (网格)
├── Block 0
│   ├── Thread (0,0) Thread (0,1) Thread (0,2)
│   ├── Thread (1,0) Thread (1,1) Thread (1,2)
│   └── Thread (2,0) Thread (2,1) Thread (2,2)
├── Block 1
│   ├── Thread (0,0) Thread (0,1) Thread (0,2)
│   ├── Thread (1,0) Thread (1,1) Thread (1,2)
│   └── Thread (2,0) Thread (2,1) Thread (2,2)
└── Block 2
    ├── Thread (0,0) Thread (0,1) Thread (0,2)
    ├── Thread (1,0) Thread (1,1) Thread (1,2)
    └── Thread (2,0) Thread (2,1) Thread (2,2)
```

**选择线程块大小的考虑因素：**

**1. 硬件限制：**
- 每个SM的最大线程数
- 每个线程块的最大线程数
- 共享内存大小

**2. 性能优化：**
- 占用率（Occupancy）
- 内存访问模式
- 计算密度

**3. 算法特性：**
- 数据依赖性
- 同步需求
- 内存访问模式

### 3. CUDA内存模型

#### 3.1 内存层次结构
```cuda
// CUDA内存层次结构
/*
GPU内存层次（从快到慢）：
1. 寄存器 (Registers) - 最快，每个线程私有
2. 共享内存 (Shared Memory) - 块内共享
3. L1缓存 (L1 Cache) - 自动管理
4. L2缓存 (L2 Cache) - 全局共享
5. 全局内存 (Global Memory) - 最慢，所有线程可访问
6. 常量内存 (Constant Memory) - 只读，有缓存
7. 纹理内存 (Texture Memory) - 只读，有缓存
*/
```

### 3. CUDA内存模型

#### 3.1 内存层次结构

**CUDA内存模型的重要性：**

CUDA内存模型是GPU编程的核心概念之一。与CPU的简单内存模型不同，GPU有复杂的内存层次结构，理解这个结构对于编写高效的CUDA程序至关重要。

**CUDA内存层次结构（从快到慢）：**

**1. 寄存器（Registers）：**
- **速度**：最快
- **容量**：每个线程私有，数量有限
- **作用**：存储局部变量和临时数据
- **特点**：编译器自动管理，程序员无法直接控制
- **限制**：寄存器溢出会影响性能

**2. 共享内存（Shared Memory）：**
- **速度**：很快
- **容量**：每个线程块共享，通常16-128KB
- **作用**：线程块内线程协作的数据交换
- **特点**：程序员显式管理
- **优势**：比全局内存快100倍以上

**3. L1缓存（L1 Cache）：**
- **速度**：快
- **容量**：自动管理
- **作用**：缓存频繁访问的数据
- **特点**：硬件自动管理
- **优势**：提高内存访问效率

**4. L2缓存（L2 Cache）：**
- **速度**：中等
- **容量**：较大，全局共享
- **作用**：缓存全局内存数据
- **特点**：硬件自动管理
- **优势**：减少全局内存访问延迟

**5. 全局内存（Global Memory）：**
- **速度**：最慢
- **容量**：最大，通常几GB到几十GB
- **作用**：存储大量数据
- **特点**：所有线程都可以访问
- **限制**：访问延迟高，需要优化访问模式

**6. 常量内存（Constant Memory）：**
- **速度**：中等（有缓存）
- **容量**：较小，通常64KB
- **作用**：存储只读数据
- **特点**：有专门的缓存
- **优势**：广播机制，一次读取可以服务多个线程

**7. 纹理内存（Texture Memory）：**
- **速度**：中等（有缓存）
- **容量**：较大
- **作用**：专门用于图像处理
- **特点**：有专门的缓存和硬件支持
- **优势**：支持2D/3D访问模式

**内存访问模式的重要性：**

**1. 合并访问（Coalesced Access）：**
- 相邻线程访问相邻内存位置
- 硬件可以将多个访问合并为一个
- 显著提高内存带宽利用率

**2. 非合并访问（Non-coalesced Access）：**
- 线程访问不连续的内存位置
- 每个访问都需要单独的内存事务
- 严重影响性能

**实际应用示例：**
```cuda
// CUDA内存层次结构使用示例
__global__ void memoryDemo(float* globalData, const float* constantData) {
    // 寄存器变量
    float registerVar = 1.0f;
    
    // 共享内存
    __shared__ float sharedData[256];
    
    // 全局内存访问
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float globalValue = globalData[tid];
    
    // 常量内存访问
    float constantValue = constantData[0];
    
    // 使用共享内存
    sharedData[threadIdx.x] = globalValue;
    __syncthreads();  // 同步块内所有线程
    
    // 计算
    float result = registerVar + globalValue + constantValue;
    globalData[tid] = result;
}
```

**内存优化策略：**

**1. 减少全局内存访问：**
- 使用共享内存缓存数据
- 合并内存访问
- 避免随机内存访问

**2. 合理使用共享内存：**
- 避免银行冲突（Bank Conflict）
- 考虑共享内存大小限制
- 使用共享内存减少全局内存访问

**3. 优化寄存器使用：**
- 避免寄存器溢出
- 使用循环展开减少寄存器使用
- 合理使用局部变量

### 4. 基本CUDA语法

#### 4.1 内核函数
```cuda
// 内核函数定义
__global__ void kernelFunction(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        data[tid] = data[tid] * 2.0f;
    }
}

// 内核函数调用
int main() {
    float* h_data = (float*)malloc(1024 * sizeof(float));
    float* d_data;
    
    // 分配设备内存
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动内核
    kernelFunction<<<4, 256>>>(d_data, 1024);
    
    // 复制数据回主机
    cudaMemcpy(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

### 4. 基本CUDA语法

#### 4.1 内核函数

**CUDA内核函数的概念：**

内核函数是CUDA编程的核心，它是在GPU上执行的函数。理解内核函数的概念和语法是学习CUDA编程的基础。

**内核函数的定义：**

**`__global__`关键字：**
- 表示这是一个CUDA内核函数
- 可以在主机上调用，在设备上执行
- 是连接主机和设备的桥梁
- 必须返回void类型

**内核函数的特点：**
1. **并行执行**：多个线程同时执行同一个内核函数
2. **线程标识**：每个线程有唯一的标识符
3. **内存访问**：可以访问全局内存和共享内存
4. **同步机制**：支持线程块内的同步

**内核函数的调用语法：**

```cuda
// 内核函数定义
__global__ void kernelFunction(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        data[tid] = data[tid] * 2.0f;
    }
}

// 内核函数调用
int main() {
    float* h_data = (float*)malloc(1024 * sizeof(float));
    float* d_data;
    
    // 分配设备内存
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动内核
    kernelFunction<<<4, 256>>>(d_data, 1024);
    
    // 复制数据回主机
    cudaMemcpy(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

**内核启动语法详解：**

**`<<<gridSize, blockSize>>>`语法：**
- `gridSize`：网格大小，指定启动多少个线程块
- `blockSize`：线程块大小，指定每个线程块包含多少个线程
- 总线程数 = gridSize × blockSize

**网格和线程块大小的选择：**
1. **硬件限制**：每个线程块最多1024个线程
2. **性能考虑**：通常选择32的倍数（warp大小）
3. **算法特性**：根据数据规模和算法特点选择

**其他函数类型：**

**`__device__`函数：**
- 在设备上定义和调用
- 只能被其他设备函数调用
- 不能从主机调用

**`__host__`函数：**
- 在主机上定义和调用
- 只能被其他主机函数调用
- 不能从设备调用

**`__host__ __device__`函数：**
- 可以在主机和设备上调用
- 编译器会生成两个版本
- 适合简单的工具函数

#### 4.2 内存管理

**CUDA内存管理的重要性：**

CUDA内存管理是GPU编程的关键技能。与CPU编程不同，GPU有独立的内存空间，需要显式管理。

**主要内存管理函数：**

**1. `cudaMalloc`：分配设备内存**
```cuda
cudaError_t cudaMalloc(void** devPtr, size_t size);
```
- `devPtr`：指向设备内存指针的指针
- `size`：要分配的字节数
- 返回值：错误代码

**2. `cudaFree`：释放设备内存**
```cuda
cudaError_t cudaFree(void* devPtr);
```
- `devPtr`：要释放的设备内存指针
- 返回值：错误代码

**3. `cudaMemcpy`：内存复制**
```cuda
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```
- `dst`：目标内存指针
- `src`：源内存指针
- `count`：复制的字节数
- `kind`：复制方向

**内存复制方向：**
- `cudaMemcpyHostToDevice`：主机到设备
- `cudaMemcpyDeviceToHost`：设备到主机
- `cudaMemcpyDeviceToDevice`：设备到设备
- `cudaMemcpyHostToHost`：主机到主机

**内存管理最佳实践：**

**1. 错误处理：**
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main() {
    float* d_data;
    
    // 使用错误检查宏
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));
    
    // 其他CUDA操作...
    
    CUDA_CHECK(cudaFree(d_data));
    
    return 0;
}
```

**2. 内存对齐：**
- 使用`cudaMallocPitch`分配2D数组
- 考虑内存对齐要求
- 避免内存碎片

**3. 内存泄漏预防：**
- 每个`cudaMalloc`都要有对应的`cudaFree`
- 使用RAII模式管理内存
- 定期检查内存使用情况

#### 4.3 错误处理

**CUDA错误处理的重要性：**

CUDA程序中的错误处理比CPU程序更重要，因为GPU错误通常不会立即导致程序崩溃，而是静默失败。

**CUDA错误类型：**

**1. 运行时错误：**
- 内存分配失败
- 内核启动失败
- 设备不可用

**2. 内核错误：**
- 越界访问
- 除零错误
- 无效内存访问

**3. 设备错误：**
- 设备超时
- 设备重置
- 硬件故障

**错误处理策略：**

**1. 检查每个CUDA调用：**
```cuda
cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
    return 1;
}
```

**2. 使用错误检查宏：**
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
```

**3. 内核错误检查：**
```cuda
kernelFunction<<<4, 256>>>(d_data, 1024);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
}
```

**常见错误和解决方案：**

**1. 内存分配失败：**
- 检查设备内存是否足够
- 使用`cudaMemGetInfo`查询可用内存
- 考虑使用统一内存

**2. 内核启动失败：**
- 检查线程块大小是否超过限制
- 检查网格大小是否合理
- 检查内核函数语法

**3. 设备不可用：**
- 检查CUDA驱动是否正确安装
- 检查设备是否被其他程序占用
- 检查设备是否支持所需的计算能力

## 实践项目

### 项目1：Hello CUDA

**项目目标：**
编写第一个CUDA程序，在GPU上打印Hello World，理解CUDA编程的基本流程。

**学习重点：**
1. 理解主机-设备编程模型
2. 掌握内核函数的定义和调用
3. 学会基本的CUDA语法
4. 理解线程标识符的使用

**实现步骤：**
1. 包含必要的头文件
2. 定义CUDA内核函数
3. 在主机函数中启动内核
4. 添加同步机制
5. 测试和调试

**代码实现：**
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA内核函数
__global__ void helloFromGPU() {
    // 获取线程标识符
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    printf("Hello from GPU! Thread ID: %d\n", tid);
}

int main() {
    printf("Hello from CPU!\n");
    
    // 启动内核：4个线程块，每个块256个线程
    helloFromGPU<<<4, 256>>>();
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    printf("GPU execution completed!\n");
    
    return 0;
}
```

**编译和运行：**
```bash
# 编译
nvcc -o hello_cuda hello_cuda.cu

# 运行
./hello_cuda
```

**预期输出：**
```
Hello from CPU!
Hello from GPU! Thread ID: 0
Hello from GPU! Thread ID: 1
...
Hello from GPU! Thread ID: 1023
GPU execution completed!
```

### 项目2：向量加法

**项目目标：**
实现GPU并行向量加法，比较CPU和GPU性能，理解GPU并行计算的优势。

**学习重点：**
1. 理解GPU并行计算的优势
2. 掌握内存管理技术
3. 学会性能比较方法
4. 理解线程索引计算

**实现步骤：**
1. 定义向量加法内核函数
2. 实现CPU版本作为对比
3. 添加性能计时
4. 比较CPU和GPU性能
5. 验证计算结果的正确性

**代码实现：**
```cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

// GPU向量加法内核
__global__ void vectorAddGPU(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// CPU向量加法
void vectorAddCPU(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1024 * 1024;  // 1M elements
    const size_t size = n * sizeof(float);
    
    // 分配主机内存
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c_cpu = (float*)malloc(size);
    float* h_c_gpu = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 分配设备内存
    float* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 复制数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // GPU计算
    clock_t start = clock();
    vectorAddGPU<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    clock_t end = clock();
    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // 复制结果回主机
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    
    // CPU计算
    start = clock();
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            correct = false;
            break;
        }
    }
    
    printf("Results: %s\n", correct ? "CORRECT" : "INCORRECT");
    printf("CPU time: %.6f seconds\n", cpu_time);
    printf("GPU time: %.6f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    
    return 0;
}
```

### 项目3：矩阵运算

**项目目标：**
实现简单的矩阵加法运算，理解2D线程索引的使用。

**学习重点：**
1. 理解2D线程索引计算
2. 掌握矩阵数据的GPU处理
3. 学会多维线程块的使用
4. 理解矩阵运算的并行化

**实现步骤：**
1. 定义2D矩阵加法内核
2. 使用2D线程块和网格
3. 实现矩阵数据的初始化
4. 验证矩阵运算结果
5. 比较不同线程块大小的性能

**代码实现：**
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

// 2D矩阵加法内核
__global__ void matrixAdd2D(float* a, float* b, float* c, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int index = row * width + col;
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const size_t size = width * height * sizeof(float);
    
    // 分配主机内存
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);
    
    // 初始化矩阵
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            h_a[index] = i + j;
            h_b[index] = i * j;
        }
    }
    
    // 分配设备内存
    float* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 复制数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 启动2D内核
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    matrixAdd2D<<<gridSize, blockSize>>>(d_a, d_b, d_c, width, height);
    
    // 复制结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            float expected = h_a[index] + h_b[index];
            if (abs(h_c[index] - expected) > 1e-6) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    
    printf("Matrix addition result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printf("Matrix size: %dx%d\n", width, height);
    printf("Thread block size: %dx%d\n", blockSize.x, blockSize.y);
    printf("Grid size: %dx%d\n", gridSize.x, gridSize.y);
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

## 每日学习任务

### 第1天：GPU架构基础

**学习目标：**
理解GPU与CPU的根本区别，掌握GPU硬件架构的基本概念。

**学习内容：**
1. **GPU与CPU的区别**
   - 核心数量对比
   - 功能复杂度对比
   - 适用场景分析
   - 性能特征对比

2. **GPU硬件架构**
   - 流多处理器（SM）的作用
   - CUDA核心的功能
   - 内存层次结构
   - 特殊功能单元

3. **计算能力理解**
   - 不同GPU架构的特点
   - 计算能力对功能的影响
   - 如何选择合适的GPU

**实践任务：**
- 使用`cudaGetDeviceProperties`查询GPU信息
- 比较不同GPU的硬件参数
- 理解硬件参数对性能的影响

**学习检查：**
- [ ] 能够解释GPU与CPU的区别
- [ ] 理解GPU硬件架构的层次结构
- [ ] 知道如何查询GPU计算能力
- [ ] 能够分析GPU硬件参数

### 第2天：CUDA编程模型

**学习目标：**
掌握CUDA编程模型的核心概念，理解主机-设备架构。

**学习内容：**
1. **主机-设备概念**
   - 主机的职责和特点
   - 设备的功能和限制
   - 主机与设备的交互方式
   - 内存分离的重要性

2. **线程层次结构**
   - 网格、线程块、线程的关系
   - 线程标识符系统
   - 线程协作机制
   - 硬件映射关系

3. **内核函数基础**
   - `__global__`关键字的作用
   - 内核启动语法
   - 线程索引计算
   - 同步机制

**实践任务：**
- 编写第一个CUDA程序
- 理解线程索引计算
- 掌握内核启动语法

**学习检查：**
- [ ] 理解主机-设备编程模型
- [ ] 掌握线程层次结构
- [ ] 能够编写基本的内核函数
- [ ] 理解线程标识符系统

### 第3天：CUDA内存模型

**学习目标：**
理解CUDA内存层次结构，掌握不同内存类型的使用。

**学习内容：**
1. **内存层次结构**
   - 寄存器内存的特点
   - 共享内存的使用
   - 全局内存的访问
   - 缓存机制

2. **内存访问模式**
   - 合并访问的优势
   - 非合并访问的问题
   - 内存对齐的重要性
   - 访问模式优化

3. **内存管理**
   - 内存分配和释放
   - 内存传输方向
   - 内存泄漏预防
   - 错误处理

**实践任务：**
- 实现向量加法程序
- 比较不同内存访问模式的性能
- 掌握内存管理技术

**学习检查：**
- [ ] 理解CUDA内存层次结构
- [ ] 掌握内存访问模式
- [ ] 能够进行内存管理
- [ ] 理解内存优化策略

### 第4天：基本CUDA语法

**学习目标：**
掌握CUDA编程的基本语法，理解内核函数和内存管理。

**学习内容：**
1. **内核函数语法**
   - 函数类型关键字
   - 内核启动语法
   - 参数传递
   - 返回值限制

2. **内存管理函数**
   - `cudaMalloc`的使用
   - `cudaFree`的重要性
   - `cudaMemcpy`的方向
   - 内存对齐

3. **错误处理**
   - CUDA错误类型
   - 错误检查方法
   - 错误处理策略
   - 调试技巧

**实践任务：**
- 实现矩阵运算程序
- 添加错误处理机制
- 掌握调试方法

**学习检查：**
- [ ] 掌握CUDA基本语法
- [ ] 能够进行内存管理
- [ ] 理解错误处理机制
- [ ] 能够调试CUDA程序

### 第5天：实践编程

**学习目标：**
通过实际编程练习，巩固所学知识，提高编程技能。

**学习内容：**
1. **Hello CUDA程序**
   - 理解CUDA编程流程
   - 掌握基本语法
   - 学会编译和运行
   - 理解输出结果

2. **向量加法实现**
   - 实现GPU并行计算
   - 比较CPU和GPU性能
   - 验证计算正确性
   - 分析性能差异

3. **矩阵运算实现**
   - 使用2D线程索引
   - 实现矩阵加法
   - 验证计算结果
   - 优化线程块大小

**实践任务：**
- 完成所有实践项目
- 分析性能结果
- 优化程序性能
- 总结学习经验

**学习检查：**
- [ ] 能够编写CUDA程序
- [ ] 理解性能比较方法
- [ ] 掌握调试技巧
- [ ] 能够优化程序

### 第6天：性能分析

**学习目标：**
学会使用CUDA性能分析工具，理解性能优化方法。

**学习内容：**
1. **性能分析工具**
   - Nsight Compute的使用
   - Nsight Systems的使用
   - 性能指标分析
   - 瓶颈识别

2. **性能优化方法**
   - 内存访问优化
   - 线程块大小优化
   - 占用率优化
   - 算法优化

3. **性能比较**
   - CPU vs GPU性能
   - 不同实现方法比较
   - 性能瓶颈分析
   - 优化效果评估

**实践任务：**
- 使用性能分析工具
- 分析程序性能
- 实施性能优化
- 评估优化效果

**学习检查：**
- [ ] 能够使用性能分析工具
- [ ] 理解性能优化方法
- [ ] 能够识别性能瓶颈
- [ ] 掌握优化技巧

### 第7天：综合练习

**学习目标：**
综合运用所学知识，完成复杂项目，准备下周学习。

**学习内容：**
1. **知识回顾**
   - 复习本周学习内容
   - 总结关键概念
   - 整理学习笔记
   - 准备问题解答

2. **综合项目**
   - 实现复杂算法
   - 优化程序性能
   - 添加错误处理
   - 完善文档

3. **学习总结**
   - 总结学习成果
   - 分析学习难点
   - 制定下周计划
   - 准备进阶学习

**实践任务：**
- 完成综合项目
- 总结学习经验
- 准备下周学习
- 解答常见问题

**学习检查：**
- [ ] 完成所有学习任务
- [ ] 掌握核心概念
- [ ] 能够独立编程
- [ ] 准备进阶学习

## 检查点

### 第1周结束时的能力要求

**核心概念掌握：**
- [ ] **理解GPU架构和CUDA编程模型**
  - 能够解释GPU与CPU的区别
  - 理解GPU硬件架构的层次结构
  - 掌握CUDA编程模型的基本概念
  - 知道如何查询GPU计算能力

- [ ] **掌握线程层次结构和索引计算**
  - 理解网格、线程块、线程的关系
  - 能够计算线程索引
  - 掌握线程标识符系统
  - 理解硬件映射关系

- [ ] **理解CUDA内存模型**
  - 掌握内存层次结构
  - 理解不同内存类型的特点
  - 知道内存访问模式的重要性
  - 能够进行内存管理

- [ ] **能够编写基本的CUDA程序**
  - 掌握内核函数定义和调用
  - 能够进行内存分配和释放
  - 理解错误处理机制
  - 能够调试CUDA程序

**实践技能要求：**
- [ ] **完成项目1-3**
  - 成功编写Hello CUDA程序
  - 实现向量加法并比较性能
  - 实现矩阵运算并验证结果
  - 能够分析性能差异

- [ ] **掌握CUDA调试工具**
  - 能够使用Nsight分析程序性能
  - 理解性能瓶颈识别
  - 掌握优化技巧
  - 能够评估优化效果

**学习成果验证：**
- [ ] **理论理解**
  - 能够解释CUDA编程的核心概念
  - 理解GPU并行计算的优势
  - 掌握内存管理的重要性
  - 知道错误处理的意义

- [ ] **实践能力**
  - 能够独立编写CUDA程序
  - 掌握编译和运行方法
  - 能够进行性能分析
  - 具备调试和优化能力

- [ ] **问题解决**
  - 能够识别和解决常见问题
  - 理解错误信息的含义
  - 掌握调试技巧
  - 能够优化程序性能

**进阶准备：**
- [ ] **知识基础**
  - 掌握CUDA基础概念
  - 理解GPU编程模型
  - 具备内存管理能力
  - 掌握基本语法

- [ ] **技能准备**
  - 能够编写基本程序
  - 掌握调试方法
  - 具备优化意识
  - 准备进阶学习

**学习建议：**
1. **巩固基础**：确保理解所有核心概念
2. **多实践**：通过编程练习加深理解
3. **多思考**：分析程序性能和优化方法
4. **多总结**：整理学习笔记和经验
5. **多交流**：与他人讨论学习心得

**常见问题解答：**

### Q: 如何确定线程块大小？
A: 线程块大小通常是32的倍数（warp大小），常见的选择是128、256、512。选择时要考虑硬件限制、占用率和算法特性。

### Q: 如何计算网格大小？
A: 网格大小 = (总线程数 + 线程块大小 - 1) / 线程块大小。这样可以确保所有数据都被处理。

### Q: 如何调试CUDA程序？
A: 使用cuda-gdb调试器，或者在内核函数中使用printf输出调试信息。注意要添加错误检查。

### Q: 内存分配失败怎么办？
A: 检查显存是否足够，使用cudaGetLastError()检查错误，确保正确释放内存。考虑使用统一内存。

### Q: 如何提高CUDA程序性能？
A: 优化内存访问模式，使用共享内存，选择合适的线程块大小，避免分支发散，使用性能分析工具。

### Q: CUDA程序编译失败怎么办？
A: 检查CUDA环境是否正确安装，确保使用正确的编译命令，检查代码语法错误，查看错误信息。

### Q: 如何选择GPU？
A: 根据应用需求选择，考虑计算能力、内存大小、功耗等因素。学习阶段建议使用计算能力6.0以上的GPU。

### Q: CUDA学习有什么建议？
A: 从基础概念开始，多实践编程，使用性能分析工具，阅读官方文档，参与社区讨论，持续学习新技术。

---

**学习时间**：第1周  
**预计完成时间**：2024-02-15  
**学习难度**：⭐⭐⭐☆☆  
**实践要求**：⭐⭐⭐⭐☆  
**理论深度**：⭐⭐⭐☆☆
