# CUDA知识库

## 工程说明

本知识库基于 `cuda-learning` 工程中的 `CUDA学习/01-基础概念/` 目录下的实际学习材料汇总而成。工程采用系统性的学习方式，当前已完成基础概念阶段的学习，包含详细的学习讲义、训练题目和实践代码。知识库会随着工程内容的持续更新而保持同步。

## 基础知识汇总

### GPU架构基础

#### GPU与CPU的根本区别

GPU（Graphics Processing Unit）最初设计用于图形渲染，但由于其强大的并行计算能力，现在广泛应用于通用计算领域。根据工程中的学习材料，GPU与CPU在架构上有根本区别。

**CPU（中央处理器）的特点**：
- **核心数量少**：通常有2-16个核心
- **每个核心功能强大**：支持复杂指令集，有分支预测、乱序执行等高级功能
- **大容量缓存**：L1、L2、L3缓存容量大，访问延迟低
- **适合串行处理**：专门优化单线程性能
- **低延迟**：单个任务响应时间短
- **通用性强**：可以处理各种类型的计算任务

**GPU（图形处理器）的特点**：
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

#### GPU硬件层次结构

GPU采用层次化的架构设计，从大到小分为以下几个层次：

**1. GPU芯片级别**：
- **全局内存（Global Memory）**：GPU的主要存储空间，容量大但访问延迟高
- **L2缓存**：连接全局内存和各个SM的高速缓存
- **常量内存（Constant Memory）**：只读内存，有专门的缓存
- **纹理内存（Texture Memory）**：专门用于图像处理的内存类型

**2. 流多处理器（SM - Streaming Multiprocessor）级别**：
- **CUDA核心**：GPU的基本计算单元，执行浮点运算和整数运算
- **共享内存（Shared Memory）**：SM内所有线程共享的高速内存
- **寄存器文件**：每个线程私有的存储空间
- **调度器**：负责线程调度和指令分发
- **特殊功能单元**：如Tensor Core（用于深度学习加速）

**3. 线程级别**：
- **线程**：GPU执行的最小单位
- **Warp**：32个线程组成一个warp，是GPU调度的基本单位
- **寄存器**：每个线程私有的存储空间

**关键概念解释**：

**流多处理器（SM）**：
- SM是GPU的核心计算单元
- 每个SM包含多个CUDA核心（通常16-128个）
- SM内部有共享内存和寄存器文件
- 多个SM可以并行执行不同的任务

**CUDA核心**：
- GPU的基本计算单元
- 可以执行浮点运算、整数运算等
- 现代GPU的CUDA核心还支持特殊运算（如Tensor Core）

**共享内存**：
- SM内所有线程共享的内存
- 访问速度比全局内存快得多
- 容量有限（通常16-128KB）
- 需要程序员显式管理

#### 计算能力（Compute Capability）

**什么是计算能力？**

计算能力是NVIDIA用来标识GPU架构和功能的版本号，它决定了GPU支持哪些CUDA功能。

**计算能力的作用**：
1. **功能支持**：不同计算能力支持不同的CUDA功能
2. **性能特征**：影响GPU的性能表现和优化策略
3. **兼容性**：决定CUDA程序能否在特定GPU上运行

**主要GPU架构和计算能力**（来自工程学习材料）：

- **Kepler架构（计算能力 3.0-3.7）**：代表产品GTX 680, GTX 780, Tesla K20
- **Maxwell架构（计算能力 5.0-5.2）**：代表产品GTX 980, GTX 1080
- **Pascal架构（计算能力 6.0-6.1）**：代表产品GTX 1080, GTX 1080 Ti, Tesla P100
- **Volta架构（计算能力 7.0-7.2）**：代表产品Tesla V100, Titan V
- **Turing架构（计算能力 7.5）**：代表产品RTX 2080, RTX 3080
- **Ampere架构（计算能力 8.0-8.6）**：代表产品RTX 3080, RTX 4080, A100
- **Hopper架构（计算能力 9.0）**：代表产品H100

**如何查看GPU计算能力**（来自工程训练题目）：

```cuda
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

### CUDA编程模型

#### 主机和设备（Host-Device）

CUDA编程模型基于**主机-设备**（Host-Device）架构，这是理解CUDA编程的基础。

**主机（Host）**：
- **定义**：运行CPU的计算机系统
- **作用**：负责程序控制、数据准备、结果处理
- **特点**：执行串行代码，管理GPU资源
- **内存**：主机内存（Host Memory），CPU可以直接访问

**设备（Device）**：
- **定义**：GPU及其相关硬件
- **作用**：执行并行计算任务
- **特点**：执行CUDA内核函数，处理大量并行数据
- **内存**：设备内存（Device Memory），GPU可以直接访问

**主机和设备的关系**：
1. **分工明确**：主机负责控制，设备负责计算
2. **内存分离**：主机内存和设备内存是独立的
3. **数据传输**：需要显式地在主机和设备间传输数据
4. **异步执行**：主机和设备可以并行工作

**实际应用场景**（来自工程学习材料）：

```cuda
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

**关键概念解释**：

**`__global__`关键字**：
- 表示这是一个CUDA内核函数
- 可以在主机上调用，在设备上执行
- 是连接主机和设备的桥梁

**`<<<1, 1>>>`语法**：
- 这是CUDA内核启动语法
- 第一个参数：网格大小（Grid Size）
- 第二个参数：线程块大小（Block Size）
- 这里表示启动1个线程块，每个线程块包含1个线程

**`cudaDeviceSynchronize()`**：
- 同步函数，等待GPU完成所有操作
- 确保主机和设备之间的同步
- 避免主机在GPU完成之前继续执行

#### 线程层次结构

CUDA的线程层次结构是理解GPU并行计算的核心概念。与CPU的单线程模型不同，GPU使用大规模并行线程模型。

**线程层次结构的三级模型**：

**1. 网格（Grid）**：
- **定义**：整个CUDA内核启动的所有线程块的集合
- **特点**：一个网格包含多个线程块
- **作用**：定义整个并行任务的规模
- **限制**：网格大小受GPU硬件限制

**2. 线程块（Block）**：
- **定义**：一组协作的线程
- **特点**：线程块内的线程可以共享内存和同步
- **作用**：线程协作的基本单位
- **限制**：每个线程块最多1024个线程（取决于GPU）

**3. 线程（Thread）**：
- **定义**：GPU执行的最小单位
- **特点**：每个线程有唯一的标识符
- **作用**：执行具体的计算任务
- **限制**：线程数量受硬件资源限制

**线程标识符系统**（来自工程学习材料）：

CUDA提供了内置变量来标识线程：
- `threadIdx`：线程在块内的索引
- `blockIdx`：线程块在网格内的索引
- `blockDim`：线程块的维度
- `gridDim`：网格的维度

**实际应用示例**（来自工程训练题目）：

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

**全局线程索引计算**：

对于1D情况，全局线程索引计算公式：
```cuda
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

对于2D情况，全局索引计算：
```cuda
int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
```

这是CUDA编程的基础，必须熟练掌握。

**线程层次结构的优势**：

**1. 可扩展性**：
- 可以根据问题规模调整线程数量
- 支持从小规模到大规模的计算任务

**2. 协作性**：
- 线程块内的线程可以协作
- 支持共享内存和同步操作

**3. 硬件映射**：
- 线程块映射到SM（流多处理器）
- 线程映射到CUDA核心

**选择线程块大小的考虑因素**：

**1. 硬件限制**：
- 每个SM的最大线程数
- 每个线程块的最大线程数
- 共享内存大小

**2. 性能优化**：
- 占用率（Occupancy）
- 内存访问模式
- 计算密度

**3. 算法特性**：
- 数据依赖性
- 同步需求
- 内存访问模式

### CUDA内存模型

#### 内存层次结构

CUDA内存模型是GPU编程的核心概念之一。与CPU的简单内存模型不同，GPU有复杂的内存层次结构，理解这个结构对于编写高效的CUDA程序至关重要。

**CUDA内存层次结构（从快到慢）**：

**1. 寄存器（Registers）**：
- **速度**：最快
- **容量**：每个线程私有，数量有限
- **作用**：存储局部变量和临时数据
- **特点**：编译器自动管理，程序员无法直接控制
- **限制**：寄存器溢出会影响性能

**2. 共享内存（Shared Memory）**：
- **速度**：很快
- **容量**：每个线程块共享，通常16-128KB
- **作用**：线程块内线程协作的数据交换
- **特点**：程序员显式管理
- **优势**：比全局内存快100倍以上

**3. L1缓存（L1 Cache）**：
- **速度**：快
- **容量**：自动管理
- **作用**：缓存频繁访问的数据
- **特点**：硬件自动管理
- **优势**：提高内存访问效率

**4. L2缓存（L2 Cache）**：
- **速度**：中等
- **容量**：较大，全局共享
- **作用**：缓存全局内存数据
- **特点**：硬件自动管理
- **优势**：减少全局内存访问延迟

**5. 全局内存（Global Memory）**：
- **速度**：最慢
- **容量**：最大，通常几GB到几十GB
- **作用**：存储大量数据
- **特点**：所有线程都可以访问
- **限制**：访问延迟高，需要优化访问模式

**6. 常量内存（Constant Memory）**：
- **速度**：中等（有缓存）
- **容量**：较小，通常64KB
- **作用**：存储只读数据
- **特点**：有专门的缓存
- **优势**：广播机制，一次读取可以服务多个线程

**7. 纹理内存（Texture Memory）**：
- **速度**：中等（有缓存）
- **容量**：较大
- **作用**：专门用于图像处理
- **特点**：有专门的缓存和硬件支持
- **优势**：支持2D/3D访问模式

**实际应用示例**（来自工程学习材料）：

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

#### 内存访问模式优化

**合并访问（Coalesced Access）**：
- 相邻线程访问相邻内存位置
- 硬件可以将多个访问合并为一个
- 显著提高内存带宽利用率

**非合并访问（Non-coalesced Access）**：
- 线程访问不连续的内存位置
- 每个访问都需要单独的内存事务
- 严重影响性能

**内存优化策略**：

**1. 减少全局内存访问**：
- 使用共享内存缓存数据
- 合并内存访问
- 避免随机内存访问

**2. 合理使用共享内存**：
- 避免银行冲突（Bank Conflict）
- 考虑共享内存大小限制
- 使用共享内存减少全局内存访问

**3. 优化寄存器使用**：
- 避免寄存器溢出
- 使用循环展开减少寄存器使用
- 合理使用局部变量

### 基本CUDA语法

#### 内核函数

**CUDA内核函数的概念**：

内核函数是CUDA编程的核心，它是在GPU上执行的函数。理解内核函数的概念和语法是学习CUDA编程的基础。

**内核函数的定义**：

**`__global__`关键字**：
- 表示这是一个CUDA内核函数
- 可以在主机上调用，在设备上执行
- 是连接主机和设备的桥梁
- 必须返回void类型

**内核函数的特点**：
1. **并行执行**：多个线程同时执行同一个内核函数
2. **线程标识**：每个线程有唯一的标识符
3. **内存访问**：可以访问全局内存和共享内存
4. **同步机制**：支持线程块内的同步

**内核函数的调用语法**（来自工程学习材料）：

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

**内核启动语法详解**：

**`<<<gridSize, blockSize>>>`语法**：
- `gridSize`：网格大小，指定启动多少个线程块
- `blockSize`：线程块大小，指定每个线程块包含多少个线程
- 总线程数 = gridSize × blockSize

**网格和线程块大小的选择**：
1. **硬件限制**：每个线程块最多1024个线程
2. **性能考虑**：通常选择32的倍数（warp大小）
3. **算法特性**：根据数据规模和算法特点选择

**其他函数类型**：

**`__device__`函数**：
- 在设备上定义和调用
- 只能被其他设备函数调用
- 不能从主机调用

**`__host__`函数**：
- 在主机上定义和调用
- 只能被其他主机函数调用
- 不能从设备调用

**`__host__ __device__`函数**：
- 可以在主机和设备上调用
- 编译器会生成两个版本
- 适合简单的工具函数

#### 内存管理

**CUDA内存管理的重要性**：

CUDA内存管理是GPU编程的关键技能。与CPU编程不同，GPU有独立的内存空间，需要显式管理。

**主要内存管理函数**：

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

**内存复制方向**：
- `cudaMemcpyHostToDevice`：主机到设备
- `cudaMemcpyDeviceToHost`：设备到主机
- `cudaMemcpyDeviceToDevice`：设备到设备
- `cudaMemcpyHostToHost`：主机到主机

**内存管理最佳实践**（来自工程训练题目）：

**1. 错误处理**：

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

**2. 内存对齐**：
- 使用`cudaMallocPitch`分配2D数组
- 考虑内存对齐要求
- 避免内存碎片

**3. 内存泄漏预防**：
- 每个`cudaMalloc`都要有对应的`cudaFree`
- 使用RAII模式管理内存
- 定期检查内存使用情况

#### 错误处理

**CUDA错误处理的重要性**：

CUDA程序中的错误处理比CPU程序更重要，因为GPU错误通常不会立即导致程序崩溃，而是静默失败。

**CUDA错误类型**：

**1. 运行时错误**：
- 内存分配失败
- 内核启动失败
- 设备不可用

**2. 内核错误**：
- 越界访问
- 除零错误
- 无效内存访问

**3. 设备错误**：
- 设备超时
- 设备重置
- 硬件故障

**错误处理策略**（来自工程学习材料）：

**1. 检查每个CUDA调用**：

```cuda
cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
    return 1;
}
```

**2. 使用错误检查宏**：

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

**3. 内核错误检查**：

```cuda
kernelFunction<<<4, 256>>>(d_data, 1024);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
}
```

**常见错误和解决方案**：

**1. 内存分配失败**：
- 检查设备内存是否足够
- 使用`cudaMemGetInfo`查询可用内存
- 考虑使用统一内存

**2. 内核启动失败**：
- 检查线程块大小是否超过限制
- 检查网格大小是否合理
- 检查内核函数语法

**3. 设备不可用**：
- 检查CUDA驱动是否正确安装
- 检查设备是否被其他程序占用
- 检查设备是否支持所需的计算能力

## 面试要点总结

### GPU架构面试要点

1. **GPU与CPU的区别**：理解两者的设计目标和适用场景，能够解释为什么GPU适合并行计算
2. **GPU硬件层次结构**：理解SM、CUDA核心、共享内存等概念
3. **计算能力**：知道如何查询GPU计算能力，理解计算能力对功能的影响

### CUDA编程模型面试要点

1. **主机-设备模型**：理解主机和设备的分工，内存分离的概念
2. **线程层次结构**：理解网格、线程块、线程的关系，能够计算全局线程索引
3. **内核函数**：理解`__global__`关键字的作用，掌握内核启动语法
4. **内存管理**：掌握`cudaMalloc`、`cudaFree`、`cudaMemcpy`的使用，理解错误处理的重要性

### CUDA内存模型面试要点

1. **内存层次结构**：理解不同内存类型的特点、访问速度和容量限制
2. **内存访问优化**：理解合并访问的重要性，知道如何优化内存访问模式
3. **共享内存使用**：理解共享内存的作用，知道如何避免银行冲突

## 引导面试者深入理解

### 理解GPU并行计算的本质

GPU的强大之处在于能够同时执行大量线程。理解线程层次结构、内存模型和同步机制是编写高效CUDA程序的基础。

### 掌握CUDA编程的基本模式

CUDA编程遵循固定的模式：
1. **内存分配**：在设备上分配内存
2. **数据传输**：将数据从主机传输到设备
3. **内核启动**：启动GPU内核函数
4. **结果传输**：将结果从设备传输回主机
5. **资源清理**：释放设备内存

### 性能优化的思考方向

1. **内存访问优化**：合并访问、使用共享内存
2. **线程配置优化**：选择合适的线程块大小，提高占用率
3. **算法优化**：减少分支发散，优化计算流程

### 实践与理论结合

理解这些概念很重要，但更重要的是能够在实际编程中正确使用。建议：
1. **编写代码验证**：通过实际代码验证理论理解
2. **性能分析**：使用Nsight等工具分析程序性能
3. **错误处理**：养成良好的错误检查习惯

### 持续学习

CUDA技术在不断发展，新的架构和特性不断出现。要保持学习习惯，关注新技术，同时也要回顾和总结已学知识，形成完整的知识体系。
