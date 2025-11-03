# 第2周：CUDA内存管理

## 学习目标

掌握CUDA内存优化技术，包括全局内存、共享内存、寄存器内存的使用和优化。

## 学习内容

### 1. 全局内存管理

#### 1.1 全局内存访问模式

**全局内存的重要性：**

全局内存是GPU中容量最大但访问延迟最高的内存类型。理解全局内存的访问模式对于编写高性能CUDA程序至关重要。

**全局内存的特点：**
- **容量大**：通常几GB到几十GB，是GPU的主要存储空间
- **访问延迟高**：比寄存器慢数百倍，比共享内存慢数十倍
- **带宽高**：虽然延迟高，但带宽很高，适合批量数据传输
- **全局可访问**：所有线程都可以访问，是线程间通信的主要方式

**内存访问模式的重要性：**

**1. 合并访问（Coalesced Access）：**
- **定义**：相邻线程访问相邻内存位置
- **优势**：硬件可以将多个访问合并为一个内存事务
- **性能提升**：可以显著提高内存带宽利用率
- **实现方法**：确保线程索引与内存访问模式匹配

**2. 非合并访问（Non-coalesced Access）：**
- **定义**：线程访问不连续的内存位置
- **问题**：每个访问都需要单独的内存事务
- **性能影响**：严重影响内存带宽利用率
- **避免方法**：重新设计数据布局或访问模式

**内存对齐的重要性：**

**1. 自然对齐：**
- 数据类型按自然边界对齐
- 提高内存访问效率
- 减少内存事务数量

**2. 强制对齐：**
- 使用`__align__`关键字
- 确保数据按特定边界对齐
- 优化内存访问性能

**实际应用示例：**
```cuda
// 合并访问示例
__global__ void coalescedAccess(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        // 合并访问：相邻线程访问相邻内存位置
        data[tid] = data[tid] * 2.0f;
    }
}

// 非合并访问示例
__global__ void uncoalescedAccess(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        // 非合并访问：跳跃式访问
        data[tid * 2] = data[tid * 2] * 2.0f;
    }
}
```

#### 1.2 内存对齐

**内存对齐的概念：**

内存对齐是指数据在内存中的存储位置必须满足特定边界要求。理解内存对齐对于优化CUDA程序性能非常重要。

**为什么需要内存对齐？**

**1. 硬件要求：**
- 现代GPU硬件去试访问对齐的数据更高效
- 未对齐的访问可能需要多次内存事务
- 对齐访问可以利用硬件的优化机制

**2. 性能优势：**
- 减少内存事务数量
- 提高内存带宽利用率
- 减少访问延迟

**3. 兼容性：**
- 确保程序在不同GPU上的一致性
- 避免硬件相关的性能问题

**内存对齐的类型：**

**1. 自然对齐：**
- 数据类型按自然边界对齐
- float按4字节边界对齐
- double按8字节边界对齐
- 结构体按最大成员对齐

**2. 强制对齐：**
- 使用`__align__`关键字
- 指定特定的对齐边界
- 可以优化特定访问模式

**对齐策略：**

**1. 结构体对齐：**
- 将相关数据打包在一起
- 使用填充避免缓存行跨越
- 考虑访问模式优化布局

**2. 数组对齐：**
- 确保数组起始地址对齐
- 考虑步长访问的对齐要求
- 使用对齐分配函数

**实际应用示例：**
```cuda
// 内存对齐结构体
struct __align__(16) AlignedStruct {
    float x, y, z, w;
};

// 使用对齐内存
__global__ void alignedMemoryDemo(AlignedStruct* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        // 对齐访问，性能更好
        data[tid].x = 1.0f;
        data[tid].y = 2.0f;
        data[tid].z = 3.0f;
        data[tid].w = 4.0f;
    }
}
```

**内存对齐的最佳实践：**

**1. 数据布局设计：**
- 将频繁访问的数据放在一起
- 考虑缓存行大小（通常128字节）
- 避免跨缓存行的访问

**2. 分配策略：**
- 使用`cudaMallocPitch`分配2D数组
- 考虑内存对齐要求
- 避免内存碎片

**3. 访问模式优化：**
- 设计合并访问模式
- 考虑步长访问的对齐
- 使用预取技术

#### 1.3 内存传输优化
```cuda
// 异步内存传输
int main() {
    float* h_data = (float*)malloc(1024 * sizeof(float));
    float* d_data;
    
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 异步传输
    cudaMemcpyAsync(d_data, h_data, 1024 * sizeof(float), 
                    cudaMemcpyHostToDevice, stream);
    
    // 在传输的同时进行其他操作
    // ...
    
    // 等待传输完成
    cudaStreamSynchronize(stream);
    
    // 清理
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

### 2. 共享内存管理

#### 2.1 共享内存基础
```cuda
// 共享内存使用示例
__global__ void sharedMemoryDemo(float* input, float* output, int size) {
    // 声明共享内存
    __shared__ float sharedData[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    
    // 加载数据到共享内存
    if (gid < size) {
        sharedData[tid] = input[gid];
    }
    
    // 同步所有线程
    __syncthreads();
    
    // 使用共享内存进行计算
    if (tid < 256) {
        sharedData[tid] = sharedData[tid] * 2.0f;
    }
    
    // 再次同步
    __syncthreads();
    
    // 写回全局内存
    if (gid < size) {
        output[gid] = sharedData[tid];
    }
}
```

#### 2.2 共享内存银行冲突
```cuda
// 避免银行冲突的矩阵转置
__global__ void transposeMatrix(float* input, float* output, int width, int height) {
    __shared__ float sharedData[32][32];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 加载到共享内存（避免银行冲突）
    if (x < width && y < height) {
        sharedData[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // 转置写入（避免银行冲突）
    int newX = blockIdx.y * blockDim.y + threadIdx.y;
    int newY = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (newX < height && newY < width) {
        output[newY * height + newX] = sharedData[threadIdx.x][threadIdx.y];
    }
}
```

#### 2.3 动态共享内存
```cuda
// 动态共享内存使用
__global__ void dynamicSharedMemoryDemo(float* data, int size) {
    // 动态共享内存声明
    extern __shared__ float dynamicShared[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    
    // 使用动态共享内存
    if (gid < size) {
        dynamicShared[tid] = data[gid];
    }
    
    __syncthreads();
    
    // 计算
    if (tid < size) {
        dynamicShared[tid] = dynamicShared[tid] * 2.0f;
    }
    
    __syncthreads();
    
    // 写回
    if (gid < size) {
        data[gid] = dynamicShared[tid];
    }
}

// 调用时指定共享内存大小
int main() {
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // 启动内核，指定共享内存大小
    dynamicSharedMemoryDemo<<<4, 256, 256 * sizeof(float)>>>(d_data, 1024);
    
    cudaFree(d_data);
    return 0;
}
```

### 3. 寄存器内存管理

#### 3.1 寄存器使用
```cuda
// 寄存器变量使用
__global__ void registerDemo(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        // 寄存器变量
        float reg1 = data[tid];
        float reg2 = reg1 * 2.0f;
        float reg3 = reg2 + 1.0f;
        
        data[tid] = reg3;
    }
}

// 避免寄存器溢出
__global__ void avoidRegisterSpill(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        // 使用局部变量而不是大量寄存器变量
        float temp = data[tid];
        
        // 分步计算，减少寄存器使用
        temp = temp * 2.0f;
        temp = temp + 1.0f;
        temp = temp * temp;
        
        data[tid] = temp;
    }
}
```

#### 3.2 寄存器优化
```cuda
// 循环展开减少寄存器使用
__global__ void unrolledLoop(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        // 循环展开
        float sum = 0.0f;
        sum += data[tid];
        sum += data[tid + 1];
        sum += data[tid + 2];
        sum += data[tid + 3];
        
        data[tid] = sum / 4.0f;
    }
}
```

### 4. 流和事件

#### 4.1 CUDA流
```cuda
// 多流并行执行
__global__ void kernel1(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = data[tid] * 2.0f;
    }
}

__global__ void kernel2(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = data[tid] + 1.0f;
    }
}

int main() {
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // 创建多个流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // 在不同流中执行内核
    kernel1<<<4, 256, 0, stream1>>>(d_data, 512);
    kernel2<<<4, 256, 0, stream2>>>(d_data + 512, 512);
    
    // 等待所有流完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // 清理
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data);
    
    return 0;
}
```

#### 4.2 CUDA事件
```cuda
// 使用事件测量时间
int main() {
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // 创建事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    
    // 执行内核
    kernel1<<<4, 256>>>(d_data, 1024);
    
    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    
    return 0;
}
```

## 实践项目

### 项目1：内存访问优化
实现矩阵乘法的内存访问优化版本。

### 项目2：共享内存应用
使用共享内存实现高效的矩阵转置。

### 项目3：多流并行
实现多流并行的向量运算。

## 每日学习任务

### 第1天：全局内存管理
- 学习全局内存访问模式
- 理解内存对齐
- 掌握内存传输优化

### 第2天：共享内存基础
- 学习共享内存使用
- 理解同步机制
- 掌握共享内存编程

### 第3天：共享内存优化
- 学习避免银行冲突
- 掌握动态共享内存
- 理解共享内存性能优化

### 第4天：寄存器内存
- 学习寄存器使用
- 理解寄存器优化
- 掌握循环展开技术

### 第5天：流和事件
- 学习CUDA流使用
- 掌握事件计时
- 理解异步执行

### 第6天：内存优化实践
- 实现内存访问优化
- 使用共享内存优化算法
- 性能分析和调优

### 第7天：综合练习
- 完成所有实践项目
- 综合运用内存管理技术
- 准备下周学习

## 检查点

### 第2周结束时的能力要求
- [ ] 掌握全局内存访问优化
- [ ] 熟练使用共享内存
- [ ] 理解寄存器内存管理
- [ ] 能够使用CUDA流和事件
- [ ] 掌握内存传输优化技术
- [ ] 能够避免内存访问瓶颈
- [ ] 完成项目1-3
- [ ] 具备内存优化能力

## 常见问题解答

### Q: 如何判断内存访问是否合并？
A: 相邻线程访问相邻内存位置时是合并访问，可以使用Nsight Compute分析内存访问模式。

### Q: 共享内存大小如何确定？
A: 共享内存大小受硬件限制，可以通过cudaDeviceGetAttribute查询最大共享内存大小。

### Q: 如何避免寄存器溢出？
A: 减少局部变量使用，使用循环展开，或者将复杂计算分解为多个简单步骤。

### Q: 流和事件的区别？
A: 流用于异步执行，事件用于同步和计时，事件可以在流中插入同步点。

---

**学习时间**：第2周  
**预计完成时间**：2024-02-22
