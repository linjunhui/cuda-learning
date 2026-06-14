# CUDA 基础回顾

## 📚 学习目标

1. 回顾 CUDA 编程模型
2. 理解 GPU 内存层次结构
3. 掌握内存访问优化技术
4. 理解 Warp 和线程块的概念

---

## 🏗️ CUDA 编程模型

### 主机-设备架构

**主机（Host）**：
- CPU 和主机内存
- 负责程序控制和数据准备
- 执行串行代码

**设备（Device）**：
- GPU 和设备内存
- 负责并行计算
- 执行 CUDA 内核函数

### CUDA 程序结构

```cuda
#include <cuda_runtime.h>

// 设备函数（在 GPU 上执行）
__global__ void kernel_function(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// 主机函数（在 CPU 上执行）
int main() {
    // 1. 分配主机内存
    float* h_data = (float*)malloc(1024 * sizeof(float));
    
    // 2. 分配设备内存
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    
    // 3. 复制数据到设备
    cudaMemcpy(d_data, h_data, 1024 * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // 4. 启动内核
    kernel_function<<<4, 256>>>(d_data, 1024);
    
    // 5. 复制结果回主机
    cudaMemcpy(h_data, d_data, 1024 * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // 6. 清理
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

---

## 🧵 线程层次结构

### 三级层次结构

```
Grid (网格)
  └── Block (线程块)
        └── Thread (线程)
```

### 线程标识符

**内置变量**：
- `threadIdx`：线程在线程块内的索引
- `blockIdx`：线程块在网格内的索引
- `blockDim`：线程块的维度
- `gridDim`：网格的维度

**全局线程 ID 计算**：
```cuda
// 1D 情况
int global_id = blockIdx.x * blockDim.x + threadIdx.x;

// 2D 情况
int global_id_x = blockIdx.x * blockDim.x + threadIdx.x;
int global_id_y = blockIdx.y * blockDim.y + threadIdx.y;

// 3D 情况
int global_id = (blockIdx.z * gridDim.y * gridDim.x + 
                 blockIdx.y * gridDim.x + 
                 blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z) +
                (threadIdx.z * blockDim.y * blockDim.x +
                 threadIdx.y * blockDim.x +
                 threadIdx.x);
```

### Warp

**Warp 定义**：
- 32 个线程组成一个 warp
- Warp 是 GPU 调度的基本单位
- Warp 内线程执行 SIMT（单指令多线程）

**Warp ID 和 Lane ID**：
```cuda
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;
```

**Warp 级操作**：
```cuda
// Shuffle 操作（warp 内数据交换）
float val = __shfl_sync(0xffffffff, val, lane_id + 1);

// Warp 级归约
float sum = __shfl_down_sync(0xffffffff, val, 1);
sum += __shfl_down_sync(0xffffffff, sum, 2);
sum += __shfl_down_sync(0xffffffff, sum, 4);
sum += __shfl_down_sync(0xffffffff, sum, 8);
sum += __shfl_down_sync(0xffffffff, sum, 16);
```

---

## 💾 GPU 内存层次结构

### 内存层次（从快到慢）

```
寄存器 (Registers)
  ↓ (~1000 TB/s)
共享内存 (Shared Memory)
  ↓ (~100 TB/s)
L1 缓存 (L1 Cache)
  ↓ (~10 TB/s)
L2 缓存 (L2 Cache)
  ↓ (~1 TB/s)
全局内存 (Global Memory)
```

### 寄存器（Registers）

**特点**：
- 最快的内存
- 每个线程私有
- 容量有限（每个线程 ~255 个寄存器）

**使用**：
```cuda
__global__ void kernel() {
    float a = 1.0f;  // 存储在寄存器
    float b = 2.0f;  // 存储在寄存器
    float c = a + b; // 存储在寄存器
}
```

**限制**：
- 寄存器溢出会使用本地内存（L1 缓存）
- 影响性能

### 共享内存（Shared Memory）

**特点**：
- 很快的内存（~100 TB/s）
- 线程块内共享
- 容量有限（每个 SM 48KB 或 164KB）

**声明**：
```cuda
__global__ void kernel() {
    __shared__ float shared_data[256];  // 共享内存
    
    int tid = threadIdx.x;
    shared_data[tid] = ...;
    __syncthreads();  // 同步所有线程
    ...
}
```

**Bank Conflict**：
- 共享内存分成 32 个 bank
- 如果多个线程访问同一个 bank，会产生冲突
- 需要避免 bank conflict

**避免 Bank Conflict**：
```cuda
// ❌ 有 bank conflict
__shared__ float data[32];
data[threadIdx.x] = ...;  // 所有线程访问不同 bank，但可能冲突

// ✅ 无 bank conflict（使用 padding）
__shared__ float data[33];  // 33 = 32 + 1 (padding)
data[threadIdx.x] = ...;  // 避免冲突
```

### 全局内存（Global Memory）

**特点**：
- 最慢的内存（~1 TB/s）
- 所有线程可访问
- 容量大（几 GB 到几十 GB）

**内存合并访问（Coalesced Access）**：
- 相邻线程访问相邻内存位置
- 硬件可以将多个访问合并为一个事务
- 显著提高内存带宽利用率

**合并访问示例**：
```cuda
// ✅ 合并访问（相邻线程访问相邻内存）
__global__ void coalesced_access(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;  // 线程 0 访问 data[0]，线程 1 访问 data[1]，...
}

// ❌ 非合并访问（线程访问不连续的内存）
__global__ void non_coalesced_access(float* data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx * stride] = data[idx * stride] * 2.0f;  // 访问间隔大
}
```

---

## 🚀 内存访问优化

### 内存合并访问规则

**规则**：
1. 线程访问的内存地址必须连续
2. 访问的起始地址必须对齐（128 字节对齐）
3. 访问大小必须是 1、2、4、8 或 16 字节

**示例**：
```cuda
// ✅ 128 字节对齐，连续访问
__global__ void aligned_access(float4* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = data[idx];  // 16 字节对齐访问
}

// ❌ 未对齐访问
__global__ void unaligned_access(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx + 1] = ...;  // 未对齐
}
```

### 共享内存优化

**使用共享内存缓存数据**：
```cuda
__global__ void shared_memory_cache(float* input, float* output, int n) {
    __shared__ float cache[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 从全局内存加载到共享内存
    if (idx < n) {
        cache[tid] = input[idx];
    }
    __syncthreads();
    
    // 从共享内存读取（更快）
    if (idx < n) {
        output[idx] = cache[tid] * 2.0f;
    }
}
```

### 预取（Prefetching）

**预取数据到共享内存**：
```cuda
__global__ void prefetch_kernel(float* data, int n) {
    __shared__ float tile[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 预取下一个块的数据
    if (idx + blockDim.x < n) {
        tile[tid] = data[idx + blockDim.x];
    }
    
    // 处理当前数据
    float val = data[idx];
    // ... 计算 ...
    
    __syncthreads();
    
    // 使用预取的数据
    if (idx + blockDim.x < n) {
        float next_val = tile[tid];
        // ... 计算 ...
    }
}
```

---

## 🔧 Flash-Attention 中的内存优化

### Q、K、V 的加载

**策略**：
1. 从全局内存加载到共享内存
2. 从共享内存加载到寄存器
3. 在寄存器中计算

**代码示例**（简化）：
```cuda
__global__ void flash_attention_kernel(...) {
    __shared__ float q_tile[64][128];  // Q 块
    __shared__ float k_tile[64][128];  // K 块
    __shared__ float v_tile[64][128];  // V 块
    
    // 1. 从全局内存加载到共享内存（合并访问）
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // 加载 Q 块
    for (int i = 0; i < 4; i++) {
        int row = warp_id * 4 + i;
        int col = lane_id * 4;
        if (row < 64 && col < 128) {
            q_tile[row][col] = q_global[row][col];
        }
    }
    __syncthreads();
    
    // 2. 从共享内存加载到寄存器
    float q_reg[4];
    for (int i = 0; i < 4; i++) {
        q_reg[i] = q_tile[warp_id][lane_id * 4 + i];
    }
    
    // 3. 计算（在寄存器中）
    // ...
}
```

### 内存访问模式优化

**Flash-Attention 的优化**：
1. **合并访问**：Q、K、V 的加载使用合并访问
2. **共享内存缓存**：使用共享内存缓存块数据
3. **寄存器优化**：中间结果存储在寄存器
4. **避免 Bank Conflict**：使用合适的共享内存布局

---

## 📊 性能分析工具

### Nsight Compute

**功能**：
- 分析内核性能
- 内存访问分析
- 占用率分析

**使用**：
```bash
ncu --set full ./your_program
```

### Nsight Systems

**功能**：
- 整体性能分析
- 时间线分析
- 内存使用分析

**使用**：
```bash
nsys profile ./your_program
```

---

## 🎯 关键要点总结

### CUDA 编程要点

1. **内存层次**：
   - 寄存器最快，但容量小
   - 共享内存快，但需要避免 bank conflict
   - 全局内存慢，但需要合并访问

2. **线程组织**：
   - Warp 是调度的基本单位
   - 线程块内可以共享内存和同步
   - 合理组织线程可以提高性能

3. **内存优化**：
   - 使用共享内存缓存数据
   - 确保内存合并访问
   - 避免 bank conflict

### Flash-Attention 中的应用

1. **Q、K、V 加载**：使用合并访问和共享内存
2. **中间结果**：存储在寄存器，避免写回全局内存
3. **输出**：只写一次，减少内存访问

---

## 📝 学习检查点

- [ ] 理解 CUDA 编程模型
- [ ] 理解 GPU 内存层次结构
- [ ] 理解内存合并访问
- [ ] 理解共享内存的使用
- [ ] 理解 Warp 的概念
- [ ] 能够分析内存访问模式

---

## 📚 参考资源

- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

**学习时间**：1-2 天  
**难度**：⭐⭐⭐☆☆
