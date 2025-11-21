# CudaSteps 项目详细总结（代码详解版）

本文档详细总结了 CudaSteps 项目中每个章节的代码，包含问题说明、实现思路和代码详解，便于背诵和复习。

---

## 第1章：CUDA架构基础（SM与Subcore）

> **注意**：本章节内容基于**H100（Hopper架构，SM 9.0）**的SM结构图。不同GPU架构的实现细节可能有所不同。

### SM（Streaming Multiprocessor）架构说明

#### SM内部结构

一个SM内可以划分为若干模块（通常为4个），有些人把这个子模块称为**subcore**，在AMD上叫SIMD单元。

**关键特性**：
- **每个subcore的寄存器上限**：64KB，即16384个int
- **SM共享内存**：一个SM内有一块共享的shared memory，它与L1 Data Cache在同一块物理内存上，一般配置为64KB或96KB

#### 线程块和Warp的驻留规则

- **线程块（Block）驻留**：一个线程块（Block）只会驻留在一个SM上
- **Warp驻留**：一个warp只会驻留在一个subcore上
- **资源调度**：只要资源足够（主要指寄存器与shared memory），一个SM可以调度多个Block

#### 资源限制与错误处理

**线程块大小限制**：
- 一个Block内最多1024个thread
- 如果thread用到的寄存器比较多，则起不了这么多线程
- 这时会出现"**too many resources requested for launch**"错误

**错误检测最佳实践**：
- 在核函数后调用`cudaGetLastError()`是一个好习惯，它可以检查核函数启动阶段的错误
- 如果直接调用`cudaDeviceSynchronize()`并检查返回码，在核函数启动失败时会出现段错误（exit code 139）
- 这种错误比较隐蔽，需要特别注意

#### Warp执行模型

**Warp调度机制**：
- 一个subcore可以调度执行多个warp
- 在一个warp访存时切换至别的warp，以隐藏访存延迟

**Warp执行特点**（基于H100架构，4个subcore）：
- **重要理解**：虽然一个warp有32个线程，但严格意义上并不是32个线程同时执行的
- **实际执行方式**（H100架构）：
  - 对于int32或fp32计算：可以同时执行4个warp的部分线程（warp0、warp1、warp2、warp3的0-15号线程），接着执行它们的16-31号线程
  - 对于double计算：需要拆分为4次执行（因为double是64位，需要更多资源）
- **执行单位**：warp内的线程是分批次执行的，根据数据类型以16个线程（或8个线程）为单位执行
- **架构说明**：这是H100（Hopper架构）的实现细节，不同架构的GPU可能有不同的执行方式

**Warp原子性**：
- 虽然warp内时分复用，先后在宽度为16（或8）的lane上执行
- 但是这个过程是原子的，warp是最小的事务单元
- 从外面看起来好像一个warp的32个thread是同步执行的，但实际上内部是分批次执行的

**Tensor Core**：
- Tensor core是针对warp而言的
- 所以基于tensor core的编程都是warp级别的

#### 线程块数配置建议

**基本配置原则**：
- **总线程数**：一般 >= 总的任务数
- **块内线程数**：一般设为32的整数倍（128、256、512）
- **线程块数**：一般设为总的线程数/blockDim并上取整
- **大规模任务**：如果任务数特别多（上亿这样），则考虑使用网格跨步循环

**性能优化考虑**：
- **线程块数多**：可以充分利用所有的SM（隐藏访存延迟等）
- **块间同步**：如果需要在线程块与块间同步，若SM上的一个线程块处于等待，SM可以立即调度别的线程块，保证SM不空闲

---

## 第2章：CUDA中的线程组织

### 代码文件：`capter2/hello.cu`

#### 问题说明
**要解决的问题**：编写第一个CUDA程序，理解CUDA中线程的组织方式和内建变量的使用。

**背景**：CUDA采用"单指令多线程"(SIMT)执行模型，需要理解如何组织线程、如何标识每个线程的身份。

#### 实现思路
1. 使用`__global__`限定符定义核函数
2. 通过`<<<grid_size, block_size>>>`指定线程配置
3. 使用内建变量（`blockIdx`, `threadIdx`, `blockDim`, `gridDim`）标识线程身份
4. 使用`cudaDeviceSynchronize()`同步主机和设备

#### 代码详解

```cuda
/*
基本思想是：演示CUDA线程组织方式，每个线程打印自己的身份信息（线程块索引和线程索引）
局限性：无，这是演示程序
*/
__global__ void hell_from__gpu()
{
    // 核函数不支持 c++ 的 iostream，但支持printf
    
    // 获取线程块在网格中的索引（三维）
    const int bx = blockIdx.x;   // 线程块在x方向的索引
    const int by = blockIdx.y;   // 线程块在y方向的索引
    const int bz = blockIdx.z;   // 线程块在z方向的索引

    // 获取线程在线程块中的索引（三维）
    const int tx = threadIdx.x;  // 线程在x方向的索引
    const int ty = threadIdx.y;  // 线程在y方向的索引
    const int tz = threadIdx.z;  // 线程在z方向的索引

    // 打印线程和线程块的身份信息
    printf("gpu: hello world! block(%d, %d, %d) -- thread(%d, %d, %d)\n", 
           bx, by, bz, tx, ty, tz);
}

int main()
{
    printf("nvcc: hello world!\n");

    // 定义线程块大小：2行4列，共8个线程
    const dim3 block_size(2, 4);
    
    // 启动核函数：1个线程块，每个线程块8个线程
    hell_from__gpu<<<1, block_size>>>();
    
    // 同步主机和设备，确保核函数执行完成，否则printf可能无法输出
    cudaDeviceSynchronize();

    return 0;
}
```

**关键点说明**：
- `__global__`：核函数限定符，由主机调用，在设备执行
- `<<<1, block_size>>>`：第一个参数是网格大小（线程块个数），第二个是线程块大小
- `dim3`：三维向量类型，未指定的维度默认为1
- `cudaDeviceSynchronize()`：强制主机等待所有设备操作完成

---

## 第3章：简单CUDA程序的基本框架

### 代码文件：`capter3/add.cu`

#### 问题说明
**要解决的问题**：实现数组相加 `z[i] = x[i] + y[i]`，展示完整的CUDA程序框架。

**背景**：CUDA程序需要遵循特定的框架：内存分配→数据传输→核函数调用→结果回传→内存释放。

#### 实现思路
1. 主机端分配内存（主机内存和设备内存）
2. 初始化主机数据
3. 将数据从主机复制到设备（H2D）
4. 调用核函数进行计算
5. 将结果从设备复制回主机（D2H）
6. 验证结果并释放内存

#### 代码详解

```cuda
/*
基本思想是：每个线程处理一个数组元素，通过线程索引计算全局索引，实现z[i] = x[i] + y[i]
局限性：需要边界检查防止越界，grid_size计算需要向上取整
*/
// 核函数：数组相加
__global__ void add(const double *x, const double *y, double *z, const int N)
{
    // 计算当前线程对应的数组索引
    // 一维线程索引 = 线程块大小 × 线程块索引 + 线程索引
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 边界检查：防止越界
    if (n > N) return;

    // 根据索引选择不同的设备函数调用方式（演示重载）
    if (n%5 == 0)
    {
        // 使用返回值的版本
        z[n] = add_in_device(x[n], y[n]);
    }
    else
    {
        // 使用引用参数的版本
        add_in_device(x[n], y[n], z[n]);
    }
}

// 设备函数：返回值的版本
__device__ double add_in_device(const double x, const double y)
{
    return x + y;
}

// 设备函数：引用参数的版本（重载）
__device__ void add_in_device(const double x, const double y, double &z)
{
    z = x + y;
}

int main()
{
    const int N = 1e4;           // 数组大小
    const int M = sizeof(double) * N;  // 内存大小（字节）

    // ========== 步骤1：分配主机内存 ==========
    double *h_x = new double[N];      // 使用new分配
    double *h_y = (double*) malloc(M); // 使用malloc分配
    double *h_z = (double*) malloc(M);

    // ========== 步骤2：初始化主机数据 ==========
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = a;  // a = 1.23
        h_y[i] = b;  // b = 2.34
    }

    // ========== 步骤3：分配设备内存 ==========
    double *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, M);  // 注意：需要传递指针的地址
    cudaMalloc((void**)&d_y, M);
    cudaMalloc((void**)&d_z, M);

    // ========== 步骤4：从主机复制数据到设备 ==========
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
    // 注意：标量参数（如int N）不需要cudaMemcpy，直接通过值传递即可
    // 核函数参数中的标量值会在核函数启动时自动从主机复制到设备

    // ========== 步骤5：调用核函数进行计算 ==========
    const int block_size = 128;                    // 每个线程块的线程数
    const int grid_size = N/128 + 1;               // 线程块个数（向上取整）
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);  // N是标量，直接传递，无需cudaMemcpy

    // ========== 步骤6：从设备复制结果到主机 ==========
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    
    // ========== 步骤7：验证结果 ==========
    check(h_z, N);  // 检查结果是否正确（期望值 c = 3.57）

    // ========== 步骤8：释放内存 ==========
    delete[] h_x;   // 释放new分配的内存
    free(h_y);      // 释放malloc分配的内存
    free(h_z);
    cudaFree(d_x);  // 释放设备内存
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}
```

**关键点说明**：
- **SIMT模型**：每个线程执行相同代码，通过线程索引区分处理的数据
- **线程索引计算**：`n = blockDim.x * blockIdx.x + threadIdx.x`
- **设备函数**：`__device__`限定符，只能被核函数或其他设备函数调用
- **函数重载**：CUDA支持C++的函数重载机制
- **标量参数传递**：核函数中的标量参数（如`int N`、`float threshold`等基本类型）不需要使用`cudaMemcpy`，CUDA运行时会自动在核函数启动时将标量值从主机复制到设备。只有数组/指针类型的参数才需要先分配设备内存并使用`cudaMemcpy`传输数据
- **标量值的设备内存存储**：虽然单个标量值很小，但如果使用`cudaMalloc`在设备上分配内存存储标量（如`real *d_y2`），仍然需要传输数据，因为数据存储在设备内存中。初始化设备内存中的标量有三种方法：
  - **`cudaMemset`**：适合初始化为0，性能最好（如`cudaMemset(d_y2, 0, sizeof(real))`）
  - **`cudaMemcpy`**：通用方法，可以初始化为任意值
  - **核函数初始化**：在核函数中初始化（适合需要复杂初始化逻辑的情况）
  - **不能直接赋值**：不能使用`d_y2 = 0.0`或`*d_y2 = 0.0`，因为主机端无法直接访问设备内存

---

## 第4章：CUDA程序的错误检测

### 代码文件：`capter4/check.cu`

#### 问题说明
**要解决的问题**：实现CUDA运行时错误的检测和报告机制，确保程序能够及时发现和定位错误。

**背景**：CUDA API返回`cudaError_t`类型，核函数调用是异步的且没有返回值，需要特殊的错误检测机制。

#### 实现思路
1. 定义CHECK宏函数检查`cudaError_t`返回值
2. 使用`cudaGetLastError()`检测核函数错误
3. 使用`cudaDeviceSynchronize()`同步主机和设备
4. 错误发生时打印详细信息并退出

#### 代码详解

```cuda
/*
基本思想是：封装CUDA API调用，自动检测错误并打印详细信息，确保程序能及时发现错误
局限性：使用exit(1)退出，可能不适合需要清理资源的场景
*/
// 错误检测宏（定义在error.cuh中）
#define CHECK(call)                                                     \
do {                                                                    \
    const cudaError_t error_code = call;                                \
    if (error_code != cudaSuccess)                                      \
    {                                                                   \
        printf("CUDA ERROR: \n");                                       \
        printf("    FILE: %s\n", __FILE__);                             \
        printf("    LINE: %d\n", __LINE__);                             \
        printf("    ERROR CODE: %d\n", error_code);                     \
        printf("    ERROR TEXT: %s\n", cudaGetErrorString(error_code)); \
        exit(1);                                                        \
    }                                                                   \
}while(0);

int main()
{
    const int N = 1e4;
    const int M = sizeof(double) * N;

    // 分配主机内存
    double *h_x = new double[N];
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = a;
        h_y[i] = b;
    }

    // ========== 使用CHECK宏检测设备内存分配 ==========
    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void**)&d_x, M));  // 如果失败会打印错误并退出
    CHECK(cudaMalloc((void**)&d_y, M));
    CHECK(cudaMalloc((void**)&d_z, M));

    // ========== 使用CHECK宏检测数据传输 ==========
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    // ========== 核函数错误检测 ==========
    const int block_size = 128;
    const int grid_size = N/128 + 1;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    
    // 核函数调用是异步的，需要先检查启动错误
    CHECK(cudaGetLastError());  // 捕捉同步前的最后一个错误
    
    // 然后同步设备，确保核函数执行完成
    CHECK(cudaDeviceSynchronize());  // 同步以捕获核函数执行错误

    // ========== 结果回传和验证 ==========
    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    // ========== 释放内存 ==========
    if (h_x) delete[] h_x;
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));

    return 0;
}
```

**关键点说明**：
- **CHECK宏**：使用`do-while(0)`确保宏可以安全地用在任何上下文中
- **核函数错误检测**：需要`cudaGetLastError()` + `cudaDeviceSynchronize()`组合
- **同步机制**：
  - `cudaMemcpy`：隐式同步（阻塞）
  - `cudaDeviceSynchronize()`：显式同步（阻塞）
  - 核函数调用：异步（非阻塞）

---

## 第5章：GPU加速的关键

### 代码文件：`capter5/add.cu` 和 `capter5/clock.cu`

#### 问题说明
**要解决的问题**：分析影响GPU加速效果的关键因素，实现精确的性能计时。

**背景**：GPU加速效果受多种因素影响，需要精确计时来分析性能瓶颈，比较单精度和双精度性能差异。

#### 实现思路
1. 使用CUDA事件精确计时
2. 分别计时内存分配、数据传输、核函数执行等步骤
3. 比较单精度和双精度性能差异
4. 分析数据传输时间与计算时间的比例

#### 代码详解

```cuda
/*
基本思想是：使用CUDA事件精确计时，分别测量主机内存分配、设备内存分配、核函数执行、数据传输等各步骤的耗时
局限性：需要手动管理事件的生命周期，累加计时需要仔细处理
*/
// clock.cu - 性能计时实现
void cuda_clock()
{
    const int N = 1e6;
    const int M = sizeof(real) * N;

    // ========== 创建CUDA事件用于计时 ==========
    float elapsed_time = 0;
    float curr_time = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));  // 创建开始事件
    CHECK(cudaEventCreate(&stop));   // 创建结束事件
    CHECK(cudaEventRecord(start));   // 记录开始时间
    cudaEventQuery(start);           // 强制刷新CUDA执行流

    // ========== 步骤1：主机内存分配和初始化 ==========
    real *h_x, *h_y, *h_z;
    h_x = new real[N];
    h_y = new real[N];
    h_z = new real[N];
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = a;  // a = 1.23
        h_y[i] = b;  // b = 2.34
    }

    // 记录步骤1的耗时
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));  // 强制同步
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));  // 计算时间差（毫秒）
    printf("host memory malloc and copy: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // ========== 步骤2：设备内存分配和数据传输 ==========
    real *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMalloc(&d_y, M));
    CHECK(cudaMalloc(&d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyDefault));

    // 记录步骤2的耗时
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    printf("device memory malloc: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // ========== 步骤3：核函数执行 ==========
    const int block_size = 128;
    const int grid_size = N/block_size + 1;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    // 记录步骤3的耗时
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    printf("kernel function : %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // ========== 步骤4：结果回传 ==========
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDefault));
    check(h_z, N);

    // 记录步骤4的耗时
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&curr_time, start, stop));
    printf("copy from device to host: %f ms.\n", curr_time - elapsed_time);
    elapsed_time = curr_time;

    // ========== 清理 ==========
    if (h_x) delete[] h_x;
    if (h_y) delete[] h_y;
    if (h_z) delete[] h_z;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
}
```

**关键点说明**：
- **CUDA事件计时**：比CPU计时更精确，直接在GPU上记录时间
- **累加计时**：使用`elapsed_time`记录累计时间，每次输出增量时间
- **单精度vs双精度**：通过`USE_DP`宏控制，双精度通常慢2倍
- **性能分析**：可以清楚看到数据传输和计算的时间占比

### 性能分析工具：nvprof

> **注意**：nvprof 在 CUDA 11.0+ 版本中已被弃用，推荐使用 Nsight Systems 和 Nsight Compute。但如果你使用的是较旧版本的 CUDA（CUDA 10.x 及更早版本），nvprof 仍然可用。

nvprof 是 NVIDIA 提供的命令行性能分析工具，可以分析 CUDA 程序的性能瓶颈。

#### 基本使用方法

```bash
# 基本用法：分析程序执行
nvprof ./a.out

# 指定输出文件
nvprof -o output.nvprof ./a.out

# 只显示摘要信息
nvprof --print-summary ./a.out

# 分析并生成 CSV 报告
nvprof --csv --log-file output.csv ./a.out
```

#### 常用分析命令

```bash
# 分析 CUDA API 调用时间
nvprof --print-api-trace ./a.out

# 分析核函数执行时间
nvprof --print-gpu-trace ./a.out

# 分析核函数性能指标（占用率、内存吞吐量等）
nvprof --print-gpu-summary ./a.out

# 分析特定指标
nvprof --metrics achieved_occupancy,shared_memory_utilization ./a.out

# 分析内存传输
nvprof --print-gpu-trace --print-api-trace ./a.out

# 设置分析时间范围（秒）
nvprof --print-gpu-trace --timeout 10 ./a.out
```

#### 常用性能指标

```bash
# 分析占用率相关指标
nvprof --metrics achieved_occupancy,theoretical_occupancy ./a.out

# 分析内存相关指标
nvprof --metrics gld_throughput,gst_throughput,shared_load_throughput ./a.out

# 分析计算相关指标
nvprof --metrics flop_count_dp,flop_count_sp ./a.out

# 分析内存合并度
nvprof --metrics gld_efficiency,gst_efficiency ./a.out

# 综合性能分析
nvprof --metrics \
    achieved_occupancy,\
    gld_efficiency,\
    gst_efficiency,\
    shared_memory_utilization,\
    flop_count_dp \
    ./a.out
```

#### 输出报告解读

**API 调用跟踪示例**：

```
==12345== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
  99.2%  109.598ms         2  54.799ms  54.799ms  54.799ms  cudaMalloc
   0.6%    1.428ms         1   1.428ms   1.428ms   1.428ms  cudaLaunchKernel
   0.2%    0.432ms         2   0.216ms   0.216ms   0.216ms  cudaMemcpy
```

**核函数跟踪示例**：

```
==12345== GPU activities:
Time(%)      Time     Calls       Avg       Min       Max  Name
  95.0%   95.000ms         1  95.000ms  95.000ms  95.000ms  add(float*, float*, float*, int)
   5.0%    5.000ms         1   5.000ms   5.000ms   5.000ms  [CUDA memcpy HtoD]
```

**性能指标示例**：

```
==12345== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "Tesla V100-PCIE-32GB (0)"
    Kernel: add(float*, float*, float*, int)
          1                    achieved_occupancy                    Achieved Occupancy         0.500000    0.500000    0.500000
          1                    gld_efficiency                      Global Load Efficiency     100.00%     100.00%     100.00%
          1                    gst_efficiency                      Global Store Efficiency    100.00%     100.00%     100.00%
```

**报告字段说明**：
- **Time(%)**：该操作占总时间的百分比
- **Time**：该操作的总耗时
- **Calls**：调用次数
- **Avg/Min/Max**：平均/最小/最大耗时
- **Metric Value**：性能指标的具体数值

#### 实用技巧

```bash
# 分析带参数的程序
nvprof ./a.out arg1 arg2

# 只分析特定设备
CUDA_VISIBLE_DEVICES=0 nvprof ./a.out

# 保存详细日志到文件
nvprof --log-file profile.log --print-gpu-trace --print-api-trace ./a.out

# 分析并导出为可视化格式
nvprof --export profile.nvprof ./a.out
# 然后使用 nvvp（NVIDIA Visual Profiler）打开 profile.nvprof

# 过滤特定核函数
nvprof --print-gpu-trace --kernels "add" ./a.out

# 分析特定时间范围
nvprof --print-gpu-trace --timeout 5 ./a.out
```

#### nvprof 与 Nsight Systems 对比

| 特性 | nvprof | Nsight Systems |
|------|--------|----------------|
| CUDA 版本支持 | CUDA 10.x 及更早 | CUDA 11.0+ |
| 命令行工具 | ✅ | ✅ |
| 图形界面 | nvvp（独立工具） | nsys-ui（集成） |
| 时间线分析 | ✅ | ✅（更强大） |
| 性能指标 | ✅ | ✅（更全面） |
| 内存分析 | 基础 | 详细 |
| 推荐使用 | CUDA 10.x 及更早 | CUDA 11.0+ |

**迁移建议**：
- 如果使用 CUDA 11.0+，建议迁移到 Nsight Systems
- 如果仍使用 CUDA 10.x，可以继续使用 nvprof
- Nsight Systems 提供更强大的分析和可视化功能

### 性能分析工具：Nsight Systems

除了使用CUDA事件进行手动计时外，还可以使用NVIDIA提供的专业性能分析工具Nsight Systems进行更全面的性能分析。

#### 基本使用方法

```bash
# 基本用法：分析程序执行
nsys profile ./a.out

# 指定输出文件名
nsys profile -o output.nsys-rep ./a.out

# 分析并生成统计报告
nsys profile --stats=true ./a.out
```

#### 常用分析命令

```bash
# 完整分析（包含所有信息）
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profile_output \
    --force-overwrite=true \
    ./a.out

# 只分析 CUDA 相关操作
nsys profile --trace=cuda ./a.out

# 包含时间线信息
nsys profile --trace=cuda,nvtx --output=timeline ./a.out

# 设置采样频率（默认1000Hz）
nsys profile --sampling-frequency=2000 ./a.out
```

#### 生成报告

```bash
# 生成文本报告
nsys stats profile_output.nsys-rep

# 生成详细报告
nsys stats --report gputrace profile_output.nsys-rep

# 生成 CUDA API 调用统计
nsys stats --report cudaapis profile_output.nsys-rep

# 生成内核执行统计
nsys stats --report gpukernsum profile_output.nsys-rep
```

#### 完整分析流程示例

```bash
# 步骤1：运行分析，生成 .nsys-rep 文件
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --output=my_profile \
    --force-overwrite=true \
    ./a.out

# 步骤2：查看统计信息
nsys stats my_profile.nsys-rep

# 步骤3：生成特定报告
nsys stats --report gputrace my_profile.nsys-rep > gpu_trace.txt
nsys stats --report gpukernsum my_profile.nsys-rep > kernel_summary.txt
```

#### 查看图形界面

```bash
# 使用 Nsight Systems GUI 打开报告文件
nsys-ui my_profile.nsys-rep

# 或者直接双击 .nsys-rep 文件（如果已安装 GUI）
```

#### 输出报告解读

生成的报告通常包含：
- **GPU 时间线**：显示核函数执行时间
- **CPU 时间线**：显示主机端活动
- **CUDA API 调用**：显示所有 CUDA API 调用
- **内存传输**：显示 H2D 和 D2H 传输
- **内核统计**：每个核函数的执行时间和调用次数

**CUDA API 调用报告示例解读**：

```
Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)  Max (ns)   StdDev (ns)           Name         
--------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ----------------------
    99.2        219196859          2  109598429.5  109598429.5      4017  219192842  154989904.5  cudaMalloc            
     0.6          1428098          1    1428098.0    1428098.0   1428098    1428098          0.0  cudaLaunchKernel      
     0.2           431833          2     215916.5     215916.5     95873     335960     169767.1  cudaMemcpy            
     0.0             2554          1       2554.0       2554.0      2554       2554          0.0  cudaDeviceSynchronize
```

**报告字段说明**：
- **Time (%)**：该API调用占总时间的百分比
- **Total Time (ns)**：该API调用的总耗时（纳秒）
- **Num Calls**：调用次数
- **Avg/Med/Min/Max (ns)**：平均/中位数/最小/最大耗时
- **StdDev (ns)**：标准差，反映耗时的波动程度

**性能问题诊断**：
- **cudaMalloc 耗时过长**（如占99%以上）：
  - 可能原因：分配了非常大的内存，或首次分配触发了设备初始化
  - 优化建议：检查分配大小，考虑内存复用，使用内存池或预分配策略
- **cudaMemcpy 耗时高**：
  - 可能原因：数据传输量大，或使用了可分页内存
  - 优化建议：减少数据传输量，使用`cudaMallocHost`分配不可分页内存，使用`cudaMemcpyAsync`异步传输
- **cudaLaunchKernel 耗时异常**：
  - 可能原因：核函数启动配置不当，或首次启动有初始化开销
  - 优化建议：检查线程配置，避免过度启动小核函数

#### 实用技巧

```bash
# 分析带参数的程序
nsys profile -o profile ./a.out arg1 arg2

# 设置环境变量并分析
CUDA_VISIBLE_DEVICES=0 nsys profile ./a.out

# 快速分析（只关注 CUDA）
nsys profile --trace=cuda -o quick_profile ./a.out
nsys stats quick_profile.nsys-rep
```

**Nsight Systems 优势**：
- **可视化时间线**：直观查看GPU和CPU的活动
- **自动统计**：自动生成详细的性能统计报告
- **无需修改代码**：不需要在代码中添加计时代码
- **全面分析**：可以分析内存使用、API调用、内核执行等各个方面

---

## 第6章：CUDA内存组织

### 代码文件1：`capter6/query.cu`

#### 问题说明
**要解决的问题**：查询GPU设备的详细规格信息，了解硬件限制和性能参数。

**背景**：不同GPU有不同的计算能力、内存大小、线程块限制等，需要查询这些信息来优化程序。

#### 实现思路
使用`cudaGetDeviceProperties()`函数获取设备属性结构体，然后打印各项规格信息。

#### 代码详解

```cuda
/*
基本思想是：查询GPU设备的详细规格信息，包括计算能力、内存大小、线程块限制等硬件参数
局限性：无，这是查询程序
*/
int main(int argc, char *argv[])
{
    // 获取设备ID（默认为0，可通过命令行参数指定）
    int device_id = 0;
    if (argc > 1) device_id = atoi(argv[1]);

    CHECK(cudaSetDevice(device_id));  // 设置当前设备

    // ========== 查询设备属性 ==========
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device_id));

    // ========== 打印设备信息 ==========
    printf("Device id: %d\n", device_id);
    printf("Device name: %s\n", prop.name);                    // GPU名称
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);  // 计算能力
    printf("Amount of global memory: %g GB\n", prop.totalGlobalMem/(1024.0*1024*1024));
    printf("Amount of constant memory: %g KB\n", prop.totalConstMem/1024.0);
    printf("Maximum grid size: %d, %d, %d\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size: %d, %d, %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);  // SM数量
    printf("Maximum amount of shared memory per block: %g KB\n", prop.sharedMemPerBlock/1024.0);
    printf("Maximum amount of shared memory per SM: %g KB\n", prop.sharedMemPerMultiprocessor/1024.0);
    printf("Maximum number of registers per block: %d K\n", prop.regsPerBlock/1024);
    printf("Maximum number of registers per SM: %d K\n", prop.regsPerMultiprocessor/1024);
    printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);

    return 0;
}
```

**关键点说明**：
- **设备属性结构体**：`cudaDeviceProp`包含所有设备规格信息
- **计算能力**：`major.minor`格式，决定支持哪些CUDA特性
- **SM数量**：影响并行度和性能上限

---

### 代码文件2：`capter6/static.cu`

#### 问题说明
**要解决的问题**：演示静态全局内存和常量内存的使用方法，理解它们与主机内存的交互方式。

**背景**：CUDA有多种内存类型，静态全局内存和常量内存可以直接在核函数中访问，但主机需要通过特殊API访问。

#### 实现思路
1. 定义静态全局内存变量（`__device__`）和常量内存变量（`__constant__`）
2. 在核函数中直接访问这些变量
3. 主机端使用`cudaMemcpyToSymbol`和`cudaMemcpyFromSymbol`进行数据传输

#### 代码详解

```cuda
/*
基本思想是：演示静态全局内存和常量内存的使用，核函数中直接访问，主机端通过Symbol函数访问
局限性：静态全局内存大小在编译期确定，常量内存有64KB限制，主机端访问必须使用特殊API
*/
// ========== 静态全局内存变量（设备内存） ==========
__device__ int d_x = 1;           // 单个变量
__device__ int d_y[2] = {2, 3};   // 数组

// ========== 常量内存变量（设备内存，只读） ==========
__constant__ double d_m = 23.33;              // 单个变量
__constant__ double d_n[] = {12.2, 34.1, 14.3};  // 数组

// ========== 核函数：修改静态全局内存 ==========
__global__ void add_array()
{
    // 静态全局内存可读可写
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_y: {%d, %d}\n", d_y[0], d_y[1]);
}

__global__ void add_var()
{
    d_x += 2;
    printf("d_x: %d\n", d_x);
}

__global__ void display()
{
    printf("d_x: %d, d_y: {%d, %d}\n", d_x, d_y[0], d_y[1]);
}

// ========== 核函数：读取常量内存 ==========
__global__ void show()
{
    // 常量内存变量在核函数中不可更改（只读）
    printf("d_m: %f, d_n: {%f, %f, %f}\n", d_m, d_n[0], d_n[1], d_n[2]);
}

int main()
{
    // ========== 核函数中直接访问 ==========
    display<<<1, 1>>>();           // 显示初始值
    add_array<<<1, 1>>>();         // 修改d_y
    add_var<<<1, 1>>>();          // 修改d_x
    CHECK(cudaDeviceSynchronize());

    show<<<1, 1>>>();              // 显示常量内存
    CHECK(cudaDeviceSynchronize());

    // ========== 主机向设备复制数据 ==========
    int h_y[2] = {10, 20};
    int h_x = 7;
    double h_m = 22.23;
    double h_n[3] = {1.1, 2.2, 3.3};

    // 使用cudaMemcpyToSymbol从主机复制到设备
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));  // 数组需要传数组名
    CHECK(cudaMemcpyToSymbol(d_x, &h_x, sizeof(int)));     // 非数组需要取地址
    display<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpyToSymbol(d_m, &h_m, sizeof(double)));
    CHECK(cudaMemcpyToSymbol(d_n, h_n, sizeof(double)*3));
    show<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    // ========== 设备向主机复制数据 ==========
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    CHECK(cudaMemcpyFromSymbol(&h_x, d_x, sizeof(int)));
    printf("host, h_y: %d, %d, h_x: %d\n", h_y[0], h_y[1], h_x);
    display<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpyFromSymbol(h_n, d_n, sizeof(double)*3));
    CHECK(cudaMemcpyFromSymbol(&h_m, d_m, sizeof(double)));
    printf("host, h_n: %f, %f, %f, h_m: %f\n", h_n[0], h_n[1], h_n[2], h_m);
    show<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    return 0;
}
```

**关键点说明**：
- **静态全局内存**：`__device__`，可读可写，所有线程可见
- **常量内存**：`__constant__`，只读，有缓存，适合小数据
- **Symbol访问**：主机端必须使用`cudaMemcpyToSymbol`/`cudaMemcpyFromSymbol`
- **数组vs变量**：数组直接传数组名，非数组需要取地址

---

## 第7章：全局内存的合理使用

### 代码文件1：`capter7/global.cu`

#### 问题说明
**要解决的问题**：演示不同的全局内存访问模式，理解合并访问与非合并访问对性能的影响。

**背景**：GPU内存访问有合并度概念，合并访问可以显著提高性能，非合并访问会浪费带宽。

#### 实现思路
实现5种不同的内存访问模式：
1. 顺序合并访问（最优）
2. 乱序合并访问
3. 不对齐非合并访问
4. 跨越式非合并访问
5. 广播式非合并访问

#### 代码详解

```cuda
/*
基本思想是：演示5种不同的全局内存访问模式，展示合并访问与非合并访问对性能的影响
局限性：不同访问模式的性能差异很大，需要根据实际场景选择最优模式
*/
// ========== 1. 顺序合并访问（最优） ==========
__global__ void add(float *x, float *y, float *z, int N)
{
    // 线程索引：连续访问，合并度100%
    int n = threadIdx.y * blockDim.x + threadIdx.x;
    if (n >= N) return;

    for (int i = 0; i < 1000 ; ++i)
    {
        z[n] = sqrt(x[n] + y[n]);
    }
}
// 说明：线程0-31访问元素0-31，需要4次传输（128字节/32字节），合并度100%

// ========== 2. 乱序合并访问 ==========
__global__ void add_permuted(float *x, float *y, float *z, int N)
{
    // 通过XOR操作打乱线程索引，但访问仍然连续
    int tid = threadIdx.x^0x1;  // 交换相邻线程
    int n = threadIdx.y * blockDim.x + tid;
    if (n >= N) return;

    for (int i = 0; i < 1000 ; ++i)
    {
        z[n] = sqrt(x[n] + y[n]);
    }
}
// 说明：虽然线程顺序打乱，但访问地址仍然连续，合并度100%，但可能增加延迟

// ========== 3. 不对齐非合并访问 ==========
__global__ void add_offset(float *x, float *y, float *z, int N)
{
    // 访问地址偏移1，导致不对齐
    int n = threadIdx.y * blockDim.x + threadIdx.x + 1;
    if (n >= N) return;

    for (int i = 0; i < 1000 ; ++i)
    {
        z[n] = sqrt(x[n] + y[n]);
    }
}
// 说明：线程0-31访问元素1-32，需要5次传输（0-31, 32-63, 64-95, 96-127, 128-159字节）
// 合并度 = 4*32/(5*32) = 80%

// ========== 4. 跨越式非合并访问 ==========
__global__ void add_stride(float *x, float *y, float *z, int N)
{
    // 访问跨度大，导致非合并
    int n = blockIdx.x + threadIdx.x*gridDim.x;
    if (n >= N) return;

    for (int i = 0; i < 1000 ; ++i)
    {
        z[n] = sqrt(x[n] + y[n]);
    }
}
// 说明：第一个线程块访问元素0, 128, 256...，每个元素不在同一32字节段
// 需要32次传输，合并度 = 4*32/(32*32) = 12.5%

// ========== 5. 广播式非合并访问 ==========
__global__ void add_broadcast(float *x, float *y, float *z, int N)
{
    int n = threadIdx.x + blockIdx.x*gridDim.x;
    if (n >= N) return;

    for (int i = 0; i < 1000 ; ++i)
    {
        // 所有线程访问同一元素x[0]
        z[n] = sqrt(x[0] + y[0]);  // 广播访问
    }
}
// 说明：所有线程访问同一地址，只传输4字节，但合并度 = 4/32 = 12.5%
// 这种情况更适合使用常量内存

int main()
{
    int N = 1.0e6;
    int M = N * sizeof(float);

    // ... 内存分配和数据初始化 ...

    // 测试各种访问模式的性能
    add<<<128, 32>>>(d_x, d_y, d_z, N);           // 顺序合并
    add_permuted<<<128, 32>>>(d_x, d_y, d_z, N);  // 乱序合并
    add_offset<<<128, 32>>>(d_x, d_y, d_z, N);     // 不对齐非合并
    add_stride<<<128, 32>>>(d_x, d_y, d_z, N);    // 跨越式非合并
    add_broadcast<<<128, 32>>>(d_x, d_y, d_z, N);  // 广播式非合并

    // ... 计时和清理 ...
}
```

**关键点说明**：
- **合并访问**：相邻线程访问相邻内存地址，合并度100%
- **内存事务**：每次传输32字节，首地址必须是32的整数倍
- **合并度计算**：请求字节数 / 实际传输字节数
- **性能影响**：非合并访问可能慢10倍以上

---

### 代码文件2：`capter7/matrix.cu`

#### 问题说明
**要解决的问题**：实现矩阵转置，优化全局内存访问模式，演示`__ldg()`函数的使用。

**背景**：矩阵转置时，读取和写入必有一个是非合并访问，需要优化策略。

#### 实现思路
1. `transpose1`：合并读取、非合并写入
2. `transpose2`：非合并读取（使用`__ldg()`优化）、合并写入
3. 优先保证写入合并，读取使用只读数据缓存

#### 代码详解

```cuda
/*
基本思想是：实现矩阵转置，transpose1优先保证读取合并，transpose2优先保证写入合并并使用__ldg()优化
局限性：矩阵转置时读写必有一个非合并，需要权衡读写性能，transpose1写入性能较差
*/
// ========== 矩阵转置1：合并读取、非合并写入 ==========
__global__ void transpose1(const real *src, real *dst, const int N)
{
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;  // 列索引
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;  // 行索引

    if (nx < N && ny < N)
    {
        // 读取：src[ny*N + nx] - 按行读取，合并访问
        // 写入：dst[nx*N + ny] - 按列写入，非合并访问
        dst[nx*N + ny] = src[ny*N + nx];
    }
}
// 说明：读取合并，写入非合并，写入性能较差

// ========== 矩阵转置2：非合并读取（优化）、合并写入 ==========
__global__ void transpose2(const real *src, real *dst, const int N)
{
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;

    if (nx < N && ny < N)
    {
        // 读取：src[nx*N + ny] - 按列读取，非合并访问
        // 使用__ldg()通过只读数据缓存优化非合并读取
        // 写入：dst[ny*N + nx] - 按行写入，合并访问
        dst[ny*N + nx] = __ldg(&src[nx*N + ny]);
    }
}
// 说明：写入合并（性能更好），读取使用__ldg()优化，整体性能更好

int main()
{
    const int N = 10000;
    const int M = N * N * sizeof(real);

    // 获取TILE_DIM（线程块大小）
    int SIZE = 0;
    CHECK(cudaMemcpyFromSymbol(&SIZE, TILE_DIM, sizeof(int)));

    const int grid_size_x = (N + SIZE - 1)/SIZE;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(SIZE, SIZE);
    const dim3 grid_size(grid_size_x, grid_size_y);

    // ... 内存分配和数据初始化 ...

    // 测试两种转置方法的性能
    transpose1<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);
    transpose2<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);

    // ... 计时和清理 ...
}
```

**关键点说明**：
- **转置问题**：矩阵转置时读写必有一个非合并
- **优化策略**：优先保证写入合并（写入更耗时）
- **`__ldg()`函数**：通过只读数据缓存优化非合并读取
- **性能对比**：`transpose2`通常比`transpose1`快

---

## 第8章：共享内存的合理使用

### 代码文件1：`capter8/reduce.cu`

#### 问题说明
**要解决的问题**：实现数组归约（求和），使用共享内存优化性能。

**背景**：归约操作需要多次访问全局内存，使用共享内存可以减少全局内存访问次数，提高性能。

#### 实现思路
1. 使用折半归约法
2. 先将全局内存数据拷贝到共享内存
3. 在共享内存中进行归约
4. 使用`__syncthreads()`同步线程块内线程

#### 代码详解

```cuda
/*
基本思想是：CPU版本串行归约，用于对比和最终结果归约
局限性：串行执行，性能较低，但逻辑简单可靠
*/
// ========== CPU版本：串行归约 ==========
real reduce_cpu(const real *x, const int N)
{
    real sum = 0.0;
    for (int i = 0; i < N ; ++i)
    {
        sum += x[i];
    }
    return sum;
}

// ========== GPU版本1：全局内存归约 ==========
/*
基本思想是：每个block计算block_size 的元素的和, 回到CPU对每个block计算的和进行求和
局限性 只能，计算 block_size 整数个的元素
*/
__global__ void reduce(real *x, real *y)
{
    const int tid = threadIdx.x;
    real *curr_x = x + blockIdx.x * blockDim.x;  // 当前线程块处理的内存首地址

    // 折半归约：每次将数组长度减半
    for (int offset = blockDim.x >> 1; offset > 0; offset >>=1)
    {
        if (tid < offset)
        {
            // 线程tid将tid和tid+offset的元素相加
            curr_x[tid] += curr_x[tid + offset];
        }
        __syncthreads();  // 同步，确保所有线程完成当前轮次
    }

    if (tid == 0)
    {
        // 线程0保存结果
        y[blockIdx.x] = curr_x[0];
    }
}

/*
基本思想是：使用静态共享内存作为中间缓冲区，先将全局内存数据拷贝到共享内存，在共享内存中进行归约，减少全局内存访问次数
局限性：静态共享内存大小在编译期确定，不够灵活，但性能比全局内存版本好
*/
// ========== GPU版本2：静态共享内存归约 ==========
__global__ void reduce_shared(const real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;  // 全局索引

    // 静态共享内存：编译期确定大小
    __shared__ real s_x[128];
    
    // 步骤1：将全局内存数据拷贝到共享内存
    s_x[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();   // 同步数据拷贝操作

    // 步骤2：在共享内存中进行折半归约
    for (int offset = blockDim.x>>1; offset > 0; offset>>=1)
    {
        if (tid < offset)
        {
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();  // 同步每轮归约
    }

    // 步骤3：保存结果到全局内存
    if (tid == 0)
    {
        y[bid] = s_x[0];
    }
}
// 说明：使用共享内存减少全局内存访问，性能更好

/*
基本思想是：使用动态共享内存，运行时指定大小，比静态共享内存更灵活，可以适应不同的block_size
局限性：需要在核函数调用时指定共享内存大小，使用extern声明
*/
// ========== GPU版本3：动态共享内存归约 ==========
__global__ void reduce_shared2(const real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    // 动态共享内存：运行时指定大小
    extern __shared__ real s_x[];
    
    s_x[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x>>1; offset > 0; offset>>=1)
    {
        if (tid < offset)
        {
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        y[bid] = s_x[0];
    }
}

int main()
{
    int N = 1e8;
    int M = N * sizeof(real);
    int block_size = 128;
    int grid_size = (N + block_size - 1)/block_size;

    // ... 内存分配和数据初始化 ...

    // 测试不同版本的性能
    reduce<<<grid_size, block_size>>>(d_x, d_y);  // 全局内存版本
    
    // 静态共享内存版本
    reduce_shared<<<grid_size, block_size>>>(d_x, d_y, N);
    
    // 动态共享内存版本（需要指定共享内存大小）
    int sharedMemSize = block_size * sizeof(real);
    reduce_shared2<<<grid_size, block_size, sharedMemSize>>>(d_x, d_y, N);

    // 主机端对各个线程块的结果进行二次归约
    real result = reduce_cpu(h_y, grid_size);

    // ... 清理 ...
}
```

**关键点说明**：
- **折半归约**：每次将数组长度减半，需要log₂(N)步
- **`__syncthreads()`**：线程块内同步，确保所有线程完成当前操作
- **共享内存优势**：访问速度比全局内存快100倍
- **静态vs动态**：静态编译期确定，动态运行时指定
- **reduce函数局限性**：全局内存版本的reduce函数没有边界检查，要求数组长度必须是blockDim.x的整数倍，否则最后一个线程块会访问越界；后续版本通过传入N参数和条件判断解决了这个问题

---

### 代码文件2：`capter8/matrix.cu`

#### 问题说明
**要解决的问题**：使用共享内存优化矩阵转置，实现读写都合并的访问模式，并避免bank冲突。

**背景**：矩阵转置时，使用共享内存作为中间缓冲区，可以将非合并访问转换为合并访问。

#### 实现思路
1. 先将全局内存数据以合并方式读取到共享内存
2. 同步线程块
3. 从共享内存以合并方式写入全局内存（转置后）
4. 通过padding避免bank冲突

#### 代码详解

```cuda
/*
基本思想是：使用共享内存作为中间缓冲区，先将数据以合并方式读取到共享内存，再以合并方式写入全局内存，实现读写都合并
局限性：transpose3可能有bank冲突，transpose4通过padding避免bank冲突，性能最优
*/
// ========== 矩阵转置3：使用共享内存（可能有bank冲突） ==========
__global__ void transpose3(const real *src, real *dst, const int N)
{
    // 二维静态共享内存，存储线程块内的一片矩阵
    __shared__ real s_mat[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x * blockDim.x;  // 当前线程块首线程在网格中列索引
    int by = blockIdx.y * blockDim.y;  // 当前线程块首线程在网格中行索引
    int tx = threadIdx.x + bx;          // 当前线程在网格中列索引
    int ty = threadIdx.y + by;          // 当前线程在网格中行索引

    // 步骤1：全局内存合并读取，共享内存合并写入
    if (tx < N && ty < N)
    {
        // 按行读取源矩阵，合并访问
        s_mat[threadIdx.y][threadIdx.x] = src[ty * N + tx];
    }
    __syncthreads();  // 同步，确保数据已拷贝到共享内存

    // 步骤2：共享内存读取，全局内存合并写入（转置）
    if (tx < N && ty < N)
    {
        // 转置：交换行列索引
        int x = by + threadIdx.x;  // 转置后的列索引
        int y = bx + threadIdx.y;  // 转置后的行索引
        // 按列读取共享内存，按行写入目标矩阵，合并访问
        dst[y * N + x] = s_mat[threadIdx.x][threadIdx.y];
    }
}
// 说明：读写都合并，但可能有bank冲突

// ========== 矩阵转置4：使用共享内存+避免bank冲突 ==========
__global__ void transpose4(const real *src, real *dst, const int N)
{
    // 通过增加列宽（padding），错开数组元素在共享内存bank中的分布
    // 避免线程束的32路bank冲突
    __shared__ real s_mat[TILE_DIM][TILE_DIM + 1];  // 列宽+1

    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int tx = threadIdx.x + bx;
    int ty = threadIdx.y + by;

    if (tx < N && ty < N)
    {
        // 由于列宽+1，相邻行的元素不会映射到同一bank
        s_mat[threadIdx.y][threadIdx.x] = src[ty * N + tx];
    }
    __syncthreads();

    if (tx < N && ty < N)
    {
        int x = by + threadIdx.x;
        int y = bx + threadIdx.y;
        dst[y * N + x] = s_mat[threadIdx.x][threadIdx.y];
    }
}
// 说明：读写都合并，且避免了bank冲突，性能最优

int main()
{
    const int N = 500;
    // ... 内存分配和数据初始化 ...

    // 测试不同转置方法的性能
    transpose1<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);  // 全局内存
    transpose2<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);  // __ldg优化
    transpose3<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);  // 共享内存
    transpose4<<<grid_size, block_size>>>(d_matrix_org, d_matrix_res, N);  // 共享内存+避免bank冲突

    // ... 计时和清理 ...
}
```

**关键点说明**：
- **共享内存转置**：通过中间缓冲区实现读写都合并
- **Bank冲突**：共享内存分为32个bank，同一线程束访问同一bank不同层会冲突
- **避免冲突**：列宽+1（padding）错开bank分布
- **性能提升**：`transpose4`通常比`transpose1`快数倍

---

## 第9章：原子函数的合理使用

### 代码文件1：`capter9/reduce.cu`

#### 问题说明
**要解决的问题**：完全在GPU中进行数组归约，使用原子函数实现最终归约。

**背景**：之前的归约需要多个核函数或主机端二次归约，使用原子函数可以在一个核函数内完成最终归约。

#### 实现思路
1. 方法1：使用多个核函数逐步归约
2. 方法2：在核函数末尾使用原子函数直接归约到全局变量

#### 代码详解

```cuda
/*
基本思想是：方法1使用多个核函数逐步归约，方法2使用原子函数在一个核函数内完成最终归约，避免主机端二次归约
局限性：原子函数是串行的，可能成为性能瓶颈，但可以减少数据传输
*/
// ========== 方法1：多核函数归约 ==========
__global__ void reduce(real *x, real *y, const int N)
{
    int tid = threadIdx.x;
    int ind = tid + blockIdx.x * blockDim.x;

    extern __shared__ real curr_x[];
    curr_x[tid] = (ind < N) ? x[ind] : 0.0;

    // 线程块内折半归约
    for (int offset = blockDim.x/2 ; offset > 0 ; offset /= 2)
    {
        if (tid < offset)
        {
            curr_x[tid] += curr_x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // 保存各个线程块的结果
        y[blockIdx.x] = curr_x[0];
    }
}
// 调用：reduce<<<gSize, bSize, ...>>>(d_x, d_y, N);
// 然后：reduce<<<1, 1024, ...>>>(d_y, d_y, gSize);  // 二次归约

// ========== 方法2：原子函数归约 ==========
__global__ void reduce2(real *x, real *y, const int N)
{
    int tid = threadIdx.x;
    int ind = tid + blockIdx.x * blockDim.x;

    extern __shared__ real curr_x[];
    curr_x[tid] = (ind < N) ? x[ind] : 0.0;

    // 线程块内折半归约
    for (int offset = blockDim.x/2 ; offset > 0 ; offset /= 2)
    {
        if (tid < offset)
        {
            curr_x[tid] += curr_x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // 使用原子函数直接累加到全局结果
        // 注意：y必须是设备内存指针，不能是主机变量地址
        // 因为原子操作需要修改设备内存，主机内存地址无法在设备上访问
        atomicAdd(y, curr_x[0]);  // 原子操作：y += curr_x[0]
    }
}
// 调用：reduce2<<<gSize, bSize, ...>>>(d_x, d_y2, N);
// 结果直接保存在d_y2[0]中

int main()
{
    int N = 1e8;
    int bSize = 32;
    int gSize = (N + bSize - 1)/bSize;

    // ... 内存分配和数据初始化 ...

    // 方法1：多核函数归约
    real *d_y;
    CHECK(cudaMalloc(&d_y, gSize*sizeof(real)));
    reduce<<<gSize, bSize, (bSize+1)*sizeof(real)>>>(d_x, d_y, N);
    reduce<<<1, 1024, 1024*sizeof(real)>>>(d_y, d_y, gSize);
    // 主机端读取d_y[0]得到结果

    // 方法2：原子函数归约
    real *d_y2, *h_y2;
    h_y2 = new real(0.0);
    CHECK(cudaMalloc(&d_y2, sizeof(real)));
    
    // 初始化设备内存为0的三种方法：
    // 方法A：使用cudaMemset（推荐，适合初始化为0，性能最好）
    CHECK(cudaMemset(d_y2, 0, sizeof(real)));
    
    // 方法B：使用cudaMemcpy（通用，可以初始化为任意值）
    // CHECK(cudaMemcpy(d_y2, h_y2, sizeof(real), cudaMemcpyHostToDevice));
    
    // 方法C：在核函数中初始化（如果核函数需要初始化逻辑）
    // init_kernel<<<1, 1>>>(d_y2, 0.0);
    
    // 注意：不能直接赋值 d_y2 = 0.0 或 *d_y2 = 0.0
    // 因为d_y2是指向设备内存的指针，主机端无法直接访问设备内存
    
    // 重要：不能使用 real d_y2（主机变量）替代 real *d_y2（设备指针）
    // 原因：1) atomicAdd需要设备内存地址，主机变量地址无法在设备上访问
    //       2) 即使传&d_y2给核函数，设备端也无法访问主机内存
    //       3) 结果需要从设备传回主机，必须使用设备内存作为中间存储
    
    reduce2<<<gSize, bSize, (bSize)*sizeof(real)>>>(d_x, d_y2, N);
    
    // 从设备内存读取标量结果，必须使用cudaMemcpy
    CHECK(cudaMemcpy(h_y2, d_y2, sizeof(real), cudaMemcpyDeviceToHost));
    cout << "reduce2 result: " << *h_y2 << endl;

    // ... 清理 ...
}
```

**关键点说明**：
- **原子函数**：`atomicAdd(address, val)` 原子地执行 `*address += val`
- **原子性**：保证操作的"读-改-写"是原子性的，不会被其他线程打断
- **性能**：原子函数是串行的，可能成为性能瓶颈
- **适用场景**：当需要减少数据传输时，原子函数很有用

---

### 代码文件2：`capter9/neighbor.cu`

#### 问题说明
**要解决的问题**：邻居列表问题——找出所有距离小于截断距离的粒子对。

**背景**：在分子动力学等应用中，需要找出每个粒子的邻居，使用原子函数可以并行更新邻居列表。

#### 实现思路
1. CPU版本：双重循环，串行查找
2. GPU版本1：每个线程处理一个粒子，但访问模式不优化
3. GPU版本2：使用原子函数更新邻居计数和列表

#### 代码详解

```cuda
/*
基本思想是：找出所有距离小于截断距离的粒子对，CPU版本串行查找，GPU版本1每个线程处理一个粒子但访问模式不优化，GPU版本2使用原子函数并行更新邻居列表
局限性：GPU版本1访问次数N*N性能较低，GPU版本2原子操作是串行的但整体仍比CPU快
*/
// ========== CPU版本：串行查找 ==========
void find_neighbor(int *NN, int *NL, const real *x, const real *y, 
                   const int N, const int M, const real minDis)
{
    // 初始化邻居计数
    for (int i = 0; i < N; ++i)
    {
        NN[i] = 0;
    }

    // 双重循环查找邻居
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)  // 只遍历上三角，避免重复
        {
            real dx = x[j] - x[i];
            real dy = y[j] - y[i];
            real dis = dx * dx + dy * dy;  // 比较平方，减少计算
            if (dis < minDis)
            {
                // 将j加入i的邻居列表
                NL[i*M + NN[i]] = j;
                NN[i]++;
                // 将i加入j的邻居列表（对称）
                NL[j*M + NN[j]] = i;
                NN[j]++;
            }
        }
    }
}

// ========== GPU版本1：非优化版本 ==========
__global__ void find_neighbor_gpu (int *NN, int *NL, const real *x, const real *y, 
                                    const int N, const int M, const real minDis)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int count = 0;  // 寄存器变量，减少对全局变量NN的访问
        for (int j = 0; j < N; ++j)  // 访问次数N*N，性能较低
        {
            real dx = x[j] - x[i];
            real dy = y[j] - y[i];
            real dis = dx * dx + dy * dy;

            if (dis < minDis && i != j)
            {
                // 修改了全局内存NL的数据排列方式，实现合并访问
                NL[(count++) * N + i] = j;
            }
        }
        NN[i] = count;
    }
}

// ========== GPU版本2：使用原子函数（优化版本） ==========
__global__ void find_neighbor_atomic(int *NN, int *NL, const real *x, const real *y, 
                                     const int N, const int M, const real minDis)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        NN[i] = 0;  // 初始化邻居计数

        // 只遍历上三角，避免重复计算
        for (int j = i + 1; j < N; ++j)
        {
            real dx = x[j] - x[i];
            real dy = y[j] - y[i];
            real dis = dx * dx + dy*dy;
            if (dis < minDis)
            {
                // 原子函数返回旧值，然后递增
                int old_i_num = atomicAdd(&NN[i], 1);  // 返回i的旧邻居数
                NL[i*M + old_i_num] = j;  // 将j加入i的邻居列表

                int old_j_num = atomicAdd(&NN[j], 1);  // 返回j的旧邻居数
                NL[j*M + old_j_num] = i;  // 将i加入j的邻居列表
            }
        }
    }
}

int main()
{
    std::string fstr = "xy.txt";
    std::vector<real> x, y;
    read_data(fstr, x, y);  // 从文件读取粒子坐标

    int N = x.size(), M = 10;
    real minDis = 1.9*1.9;  // 截断距离的平方

    int *NN = new int[N];      // 邻居计数数组
    int *NL = new int[N*M];    // 邻居列表数组

    // ... 内存分配和数据传输 ...

    // CPU版本
    find_neighbor(NN, NL, x.data(), y.data(), N, M, minDis);

    // GPU版本（原子函数）
    find_neighbor_atomic<<<grid_size, block_size>>>(d_NN, d_NL, d_x, d_y, N, M, minDis);

    // ... 结果回传和验证 ...
}
```

**关键点说明**：
- **原子函数返回值**：`atomicAdd`返回操作前的旧值，可以用来作为数组索引
- **对称更新**：粒子i和j互为邻居，需要同时更新两个粒子的邻居列表
- **性能优化**：
  - 使用寄存器变量`count`减少全局内存访问
  - 距离判断优先，提高"假"的命中率
  - 比较平方距离，避免开方运算
- **原子函数代价**：原子操作是串行的，但在这个场景下仍然比CPU快很多

---

## 第10章：线程束基本函数与协作组

### 代码文件1：`capter10/reduce.cu`

#### 问题说明
**要解决的问题**：进一步优化数组归约，使用线程束内函数和协作组提高性能。

**背景**：线程束（warp）是GPU执行的基本单位，使用线程束级别的函数可以避免线程块级别的同步开销。

#### 实现思路
1. 使用`__syncwarp()`替代`__syncthreads()`（仅线程束内）
2. 使用线程束洗牌函数实现线程束内归约
3. 使用协作组提供更灵活的同步机制
4. 提高线程利用率（每个线程处理多个数据）

#### 代码详解

```cuda
/*
基本思想是：使用线程束级别函数优化归约，版本1使用__syncwarp()，版本2使用洗牌函数，版本3使用协作组，版本4提高线程利用率
局限性：线程束函数只能在32个线程的线程束内使用，需要理解线程束的执行模型
*/
// ========== 版本1：使用线程束同步 ==========
__global__ void reduce_syncwarp(real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    // 线程块之间的二分求和（offset >= 32）
    for (int offset = blockDim.x/2; offset >= 32; offset /=2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads();  // 线程块同步
    }

    // 线程束内的二分求和（offset < 32）
    for (int offset = 16; offset > 0; offset /=2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncwarp();  // 线程束同步，比__syncthreads()更轻量
    }

    if (tid == 0)
    {
        atomicAdd(y, block_arr[0]);
    }
}

// ========== 版本2：使用线程束洗牌函数 ==========
__global__ void reduce_shfl_down(real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    // 线程块间归约
    for (int offset = blockDim.x /2 ; offset >= 32; offset /= 2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads();
    }

    // 在线程寄存器上定义变量
    real curr_y = block_arr[tid];

    // 使用线程束洗牌函数实现线程束内归约
    for (int offset = 16; offset > 0; offset /= 2)
    {
        // __shfl_down_sync：将高线程号的值平移到低线程号
        // 等价于：curr_y += block_arr[tid + offset]（如果tid + offset < 32）
        curr_y += __shfl_down_sync(FULL_MASK, curr_y, offset);
    }

    if (tid == 0)
    {
        atomicAdd(y, curr_y);
    }
}

// ========== 版本3：使用协作组 ==========
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void reduce_cp(real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int ind = bid * blockDim.x + tid;

    extern __shared__ real block_arr[];
    block_arr[tid] = (ind < N) ? x[ind] : 0.0;
    __syncthreads();

    // 线程块间归约
    for (int offset = blockDim.x /2 ; offset >= 32; offset /= 2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads();
    }

    real curr_y = block_arr[tid];

    // 创建线程块片（32个线程）
    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());

    // 使用协作组的洗牌函数
    for (int offset = 16; offset > 0; offset /= 2)
    {
        curr_y += g32.shfl_down(curr_y, offset);  // 不需要指定mask
    }

    if (tid == 0)
    {
        atomicAdd(y, curr_y);
    }
}

// ========== 版本4：提高线程利用率 ==========
__global__ void reduce_cp_grid(const real *x, real *y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real block_arr[];

    real curr_y = 0.0;

    // 在归约前处理计算，提高线程利用率
    // 每个线程处理多个数据，跨度为blockDim.x * gridDim.x
    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < N; n += stride)
    {
        curr_y += x[n];  // 合并访问
    }

    block_arr[tid] = curr_y;
    __syncthreads();

    // 后续归约过程（线程块间+线程束内）
    for (int offset = blockDim.x /2 ; offset >= 32; offset /= 2)
    {
        if (tid < offset)
        {
            block_arr[tid] += block_arr[tid + offset];
        }
        __syncthreads();
    }

    curr_y = block_arr[tid];
    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());
    for (int offset = 16; offset > 0; offset /= 2)
    {
        curr_y += g32.shfl_down(curr_y, offset);
    }

    if (tid == 0)
    {
        y[bid] = curr_y;
    }
}

int main()
{
    int N = 1e8;
    int bSize = 32;
    int gSize = (N + bSize - 1)/bSize;

    // ... 内存分配和数据初始化 ...

    // 测试不同版本的性能
    reduce_syncwarp<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_res, N);
    reduce_shfl_down<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_res, N);
    reduce_cp<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_res, N);
    reduce_cp_grid<<<gSize, bSize, bSize*sizeof(real)>>>(d_x, d_y, N);

    // ... 清理 ...
}
```

**关键点说明**：
- **`__syncwarp()`**：线程束内同步，比`__syncthreads()`更轻量
- **线程束洗牌函数**：`__shfl_down_sync`可以在线程束内直接交换数据，无需共享内存
- **协作组**：提供更灵活的线程组织方式，代码更清晰
- **线程利用率**：让每个线程处理多个数据，提高利用率

---

### 代码文件2：`capter10/warp.cu`

#### 问题说明
**要解决的问题**：演示各种线程束基本函数的使用方法和行为。

**背景**：理解线程束函数的行为对于优化CUDA程序很重要。

#### 实现思路
实现各种线程束函数并打印它们的输出，观察行为。

#### 代码详解

```cuda
/*
基本思想是：演示各种线程束基本函数的使用方法和行为，包括掩码生成、逻辑判断、数据洗牌等操作
局限性：无，这是演示程序，用于理解线程束函数的行为
*/
const unsigned WIDTH = 8;  // 逻辑线程束大小
const unsigned BLOCK_SIZE = 16;
const unsigned FULL_MASK = 0xffffffff;  // 全掩码（32位全1）

__global__ void test_warp_primitives(void)
{
    int tid = threadIdx.x;
    int laneId = tid % WIDTH;  // 束内索引

    // ========== 1. __ballot_sync：生成掩码 ==========
    // 如果线程束内第n个线程参与计算且predicate值非零，则返回值的第n位为1
    unsigned mask1 = __ballot_sync(FULL_MASK, tid>0);   // 排除0号线程的掩码
    unsigned mask2 = __ballot_sync(FULL_MASK, tid==0);  // 仅0号线程的掩码

    // ========== 2. __all_sync：全真判断 ==========
    // 线程束内所有参与线程的predicate值均非零，则返回1
    int result = __all_sync(FULL_MASK, tid);      // 返回0（因为tid=0时predicate=0）
    result = __all_sync(mask1, tid);              // 返回1（mask1排除了0号线程）

    // ========== 3. __any_sync：存在判断 ==========
    // 线程束内所有参与线程的predicate值存在非零，则返回1
    result = __any_sync(FULL_MASK, tid);          // 返回1（存在tid>0）
    result = __any_sync(mask2, tid);              // 返回0（mask2只有0号线程，且tid=0时predicate=0）

    // ========== 4. __shfl_sync：广播 ==========
    // 所有参与线程返回标号为srcLane的线程中变量v的值
    int value = __shfl_sync(FULL_MASK, tid, 2, WIDTH);
    // 所有8个线程都返回laneId=2的线程的tid值

    // ========== 5. __shfl_up_sync：向上平移 ==========
    // 标号为t的线程返回标号为t-d的线程的值，t-d<0时返回自己的值
    value = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH);
    // 0号线程返回自己的tid，1-7号线程返回前一个线程的tid

    // ========== 6. __shfl_down_sync：向下平移 ==========
    // 标号为t的线程返回标号为t+d的线程的值，t+d≥w时返回自己的值
    value = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH);
    // 0-6号线程返回后一个线程的tid，7号线程返回自己的tid

    // ========== 7. __shfl_xor_sync：两两交换 ==========
    // 标号为t的线程返回标号为t^laneMask的线程的值
    value = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH);
    // 0号线程返回1号的tid，1号返回0号的tid，2号返回3号的tid，等等
}

int main()
{
    test_warp_primitives<<<1, BLOCK_SIZE>>>();
    CHECK(cudaDeviceSynchronize());
    return 0;
}
```

**关键点说明**：
- **掩码（Mask）**：32位无符号整数，每一位代表一个线程是否参与
- **`__ballot_sync`**：根据predicate值生成掩码
- **`__all_sync`/`__any_sync`**：线程束级别的逻辑判断
- **洗牌函数**：在线程束内交换数据，无需共享内存

---

## 第11章：CUDA流

### 代码文件：`capter11/stream.cu`

#### 问题说明
**要解决的问题**：实现主机和设备计算重叠、多个核函数并行执行、核函数执行与数据传输重叠。

**背景**：CUDA操作默认是顺序执行的，使用CUDA流可以实现并行执行，提高GPU利用率。

#### 实现思路
1. 主机-设备重叠：核函数调用后执行主机计算
2. 多核函数并行：创建多个非默认CUDA流
3. 核函数-数据传输重叠：使用`cudaMemcpyAsync`和不可分页内存

#### 代码详解

```cuda
/*
基本思想是：使用CUDA流实现并行执行，包括主机-设备计算重叠、多个核函数并行、核函数与数据传输重叠
局限性：异步操作需要不可分页内存，流的创建和销毁需要配对管理，重叠效果取决于操作的可并行性
*/
const int NUM_REPEATS = 10;
const int MAX_NUM_STREAMS = 30;
cudaStream_t streams[MAX_NUM_STREAMS];  // CUDA流数组

// ========== 1. 主机和设备计算重叠 ==========
void timing(const real *h_x, const real *h_y, real *h_z,
            const real *d_x, const real *d_y, real *d_z,
            const int ratio, bool overlap)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    if (!overlap)
    {
        // 不重叠：先执行CPU计算
        cpu_sum(h_x, h_y, h_z, N / ratio);
    }

    // GPU计算（异步）
    gpu_sum<<<grid_size, block_size>>>(d_x, d_y, d_z);

    if (overlap)
    {
        // 重叠：CPU计算与GPU计算并行执行
        cpu_sum(h_x, h_y, h_z, N / ratio);
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    // ... 计算时间 ...
}
// 说明：当CPU和GPU计算量相当时，重叠可以显著提高性能

// ========== 2. 多个核函数并行执行 ==========
void timing(const real *d_x, const real *d_y, real *d_z, const int num)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    for (int n = 0; n < num; ++n)
    {
        int offset = n * N1;
        // 指定各个核函数的CUDA流，实现核函数的并行
        add<<<grid_size, block_size, 0, streams[n]>>>(
            d_x + offset, d_y + offset, d_z + offset);
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    // ... 计算时间 ...
}
// 说明：使用多个流可以让多个核函数并行执行，提高GPU利用率

// ========== 3. 核函数执行与数据传输重叠 ==========
void timing(const real *h_x, const real *h_y, real *h_z,
            real *d_x, real *d_y, real *d_z,
            const int num)
{
    int N1 = N / num;
    int M1 = M / num;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    for (int i = 0; i < num; i++)
    {
        int offset = i * N1;

        // 异步数据传输（H2D）
        CHECK(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, 
                              cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, 
                              cudaMemcpyHostToDevice, streams[i]));

        int block_size = 128;
        int grid_size = (N1 - 1) / block_size + 1;

        // 核函数执行（在同一流中）
        add2<<<grid_size, block_size, 0, streams[i]>>>(
            d_x + offset, d_y + offset, d_z + offset, N1);

        // 异步数据传输（D2H）
        CHECK(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, 
                              cudaMemcpyDeviceToHost, streams[i]));
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    // ... 计算时间 ...
}
// 说明：不同流中的数据传输和核函数执行可以重叠，理论上最大加速比为3

int main(void)
{
    // ========== 创建CUDA流 ==========
    for (int n = 0 ; n < MAX_NUM_STREAMS; ++n)
    {
        CHECK(cudaStreamCreate(&(streams[n])));
    }

    // ========== 测试1：主机-设备重叠 ==========
    printf("Without CPU-GPU overlap\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, false);
    printf("With CPU-GPU overlap\n");
    timing(h_x, h_y, h_z, d_x, d_y, d_z, 10, true);

    // ========== 测试2：多核函数并行 ==========
    for (int num = 1; num <= MAX_NUM_STREAMS; ++num)
    {
        timing(d_x, d_y, d_z, num);
    }

    // ========== 测试3：核函数-数据传输重叠 ==========
    // 分配不可分页内存（固定内存）
    real *h_x2, *h_y2, *h_z2;
    CHECK(cudaMallocHost(&h_x2, M));  // 不可分页内存
    CHECK(cudaMallocHost(&h_y2, M));
    CHECK(cudaMallocHost(&h_z2, M));

    for (int num = 1; num <= MAX_NUM_STREAMS; num *= 2)
    {
        timing(h_x2, h_y2, h_z2, d_x, d_y, d_z, num);
    }

    // ========== 销毁CUDA流 ==========
    for (int n = 0 ; n < MAX_NUM_STREAMS; ++n)
    {
        CHECK(cudaStreamDestroy(streams[n]));
    }

    // ========== 释放不可分页内存 ==========
    CHECK(cudaFreeHost(h_x2));
    CHECK(cudaFreeHost(h_y2));
    CHECK(cudaFreeHost(h_z2));

    // ... 清理 ...
}
```

**关键点说明**：
- **CUDA流**：一个CUDA操作序列，同一流内顺序执行，不同流可以并行
- **异步操作**：`cudaMemcpyAsync`需要不可分页内存（`cudaMallocHost`）
- **重叠效果**：当操作可以并行时，重叠可以显著提高性能
- **流管理**：创建和销毁流需要配对，避免资源泄漏

---

## 总结

### 代码组织模式

1. **基本框架**：内存分配→数据传输→核函数调用→结果回传→内存释放
2. **错误检测**：使用CHECK宏检测所有CUDA API调用
3. **性能分析**：使用CUDA事件精确计时
4. **优化策略**：合并访问→共享内存→线程束函数→CUDA流

### 关键记忆点

- **线程组织**：Grid → Block → Thread → Warp(32)
- **内存层次**：寄存器 > 共享内存 > L1/L2缓存 > 常量内存/纹理内存 > 全局内存
- **优化原则**：合并访问、避免bank冲突、减少分支发散、合理使用流
- **常用API**：`cudaMalloc`, `cudaMemcpy`, `cudaFree`, `__syncthreads()`, `atomicAdd`, `cudaStreamCreate`

---

*本文档基于《CUDA编程——基础与实践》（樊哲勇）和CudaSteps项目代码整理*
