# CUDA Hello World - 第一个CUDA程序

入门题1：实现第一个CUDA程序，学习CUDA基础编程模型。

## 项目目标

完成以下三个核心任务：
1. ✅ 输出CUDA设备信息（设备名称、计算能力、显存大小等）
2. ✅ 检查CUDA运行时是否正确初始化
3. ✅ 实现简单的向量相加kernel函数

## 项目结构

```
proj1_hello_world/
├── CMakeLists.txt           # 构建配置文件
├── README.md                # 项目说明
├── include/
│   └── cuda_utils.h         # CUDA工具函数头文件
└── src/
    ├── main.cu              # 主程序（向量相加kernel实现）
    └── device_info.cu       # 设备信息查询功能
```

## 编译和运行

### 编译
```bash
mkdir -p build
cd build
cmake ..
make -j4
```

### 运行
```bash
./hello_cuda
```

### 预期输出

程序将执行以下步骤并输出详细信息：

1. **检查CUDA运行时初始化** - 检测可用设备数量
2. **打印设备详细信息** - GPU型号、计算能力、显存等完整信息
3. **向量相加测试** - 启动GPU kernel执行向量运算
4. **验证结果** - 与CPU计算结果对比，确保正确性
5. **释放资源** - 清理所有分配的内存

## 核心知识点

### 1. CUDA Kernel函数
```cuda
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### 2. Kernel启动配置
```cuda
// 配置：grid维度 × block维度
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

### 3. 内存管理流程
```
CPU分配内存 → cudaMalloc(设备内存) → cudaMemcpy(H2D) 
→ 启动kernel → cudaMemcpy(D2H) → cudaFree → CPU释放内存
```

### 4. CUDA错误检查
```cuda
#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)
CHECK_CUDA_ERROR(cudaMalloc(&d_ptr, size));
CHECK_CUDA_ERROR(cudaGetLastError());  // 检查kernel启动错误
CHECK_CUDA_ERROR(cudaDeviceSynchronize());  // 等待kernel完成
```

### 5. 设备信息查询
- `cudaGetDeviceCount()` - 获取设备数量
- `cudaGetDeviceProperties()` - 获取设备属性
- `cudaSetDevice()` - 设置当前设备

## 运行结果示例

```
========================================
CUDA Hello World - 第一个CUDA程序
========================================

[步骤1] 检查CUDA运行时初始化...
检测到 1 个CUDA设备

[步骤2] 打印设备详细信息...

========================================
设备 0 详细信息
========================================
设备名称:          NVIDIA GeForce GTX 1660 SUPER
计算能力:          7.5
显存大小:          6.00 GB (6144 MB)
多处理器数量:      22
每块最大线程数:    1024
...

[步骤4] 向量相加kernel测试...
启动配置: Grid=(4), Block=(256)
总线程数: 1024

[步骤5] 验证GPU计算结果...
✓ 验证成功! GPU结果与CPU结果完全一致
```

## 验收标准

- ✅ 程序能够成功编译和运行
- ✅ 正确输出设备信息
- ✅ 向量相加结果正确
- ✅ 正确释放所有内存资源

## 学习收获

通过这个项目，你将学习到：
- CUDA基本编程模型（host/device、kernel、线程组织）
- CUDA内存管理（设备内存分配、数据传输）
- CUDA线程模型（grid、block、thread的关系）
- 设备信息查询和错误处理
- CUDA程序的基本编译流程

## 下一步

完成入门题1后，可以继续学习：
- **入门题2**: 向量加法性能对比 - 学习性能分析和优化
- **入门题3**: 数组元素平方计算 - 学习多维数组处理
- **入门题4**: 简单的归约运算 - 学习共享内存和同步

