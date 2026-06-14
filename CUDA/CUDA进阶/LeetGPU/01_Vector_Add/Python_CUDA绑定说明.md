# Python 与 CUDA 函数绑定说明文档

## 目录
1. [概述](#概述)
2. [绑定原理](#绑定原理)
3. [技术栈](#技术栈)
4. [实现步骤](#实现步骤)
5. [代码结构解析](#代码结构解析)
6. [编译流程](#编译流程)
7. [使用示例](#使用示例)
8. [性能优化技巧](#性能优化技巧)
9. [常见问题](#常见问题)
10. [最佳实践](#最佳实践)

---

## 概述

Python 与 CUDA 函数绑定是指将用 CUDA C++ 编写的 GPU 内核函数暴露给 Python，使其可以在 Python 中直接调用。这种绑定机制使得我们可以：

- **利用 GPU 加速**：在 Python 中调用高性能的 CUDA 内核
- **保持 Python 的灵活性**：使用 Python 进行数据预处理、后处理和流程控制
- **无缝集成**：与 PyTorch、NumPy 等 Python 科学计算库无缝集成

### 为什么需要绑定？

1. **性能需求**：Python 本身执行速度较慢，对于计算密集型任务需要 GPU 加速
2. **代码复用**：已有的 CUDA 内核代码可以在 Python 项目中使用
3. **开发效率**：Python 的易用性与 CUDA 的高性能相结合

---

## 绑定原理

### 整体架构

```
┌─────────────────┐
│   Python 代码   │
│  (vector_add.py)│
└────────┬────────┘
         │ 调用
         ▼
┌─────────────────┐
│  PyTorch 扩展   │
│  (pybind11)     │
└────────┬────────┘
         │ 转换
         ▼
┌─────────────────┐
│  C++ 包装函数   │
│  (vector_add.cu)│
└────────┬────────┘
         │ 启动
         ▼
┌─────────────────┐
│  CUDA 内核      │
│  (GPU 执行)     │
└─────────────────┘
```

### 关键组件

1. **PyTorch cpp_extension**：提供动态编译和加载机制
2. **pybind11**：C++ 与 Python 之间的绑定库
3. **CUDA Runtime API**：GPU 内存管理和内核启动
4. **nvcc 编译器**：编译 CUDA 代码

---

## 技术栈

### 1. PyTorch cpp_extension

PyTorch 提供的扩展工具，支持两种方式：

- **`load()`**：运行时动态编译（JIT - Just-In-Time）
  - 优点：开发方便，无需手动编译
  - 缺点：首次运行需要编译时间

- **`BuildExtension`**：预编译扩展（setup.py）
  - 优点：安装后直接使用，无需编译
  - 缺点：需要打包和安装步骤

本示例使用 `load()` 方式，适合开发和快速迭代。

### 2. pybind11

pybind11 是一个轻量级的 C++ 库，用于在 C++ 和 Python 之间创建绑定。

**核心特性**：
- 自动类型转换（Python 对象 ↔ C++ 类型）
- 支持 NumPy 数组和 PyTorch Tensor
- 简洁的 API（`m.def()` 即可绑定函数）

**基本用法**：
```cpp
PYBIND11_MODULE(module_name, m) {
    m.def("function_name", &cpp_function, "docstring");
}
```

### 3. CUDA 编程模型

- **主机（Host）**：CPU 和 CPU 内存
- **设备（Device）**：GPU 和 GPU 内存
- **内核（Kernel）**：在 GPU 上执行的函数
- **线程层次**：Grid → Block → Thread

---

## 实现步骤

### 步骤 1：编写 CUDA 内核函数

```cpp
__global__ void vector_add_kernel(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**关键点**：
- `__global__`：标识这是一个 CUDA 内核函数
- 线程索引计算：`blockIdx.x * blockDim.x + threadIdx.x`
- 边界检查：确保 `idx < N`

### 步骤 2：编写 C++ 包装函数

```cpp
void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    // 1. 数据类型检查
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32);
    
    // 2. 获取数据指针
    float *a_ptr = reinterpret_cast<float *>(a.data_ptr());
    
    // 3. 计算线程块和网格大小
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    
    // 4. 启动 CUDA 内核
    vector_add_kernel<<<grid, block>>>(a_ptr, b_ptr, c_ptr, N);
}
```

**关键点**：
- 使用 `torch::Tensor` 接收 Python 的 PyTorch Tensor
- 通过 `data_ptr()` 获取底层数据指针
- 使用 `dim3` 定义线程块和网格大小
- 使用 `<<<grid, block>>>` 启动内核

### 步骤 3：使用 pybind11 绑定函数

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Vector addition on GPU");
}
```

**关键点**：
- `TORCH_EXTENSION_NAME`：PyTorch 自动定义的模块名
- `m.def()`：绑定函数到 Python

### 步骤 4：在 Python 中加载和使用

```python
from torch.utils.cpp_extension import load

lib = load(name="vector_add", sources=["vector_add.cu"])
result = lib.vector_add(a, b, c)
```

---

## 代码结构解析

### Python 端（vector_add.py）

#### 1. 模块加载

```python
lib = load(
    name="vector_add",
    sources=["vector_add.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-std=c++17"],
    verbose=True
)
```

**参数说明**：
- `name`：模块名称，用于生成临时文件
- `sources`：源文件列表（.cu 或 .cpp）
- `extra_cuda_cflags`：传递给 nvcc 的编译选项
- `extra_cflags`：传递给 C++ 编译器的选项
- `verbose`：显示编译信息

**编译选项详解**：
- `-O3`：最高级别优化
- `--use_fast_math`：使用快速数学库（可能降低精度）
- `-std=c++17`：C++ 标准版本

#### 2. 性能测试函数

```python
def run_benchmark(perf_func, a, b, tag, out=None, warmup=10, iters=1000):
    # 预热阶段
    for i in range(warmup):
        perf_func(a, b, out)
    
    # 同步 GPU
    torch.cuda.synchronize()
    
    # 计时阶段
    start = time.time()
    for i in range(iters):
        perf_func(a, b, out)
    torch.cuda.synchronize()
    end = time.time()
    
    # 计算平均时间
    mean_time = (end - start) * 1000 / iters
```

**关键点**：
- **预热（Warmup）**：消除首次运行的开销（初始化、JIT 编译等）
- **同步（Synchronize）**：CUDA 操作默认异步，需要同步才能准确计时
- **多次迭代**：取平均值减少测量误差

### CUDA 端（vector_add.cu）

#### 1. CUDA 内核函数

**基础版本（逐个元素）**：
```cpp
__global__ void vector_add_f32_kernel(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**向量化版本（float4）**：
```cpp
__global__ void vector_add_f32x4_kernel(float *a, float *b, float *c, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    }
}
```

**向量化的优势**：
- 减少内存访问次数（4个元素一次读取）
- 提高内存带宽利用率
- 减少线程数量（每个线程处理4个元素）

#### 2. 宏定义系统

**字符串化宏**：
```cpp
#define STRINGFY(str) #str
// STRINGFY(vector_add) → "vector_add"
```

**函数绑定宏**：
```cpp
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));
```

**模板宏（生成多个版本）**：
```cpp
#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
void vector_add_##packed_type(torch::Tensor a, torch::Tensor b, torch::Tensor c) { \
    // 数据类型检查 \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type)); \
    // 计算线程配置 \
    // 启动内核 \
}
```

**宏展开示例**：
```cpp
TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
// 展开为：
void vector_add_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    // ... 实现
}
```

#### 3. 线程配置策略

**1维张量**：
```cpp
dim3 block(256 / n_elements);  // 每个线程块256个线程（考虑向量化）
dim3 grid((N + 255) / 256);     // 向上取整
```

**2维张量（优化）**：
```cpp
if (K / n_elements <= 1024) {
    dim3 block(K / n_elements);  // 每个线程块处理一行
    dim3 grid(S);                // 每个线程块对应一行
} else {
    // 回退到1维处理
}
```

**配置原则**：
- 线程块大小：通常是 32 的倍数（warp 大小）
- 网格大小：确保覆盖所有元素
- 考虑向量化：`n_elements` 影响实际线程数

---

## 编译流程

### 动态编译（load）

```
Python 调用 load()
    ↓
检查缓存（.so 文件是否存在）
    ↓
如果不存在，开始编译：
    1. nvcc 编译 .cu → .o
    2. g++ 链接 → .so
    ↓
加载 .so 到 Python
    ↓
返回模块对象
```

### 编译产物

- **临时目录**：`~/.cache/torch_extensions/vector_add/`
- **共享库**：`vector_add.so`（Linux）或 `vector_add.pyd`（Windows）
- **编译日志**：如果 `verbose=True`，会显示编译命令和输出

### 缓存机制

- 如果源文件未修改，直接加载已编译的 `.so`
- 如果源文件修改，重新编译
- 可以通过修改 `name` 参数强制重新编译

---

## 使用示例

### 基本使用

```python
import torch
from torch.utils.cpp_extension import load

# 加载扩展
lib = load(name="vector_add", sources=["vector_add.cu"])

# 准备数据
a = torch.randn(1000, 1000).cuda().float()
b = torch.randn(1000, 1000).cuda().float()
c = torch.zeros_like(a)

# 调用 CUDA 函数
lib.vector_add_f32(a, b, c)

# 验证结果
print(c[:5, :5])
```

### 性能对比

```python
# 基础版本
lib.vector_add_f32(a, b, c)

# 向量化版本（通常更快）
lib.vector_add_f32x4(a, b, c)
```

### 错误处理

```python
try:
    lib.vector_add_f32(a, b, c)
except RuntimeError as e:
    print(f"CUDA 错误: {e}")
```

---

## 性能优化技巧

### 1. 内存访问优化

**合并访问（Coalesced Access）**：
- 确保线程访问连续的内存地址
- 使用向量化类型（float4, int4 等）

**示例**：
```cpp
// 好的：连续访问
int idx = blockIdx.x * blockDim.x + threadIdx.x;
c[idx] = a[idx] + b[idx];

// 差的：非连续访问
int idx = threadIdx.x * gridDim.x + blockIdx.x;
c[idx] = a[idx] + b[idx];
```

### 2. 向量化

**使用 float4/int4 等向量类型**：
```cpp
float4 reg_a = FLOAT4(a[idx]);  // 一次读取4个 float
float4 reg_b = FLOAT4(b[idx]);
// 计算...
FLOAT4(c[idx]) = reg_c;         // 一次写入4个 float
```

**优势**：
- 减少内存事务数
- 提高内存带宽利用率
- 减少指令数

### 3. 线程块大小选择

**经验法则**：
- 线程块大小：128, 256, 512, 1024
- 通常是 32 的倍数（warp 大小）
- 根据 GPU 架构调整（如 SM 版本）

**测试不同配置**：
```python
for block_size in [128, 256, 512, 1024]:
    # 测试性能
    pass
```

### 4. 共享内存使用

对于需要数据重用的场景：
```cpp
__shared__ float shared_data[256];

// 加载数据到共享内存
shared_data[threadIdx.x] = global_data[threadIdx.x];
__syncthreads();

// 使用共享内存中的数据
```

### 5. 编译优化选项

```python
extra_cuda_cflags=[
    "-O3",                    # 最高优化
    "--use_fast_math",        # 快速数学（可能降低精度）
    "--ptxas-options=-v",     # 显示寄存器使用情况
    "-arch=sm_75",            # 指定 GPU 架构
]
```

---

## 常见问题

### 1. 编译错误

**问题**：找不到 CUDA 头文件
```
fatal error: cuda_runtime.h: No such file or directory
```

**解决**：
- 确保 CUDA 已正确安装
- 设置 `CUDA_HOME` 环境变量
- 检查 `nvcc` 是否在 PATH 中

### 2. 运行时错误

**问题**：`CUDA error: invalid device function`
```
原因：编译的 GPU 架构与运行时的 GPU 不匹配
解决：指定正确的架构，如 `-arch=sm_75`
```

**问题**：`CUDA error: out of memory`
```
原因：GPU 内存不足
解决：减少数据大小或使用更小的批次
```

### 3. 性能问题

**问题**：CUDA 函数比 CPU 还慢
```
可能原因：
1. 数据太小，GPU 启动开销大于计算时间
2. 内存传输开销（CPU ↔ GPU）
3. 线程配置不合理

解决：
1. 确保数据足够大（通常 > 1MB）
2. 减少 CPU-GPU 数据传输
3. 优化线程块大小
```

### 4. 数据类型不匹配

**问题**：`RuntimeError: values must be torch::kFloat32`
```
原因：Python 传入的 Tensor 类型与 C++ 期望的不匹配
解决：确保 Tensor 类型正确，如 `.float()` 或 `.cuda().float()`
```

### 5. 内存布局问题

**问题**：结果不正确或崩溃
```
原因：Tensor 内存不连续
解决：使用 `.contiguous()` 确保内存连续
```

---

## 最佳实践

### 1. 代码组织

```
project/
├── cuda_kernels/
│   ├── vector_add.cu
│   └── other_kernels.cu
├── python/
│   └── main.py
└── README.md
```

### 2. 错误处理

```cpp
// C++ 端
void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    // 检查输入
    TORCH_CHECK(a.dim() == b.dim(), "维度不匹配");
    TORCH_CHECK(a.sizes() == b.sizes(), "形状不匹配");
    
    // 检查设备
    TORCH_CHECK(a.device().is_cuda(), "必须在 GPU 上");
    
    // 执行内核
    // ...
}
```

### 3. 文档和注释

- 为每个函数添加文档字符串
- 解释线程配置的选择原因
- 记录性能特性（如内存带宽、计算强度）

### 4. 测试

```python
def test_vector_add():
    a = torch.randn(1000).cuda().float()
    b = torch.randn(1000).cuda().float()
    c = torch.zeros_like(a)
    
    lib.vector_add_f32(a, b, c)
    
    expected = a + b
    assert torch.allclose(c, expected, rtol=1e-5)
```

### 5. 版本控制

- 不要提交编译产物（`.so`, `.pyd`）
- 在 `.gitignore` 中添加：
```
*.so
*.pyd
__pycache__/
.cache/
```

### 6. 性能分析

使用 CUDA 性能分析工具：
```bash
# 使用 nvprof
nvprof python script.py

# 使用 Nsight Compute
nsys profile python script.py
```

---

## 总结

Python 与 CUDA 函数绑定是一个强大的技术，它结合了：

- **Python 的易用性**：快速开发和原型设计
- **CUDA 的高性能**：GPU 加速计算
- **PyTorch 的生态**：与深度学习框架无缝集成

通过本示例，我们学习了：

1. ✅ 如何使用 `torch.utils.cpp_extension.load()` 动态编译
2. ✅ 如何编写 CUDA 内核函数
3. ✅ 如何使用 pybind11 绑定 C++ 函数到 Python
4. ✅ 如何优化 CUDA 内核性能（向量化、线程配置）
5. ✅ 如何进行性能基准测试

**下一步学习方向**：
- 更复杂的 CUDA 内核（矩阵乘法、卷积等）
- 共享内存和常量内存的使用
- 多 GPU 编程
- CUDA Streams 和异步执行

---

## 参考资料

- [PyTorch C++ Extensions 官方文档](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [pybind11 文档](https://pybind11.readthedocs.io/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**文档版本**：1.0  
**最后更新**：2024年
