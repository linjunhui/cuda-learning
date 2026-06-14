# CUDA GEMM 项目文档

本文档详细解释两个核心主题：
1. **Pybind11 函数绑定与参数转换机制**
2. **Benchmark 设计：Warmup 与算力计算**

---

## 一、Pybind11 函数绑定与参数转换

### 1.1 Pybind11 简介

Pybind11 是一个轻量级的 C++ 库，用于在 Python 和 C++ 之间创建绑定。它允许 C++ 代码被 Python 直接调用，无需手动编写 Python C API。

### 1.2 基本绑定语法

在我们的项目中，使用 PyTorch 的扩展系统，它基于 pybind11。基本语法如下：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("函数名", &C++函数指针, "函数文档字符串");
}
```

**关键点：**
- `TORCH_EXTENSION_NAME` 是 PyTorch 自动定义的宏，对应 Python 模块名
- `m` 是模块对象，用于定义函数
- `m.def()` 用于绑定 C++ 函数到 Python

### 1.3 函数签名绑定

#### 1.3.1 基本类型绑定

Pybind11 支持自动类型转换，常见类型映射如下：

| C++ 类型 | Python 类型 | 说明 |
|---------|------------|------|
| `int` | `int` | 整数 |
| `float` | `float` | 浮点数 |
| `double` | `float` | 双精度浮点数 |
| `bool` | `bool` | 布尔值 |
| `std::string` | `str` | 字符串 |
| `std::vector<T>` | `list` | 列表 |

#### 1.3.2 张量类型绑定（PyTorch 扩展）

在我们的 GEMM 实现中，使用 `torch::Tensor` 类型：

```cpp
void naive_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C)
```

**参数转换过程：**

1. **Python 端调用：**
   ```python
   lib.naive_gemm(A, B, C)  # A, B, C 是 torch.Tensor
   ```

2. **Pybind11 自动转换：**
   - Python `torch.Tensor` → C++ `torch::Tensor`
   - 无需手动转换，pybind11 自动处理

3. **C++ 端接收：**
   ```cpp
   void naive_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
       // A, B, C 已经是 torch::Tensor 类型
   }
   ```

### 1.4 参数验证与转换

#### 1.4.1 类型检查

在函数内部，我们需要验证张量的类型和维度：

```cpp
// 检查数据类型
TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

// 检查维度
TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
```

**TORCH_CHECK 宏：**
- 如果条件为假，抛出 Python 异常
- 错误信息会传递到 Python 端

#### 1.4.2 维度提取

从 `torch::Tensor` 提取维度信息：

```cpp
int M = A.size(0);  // 第 0 维的大小（行数）
int N = A.size(1);  // 第 1 维的大小（列数）
```

#### 1.4.3 数据指针获取

获取底层数据指针（用于 CUDA kernel）：

```cpp
const float* d_A = A.data_ptr<float>();  // 只读指针
float* d_C = C.data_ptr<float>();        // 可写指针
```

**重要说明：**
- `data_ptr<T>()` 返回指向张量数据的原始指针
- 类型 `T` 必须与张量的数据类型匹配
- 对于 GPU 张量，返回的是设备内存指针

### 1.5 完整绑定示例

```cpp
// 1. 定义 C++ 函数
void naive_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // 参数验证
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
    
    // 提取维度
    int M = A.size(0);
    int N = A.size(1);
    
    // 获取数据指针
    const float* d_A = A.data_ptr<float>();
    float* d_C = C.data_ptr<float>();
    
    // 调用 CUDA kernel
    matrix_multi_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
}

// 2. 绑定到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_gemm", &naive_gemm, "朴素矩阵乘法 (C = A * B)");
}
```

### 1.6 为什么 torch::Tensor 可以自动转换？

在 PyTorch 的 `cpp_extension` 系统中，`torch::Tensor` 之所以可以自动转换，是因为 **PyTorch 已经为 pybind11 提供了预定义的绑定**。

#### 1.6.1 PyTorch 的预定义绑定

PyTorch 在编译时已经注册了 `torch::Tensor` 与 Python `torch.Tensor` 之间的转换规则：

```cpp
// PyTorch 内部已经实现了类似这样的绑定（简化版）
namespace pybind11 {
    template <>
    struct type_caster<torch::Tensor> {
        // 自动处理 Python torch.Tensor → C++ torch::Tensor 的转换
        // 自动处理 C++ torch::Tensor → Python torch.Tensor 的转换
    };
}
```

因此，当我们使用 `torch::Tensor` 作为函数参数时，pybind11 会自动识别并使用这些预定义的转换规则。

#### 1.6.2 其他 PyTorch 类型的自动转换

除了 `torch::Tensor`，PyTorch 还提供了其他类型的自动转换：

| C++ 类型 | Python 类型 | 说明 |
|---------|------------|------|
| `torch::Tensor` | `torch.Tensor` | 张量 |
| `torch::Scalar` | `int/float` | 标量值 |
| `torch::IntArrayRef` | `list/tuple` | 整数数组 |
| `torch::optional<T>` | `T` 或 `None` | 可选类型 |
| `std::vector<torch::Tensor>` | `list[torch.Tensor]` | 张量列表 |

### 1.7 自定义类型和类的转换

**是的，自定义类型和类也可以转换！** 但需要手动编写绑定代码。

#### 1.7.1 基本自定义类型绑定

```cpp
// 定义自定义结构体
struct MatrixSize {
    int M, N, K;
    MatrixSize(int m, int n, int k) : M(m), N(n), K(k) {}
};

// 绑定到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 注册结构体
    py::class_<MatrixSize>(m, "MatrixSize")
        .def(py::init<int, int, int>())  // 构造函数
        .def_readwrite("M", &MatrixSize::M)
        .def_readwrite("N", &MatrixSize::N)
        .def_readwrite("K", &MatrixSize::K);
    
    // 使用自定义类型作为参数
    m.def("my_gemm", [](torch::Tensor A, torch::Tensor B, MatrixSize size) {
        // size.M, size.N, size.K 可以直接使用
        // ...
    });
}
```

**Python 端使用：**
```python
import lib
size = lib.MatrixSize(1024, 1024, 1024)
lib.my_gemm(A, B, size)
```

#### 1.7.2 自定义类的完整示例

```cpp
// 定义矩阵乘法配置类
class GEMMConfig {
public:
    int block_size;
    bool use_shared_memory;
    float alpha;
    
    GEMMConfig(int bs = 16, bool usm = true, float a = 1.0f)
        : block_size(bs), use_shared_memory(usm), alpha(a) {}
    
    void print() const {
        std::cout << "Block size: " << block_size << std::endl;
    }
};

// 绑定类
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GEMMConfig>(m, "GEMMConfig")
        .def(py::init<>())  // 默认构造函数
        .def(py::init<int, bool, float>())  // 带参数的构造函数
        .def_readwrite("block_size", &GEMMConfig::block_size)
        .def_readwrite("use_shared_memory", &GEMMConfig::use_shared_memory)
        .def_readwrite("alpha", &GEMMConfig::alpha)
        .def("print", &GEMMConfig::print);  // 绑定成员函数
    
    // 使用配置类
    m.def("configured_gemm", [](torch::Tensor A, torch::Tensor B, 
                                torch::Tensor C, GEMMConfig config) {
        // 使用 config.block_size, config.use_shared_memory 等
        // ...
    });
}
```

**Python 端使用：**
```python
config = lib.GEMMConfig(block_size=32, use_shared_memory=True, alpha=1.0)
config.print()
lib.configured_gemm(A, B, C, config)
```

#### 1.7.3 自定义类型转换器（高级）

如果需要更复杂的转换（例如 Python dict → C++ struct），可以自定义类型转换器：

```cpp
// C++ 结构体
struct KernelParams {
    int threads_per_block;
    int blocks_per_grid;
    bool enable_fast_math;
};

// 自定义转换器
namespace pybind11 {
    template <>
    struct type_caster<KernelParams> {
        PYBIND11_TYPE_CASTER(KernelParams, _("KernelParams"));
        
        bool load(handle src, bool) {
            if (!py::isinstance<py::dict>(src)) return false;
            
            auto d = py::cast<py::dict>(src);
            value.threads_per_block = py::cast<int>(d["threads_per_block"]);
            value.blocks_per_grid = py::cast<int>(d["blocks_per_grid"]);
            value.enable_fast_math = py::cast<bool>(d.get("enable_fast_math", false));
            return true;
        }
        
        static handle cast(const KernelParams& src, return_value_policy, handle) {
            py::dict d;
            d["threads_per_block"] = src.threads_per_block;
            d["blocks_per_grid"] = src.blocks_per_grid;
            d["enable_fast_math"] = src.enable_fast_math;
            return d.release();
        }
    };
}

// 使用
m.def("kernel_with_params", [](torch::Tensor A, KernelParams params) {
    // params 可以直接使用
});
```

**Python 端使用：**
```python
params = {"threads_per_block": 16, "blocks_per_grid": 64, "enable_fast_math": True}
lib.kernel_with_params(A, params)  # 自动转换为 KernelParams
```

#### 1.7.4 STL 容器的自动转换

pybind11 已经为常见的 STL 容器提供了自动转换：

```cpp
// 这些类型可以直接使用，无需额外绑定
m.def("process_tensors", [](std::vector<torch::Tensor> tensors) {
    // Python list[torch.Tensor] → C++ std::vector<torch::Tensor>
});

m.def("get_sizes", []() -> std::vector<int> {
    return {128, 256, 512};  // 自动转换为 Python list
});

m.def("process_map", [](std::map<std::string, int> config) {
    // Python dict[str, int] → C++ std::map<std::string, int>
});
```

**Python 端使用：**
```python
tensors = [A, B, C]
lib.process_tensors(tensors)

sizes = lib.get_sizes()  # 返回 [128, 256, 512]

config = {"block_size": 16, "grid_size": 64}
lib.process_map(config)
```

#### 1.7.5 总结：类型转换能力

| 类型 | 是否需要手动绑定 | 说明 |
|------|----------------|------|
| `torch::Tensor` | ❌ 否 | PyTorch 已提供 |
| 基本类型 (`int`, `float`, `bool`) | ❌ 否 | pybind11 内置支持 |
| STL 容器 (`vector`, `map`, `set`) | ❌ 否 | pybind11 内置支持 |
| 自定义结构体/类 | ✅ 是 | 需要 `py::class_` 绑定 |
| 自定义转换器 | ✅ 是 | 需要 `type_caster` 特化 |

### 1.8 参数转换流程图

```
Python 调用
    ↓
lib.naive_gemm(A, B, C)
    ↓
Pybind11 类型检查与转换
    ├─ torch.Tensor → torch::Tensor (自动，PyTorch 提供)
    ├─ int → int (自动，pybind11 内置)
    ├─ list → std::vector (自动，pybind11 内置)
    └─ 自定义类型 → 需要手动绑定
    ↓
C++ 函数接收
    ↓
void naive_gemm(torch::Tensor A, ...)
    ↓
提取维度、获取指针
    ↓
调用 CUDA kernel
```

---

## 二、Benchmark 设计：Warmup 与算力计算

### 2.1 Benchmark 的重要性

性能测试是优化 CUDA 程序的关键步骤。准确的 benchmark 可以帮助我们：
- 评估优化效果
- 发现性能瓶颈
- 对比不同实现的性能

### 2.2 Warmup（预热）机制

#### 2.2.1 为什么需要 Warmup？

GPU 执行存在以下特点，需要预热：

1. **首次启动延迟：**
   - GPU 驱动初始化
   - CUDA context 创建
   - 内存分配和传输

2. **缓存预热：**
   - GPU L1/L2 缓存未命中
   - 指令缓存未填充

3. **频率提升：**
   - GPU 动态频率调整（boost clock）
   - 需要时间达到最高频率

#### 2.2.2 Warmup 实现

```python
def benchmark_single_size(M, N, K, num_iterations=10, warmup=5):
    # 准备数据
    A_gpu = torch.randn(M, N, dtype=torch.float32, device='cuda')
    B_gpu = torch.randn(N, K, dtype=torch.float32, device='cuda')
    C_gpu = torch.zeros(M, K, dtype=torch.float32, device='cuda')
    
    # 预热阶段：执行 warmup 次，但不计时
    for _ in range(warmup):
        lib.naive_gemm(A_gpu, B_gpu, C_gpu)
    torch.cuda.synchronize()  # 确保所有预热操作完成
    
    # 正式计时阶段
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()  # 同步，确保上次操作完成
        start = time.time()
        lib.naive_gemm(A_gpu, B_gpu, C_gpu)
        torch.cuda.synchronize()  # 同步，确保本次操作完成
        times.append(time.time() - start)
    
    return np.mean(times)
```

**关键点：**
- `warmup=5`：预热 5 次（可根据需要调整）
- `torch.cuda.synchronize()`：确保 GPU 操作完成
- 预热结果不计入性能统计

#### 2.2.3 同步的重要性

```python
torch.cuda.synchronize()  # 必须！
start = time.time()
lib.naive_gemm(A, B, C)   # 异步执行
torch.cuda.synchronize()  # 必须！
end = time.time()
```

**为什么需要同步？**
- CUDA kernel 是**异步执行**的
- 不同步会导致计时不准确（可能只测量了启动时间）
- 同步确保 kernel 完全执行完毕

### 2.3 算力计算（GFLOPS）

#### 2.3.1 什么是 GFLOPS？

**GFLOPS** = Giga Floating Point Operations Per Second（每秒十亿次浮点运算）

用于衡量计算性能的指标。

#### 2.3.2 矩阵乘法的运算量

对于矩阵乘法 C = A × B：
- A: M × N
- B: N × K
- C: M × K

**浮点运算次数计算：**
- 每个输出元素 C[i][j] 需要 N 次乘法和 N 次加法
- 总元素数：M × K
- **总运算量 = M × N × K × 2**（乘法和加法各 N 次）

**公式：**
```
FLOPS = M × N × K × 2
GFLOPS = FLOPS / (时间_秒) / 1e9
```

#### 2.3.3 实现代码

```python
def benchmark_single_size(M, N, K, num_iterations=10, warmup=5):
    # ... 预热和计时代码 ...
    
    avg_time = np.mean(times)  # 平均时间（秒）
    
    # 计算 GFLOPS
    flops = 2.0 * M * N * K  # 总浮点运算次数
    gflops = flops / avg_time / 1e9  # 转换为 GFLOPS
    
    print(f"大小 {M}x{N} × {N}x{K}: {avg_time*1000:.3f} ms, {gflops:.2f} GFLOPS")
    
    return avg_time
```

#### 2.3.4 性能分析示例

假设测试 1024×1024 矩阵乘法：

```
M = 1024, N = 1024, K = 1024
FLOPS = 2 × 1024 × 1024 × 1024 = 2,147,483,648
如果执行时间 = 0.01 秒
GFLOPS = 2,147,483,648 / 0.01 / 1e9 = 214.75 GFLOPS
```

### 2.4 完整的 Benchmark 流程

```python
def plot_performance_comparison():
    problem_sizes = [128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072]
    num_iterations = 10
    warmup = 5
    
    my_times = []
    sizes_list = []
    
    for size in problem_sizes:
        # 1. 预热 + 计时
        avg_time = benchmark_single_size(size, size, size, num_iterations, warmup)
        
        # 2. 计算 GFLOPS
        gflops = (2.0 * size * size * size) / avg_time / 1e9
        
        # 3. 记录结果
        my_times.append(avg_time)
        sizes_list.append(size)
        
        print(f"大小 {size}x{size}: {avg_time*1000:.3f} ms, {gflops:.2f} GFLOPS")
    
    # 4. 绘制性能曲线
    plt.semilogy(sizes_array, my_times, 'r-s', label='我的矩阵乘法')
    plt.show()
```

### 2.5 Benchmark 最佳实践

#### 2.5.1 迭代次数选择

- **warmup**: 5-10 次（确保 GPU 稳定）
- **iterations**: 10-100 次（根据测试时间调整）
- 更多迭代 → 更准确，但更耗时

#### 2.5.2 统计方法

```python
times = []
for _ in range(num_iterations):
    # ... 计时 ...
    times.append(time)

avg_time = np.mean(times)      # 平均值
std_time = np.std(times)       # 标准差
min_time = np.min(times)       # 最小值
max_time = np.max(times)       # 最大值

print(f"{avg_time:.3f} ± {std_time:.3f} ms")
```

#### 2.5.3 避免干扰因素

1. **关闭其他 GPU 程序**
2. **固定 GPU 频率**（可选）
3. **使用相同的数据**（或固定随机种子）
4. **多次运行取平均**

### 2.6 性能分析指标

除了 GFLOPS，还可以关注：

1. **带宽利用率：**
   ```
   理论带宽 = GPU 内存带宽（GB/s）
   实际带宽 = 数据传输量 / 时间
   利用率 = 实际带宽 / 理论带宽 × 100%
   ```

2. **计算强度（Compute Intensity）：**
   ```
   CI = FLOPS / 字节数
   ```
   用于判断是计算瓶颈还是内存瓶颈

3. **加速比：**
   ```
   加速比 = CPU时间 / GPU时间
   ```

---

## 三、总结

### 3.1 Pybind11 绑定要点

1. **自动类型转换**：Python 类型自动转换为 C++ 类型
2. **参数验证**：使用 `TORCH_CHECK` 验证输入
3. **数据访问**：使用 `data_ptr<T>()` 获取原始指针
4. **模块定义**：使用 `PYBIND11_MODULE` 定义 Python 模块

### 3.2 Benchmark 设计要点

1. **Warmup 必不可少**：消除首次启动延迟
2. **同步是关键**：使用 `torch.cuda.synchronize()` 确保准确计时
3. **多次迭代**：取平均值提高准确性
4. **算力计算**：使用 GFLOPS 量化性能

### 3.3 性能优化路径

```
Naive 实现
    ↓
Warmup + 准确 Benchmark
    ↓
识别瓶颈（计算/内存）
    ↓
针对性优化
    ↓
重新 Benchmark 验证
```

---

## 参考资料

- [Pybind11 官方文档](https://pybind11.readthedocs.io/)
- [PyTorch C++ 扩展文档](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA 性能分析最佳实践](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

