# SGL Kernel

SGLang 的 [核心库](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)

<div align="center">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sgl-project/sglang/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sgl-kernel)](https://pypi.org/project/sgl-kernel)

</div>

SGL Kernel 为 SGLang 框架提供优化的计算原语，通过自定义内核操作实现大语言模型和视觉语言模型的高效推理。

## 安装
需要 torch == 2.8.0

```bash
# 最新版本
pip3 install sgl-kernel --upgrade
```

## 从源码构建
需要
- CMake ≥3.31
- Python ≥3.10
- scikit-build-core
- ninja（可选）

### 使用 Makefile 构建 sgl-kernel

```bash
make build
```

## 贡献

### 添加新内核的步骤：

1. 在 [csrc](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc) 中实现内核
2. 在 [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/include/sgl_kernel_ops.h) 中暴露接口
3. 在 [csrc/common_extension.cc](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/common_extension.cc) 中创建 torch 扩展
4. 更新 [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/CMakeLists.txt) 以包含新的 CUDA 源文件
5. 在 [python](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/python/sgl_kernel) 中暴露 Python 接口
6. 添加测试和基准测试

### 开发提示

1. 创建 torch 扩展时，使用 `m.def` 添加函数定义，使用 `m.impl` 进行设备绑定：

- 如何编写 schema：[Schema 参考](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // 这里需要带 schema 的 def 以便 torch.compile 使用
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### 适配 C++ 原生类型以兼容 Torch

第三方 C++ 库通常使用 int 和 float，但由于 Python 的类型映射，PyTorch 绑定需要 int64_t 和 double。

使用 sgl_kernel_torch_shim.h 中的 make_pytorch_shim 来自动处理类型转换：

```cpp

// 添加 int -> int64_t 的类型转换
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "value too large");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "value too small");
    return arg;
  }
};
```
```cpp
// 包装你的函数
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### 测试和基准测试

1. 在 [tests/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests) 中添加 pytest 测试，如果需要跳过某些测试，请使用 `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 需要计算能力为 10 或更高。"
)
```

2. 在 [benchmark/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark) 中使用 [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) 添加基准测试

   **我们推荐使用 `triton.testing.do_bench_cudagraph` 进行内核基准测试**：

   与 `triton.testing.do_bench` 相比，`do_bench_cudagraph` 提供：
   - 减少 CPU 开销影响，获得更准确的内核性能测量
   - 将 PDL（程序化依赖启动）效应纳入单个内核结果
   - 在支持 PDL 的架构（SM >= 90）上提供更真实的性能数据

3. 运行测试套件

## 常见问题
- Q: 使用 CUDA 12.6 时出现段错误
- A: 将 ptxas 更新到 12.8，参考：[段错误错误](https://github.com/Dao-AILab/flash-attention/issues/1453)

