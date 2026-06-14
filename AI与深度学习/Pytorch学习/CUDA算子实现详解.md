# PyTorch CUDA 算子实现详解

## 📖 概述

本文档从简单到复杂详细介绍 PyTorch 中 CUDA 算子的实现方式。CUDA 算子是 PyTorch 在 GPU 上执行计算的核心，理解它们的实现有助于：

- 理解 PyTorch 的 GPU 计算机制
- 学习如何实现高效的 CUDA 算子
- 优化自定义算子性能

## 🎯 核心概念：Functor（函数对象）

在深入 CUDA 算子之前，需要理解一个重要的 C++ 概念：**Functor（函数对象）**。

### 什么是 Functor？

**Functor** 是通过重载 `operator()` 操作符来使对象可以像函数一样被调用的 C++ 类或结构体。

### 示例说明

```cpp
// 定义一个 Functor
struct AddFunctor {
    int value;
    AddFunctor(int v) : value(v) {}
    
    // 重载 () 操作符，使对象可像函数一样调用
    int operator() (int x) const {
        return x + value;
    }
};

// 使用方式
AddFunctor add_5(5);  // 创建函数对象，value = 5
int result = add_5(10);  // 调用 operator(10)，返回 15
// 等价于：result = 10 + 5
```

### 为什么使用 Functor？

1. **携带状态**：可以将数据存储在对象中（如 `FillFunctor` 存储填充值）
2. **模板友好**：可以用模板参数指定类型，支持多种数据类型
3. **性能优化**：编译器可以更好地内联优化
4. **CUDA 兼容**：可以在 GPU 上执行（使用 `__device__` 标记）

### PyTorch 中的 Functor 模式

PyTorch 中的所有 CUDA 算子都使用 Functor 模式：
- **FillFunctor**：`operator() ()` - 无参数，返回固定值
- **AbsFunctor**：`operator() (scalar_t a)` - 一个参数，返回绝对值
- **MulFunctor**：`operator() (scalar_t a, scalar_t b)` - 两个参数，返回乘积

这种统一的设计使得所有算子都可以通过相同的接口（`gpu_kernel`）来启动。

---

## 🏗️ CUDA 算子的基本架构

PyTorch 中的 CUDA 算子实现遵循以下层次结构：

```
Python API 调用 (torch.fill, torch.abs 等)
    ↓
ATen 分发系统 (根据设备类型选择实现)
    ↓
CUDA 算子入口函数 (如 fill_kernel_cuda, abs_kernel_cuda)
    ↓
gpu_kernel/gpu_reduce_kernel (通用内核启动函数)
    ↓
实际 CUDA Kernel (在 GPU 上执行的代码)
```

### 关键组件

1. **TensorIterator**：统一处理张量迭代的工具，自动处理广播、内存布局等
2. **Loops.cuh / CUDALoops.cuh**：提供通用的内核启动框架
3. **Functor**：封装实际计算逻辑的函数对象
4. **Dispatch 系统**：根据数据类型自动选择实现

---

## 1️⃣ 最简单的算子：Fill（填充）

### 1.1 算子说明

**Fill** 算子是最简单的 CUDA 算子之一，它的功能是使用指定的标量值填充整个张量。

**功能**：`out[i] = value` 对所有 `i`

**特点**：
- 单输出，无输入张量（只有标量参数）
- 每个输出元素的计算相互独立
- 内存访问模式简单（只写输出）

### 1.2 源码实现

#### 入口函数：`fill_kernel_cuda`

```1:30:Pytorch学习/pytorch/aten/src/ATen/native/cuda/FillKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>
#include <c10/core/Scalar.h>

namespace at::native {

template<typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v): value(v) {}
  __device__ __forceinline__ scalar_t operator() () const {
    return value;
  }
  private:
    scalar_t value;
};

void fill_kernel_cuda(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_V2(iter.dtype(), "fill_cuda", AT_WRAP([&]() {
    gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kBool, kHalf, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_cuda)

} // namespace at::native
```

#### 代码解析

1. **FillFunctor 结构体**：
   - 这是一个**函数对象（Functor）**，用于封装填充操作
   - 构造函数接收填充值 `value` 并存储在成员变量中
   - **`operator() ()` 重载**：这是函数调用操作符的重载，使得 `FillFunctor` 对象可以像函数一样被调用
     ```cpp
     FillFunctor<float> functor(5.0f);
     float result = functor();  // 调用 operator()，返回 5.0f
     ```
   - `const` 关键字表示该函数不会修改对象状态
   - `__device__` 表示该函数可以在 GPU 上执行
   - `__forceinline__` 提示编译器尽可能内联该函数
   - 这是 CUDA kernel 中实际调用的函数，每个线程都会调用它来获取填充值

2. **`AT_DISPATCH_V2` 宏**：
   - 根据张量的数据类型自动分发到对应的模板实例化
   - 支持多种数据类型（float32, int64, complex 等）

3. **`gpu_kernel` 函数**：
   - 通用的 CUDA 内核启动函数
   - 自动处理内存访问、线程分配等细节
   - 接收 `TensorIterator` 和 `Functor`

4. **`REGISTER_DISPATCH`**：
   - 注册算子到分发系统，使其能被调用

---

## 2️⃣ 一元算子：Abs（绝对值）

### 2.1 算子说明

**Abs** 算子计算输入张量中每个元素的绝对值。

**功能**：`out[i] = |in[i]|`

**特点**：
- 单输入单输出
- 元素级操作（element-wise），每个元素独立计算
- 支持多种数据类型，包括复数（复数的绝对值是其模长）

### 2.2 源码实现

#### 入口函数：`abs_kernel_cuda`

```1:51:Pytorch学习/pytorch/aten/src/ATen/native/cuda/AbsKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at::native {

template<typename scalar_t>
struct AbsFunctor {
  __device__ __forceinline__ scalar_t operator() (const scalar_t a) const {
    return std::abs(a);
  }
};

constexpr char abs_name[] = "abs_kernel";
void abs_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto abs_string = jiterator_stringify(
        template <typename T> T abs_kernel(T x) { return std::abs(x); });
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/abs_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, abs_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_cuda", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, AbsFunctor<opmath_t>());
    });
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Half,
        ScalarType::BFloat16,
        ScalarType::Bool,
        iter.dtype(),
        "abs_cuda",
        [&]() { gpu_kernel(iter, AbsFunctor<scalar_t>()); });
  }
}

  REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda)

} // namespace at::native
```

#### 代码解析

1. **AbsFunctor 结构体**：
   - `operator()` 接收一个参数 `a`（输入元素）
   - 使用 `std::abs()` 计算绝对值
   - `__device__ __forceinline__` 表示这是设备端函数，编译器会尝试内联

2. **复数类型特殊处理**：
   - 复数类型可以使用 JIT 编译（JIterator）来提高性能
   - JIterator 是运行时编译系统，可以减少二进制大小

3. **数据类型分发**：
   - 对复数类型使用 `AT_DISPATCH_COMPLEX_TYPES_AND`
   - 对实数类型使用 `AT_DISPATCH_ALL_TYPES_AND3`

### 2.3 底层实现：gpu_kernel 如何工作

虽然我们在实现中只写了 Functor，但 `gpu_kernel` 函数会处理所有的底层细节：

```105:126:Pytorch学习/pytorch/aten/src/ATen/native/cuda/Loops.cuh
template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl(iter, f);
}
```

**关键步骤**：
1. **设备检查**：确保所有张量都在 CUDA 设备上
2. **空张量处理**：如果元素数为 0，直接返回
3. **32 位索引限制**：如果张量太大无法用 32 位索引，需要分块处理
4. **实际内核启动**：调用 `gpu_kernel_impl` 启动 CUDA kernel

---

## 3️⃣ 二元算子：Mul（乘法）

### 3.1 算子说明

**Mul** 算子计算两个输入张量的逐元素乘积。

**功能**：`out[i] = a[i] * b[i]`

**特点**：
- 双输入单输出
- 支持标量广播（如 `tensor * 5`）
- 支持不同类型之间的运算（如 `float32 * float64`）
- 对于复数，计算复数的乘法

### 3.2 源码实现

#### 入口函数：`mul_kernel_cuda`

```1:48:Pytorch学习/pytorch/aten/src/ATen/native/cuda/BinaryMulKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/BinaryInternal.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

constexpr char mul_name[] = "mul_kernel";
void mul_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
#if AT_USE_JITERATOR()
    static const auto mul_string = jiterator_stringify(
        template <typename T> T mul_kernel(T a, T b) { return a * b; });
    opmath_jitted_gpu_kernel_with_scalars<mul_name, scalar_t, scalar_t>(
        iter, mul_string);
#else
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, binary_internal::MulFunctor<opmath_t>());
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, binary_internal::MulFunctor<opmath_t>());
        });
  }
}

REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda)

} // namespace at::native
```

#### 代码解析

1. **`opmath_symmetric_gpu_kernel_with_scalars`**：
   - 这是专门处理二元操作的函数
   - **symmetric** 表示操作是对称的（`a * b == b * a`），可以优化标量参数的位置
   - **with_scalars** 表示支持其中一个参数是标量（CPU 上的标量）

2. **`common_dtype`**：
   - 二元操作需要处理两个输入可能类型不同的情况
   - `common_dtype` 是两个输入类型提升后的共同类型

3. **`opmath_type`**：
   - 为了提高数值精度，PyTorch 使用操作数学类型
   - 例如，`float16` 的计算可能使用 `float32` 进行，最后再转换回 `float16`

#### MulFunctor 实现（在 BinaryInternal.h 中）

虽然源码中没有直接显示，但 `binary_internal::MulFunctor` 大致如下：

```cpp
template<typename scalar_t>
struct MulFunctor {
  __device__ __forceinline__ scalar_t operator() (const scalar_t a, const scalar_t b) const {
    return a * b;
  }
};
```

#### 标量处理

`opmath_symmetric_gpu_kernel_with_scalars` 内部会检查是否有标量参数：

```203:241:Pytorch学习/pytorch/aten/src/ATen/native/cuda/Loops.cuh
template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  // Use symmetric property of the functor to reduce number of kernels,
  // requires f(a, b) == f(b, a)
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg_t = typename traits::template arg<0>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  static_assert(std::is_same_v<opmath_arg_t, typename traits::template arg<1>::type>,
                "f is not symmetric");

  OptionalDeviceGuard device_guard;
  opmath_arg_t scalar_val{};

  if (iter.is_cpu_scalar(1)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(1);
    iter.remove_operand(1);

    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    device_guard.reset_device(iter.device(1));
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(2);
    iter.remove_operand(2);
  }

  if (iter.ninputs() == 2) {
    gpu_kernel(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
    gpu_kernel(iter, unary_f);
  }
}
```

**关键点**：
- 如果检测到标量参数，将其提取出来
- 将二元 Functor 转换为一元 Functor（其中一个参数已固定为标量值）
- 这样可以避免在 GPU 内存中存储标量，提高性能

---

## 4️⃣ 归约算子：Sum（求和）

### 4.1 算子说明

**Sum** 算子是一个归约操作（reduction），它将输入张量的多个元素归约为一个或多个标量值。

**功能**：`out = sum(in[i])` 对所有 `i`

**特点**：
- **归约操作**：多个输入元素映射到一个输出元素
- **需要同步**：不同线程处理的数据需要合并结果
- **使用共享内存**：在 block 内部进行部分归约
- **可能需要多级归约**：如果数据量很大，需要多个 kernel launch

### 4.2 源码实现

#### 入口函数：`sum_functor`

```13:30:Pytorch学习/pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct sum_functor {
  void operator()(TensorIterator& iter) {
    const auto sum_combine = [] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
      return a + b;
    };
    constexpr bool is_16_bits = sizeof(scalar_t) == 2;
    if constexpr (is_16_bits) {
      gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(
        iter, func_wrapper<out_t>(sum_combine)
      );
    } else {
      gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>(sum_combine)
      );
    }
  }
};
```

#### 代码解析

1. **`sum_combine` 函数**：
   - 这是归约操作的合并函数
   - `GPU_LAMBDA` 宏确保这个 lambda 可以在 GPU 上执行
   - 函数定义了如何合并两个值：`a + b`

2. **`gpu_reduce_kernel`**：
   - 这是归约操作的通用内核启动函数
   - 与 `gpu_kernel` 不同，它需要处理线程间的数据合并
   - **模板参数**：
     - `scalar_t`：输入元素类型
     - `out_t`：输出元素类型
     - `vt0`：每个线程处理的元素数量（向量化参数）
     - `input_vec_size`：向量化加载的元素数量

### 4.2.1 `gpu_reduce_kernel` 核心代码分析

#### 入口函数签名

```1222:1224:Pytorch学习/pytorch/aten/src/ATen/native/cuda/Reduce.cuh
template <typename scalar_t, typename out_scalar_t, int vt0=4, int input_vec_size=vt0, typename ops_t, typename ident_t=double>
inline void gpu_reduce_kernel(TensorIterator& iter, const ops_t& ops, ident_t ident=0,
                              AccumulationBuffer* acc_buf_ptr=nullptr, int64_t base_idx=0) {
```

**关键参数说明**：
- `iter`: TensorIterator，包含输入输出张量信息
- `ops`: 归约操作函数对象（如 sum 的 `combine` 函数）
- `ident`: 归约的单位元（sum 为 0，prod 为 1）
- `vt0`: 向量化参数，每个线程处理的元素数

#### 核心执行流程

```1280:1323:Pytorch学习/pytorch/aten/src/ATen/native/cuda/Reduce.cuh
  const char* in_data = (char*)iter.data_ptr(iter.ntensors() - 1);
  char* out_data = (char*)iter.data_ptr(0);
  const auto noutputs = iter.noutputs();
  std::optional<char*> out_data_extra;
  if (noutputs > 1) {
    out_data_extra = (char*)iter.data_ptr(1);
  } else {
    out_data_extra = std::nullopt;
  }
  char* acc_data = acc_buf_ptr->get_acc_slice(out_data);

  ReduceConfig config = setReduceConfig<arg_t, scalar_t, vt0, input_vec_size>(iter);
  at::DataPtr buffer;
  at::DataPtr semaphores;
  if (config.should_global_reduce()) {
    auto& allocator = *c10::cuda::CUDACachingAllocator::get();
    buffer = allocator.allocate(config.global_memory_size());
    semaphores = allocator.allocate(config.semaphore_size());

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemsetAsync(semaphores.get(), 0, config.semaphore_size(), stream));
  }

  AT_ASSERT(can_use_32bit_indexing);
  auto output_calc = make_output_calculator<uint32_t>(iter);
  auto input_calc = make_input_calculator<uint32_t>(iter);
  auto reduce = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, vt0, input_vec_size>(
      ops,
      config,
      input_calc,
      output_calc,
      in_data,
      out_data,
      out_data_extra,
      acc_data,
      buffer.get(),
      (int*)semaphores.get(),
      ident,
      noutputs,
      base_idx);
  reduce.accumulate = iter.should_accumulate();
  reduce.final_output = iter.is_final_output();

  launch_reduce_kernel<mnt_wrapper<scalar_t>::MAX_NUM_THREADS>(config, reduce);
```

**关键步骤解析**：

1. **获取数据指针**：
   ```cpp
   const char* in_data = (char*)iter.data_ptr(iter.ntensors() - 1);  // 输入数据
   char* out_data = (char*)iter.data_ptr(0);  // 输出数据
   ```
   - 输入在最后一个张量位置
   - 输出在第一个张量位置

2. **配置归约参数**（`ReduceConfig`）：
   - 计算 block 和 grid 大小
   - 确定是否需要全局归约（多 block）
   - 设置共享内存大小

3. **创建归约操作对象**（`ReduceOp`）：
   - 封装所有归约所需的信息
   - 包含操作函数、配置、内存指针等

4. **启动内核**：
   ```cpp
   launch_reduce_kernel<mnt_wrapper<scalar_t>::MAX_NUM_THREADS>(config, reduce);
   ```

#### ReduceOp::run() - 核心归约执行逻辑

```401:477:Pytorch学习/pytorch/aten/src/ATen/native/cuda/Reduce.cuh
  template <int output_vec_size>
  C10_DEVICE void run() const {
    extern __shared__ char shared_memory[];
    index_t output_idx = config.output_idx<output_vec_size>();
    index_t input_idx = config.input_idx();
    auto base_offsets1 = output_calc.get(output_idx)[1];

    using arg_vec_t = std::array<arg_t, output_vec_size>;
    arg_vec_t value;

    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      const scalar_t* input_slice = (const scalar_t*)((const char*)src + base_offsets1);
      value = thread_reduce<output_vec_size>(input_slice);
    }

    if (config.should_block_x_reduce()) {
      value = block_x_reduce<output_vec_size>(value, shared_memory);
    }
    if (config.should_block_y_reduce()) {
      value = block_y_reduce<output_vec_size>(value, shared_memory);
    }
    using out_ptr_vec_t = std::array<out_scalar_t*, output_vec_size>;
    using offset_vec_t = std::array<index_t, output_vec_size>;
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = (out_scalar_t*)((char*)dst[0] + base_offsets[i]);
    }

    arg_vec_t* acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(numerator, denominator);
      acc = (arg_vec_t*)((char*)acc_buf + (base_offsets[0] * numerator / denominator));
    }

    if (config.should_global_reduce()) {
      value = global_reduce<output_vec_size>(value, acc, shared_memory);
    } else if (config.should_store(output_idx)) {
      if (accumulate) {
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = ops.translate_idx(value[i], base_idx);
        }
      }

      if (acc == nullptr) {
        if (accumulate) {
          value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
        }
        if (final_output) {
          set_results_to_output<output_vec_size>(value, base_offsets);
        } else {
          #pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
          }
        }
      } else {
        if (accumulate) {
          #pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = ops.combine((*acc)[i], value[i]);
          }
        }
        if (final_output) {
          set_results_to_output<output_vec_size>(value, base_offsets);
        } else {
          *acc = value;
        }
      }
    }
  }
```

**执行流程解析**：

1. **线程级归约**（`thread_reduce`）：
   ```cpp
   value = thread_reduce<output_vec_size>(input_slice);
   ```
   - 每个线程加载并归约自己负责的输入元素
   - 使用向量化加载提高带宽利用率

2. **Block 内归约**（`block_x_reduce` / `block_y_reduce`）：
   ```cpp
   if (config.should_block_x_reduce()) {
     value = block_x_reduce<output_vec_size>(value, shared_memory);
   }
   if (config.should_block_y_reduce()) {
     value = block_y_reduce<output_vec_size>(value, shared_memory);
   }
   ```
   - 使用共享内存在 block 内合并所有线程的结果
   - X 和 Y 维度分别归约（支持多维归约）

3. **全局归约**（如需要，`global_reduce`）：
   ```cpp
   if (config.should_global_reduce()) {
     value = global_reduce<output_vec_size>(value, acc, shared_memory);
   }
   ```
   - 如果数据量很大，需要多个 block
   - 使用全局内存和原子操作合并多个 block 的结果

4. **写回结果**（`set_results_to_output`）：
   - 将最终结果写回输出张量
   - 支持累积模式（accumulate）和最终输出模式

#### 线程级归约核心代码

```479:558:Pytorch学习/pytorch/aten/src/ATen/native/cuda/Reduce.cuh
  C10_DEVICE arg_t input_vectorized_thread_reduce_impl(const scalar_t* data) const {
    index_t end = config.num_inputs;

    // Handle the head of input slice where data is not aligned
    arg_t value = ident;
    constexpr int align_bytes = alignof(at::native::memory::aligned_vector<scalar_t, input_vec_size>);
    constexpr int align_elements = align_bytes / sizeof(scalar_t);
    int shift = ((uint64_t)data) % align_bytes / sizeof(scalar_t);
    if (shift > 0) {
      data -= shift;
      end += shift;
      if(threadIdx.x >= shift && threadIdx.x < align_elements && config.should_reduce_tail()){
        value = ops.reduce(value, c10::load(data + threadIdx.x), threadIdx.x - shift);
      }
      end -= align_elements;
      data += align_elements;
      shift = align_elements - shift;
    }

    // Do the vectorized reduction
    using load_t = at::native::memory::aligned_vector<scalar_t, input_vec_size>;

    index_t idx = config.input_idx();
    const index_t stride = config.step_input;

    // Multiple accumulators to remove dependency between unrolled loops.
    arg_t value_list[input_vec_size];
    value_list[0] = value;

    #pragma unroll
    for (int i = 1; i < input_vec_size; i++) {
      value_list[i] = ident;
    }

    while (idx * input_vec_size + input_vec_size - 1 < end) {
      const auto values_vec = memory::load_vector<input_vec_size>(data, idx);
      #pragma unroll
      for (index_t i = 0; i < input_vec_size; i++) {
        value_list[i] = ops.reduce(value_list[i], values_vec.val[i], shift + idx * input_vec_size + i);
      }
      idx += stride;
    }

    // tail
    index_t tail_start = end - end % input_vec_size;
    if (config.should_reduce_tail()) {
      int idx = tail_start + threadIdx.x;
      if (idx < end) {
        const auto value = c10::load(data + idx);
        value_list[0] = ops.reduce(value_list[0], value, idx + shift);
      }
    }

    // combine accumulators
    #pragma unroll
    for (int i = 1; i < input_vec_size; i++) {
      value_list[0] = ops.combine(value_list[0], value_list[i]);
    }
    return value_list[0];
  }
```

**关键优化技术**：

1. **内存对齐处理**：
   - 检查数据是否对齐到向量化边界
   - 不对齐时先处理头部数据

2. **向量化加载**：
   ```cpp
   const auto values_vec = memory::load_vector<input_vec_size>(data, idx);
   ```
   - 一次加载多个元素（如 `input_vec_size=8` 表示一次加载 8 个 float）
   - 提高内存带宽利用率

3. **多个累加器**：
   ```cpp
   arg_t value_list[input_vec_size];  // 多个累加器
   ```
   - 减少循环间的数据依赖
   - 允许编译器更好地并行化循环

4. **处理尾部数据**：
   - 向量化处理完整部分
   - 单独处理剩余的不完整向量

#### 内核启动代码

```912:933:Pytorch学习/pytorch/aten/src/ATen/native/cuda/Reduce.cuh
template<int max_threads, typename R>
static void launch_reduce_kernel(const ReduceConfig& config, const R& reduction) {
  dim3 block = config.block();
  dim3 grid = config.grid();

  auto stream = at::cuda::getCurrentCUDAStream();
  int shared_memory = config.shared_memory_size();

  switch(config.output_vec_size) {
  case 4:
    reduce_kernel<max_threads / 4, 4, R><<<grid, block, shared_memory, stream>>>(reduction);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 2:
    reduce_kernel<max_threads / 2, 2, R><<<grid, block, shared_memory, stream>>>(reduction);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  default:
    reduce_kernel<max_threads / 1, 1, R><<<grid, block, shared_memory, stream>>>(reduction);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
```

**关键点**：
- **output_vec_size**：输出向量化大小（1、2、4）
- **共享内存**：动态分配用于 block 内归约
- **模板实例化**：根据向量化大小选择不同的内核版本

### 4.2.2 归约操作的完整流程图

```
输入张量 [N 个元素]
    ↓
[线程级归约]
每个线程: 加载多个元素 → 向量化归约 → 局部累加器
    ↓
[Block 内归约] (共享内存)
Block 0: [thread0, thread1, ...] → tree reduction → 结果0
Block 1: [thread0, thread1, ...] → tree reduction → 结果1
...
    ↓
[全局归约] (如果需要)
多个 Block 的结果 → 原子操作/全局内存 → 最终结果
    ↓
输出张量 [1 个或少量元素]
```

### 4.2.3 关键性能优化

1. **向量化加载**：一次加载 4/8 个元素，提高带宽
2. **多个累加器**：减少循环依赖，提高并行度
3. **共享内存**：Block 内快速通信
4. **树形归约**：O(log n) 复杂度，高效的 block 内归约
5. **对齐优化**：处理非对齐内存访问

3. **16 位类型的特殊处理**：
   - `float16` 和 `bfloat16` 使用更激进的向量化策略
   - `input_vec_size=8` 表示一次加载 8 个元素

### 4.3 归约内核的工作流程

归约操作比元素级操作复杂得多，因为它需要：

1. **加载阶段**：每个线程加载多个输入元素
2. **局部归约**：每个线程在寄存器中归约自己加载的数据
3. **Block 级归约**：使用共享内存（shared memory）在 block 内部归约
4. **全局归约**（如果需要）：多个 block 的结果需要再次归约

#### 简化的归约流程示意

```cpp
// 伪代码示意
__global__ void reduce_kernel(...) {
  // 1. 加载数据到寄存器
  acc_t local_sum = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    local_sum += input[i];
  }
  
  // 2. 存储到共享内存
  __shared__ acc_t shared_sum[blockDim.x];
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();
  
  // 3. Block 内归约（树形归约）
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  // 4. 第一个线程写入结果
  if (threadIdx.x == 0) {
    output[blockIdx.x] = shared_sum[0];
  }
}
```

#### 实际的归约实现

实际的归约实现更复杂，在 `Reduce.cuh` 中：

- **向量化加载**：一次加载多个元素以提高带宽利用率
- **多级归约**：支持 block 内的多维度归约
- **原子操作**：处理多 block 之间的归约
- **不同归约类型**：sum, prod, min, max 等使用不同的合并函数

---

## 5️⃣ 复杂算子：索引操作（Index）

### 5.1 算子说明

**索引操作**允许使用张量索引来访问或修改另一个张量的元素。这是一个相对复杂的操作，因为：

- **不规则访问模式**：每个输出元素可能需要访问输入的不同位置
- **需要边界检查**：索引可能越界
- **内存访问模式复杂**：难以向量化

**功能示例**：`out[i] = input[indices[i]]`

### 5.2 源码实现片段

```28:54:Pytorch学习/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu
template<int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel(const int64_t N, const func_t f) {
  const auto tid = threadIdx.x;
  const auto nv = nt * vt;
  auto idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t>
static void launch_kernel(const int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  const auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

#### 代码解析

1. **`nt` 和 `vt` 模板参数**：
   - `nt`：每个 block 的线程数（number of threads）
   - `vt`：每个线程处理的元素数（values per thread）
   - 这种设计允许调整每个线程的工作量

2. **Grid-Stride Loop 模式**：
   ```cpp
   auto idx = nv * blockIdx.x + tid;
   for (int i = 0; i < vt; i++) {
     if (idx < N) {
       f(idx);
       idx += nt;  // 下一个元素
     }
   }
   ```
   - 这种模式允许任意大小的数组，不需要完美对齐
   - 每个线程处理 `vt` 个元素，元素之间的间隔是 `nt`

3. **`C10_LAUNCH_BOUNDS_2`**：
   - 这是 CUDA 的 launch bounds，告诉编译器每个 block 的最大线程数
   - 帮助编译器进行寄存器分配优化

---

## 🔧 TensorIterator：统一的内存访问抽象

### 什么是 TensorIterator？

`TensorIterator` 是 PyTorch 中用于统一处理不同形状、内存布局的张量迭代的工具。它自动处理：

1. **广播**：自动扩展维度以匹配形状
2. **内存布局**：处理 contiguous、channels_last 等不同布局
3. **类型转换**：处理不同类型之间的运算
4. **偏移计算**：自动计算每个元素的正确内存偏移

### TensorIterator 的工作方式

当我们调用 `gpu_kernel(iter, functor)` 时：

1. `TensorIterator` 分析输入输出张量的形状和内存布局
2. 生成统一的内存访问模式
3. 自动处理广播和类型转换
4. 启动 CUDA kernel，每个线程处理一个或多个元素

### 示例：为什么需要 TensorIterator

考虑两个形状不同的张量相加：

```python
a = torch.randn(3, 1, 5)  # shape: (3, 1, 5)
b = torch.randn(3, 4, 5)  # shape: (3, 4, 5)
c = a + b  # 广播后 shape: (3, 4, 5)
```

`TensorIterator` 会自动：
- 扩展 `a` 的第二个维度（从 1 到 4）
- 计算正确的内存偏移（`a` 的 stride 与 `b` 不同）
- 确保每个线程访问正确的元素

---

## 🚀 性能优化技巧

### 1. 向量化（Vectorization）

PyTorch 大量使用向量化来提高性能：

- **加载向量化**：一次加载 4 个或 8 个元素（如 `float4`, `float8`）
- **计算向量化**：在寄存器中并行处理多个元素
- **存储向量化**：一次存储多个结果

### 2. 共享内存（Shared Memory）

归约操作使用共享内存：
- **快速通信**：Block 内线程之间共享数据
- **减少全局内存访问**：先在共享内存中归约，再写回全局内存

### 3. 寄存器优化

- **Launch Bounds**：告诉编译器预期的线程配置，优化寄存器使用
- **减少寄存器压力**：避免在 kernel 中使用过多局部变量

### 4. 内存合并访问（Coalesced Access）

- **连续访问**：尽量让线程访问连续的内存地址
- **对齐访问**：对齐到内存边界以提高带宽

---

## 📝 如何实现自定义 CUDA 算子

### 步骤 1：定义 Functor

```cpp
template<typename scalar_t>
struct MyCustomFunctor {
  __device__ __forceinline__ scalar_t operator() (const scalar_t a) const {
    // 你的计算逻辑
    return a * 2.0f + 1.0f;
  }
};
```

### 步骤 2：实现入口函数

```cpp
void my_custom_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "my_custom_cuda", [&]() {
    gpu_kernel(iter, MyCustomFunctor<scalar_t>());
  });
}
```

### 步骤 3：注册到分发系统

```cpp
REGISTER_DISPATCH(my_custom_stub, &my_custom_kernel_cuda)
```

### 步骤 4：在 native_functions.yaml 中定义

```yaml
- func: my_custom(Tensor self) -> Tensor
  dispatch:
    CUDA: my_custom_stub
```

---

## 📊 算子复杂度对比

| 算子类型 | 复杂度 | 主要挑战 | 示例 |
|---------|--------|----------|------|
| **Fill** | ⭐ | 无 | Fill |
| **一元元素级** | ⭐⭐ | 类型分发 | Abs, Sin, Log |
| **二元元素级** | ⭐⭐⭐ | 广播、标量处理 | Add, Mul, Div |
| **归约** | ⭐⭐⭐⭐ | 同步、共享内存 | Sum, Max, Mean |
| **索引** | ⭐⭐⭐⭐ | 不规则访问 | Index, Gather |
| **卷积/矩阵乘法** | ⭐⭐⭐⭐⭐ | 复杂的内存访问模式 | Conv2d, Matmul |

---

## 🔍 总结

PyTorch 的 CUDA 算子实现遵循以下设计原则：

1. **分层抽象**：
   - 上层：Python API 和类型分发
   - 中层：TensorIterator 和通用启动函数
   - 底层：实际的 CUDA kernel

2. **代码复用**：
   - 使用模板和 Functor 减少重复代码
   - `gpu_kernel` 和 `gpu_reduce_kernel` 处理通用逻辑

3. **性能优化**：
   - 向量化内存访问
   - 共享内存用于归约
   - JIT 编译减少二进制大小

4. **灵活性**：
   - 支持多种数据类型
   - 自动处理广播和内存布局
   - 支持标量参数

理解这些算子的实现方式，有助于：
- **调试性能问题**：了解底层实现，找到瓶颈
- **实现自定义算子**：遵循相同的模式
- **优化模型性能**：理解不同操作的代价

---

## 📚 参考资料

- **PyTorch 源码位置**：`aten/src/ATen/native/cuda/`
- **关键头文件**：
  - `Loops.cuh`：元素级操作的通用框架
  - `Reduce.cuh`：归约操作的实现
  - `CUDALoops.cuh`：向量化的元素级操作
- **文档**：
  - `torch/csrc/README.md`：C++ 代码说明
  - `aten/src/ATen/native/README.md`：操作符实现指南

