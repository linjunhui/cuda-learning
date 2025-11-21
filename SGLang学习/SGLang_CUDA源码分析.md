# SGLang CUDA 源码分析

## 📖 为什么 SGLang 更适合学习 CUDA？

相比 PyTorch，SGLang 的 CUDA 代码有以下优势：

### ✅ **更直接**：代码更接近 CUDA 内核的本质
- **PyTorch**：多层封装（TensorIterator、Dispatch、Functor 等），抽象层次高
- **SGLang**：直接写 CUDA kernel，逻辑清晰，容易理解

### ✅ **更实用**：专注于 LLM 推理的实际需求
- Attention 解码
- RoPE 位置编码
- 激活函数（SiLU、GELU）
- TopK 采样
- 这些都是 LLM 推理中的核心操作

### ✅ **更容易理解**：代码结构简单
- 一个文件一个功能
- Kernel 实现直观
- 共享内存使用清晰

---

## 📁 SGLang CUDA 代码结构

### 核心目录

```
sgl-kernel/csrc/
├── attention/          # 注意力机制
│   ├── lightning_attention_decode_kernel.cu  # 解码阶段注意力
│   ├── cutlass_mla_kernel.cu                # CUTLASS 优化的 MLA
│   └── merge_attn_states.cu                 # 合并注意力状态
├── elementwise/        # 逐元素操作
│   ├── activation.cu   # 激活函数（SiLU、GELU）
│   ├── rope.cu         # RoPE 位置编码
│   ├── copy.cu         # 内存拷贝
│   └── topk.cu         # TopK 采样
├── gemm/              # 矩阵乘法（GEMM）
│   └── [各种量化 GEMM]
├── moe/               # 混合专家（MoE）
└── quantization/      # 量化相关
```

---

## 1️⃣ 最简单的例子：激活函数（SiLU）

### 1.1 算子说明

**SiLU (Swish)**：`silu(x) = x / (1 + exp(-x))`

这是 LLM 中最常用的激活函数之一（如 Llama 使用 SiLU）。

### 1.2 源码实现

```56:60:SGLang学习/sglang/sgl-kernel/csrc/elementwise/activation.cu
template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}
```

**代码解析**：
- **直接明了**：就是数学公式的代码实现
- **类型转换**：先转 float32 计算，再转回原类型（避免精度问题）
- `__device__ __forceinline__`：设备端函数，强制内联

### 1.3 主机端调用

```85:104:SGLang学习/sglang/sgl-kernel/csrc/elementwise/activation.cu
void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}
```

**关键点**：
- **直接调用**：`<<<grid, block>>>` 启动 kernel
- **向量化**：`vec_size = 16 / sizeof(c_type)` 自动计算向量大小
- **Grid 配置**：每个 token 一个 block（`grid(num_tokens)`）
- **Block 配置**：根据维度大小动态调整

---

## 2️⃣ 核心例子：Lightning Attention Decode（解码阶段注意力）

### 2.1 算子说明

这是 LLM **解码阶段**的注意力计算，比预填充阶段简单得多：
- 输入：当前 step 的 q, k, v（形状 `[batch, heads, 1, dim]`）
- 过去：past_kv（形状 `[batch, heads, qk_dim, v_dim]`）
- 输出：注意力结果和更新的 kv cache

**关键特点**：
- **增量计算**：只计算当前 token 的注意力
- **使用共享内存**：q, k, v 载入共享内存复用
- **KV Cache 更新**：融合更新操作

### 2.2 核心 Kernel 代码

```25:113:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
template <typename T>
__global__ void lightning_attention_decode_kernel(
    const T* __restrict__ q,            // [b, h, 1, d]
    const T* __restrict__ k,            // [b, h, 1, d]
    const T* __restrict__ v,            // [b, h, 1, e]
    const float* __restrict__ past_kv,  // [b, h, d, e]
    const float* __restrict__ slope,    // [h, 1, 1]
    T* __restrict__ output,             // [b, h, 1, e]
    float* __restrict__ new_kv,         // [b, h, d, e]
    const int batch_size,
    const int num_heads,
    const int qk_dim,
    const int v_dim) {
  extern __shared__ char smem[];
  T* __restrict__ q_shared = reinterpret_cast<T*>(smem);
  T* __restrict__ k_shared = reinterpret_cast<T*>(smem + qk_dim * sizeof(T));
  T* __restrict__ v_shared = reinterpret_cast<T*>(smem + 2 * qk_dim * sizeof(T));
  float* __restrict__ new_kv_shared = reinterpret_cast<float*>(smem + (2 * qk_dim + v_dim) * sizeof(T));
  T* __restrict__ output_shared =
      reinterpret_cast<T*>(smem + (2 * qk_dim + v_dim) * sizeof(T) + qk_dim * (v_dim + 1) * sizeof(float));

  const int32_t tid = threadIdx.x;
  const int32_t current_head = blockIdx.x;
  const int32_t b = current_head / num_heads;
  const int32_t h = current_head % num_heads;

  if (b >= batch_size) return;

  const int32_t qk_offset = b * num_heads * qk_dim + h * qk_dim;
  const int32_t v_offset = b * num_heads * v_dim + h * v_dim;
  const int32_t kv_offset = b * num_heads * qk_dim * v_dim + h * qk_dim * v_dim;

  // Load q, k, v into shared memory
  for (int d = tid; d < qk_dim; d += blockDim.x) {
    q_shared[d] = q[qk_offset + d];
    k_shared[d] = k[qk_offset + d];
  }
  for (int e = tid; e < v_dim; e += blockDim.x) {
    v_shared[e] = v[v_offset + e];
  }

  __syncthreads();

  const float ratio = expf(-1.0f * slope[h]);

  // Compute new_kv
  for (int d = tid; d < qk_dim; d += blockDim.x) {
    const T k_val = k_shared[d];
    for (int e = 0; e < v_dim; ++e) {
      const int past_kv_idx = kv_offset + d * v_dim + e;
      const T v_val = v_shared[e];
      const float new_val = ratio * past_kv[past_kv_idx] + k_val * v_val;
      const int shared_idx = d * (v_dim + 1) + e;
      new_kv_shared[shared_idx] = new_val;
    }
  }

  __syncthreads();

  // Store new_kv to global memory
  for (int idx = tid; idx < qk_dim * v_dim; idx += blockDim.x) {
    const int d = idx / v_dim;
    const int e = idx % v_dim;
    const int shared_idx = d * (v_dim + 1) + e;
    const int global_idx = kv_offset + idx;
    new_kv[global_idx] = new_kv_shared[shared_idx];
  }

  __syncthreads();

  // Compute output
  for (int e = tid; e < v_dim; e += blockDim.x) {
    float sum = 0.0f;
    for (int d = 0; d < qk_dim; ++d) {
      const int shared_idx = d * (v_dim + 1) + e;
      sum += q_shared[d] * new_kv_shared[shared_idx];
    }
    output_shared[e] = static_cast<T>(sum);
  }

  __syncthreads();

  // Store output to global memory
  if (tid == 0) {
    for (int e = 0; e < v_dim; ++e) {
      output[v_offset + e] = output_shared[e];
    }
  }
}
```

### 2.3 代码流程详解

#### 第一步：分配共享内存

```cpp
extern __shared__ char smem[];
T* q_shared = reinterpret_cast<T*>(smem);
T* k_shared = reinterpret_cast<T*>(smem + qk_dim * sizeof(T));
T* v_shared = reinterpret_cast<T*>(smem + 2 * qk_dim * sizeof(T));
float* new_kv_shared = reinterpret_cast<float*>(smem + (2 * qk_dim + v_dim) * sizeof(T));
```

**关键点**：
- **动态共享内存**：`extern __shared__ char smem[]`
- **手动布局**：在共享内存中手动分配各个数组的位置
- **类型转换**：使用 `reinterpret_cast` 在不同类型间切换

#### 第二步：计算线程索引

```cpp
const int32_t tid = threadIdx.x;              // 线程在 block 内的索引
const int32_t current_head = blockIdx.x;      // 当前 block（每个 head 一个 block）
const int32_t b = current_head / num_heads;   // batch 索引
const int32_t h = current_head % num_heads;   // head 索引
```

**设计模式**：
- **每个 head 一个 block**：简化同步，每个 head 独立处理
- **Grid 配置**：`grid(batch_size * num_heads)`

#### 第三步：加载数据到共享内存

```cpp
for (int d = tid; d < qk_dim; d += blockDim.x) {
  q_shared[d] = q[qk_offset + d];
  k_shared[d] = k[qk_offset + d];
}
```

**关键点**：
- **协作加载**：多个线程协作加载一个数组
- **合并访问**：如果连续访问，会产生合并内存访问
- **同步**：加载完成后 `__syncthreads()`

#### 第四步：更新 KV Cache

```cpp
const float ratio = expf(-1.0f * slope[h]);  // 衰减因子

for (int d = tid; d < qk_dim; d += blockDim.x) {
  const T k_val = k_shared[d];
  for (int e = 0; e < v_dim; ++e) {
    const float new_val = ratio * past_kv[past_kv_idx] + k_val * v_val;
    new_kv_shared[shared_idx] = new_val;
  }
}
```

**计算公式**：
```
new_kv = ratio * old_kv + k * v
```
- **ratio**：衰减因子（基于 slope），实现滑动窗口注意力
- **增量更新**：直接更新 KV cache，不需要重新计算

#### 第五步：计算注意力输出

```cpp
for (int e = tid; e < v_dim; e += blockDim.x) {
  float sum = 0.0f;
  for (int d = 0; d < qk_dim; ++d) {
    sum += q_shared[d] * new_kv_shared[shared_idx];  // q * kv
  }
  output_shared[e] = static_cast<T>(sum);
}
```

**关键点**：
- **矩阵向量乘法**：`q * kv` 得到输出
- **寄存器累加**：使用 `float sum` 在寄存器中累加
- **共享内存复用**：`q_shared` 和 `new_kv_shared` 都在共享内存中

### 2.4 主机端调用

```115:154:SGLang学习/sglang/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu
void lightning_attention_decode(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& past_kv,
    const torch::Tensor& slope,
    torch::Tensor output,
    torch::Tensor new_kv) {
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(past_kv.is_contiguous(), "past_kv must be contiguous");

  auto batch_size = q.size(0);
  auto num_heads = q.size(1);
  auto qk_dim = q.size(3);
  auto v_dim = v.size(3);

  dim3 block(THREADS_PER_BLOCK);  // 128 个线程
  dim3 grid(batch_size * num_heads);  // 每个 head 一个 block

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "lightning_attention_decode_kernel", ([&] {
        size_t smem_size = (2 * qk_dim + 2 * v_dim) * sizeof(scalar_t) + qk_dim * (v_dim + 1) * sizeof(float);
        lightning_attention_decode_kernel<scalar_t><<<grid, block, smem_size, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            past_kv.data_ptr<float>(),
            slope.data_ptr<float>(),
            output.data_ptr<scalar_t>(),
            new_kv.data_ptr<float>(),
            batch_size,
            num_heads,
            qk_dim,
            v_dim);
      }));
}
```

**关键配置**：
- **Block 大小**：`THREADS_PER_BLOCK = 128`
- **Grid 大小**：`batch_size * num_heads`（每个 head 一个 block）
- **共享内存大小**：动态计算，包含所有中间数组

---

## 3️⃣ 实用技巧：TopK 采样

TopK 是 LLM 推理中的关键操作，SGLang 的实现非常高效。

### 3.1 核心代码片段

```76:142:SGLang学习/sglang/sgl-kernel/csrc/elementwise/topk.cu
__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();
```

**关键算法**：**基数排序（Radix Sort）** 的思想
1. **直方图**：将 float 转换为 uint8，构建直方图
2. **累积和**：计算每个 bin 的累积数量
3. **阈值查找**：找到包含 TopK 的 bin
4. **精细排序**：只在阈值 bin 内做完整排序

---

## 📊 SGLang vs PyTorch：代码对比

### 相同功能的实现对比

#### 例子 1：激活函数

**PyTorch 方式**：
```cpp
// 多层封装
FillFunctor -> gpu_kernel -> TensorIterator -> CUDALoops -> 实际 kernel
```

**SGLang 方式**：
```cpp
// 直接定义 kernel 函数
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}

// 直接启动
silu_kernel<<<grid, block>>>(...);
```

**优势**：
- ✅ **直观**：一看就懂
- ✅ **简单**：没有抽象层
- ✅ **可控**：完全控制 kernel 的行为

### 代码复杂度对比

| 方面 | PyTorch | SGLang |
|------|---------|--------|
| **抽象层次** | 5+ 层 | 1-2 层 |
| **代码行数** | ~200 行（包含所有封装） | ~50 行 |
| **理解难度** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **性能优化** | 自动化，但难以控制 | 手动优化，完全可控 |
| **学习价值** | 理解系统设计 | 理解 CUDA 本质 |

---

## 🎯 适合学习的 SGLang Kernel

### 1. **激活函数**（`elementwise/activation.cu`）
- **难度**：⭐⭐
- **学习点**：
  - 简单的设备端函数
  - 类型转换技巧
  - 向量化调用

### 2. **Lightning Attention Decode**（`attention/lightning_attention_decode_kernel.cu`）
- **难度**：⭐⭐⭐⭐
- **学习点**：
  - 共享内存使用
  - 线程协作模式
  - KV Cache 更新
  - 矩阵向量乘法优化

### 3. **TopK 采样**（`elementwise/topk.cu`）
- **难度**：⭐⭐⭐⭐⭐
- **学习点**：
  - 基数排序
  - 共享内存优化
  - 原子操作
  - 复杂算法在 GPU 上的实现

### 4. **Copy**（`elementwise/copy.cu`）
- **难度**：⭐
- **学习点**：
  - 最简单的 kernel
  - 模板参数使用
  - Grid-Stride Loop

### 5. **RoPE**（`elementwise/rope.cu`）
- **难度**：⭐⭐⭐
- **学习点**：
  - 旋转位置编码
  - 复数运算
  - 内存访问模式优化

---

## 🔍 关键学习点总结

### 1. **共享内存的使用模式**

```cpp
// 动态分配共享内存
extern __shared__ char smem[];

// 手动布局
T* q_shared = reinterpret_cast<T*>(smem);
T* k_shared = reinterpret_cast<T*>(smem + offset1);
float* kv_shared = reinterpret_cast<float*>(smem + offset2);
```

**优势**：
- 完全控制内存布局
- 最大化共享内存利用率
- 避免重复内存访问

### 2. **线程索引计算**

```cpp
const int32_t tid = threadIdx.x;           // Block 内索引
const int32_t bid = blockIdx.x;            // Block 索引
const int32_t global_id = bid * blockDim.x + tid;  // 全局索引
```

**模式**：
- **每个 head 一个 block**：简化同步
- **Grid-Stride Loop**：处理任意大小数组
- **多维索引**：`b = bid / num_heads`, `h = bid % num_heads`

### 3. **向量化技巧**

```cpp
uint32_t vec_size = 16 / sizeof(c_type);  // 自动计算向量大小
dim3 block(std::min(d / vec_size, 1024U));  // 根据向量大小调整 block
```

**关键**：
- 一次加载/存储多个元素
- 提高内存带宽利用率
- 减少指令数

### 4. **融合操作（Fused Operations）**

SGLang 大量使用融合操作，如：
- **SiLU and Mul**：`out = silu(x[:d]) * x[d:]`
- **Attention + KV Update**：同时计算注意力和更新 cache

**优势**：
- 减少内存访问
- 提高缓存利用率
- 降低 kernel 启动开销

---

## 📚 学习路径建议

### 阶段 1：基础操作（⭐⭐）

1. **Copy Kernel**：理解最基本的 CUDA kernel
2. **激活函数**：学习设备端函数和类型转换
3. **简单的逐元素操作**：理解 Grid-Stride Loop

### 阶段 2：中级操作（⭐⭐⭐）

1. **RoPE**：学习复杂的数学运算和内存访问
2. **Lightning Attention Decode**：学习共享内存和线程协作
3. **TopK（简化版）**：学习排序算法

### 阶段 3：高级操作（⭐⭐⭐⭐⭐）

1. **完整的 TopK**：学习基数排序和复杂算法
2. **GEMM 优化**：学习矩阵乘法的优化技巧
3. **MoE 相关**：学习混合专家的实现

---

## 💡 实践建议

### 1. 从简单开始

先理解简单的 kernel（如 copy、activation），再学习复杂的（如 attention）。

### 2. 画图理解

对于复杂的 kernel，画出：
- 线程分配图
- 共享内存布局图
- 数据流图

### 3. 运行调试

使用 `nsight-compute` 或 `cuda-gdb` 调试：
```bash
ncu --set full ./your_program
cuda-gdb ./your_program
```

### 4. 对比学习

对比 SGLang 和 PyTorch 的实现：
- 理解为什么 SGLang 更简单
- 学习两者的优化思路
- 找到适合自己的编程风格

---

## 🔗 相关资源

- **SGLang 官方文档**：https://github.com/sgl-project/sglang
- **FlashInfer**：SGLang 使用的注意力库
- **CUTLASS**：NVIDIA 的矩阵乘法库

---

## 📝 总结

SGLang 的 CUDA 代码相比 PyTorch：

✅ **更直接**：没有多层抽象  
✅ **更易学**：代码结构清晰  
✅ **更实用**：专注 LLM 推理的核心操作  
✅ **更可控**：完全控制优化细节  

**推荐学习顺序**：
1. Copy → Activation → RoPE
2. Lightning Attention Decode
3. TopK → GEMM → MoE

这样的学习路径能让你：
- 快速掌握 CUDA 编程的核心概念
- 理解 GPU 优化的实际技巧
- 学会如何编写高性能的 kernel

