// ========== 头文件包含 ==========
#include <algorithm>        // C++算法库，用于std::max等操作
#include <cuda_bf16.h>      // CUDA bfloat16（BF16）数据类型支持，用于混合精度训练
#include <cuda_fp16.h>      // CUDA half precision（FP16）数据类型支持，16位浮点数
#include <cuda_fp8.h>       // CUDA 8位浮点数支持（较新的数据类型）
#include <cuda_runtime.h>   // CUDA运行时API，提供cudaMalloc、cudaMemcpy等函数
#include <float.h>          // 浮点数常量定义，如FLT_MAX、FLT_MIN等
#include <stdio.h>          // 标准输入输出，用于printf等
#include <stdlib.h>         // 标准库函数，如malloc、free等
#include <torch/extension.h> // PyTorch C++扩展接口，用于与Python交互
#include <torch/types.h>     // PyTorch类型定义，如torch::Tensor
#include <vector>           // C++向量容器

// ========== 常量定义 ==========
#define WARP_SIZE 32  // Warp大小：CUDA中一个warp包含32个线程，这是GPU执行的基本单位
                      // 知识点：warp是SIMT（单指令多线程）执行的最小单元，同一warp内的线程执行相同指令

// ========== 向量化类型转换宏定义 ==========
// 这些宏用于将标量指针重新解释为向量类型，实现向量化内存访问
// 优化点：向量化访问可以减少内存事务次数，提高带宽利用率

// INT4: 将int指针重新解释为int4类型（128位，4个int）
// 知识点：int4是CUDA内置的向量类型，包含4个int32，总共128位
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])

// FLOAT4: 将float指针重新解释为float4类型（128位，4个float）
// 优化点：一次访问4个float（16字节），符合128位对齐要求，可以实现合并访问
// 知识点：float4包含x、y、z、w四个float成员，常用于向量化操作
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// HALF2: 将half指针重新解释为half2类型（32位，2个half）
// 优化点：half2是CUDA原生支持的向量类型，可以使用__hadd2等向量化指令
// 知识点：half2包含x、y两个half成员，GPU有专门的half2运算单元
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

// BFLOAT2: 将bfloat16指针重新解释为__nv_bfloat162类型（32位，2个bfloat16）
// 知识点：bfloat16是另一种16位浮点格式，常用于深度学习训练
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

// LDST128BITS: 用于128位（16字节）的向量化加载和存储
// 优化点：一次加载/存储128位数据，充分利用内存带宽
// 知识点：现代GPU的内存事务通常是128位对齐的，使用float4可以匹配这个宽度
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// ========== FP32（单精度浮点数）版本 ==========

// 版本1：标量版本（最基础的实现）
// 功能：逐元素相加 c[i] = a[i] + b[i]
// 线程配置：grid(N/256), block(256)
// 说明：每个线程处理一个元素，这是最直观但性能较低的实现方式
// LeetGPU 题目 1 
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  // 计算当前线程对应的全局索引
  // 知识点：这是CUDA中最常用的线程索引计算方式
  // blockIdx.x: 当前线程块在网格中的x方向索引
  // blockDim.x: 每个线程块在x方向的线程数（这里是256）
  // threadIdx.x: 当前线程在线程块中的x方向索引
  // 公式：全局索引 = 线程块索引 × 每块线程数 + 线程索引
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 边界检查：防止数组越界
  // 优化点：使用if判断而不是循环，避免分支发散（同一warp内所有线程通常都满足或不满足条件）
  if (idx < N)
    // 逐元素相加：每个线程处理一个元素
    // 性能分析：这是标量操作，每个线程需要3次内存访问（读a、读b、写c）
    // 如果内存访问是合并的，性能尚可，但不如向量化版本
    c[idx] = a[idx] + b[idx];
}

// 版本2：向量化版本（使用float4）
// 功能：逐元素相加，但每个线程处理4个元素
// 线程配置：grid(N/256), block(256/4) = block(64)
// 优化点：向量化访问可以减少内存事务次数，提高带宽利用率
// 性能提升：理论上可以减少约75%的内存事务次数（4个元素合并为1次事务）
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N) {
  // 计算起始索引：每个线程处理4个连续元素
  // 注意：blockDim.x现在是64（256/4），所以每个线程块处理256个元素
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  
  if (idx < N) {
    // 向量化加载：使用FLOAT4宏一次性加载4个float（16字节，128位）
    // 优化点：一次内存事务加载16字节，比4次独立的4字节加载效率高得多
    // 知识点：float4是CUDA内置的向量类型，编译器会生成向量化指令
    // 内存对齐：要求a[idx]地址是16字节对齐的，否则可能触发非对齐访问
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    
    // 向量化计算：对4个元素分别进行加法运算
    // 优化点：现代GPU的ALU（算术逻辑单元）可能支持向量化指令
    // 编译器优化：这里很可能被融合为一条或几条向量化指令序列
    // 性能：执行时间远低于4条独立的标量加法的时间总和
    // 知识点：虽然看起来是4条独立语句，但编译器可能优化为SIMD指令
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    
    // 向量化存储：一次性写入4个float（16字节）
    // 优化点：减少存储事务次数，提高写入带宽利用率
    FLOAT4(c[idx]) = reg_c;
  }
}

// ========== FP16（半精度浮点数）版本 ==========

// 版本1：FP16标量版本
// 功能：逐元素相加，使用half精度（16位浮点数）
// 线程配置：grid(N/256), block(256)
// 优化点：FP16数据量是FP32的一半，可以节省内存带宽和存储空间
// 应用场景：深度学习推理中常用FP16，在保持精度的同时提高性能
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  // 计算全局索引（与FP32版本相同）
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < N)
    // 使用CUDA内置的half加法函数__hadd
    // 知识点：half类型不能直接使用+运算符，必须使用CUDA提供的half运算函数
    // __hadd: half add，专门用于half类型的加法，保证精度和性能
    // 性能：FP16运算通常比FP32快，但精度较低
    c[idx] = __hadd(a[idx], b[idx]);
}

// 版本2：FP16向量化版本（使用half2）
// 功能：每个线程处理2个half元素
// 线程配置：grid(N/256), block(256/2) = block(128)
// 优化点：使用half2向量类型，可以利用GPU的half2运算单元
// 知识点：half2是CUDA原生支持的向量类型，GPU有专门的half2 ALU
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  // 计算起始索引：每个线程处理2个连续元素
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  
  if (idx < N) {
    // 向量化加载：使用HALF2宏一次性加载2个half（4字节，32位）
    // 优化点：一次内存事务加载4字节，比2次独立的2字节加载效率高
    // 知识点：half2包含x、y两个half成员，总共32位
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;
    
    // 标量方式计算：分别对x和y进行加法
    // 注意：这里没有使用__hadd2（half2向量加法），而是分别计算
    // 优化空间：可以使用__hadd2进一步优化（见f16x8_pack版本）
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    
    // 向量化存储：一次性写入2个half
    HALF2(c[idx]) = reg_c;
  }
}

// 版本3：FP16x8版本（每个线程处理8个half元素）
// 功能：每个线程处理8个half元素（4个half2向量）
// 线程配置：grid(N/256), block(256/8) = block(32)
// 优化点：提高线程利用率，减少线程块数量，降低调度开销
// 性能分析：每个线程处理更多数据，可以更好地隐藏内存延迟
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  // 计算起始索引：每个线程处理8个连续元素
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  
  // 加载4个half2向量（共8个half元素）
  // 优化点：使用多个寄存器存储数据，减少内存访问次数
  // 知识点：寄存器是GPU上最快的存储，但数量有限（每个线程约255个32位寄存器）
  half2 reg_a_0 = HALF2(a[idx + 0]);  // 加载元素0-1
  half2 reg_a_1 = HALF2(a[idx + 2]);  // 加载元素2-3
  half2 reg_a_2 = HALF2(a[idx + 4]);  // 加载元素4-5
  half2 reg_a_3 = HALF2(a[idx + 6]);  // 加载元素6-7
  half2 reg_b_0 = HALF2(b[idx + 0]);
  half2 reg_b_1 = HALF2(b[idx + 2]);
  half2 reg_b_2 = HALF2(b[idx + 4]);
  half2 reg_b_3 = HALF2(b[idx + 6]);
  
  // 声明结果寄存器
  half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
  
  // 分别计算4个half2向量的加法
  // 注意：这里仍然使用标量方式（__hadd），而不是向量化方式（__hadd2）
  // 优化空间：可以使用__hadd2进一步优化（见f16x8_pack版本）
  reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
  reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
  reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
  reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
  reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
  reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
  reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
  reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
  
  // 边界检查：分别检查每个half2向量的写入是否越界
  // 优化点：使用多个独立的if判断，避免分支发散
  // 知识点：如果同一warp内的线程都满足或不满足条件，分支开销很小
  if ((idx + 0) < N) {
    HALF2(c[idx + 0]) = reg_c_0;
  }
  if ((idx + 2) < N) {
    HALF2(c[idx + 2]) = reg_c_1;
  }
  if ((idx + 4) < N) {
    HALF2(c[idx + 4]) = reg_c_2;
  }
  if ((idx + 6) < N) {
    HALF2(c[idx + 6]) = reg_c_3;
  }
}

// 版本4：FP16x8_pack版本（最优化的实现）
// 功能：每个线程处理8个half元素，使用128位向量化加载/存储和__hadd2向量化指令
// 线程配置：grid(N/256), block(256/8) = block(32)
// 优化点：结合了向量化内存访问和向量化计算，是性能最优的实现
__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c,
                                                  int N) {
  // 计算起始索引：每个线程处理8个连续元素
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  
  // 声明局部数组：存储在寄存器中（.local空间）
  // 知识点：这些数组存储在PTX的.local空间，实际上是寄存器数组
  // 优化点：使用局部数组可以方便地进行向量化操作
  // 大小：8个half × 16位 = 128位，正好是一个float4的大小
  half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits = 128 bits
  
  // ========== 向量化加载（128位） ==========
  // 使用LDST128BITS宏将8个half重新解释为float4，一次性加载128位（16字节）
  // 优化点：一次内存事务加载16字节，比8次独立的2字节加载效率高得多
  // 知识点：现代GPU的内存事务通常是128位对齐的，使用float4可以匹配这个宽度
  // 性能：理论上可以减少约87.5%的内存事务次数（8个元素合并为1次事务）
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits

  // ========== 向量化计算 ==========
  // #pragma unroll: 循环展开指令，告诉编译器展开循环
  // 优化点：循环展开可以减少循环控制开销，提高指令级并行度
  // 知识点：编译器会生成4次__hadd2指令，而不是循环结构
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    // 使用__hadd2进行half2向量加法（这是关键优化！）
    // 优化点：__hadd2是CUDA提供的half2向量加法指令，一次计算2个half
    // 性能：比2次__hadd快，因为GPU有专门的half2 ALU单元
    // 知识点：half2向量运算可以充分利用GPU的SIMD能力
    // 循环4次，共处理8个half元素（4个half2向量）
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
  }
  
  // ========== 向量化存储（128位） ==========
  // 如果8个元素都在边界内，使用向量化存储
  // 优化点：一次内存事务写入16字节，比8次独立的2字节写入效率高得多
  if ((idx + 7) < N) {
    // 使用LDST128BITS宏一次性写入128位（8个half）
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  } else {
    // 边界情况：如果部分元素越界，使用标量方式处理剩余元素
    // 优化点：避免写入越界内存，保证正确性
    // 性能：边界情况性能较差，但保证正确性更重要
    for (int i = 0; idx + i < N; i++) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
    }
  }
}

// ========== PyTorch绑定宏定义 ==========

// STRINGFY: 将宏参数转换为字符串字面量
// 知识点：使用#运算符可以将宏参数转换为字符串
// 示例：STRINGFY(func) -> "func"
#define STRINGFY(str) #str

// TORCH_BINDING_COMMON_EXTENSION: 通用的PyTorch扩展绑定宏
// 功能：将C++函数绑定到Python，使其可以在Python中调用
// 知识点：m.def是Pybind11的API，用于定义Python函数
// 参数：func - 要绑定的C++函数名
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// CHECK_TORCH_TENSOR_DTYPE: 检查PyTorch张量的数据类型
// 功能：验证张量的数据类型是否符合要求，不符合则抛出异常
// 优化点：在核函数启动前进行类型检查，避免运行时错误
// 知识点：PyTorch张量的类型信息存储在options().dtype()中
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// TORCH_BINDING_ELEM_ADD: 生成PyTorch绑定的元素相加函数
// 参数说明：
//   packed_type: 打包类型名称（如f32、f32x4、f16x2等），用于函数命名
//   th_type: PyTorch数据类型（如torch::kFloat32、torch::kHalf）
//   element_type: C++元素类型（如float、half）
//   n_elements: 每个线程处理的元素数量（1、2、4、8等）
// 功能：自动生成类型检查和核函数启动代码
// 注意：宏定义内部不能有注释在反斜杠后面，注释必须在反斜杠之前
#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

// ========== 宏展开：生成所有版本的PyTorch绑定函数 ==========
// 这些宏会展开为完整的函数定义，每个函数对应一个kernel实现

// FP32标量版本：每个线程处理1个float元素
TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)

// FP32向量化版本：每个线程处理4个float元素（使用float4）
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)

// FP16标量版本：每个线程处理1个half元素
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)

// FP16向量化版本（half2）：每个线程处理2个half元素
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)

// FP16向量化版本（x8）：每个线程处理8个half元素（4个half2）
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)

// FP16最优版本（x8_pack）：每个线程处理8个half元素，使用128位向量化加载/存储和__hadd2
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)

// ========== Pybind11模块定义 ==========
// PYBIND11_MODULE: 定义Python扩展模块
// TORCH_EXTENSION_NAME: PyTorch自动定义的模块名（通常在setup.py中指定）
// 功能：将所有C++函数绑定到Python，使其可以在Python中调用
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 绑定所有版本的elementwise_add函数到Python
  // 知识点：m.def用于定义Python函数，第一个参数是Python函数名，第二个是C++函数指针
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)        // Python: elementwise_add_f32()
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)      // Python: elementwise_add_f32x4()
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)        // Python: elementwise_add_f16()
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)      // Python: elementwise_add_f16x2()
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)     // Python: elementwise_add_f16x8()
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack) // Python: elementwise_add_f16x8_pack()
}
