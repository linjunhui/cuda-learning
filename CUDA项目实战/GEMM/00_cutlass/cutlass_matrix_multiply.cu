#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// CUTLASS 头文件
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// PyTorch 绑定的矩阵乘法函数
// 实现 C = A * B，其中 A 是 M x N，B 是 N x K，C 是 M x K
void cutlass_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // 必须是 float32
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");

    // 必须是二维矩阵
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor (M x N)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor (N x K)");
    TORCH_CHECK(C.dim() == 2, "C must be 2D tensor (M x K)");

    // 检查矩阵维度判断是否能进行矩阵乘法
    int M = A.size(0);
    int N = A.size(1);
    int N_B = B.size(0);
    int K = B.size(1);

    // A的列数 要求等于  B的行数
    TORCH_CHECK(N == N_B, "Dimension mismatch: A's columns (", N, ") != B's rows (", N_B, ")");
    // A的行数 要求等于 C的行数， B的列数要求等于C的列数
    TORCH_CHECK(M == C.size(0) && K == C.size(1), "C's shape must be (", M, ",", K, "), but got (", C.size(0), ", ", C.size(1), ")");

    // 强制设备必须得在CUDA上
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "A, B, C must be on CUDA device (CPU is not supported) !");

    // 检查设备一致性， 必须 CPU/GPU必须统一， 这里其实要求都在GPU
    TORCH_CHECK(A.device() == B.device() && B.device() == C.device(), "A, B, C must be on the same device (CPU/GPU)");

    const at::Device device = A.device();
    at::DeviceGuard guard(device);

    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // 获取 CUDA stream
    // 使用 nullptr 让 CUTLASS 使用默认的 CUDA stream
    // 或者可以使用 c10::cuda::getCurrentCUDAStream() 如果可用
    cudaStream_t stream = nullptr;

    // 定义 CUTLASS GEMM 类型
    // 使用 float32 数据类型，行主序布局
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    
    // 使用 CUTLASS 的 GEMM 操作
    // 定义 GEMM 操作：C = alpha * A * B + beta * C
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    
    // 定义 CUTLASS GEMM 操作
    // 使用 CUTLASS 4.3.1 的 Gemm API
    using GemmOperation = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,  // SIMT operator class
        cutlass::arch::Sm75          // 默认架构，可根据实际 GPU 调整
    >;

    // 计算 leading dimensions
    int lda = N;  // A 是 M x N，行主序
    int ldb = K;  // B 是 N x K，行主序
    int ldc = K;  // C 是 M x K，行主序

    // GEMM 参数
    // CUTLASS 4.3.1 的 Arguments 结构
    typename GemmOperation::Arguments arguments{
        {M, N, K},                    // problem size (GemmCoord)
        {A_ptr, lda},                 // A 矩阵引用 (TensorRef)
        {B_ptr, ldb},                 // B 矩阵引用 (TensorRef)
        {C_ptr, ldc},                 // C 矩阵引用 (TensorRef) - 输入
        {C_ptr, ldc},                 // D 矩阵引用 (TensorRef) - 输出
        {1.0f, 0.0f}                  // Epilogue 参数 (alpha, beta)
    };

    // 创建 GEMM 操作实例
    GemmOperation gemm_op;

    // 执行 GEMM 操作（会自动初始化和运行）
    cutlass::Status status = gemm_op(arguments, nullptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM execution failed");
    
    // 同步等待完成
    // cudaStreamSynchronize(nullptr) 会同步默认 stream
    cudaStreamSynchronize(stream);
}

// PyTorch 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_gemm", &cutlass_gemm, "CUTLASS GEMM (C = A * B)");
}
