#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

/*
使用 Tensor Core 的 WMMA (Warp-level Matrix Multiply-Accumulate) API 实现矩阵乘法

parameters:
    A: 矩阵A (M x N)
    B: 矩阵B (N x K)
    C: 矩阵C (M x K)
    M: 矩阵A的行数
    N: 矩阵A的列数
    K: 矩阵B的列数

C = A * B

WMMA API 支持的矩阵大小:
- 16x16x16 (half precision)
- 8x32x16 (half precision)
- 32x8x16 (half precision)

这里使用 16x16x16 的 tile 大小
*/

// WMMA tile 大小
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// 每个 warp 处理一个 16x16 的输出 tile
// block 大小: (32, 4) = 4 个 warp，每个 warp 32 个线程
// 每个 block 处理 4 行 x 1 列的 tile (64 x 16)

__global__ void mma_kernel_simple(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // 计算当前 warp 在 block 内的位置
    int warp_id = threadIdx.y;  // 0-3，表示当前 warp 在 block 内的行位置
    
    // 计算当前 warp 负责的输出 tile
    // 每个 block 处理 4 行 tile (4 * 16 = 64 行)
    int c_row = blockIdx.y * 64 + warp_id * WMMA_M;
    int c_col = blockIdx.x * WMMA_N;

    // 边界检查
    if (c_row >= M || c_col >= K) {
        return;
    }

    __half zero = __float2half(0.0f);

    // WMMA 片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, zero);

    // 执行矩阵乘法: C = A * B
    // A: M x N, B: N x K, C: M x K
    for (int k = 0; k < N; k += WMMA_K) {
        // 加载 A tile (row major)
        // A 的 tile: [c_row : c_row+16, k : k+16]
        if (c_row + WMMA_M <= M && k + WMMA_K <= N) {
            wmma::load_matrix_sync(a_frag, &A[c_row * N + k], N);
        } else {
            
            wmma::fill_fragment(a_frag, zero);
        }

        // 加载 B tile (col major)
        // B 的 tile: [k : k+16, c_col : c_col+16]
        if (k + WMMA_K <= N && c_col + WMMA_N <= K) {
            wmma::load_matrix_sync(b_frag, &B[k * K + c_col], K);
        } else {
            wmma::fill_fragment(b_frag, zero);
        }

        // 执行 MMA 操作: c_frag += a_frag * b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储结果
    if (c_row + WMMA_M <= M && c_col + WMMA_N <= K) {
        wmma::store_matrix_sync(&C[c_row * K + c_col], c_frag, K, wmma::mem_row_major);
    }
}

// PyTorch 绑定的矩阵乘法函数
void mma_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // 检查张量类型和维度
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16 (half)");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16 (half)");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be float16 (half)");
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
    TORCH_CHECK(C.dim() == 2, "C must be 2D tensor");
    
    int M = A.size(0);
    int N = A.size(1);
    int N_B = B.size(0);
    int K = B.size(1);
    
    TORCH_CHECK(N == N_B, "A的列数必须等于B的行数");
    TORCH_CHECK(C.size(0) == M && C.size(1) == K, "C的维度必须为 [M, K]");
    
    // 获取数据指针
    const half* d_A = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half* d_B = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    half* d_C = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    
    // 配置线程块和网格
    // block 大小: (32, 4) = 4 个 warp，每个 warp 32 个线程
    // 每个 block 处理 64 行 x 16 列
    dim3 threads(32, 4);  // 4 个 warp，每个 warp 32 个线程
    dim3 blocks(
        (K + WMMA_N - 1) / WMMA_N,  // 每个 block 处理 16 列
        (M + 64 - 1) / 64            // 每个 block 处理 64 行
    );
    
    // 启动核函数
    mma_kernel_simple<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(err));
}

// Pybind11 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_gemm", &mma_gemm, "使用 Tensor Core WMMA API 的矩阵乘法 (C = A * B)");
}
