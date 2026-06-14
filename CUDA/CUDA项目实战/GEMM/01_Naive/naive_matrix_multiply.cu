#include <torch/extension.h>
#include <cuda_runtime.h>

/*
编写一个最朴素、最简单的矩阵乘法核函数
parameters:
    A: 矩阵A
    B: 矩阵B
    C: 矩阵C
    M: 矩阵A的行数
    N: 矩阵A的列数
    K: 矩阵B的列数

A is M x N
B is N x K
C is M x Ks

C = A * B

C[m][k] = A[m][n] * B[n][k]
*/
__global__ void matrix_multi_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 从C的视角出发， 就是计算  M x K 个元素， M行， K列

    // 当前线程的全局 坐标 
    int global_tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    // 在当前block中的坐标 
    int local_tid_x = threadIdx.x;
    int local_tid_y = threadIdx.y;

    // 计算 C(m, k) 的值, A的第m行，B的k列, 计算元素的个数 N
    int m = global_tid_y;
    int k = global_tid_x;
    
    // 边界检查
    if (m >= M || k >= K) {
        return;
    }
    
    float c_reg = 0.0f;

    for(int i = 0; i < N; i++) {
        // A(m, i) * B(i, k), 这里要做一个二维到一维的转换
        c_reg += A[m * N + i] * B[i * K + k];
    }

    // 将结果写入到C中
    C[m * K + k] = c_reg;
}

// PyTorch 绑定的矩阵乘法函数
void naive_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // 检查张量类型和维度
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
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
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();
    
    // 配置线程块和网格
    int threads_per_block = 32;
    int blocks_per_grid_x = (K + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid_y = (M + threads_per_block - 1) / threads_per_block;
    dim3 blocks(blocks_per_grid_x, blocks_per_grid_y);
    dim3 threads(threads_per_block, threads_per_block);
    
    // 启动核函数
    matrix_multi_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// Pybind11 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_gemm", &naive_gemm, "朴素矩阵乘法 (C = A * B)");
}