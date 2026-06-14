#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

template<int TILE_SIZE>
__global__ void coalesced_gemm_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // 1. 声明共享内存（使用 Padding 规避 Bank Conflict）
    __shared__ float S_A[TILE_SIZE][TILE_SIZE];
    __shared__ float S_B[TILE_SIZE][TILE_SIZE + 1];

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    // 计算当前线程对应的 C 矩阵全局坐标
    int m = blockIdx.y * TILE_SIZE + tid_y;
    int n = blockIdx.x * TILE_SIZE + tid_x;

    float c_reg = 0.0f;
    int TILE_NUM = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < TILE_NUM; i++) {
        int k_base = i * TILE_SIZE;

        // 2. 加载 A 到共享内存
        // 线程(tx, ty) 负责搬运 A[m][k_base + tx]
        if (m < M && (k_base + tid_x) < K) {
            S_A[tid_y][tid_x] = A[m * K + k_base + tid_x];
        } else {
            S_A[tid_y][tid_x] = 0.0f;
        }

        // 3. 加载 B 到共享内存
        // 线程(tx, ty) 负责搬运 B[k_base + ty][n]
        int B_k = k_base + tid_y;
        if (B_k < K && n < N) {
            S_B[tid_y][tid_x] = B[B_k * N + n];
        } else {
            S_B[tid_y][tid_x] = 0.0f;
        }

        // 必须同步，等待当前 Tile 所有数据搬运完成
        __syncthreads();

        // 4. 计算累加
        #pragma unroll
        for (int t = 0; t < TILE_SIZE; t++) {
            c_reg += S_A[tid_y][t] * S_B[t][tid_x];
        }

        // 必须同步，确保计算完成后再搬运下一个 Tile 覆盖 S_A/S_B
        __syncthreads();
    }

    // 5. 写回结果
    if (m < M && n < N) {
        C[m * N + n] = c_reg;
    }
}

void gemm_launch(torch::Tensor input_a, torch::Tensor input_b, torch::Tensor output) {
    // 检查输入是否在 CUDA 上
    TORCH_CHECK(input_a.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(input_b.is_cuda(), "Input B must be a CUDA tensor");
    
    // 强制连续内存，防止由于 stride 导致的索引错误
    auto a = input_a.contiguous();
    auto b = input_b.contiguous();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int TILE_SIZE = 32;

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = a.data_ptr<float>();
    const float* B_ptr = b.data_ptr<float>();
    float* C_ptr = output.data_ptr<float>();

    // 启动 Kernel
    coalesced_gemm_kernel<TILE_SIZE><<<grid_size, block_size>>>(A_ptr, B_ptr, C_ptr, M, K, N);
    // 错误检查
    cudaError_t err = cudaGetLastError();
    fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_launch, "Coalesced GEMM with Shared Memory");
}