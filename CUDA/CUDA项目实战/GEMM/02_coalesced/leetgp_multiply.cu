#include <cuda_runtime.h>
#include <stdio.h>

const int TILE_SIZE = 32;
// PADDED_SIZE 用于避免 Shared Memory Bank Conflict，
// 尤其是在列访问模式下。虽然这里我们修正为标准访问，保留它作为好习惯。
const int PADDED_SIZE = TILE_SIZE + 1; 

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 声明 Shared Memory
    __shared__ float S_A[TILE_SIZE][PADDED_SIZE]; // 这里也可以加 padding
    __shared__ float S_B[TILE_SIZE][PADDED_SIZE];

    // 1. 计算当前线程负责计算的 C 矩阵坐标 (Global Row m, Global Col k)
    // 对应关系: tid_y -> 行 (m), tid_x -> 列 (k)
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    int m = blockDim.y * blockIdx.y + tid_y; // 对应 A 的行，C 的行
    int k = blockDim.x * blockIdx.x + tid_x; // 对应 B 的列，C 的列

    // 局部寄存器用于累加结果
    float c_reg = 0.0f;

    // 2. 循环遍历所有的 Tile (在维度 N 上移动)
    // num_tiles 计算方式：向上取整
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < num_tiles; ++i) {
        
        // --- 加载 S_A ---
        // 我们需要 A 的第 m 行。
        // 当前 Tile 的列索引由 i * TILE_SIZE 开始，偏移量是 tid_x
        // 这里的 tid_x 负责把 A 的一行中的一部分搬运到 Shared Memory
        // 这种方式保证了 Global Memory 的合并访问 (Coalesced Access)
        int A_row = m;
        int A_col = i * TILE_SIZE + tid_x;

        // 只有在矩阵范围内才加载数据，否则补 0
        if (A_row < M && A_col < N) {
            // 注意：S_A 使用 [tid_y][tid_x] 布局
            S_A[tid_y][tid_x] = A[A_row * N + A_col];
        } else {
            S_A[tid_y][tid_x] = 0.0f;
        }

        // --- 加载 S_B ---
        // 我们需要 B 的第 k 列。
        // 当前 Tile 的行索引由 i * TILE_SIZE 开始，偏移量是 tid_y
        // 这里的 tid_x 依然对应 B 的列 (k)，保证合并访问 (tid_x 连续，地址连续)
        int B_row = i * TILE_SIZE + tid_y;
        int B_col = k;

        if (B_row < N && B_col < K) {
            // 注意：S_B 使用 [tid_y][tid_x] 布局，保持和 Global Memory 一样的形态
            S_B[tid_y][tid_x] = B[B_row * K + B_col];
        } else {
            S_B[tid_y][tid_x] = 0.0f;
        }

        // --- 同步 ---
        // 必须等待 Block 内所有线程都加载完 S_A 和 S_B
        __syncthreads();

        // --- 计算 (核心点) ---
        // 现在 S_A 和 S_B 都在 Shared Memory 中
        // 我们的目标是计算 C(m, k)
        // C(m, k) += sum( A(m, t) * B(t, k) )
        // 
        // S_A 的行是 tid_y (对应 m)
        // S_B 的列是 tid_x (对应 k)
        // t 是中间维度，在当前 Tile 中从 0 到 31
        
        for (int t = 0; t < TILE_SIZE; ++t) {
            // S_A[tid_y][t]: 取 A 的当前行，第 t 个元素
            // S_B[t][tid_x]: 取 B 的当前列，第 t 个元素
            c_reg += S_A[tid_y][t] * S_B[t][tid_x];
        }

        // --- 同步 ---
        // 必须等待所有线程计算完，才能进入下一轮加载，避免覆写尚未使用的 Shared Memory
        __syncthreads();
    }

    // 3. 将结果写回 Global Memory
    // 只有在有效范围内的线程才写入
    if (m < M && k < K) {
        C[m * K + k] = c_reg;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    // 检查 Kernel 启动错误（建议在开发阶段加上）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}