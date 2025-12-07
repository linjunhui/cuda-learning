#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>


/*
从C的视角出发， 就是计算  M x K 个元素， M行， K列
C = A * B，其中 A是M×N，B是N×K，C是M×K
*/
__global__ void matrix_multiplication_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // x 变化的快，就对应 列
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    // y变化慢 对应的是行
    int m = blockDim.y * blockIdx.y + threadIdx.y;


    if(k < K && m < M) {
        // C[m][k] = A的第m行 点乘 B的第k列
        // A是M×N，B是N×K，C是M×K
        // 做二维到一维的转换
        float c_reg = 0.0f;
        for(int n = 0; n < N; n++) {
            // A[m][n] * B[n][k]
            // A[m][n] = A[m * N + n] (A是M×N)
            // B[n][k] = B[n * K + k] (B是N×K)
            c_reg += A[m * N + n] * B[n * K + k];
        }
        // C[m][k] = C[m * K + k] (C是M×K)
        C[m * K + k] = c_reg;
    }
}

#include <cuda_runtime.h>

const int TILE_SIZE = 16;

/*
目标: 计算 C = A * B，其中 A(M x N) * B(N x K) = C(M x K)
优化策略: 使用共享内存分块，并且在加载 B 时进行转置存储 (S_B 存储 B^T)
*/
__global__ void matrix_multiplication_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 线程块的基址：确定当前块计算 C 矩阵中哪个 TILE
    int m_base = blockIdx.y * TILE_SIZE; // C 的行基址 (m)
    int k_base = blockIdx.x * TILE_SIZE; // C 的列基址 (k)

    // 线程的局部索引 (0 到 TILE_SIZE - 1)
    int tx = threadIdx.x; // TILE 列索引
    int ty = threadIdx.y; // TILE 行索引

    // C 矩阵的最终全局索引
    int m = m_base + ty; // C 的全局行索引
    int k = k_base + tx; // C 的全局列索引

    // N 维度上的分块数量 (内循环次数)
    int TILE_NUM = (N + TILE_SIZE - 1) / TILE_SIZE;

    // 共享内存分配 (16x16)
    __shared__ float S_A[TILE_SIZE][TILE_SIZE]; // 存储 A 的当前分块
    __shared__ float S_B[TILE_SIZE][TILE_SIZE]; // 存储 B 的当前分块的转置 (B^T)

    float C_value = 0.0f; // 累加器（使用 float 以最大化吞吐量）

    // 1. 外部循环：遍历 N 维度上的所有分块
    for(int i = 0; i < TILE_NUM; i++) {
        int n_base = i * TILE_SIZE; // A 的列基址 / B 的行基址

        // 线程(ty, tx) 合作加载 S_A[ty][tx] 和 S_B[tx][ty] (转置)

        // 1.1 加载 A 的分块到 S_A (行主序加载)
        // 线程 (ty, tx) 负责 S_A[ty][tx]
        // 对应的 A 元素在全局索引 (m, n_base + tx)
        if(m < M && n_base + tx < N) {
            S_A[ty][tx] = A[m * N + n_base + tx];
        } else {
            S_A[ty][tx] = 0.0f; // 边界填充零
        }

        // 1.2 加载 B 的分块到 S_B (转置加载 B -> S_B 存储 B^T)
        // 线程 (ty, tx) 负责 S_B[tx][ty]
        // 对应的 B 元素在全局索引 (n_base + ty, k)
        if(n_base + ty < N && k < K) {
            // S_B[tx][ty] = B[row][col]
            S_B[tx][ty] = B[(n_base + ty) * K + k];
        } else {
            S_B[tx][ty] = 0.0f; // 边界填充零
        }

        __syncthreads(); // 等待所有线程完成加载

        // 2. 内部计算：点积 S_A 的 ty 行 与 S_B 的 tx 行
        // 由于 S_B 存储的是 B^T，所以我们点乘 S_A 的行 (ty) 和 S_B 的行 (tx)
        for(int j = 0; j < TILE_SIZE; j++) {
            // S_A[ty][j] -> A 的行 m 的元素
            // S_B[tx][j] -> B 的列 k 的元素 (通过 B^T 的行 tx 访问)
            C_value += S_A[ty][j] * S_B[tx][j];
        }

        __syncthreads(); // 等待所有线程完成当前分块的计算
    }

    // 3. 写入最终结果到全局内存 C[m][k]
    if(m < M && k < K) {
        C[m * K + k] = C_value;
    }
}