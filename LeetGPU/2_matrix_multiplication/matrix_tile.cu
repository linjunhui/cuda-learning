#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 32;

/*
目标: 计算 C = A * B，其中 A(M x N) * B(N x K) = C(M x K)
优化策略: 使用共享内存分块，并且在加载 B 时进行转置存储 (S_B 存储 B^T)
*/
__global__ void matrix_multiplication_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    
    // 从C的元素的视角来计算
    // 全局索引
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    // 当前block的base地址 (用于计算全局索引)
    int m_base = blockDim.y * blockIdx.y;
    int k_base = blockDim.x * blockIdx.x;

    // 当前block中的索引
    int tid_x = threadIdx.x; // TILE 列索引
    int tid_y = threadIdx.y; // TILE 行索引

    // 将矩阵分成 TILE, C的一个元素是A的一行N个元素和B的一列N个元素计算, 需要注意的是TILE_SIZE == BLOCK_SIZE
    int TILE_NUM = (N + TILE_SIZE - 1) / TILE_SIZE; // 在N 个元素上分块的个数
    
    /*
        每个块要干什么?
        1. 首先 TILE_SIZE == BLOCK_SIZE 所以这里设计，一个线程处理 TILE上的一个元素
        1.1 举例：ty 行 88个元素(N = 88) 第0个TILE (tid_x = 0, 1, 2..31)
        1.2 所以 tid_y 固定，看作一维时，需要TILE_SIZE的数组来加载 A的一个TILE_SIZE个元素到TILE
        2. 所以循环 TILE_NUM次
        3. 实际一个BLOCK 是 TILE_SIZE x TILE_SIZE， 所以这里共享内存数组大小 TILE_SIZE x TILE_SIZE
    */

    __shared__ float S_A[TILE_SIZE][TILE_SIZE];
    __shared__ float S_B[TILE_SIZE][TILE_SIZE];

    // C[m][k] 的累加器
    float C_value = 0.0f;

    // 外部循环：遍历 N 维度上的所有分块
    for(int i = 0; i < TILE_NUM; i++) {
        // N 维度的基地址
        int n_tile_base = i * TILE_SIZE; 

        // --- 错误对比区域 (保留您的原始注释和思考痕迹) ---
        
        //C(m, k) 就是在B的k列加载B的一个TILE，按列加载, 行 基坐标 i*TILE_SIZE + tid_x ,  列坐标就是k
        /*
        if(m < M && n_tile_base + TILE_SIZE < N) {
            S_B[tid_y][tid_x] = B[(i*TILE_SIZE + tid_x ) * K + k]; // 这里就有一个问题 B是跳着访问的，Strip是K，全局内存访问不合并
        }
        */
        
        /*
            从一维考虑， 加载B的一列，要跳着操作，全局内存访问不合并
            现在期望对于B按行访问， B[]
        */
        
        // --- 正确的 I/O 阶段 ---

        // A 的全局列索引 n_col (用于加载 A[m][n_col])
        int n_col = n_tile_base + tid_x;  // n_tile_base 是当前TILE的其实地址

        // 1.1 加载 A 的分块到 S_A (行主序加载，天然合并)
        if(m < M && n_col < N) {
            // 加载A的 A[m][n_col] 就是A的m行，取TILE tid_x 表示一个线程，一个线程处理TILE中的一个元素
            S_A[tid_y][tid_x] = A[m * N + n_col]; // Global: A[m][n_col] -> S_A[ty][tx]
        } else {
            S_A[tid_y][tid_x] = 0.0f; // 边界填充零
        }

        /*
            从矩阵乘法来说，加载了A的m行，现在要加载B的k列
            加载列就是跨很多行，导致全局内存访问不合并。

            这里从二维全局考虑，实际是
        */

        // B 的全局行索引 r_row
        int r_row = n_tile_base + tid_y;
        // B 的全局列索引 k_col
        int k_col = k_base + tid_x;
        
        // 1.2 加载 B 的分块到 S_B (转置存储，按行合并访问)
        // 边界检查：确保 r_row < N 且 k_col < K
        /*
        B的形状是 N x K
        当前全局线程(m, k) 需要 A的m行, B的k列
        B(, k) = 
        */
        if(r_row < N && k_col < K) {
            // S_B[tx][ty] = B[r_row][k_col] 
            // 这种加载方式实现了对 B 的按行合并访问。
            S_B[tid_x][tid_y] = B[r_row * K + k_col];
        } else {
            S_B[tid_x][tid_y] = 0.0f; // 边界填充零
        }

        /*
            这里是二维BLOCK, 同步前是完成了一整个TILE的加载， TILE_SIZE * TILE_SIZE 的元素加载
            就是 A 和 B 一个TILE的行和列 都加载了
        */
        __syncthreads(); // 关键同步点：等待所有线程完成全局内存到共享内存的加载

        // --- 计算阶段 ---
        
        // 2. 内部计算：点积 S_A 的 tid_y 行 与 S_B 的 tid_x 行
        // S_B 的行 tid_x 存储的是 B 的列 k 的数据。
        for(int j = 0; j < TILE_SIZE; j++) {
            C_value += S_A[tid_y][j] * S_B[tid_x][j];
        }

        __syncthreads(); // 关键同步点：等待所有线程完成当前分块的点积计算
    }

    // 3. 写入最终结果到全局内存 C[m][k]
    if(m < M && k < K) {
        C[m * K + k] = C_value;
    }
}