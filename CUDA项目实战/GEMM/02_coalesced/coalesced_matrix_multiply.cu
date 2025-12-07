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

const int TILE_SIZE = 32;
const int PADDED_SIZE = TILE_SIZE + 1; // 增加一列或一行用于填充

__global__ void matrix_multi_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float S_A[TILE_SIZE][TILE_SIZE];
    __shared__ float S_B[TILE_SIZE][PADDED_SIZE];

    // 从C的视角出发， 就是计算  M x K 个元素， M行， K列

    // 加载到 A 的一个 TILE 到 S_A
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    // 当前线程的全局 坐标, 计算 C(m, k) 的值, A的第m行，B的k列, 计算元素的个数 N
    int m = blockDim.y * blockIdx.y + tid_y;
    int k = blockDim.x * blockIdx.x + tid_x;

    // 从 A、B划分TILE的角度思考, 当前block 计算的元素的范围 C(m_base, k_base) -> C(m_base+31, k_base+31) 正方形
    int m_base = blockDim.y * blockIdx.y;
    int k_base = blockDim.x * blockIdx.x;




    // 边界检查
    if (m >= M || k >= K) {
        return;
    }



    // 要计算 N 个元素，这里 分TILE计算
    int TILE_NUM = (N + TILE_SIZE - 1) / TILE_SIZE;
    float c_reg = 0.0f;

    /*
        此时视角要转换到 A、B划分 TILE的视角， 从Block角度出发
        tid_x (0, 31), tid_y (0, 31)
        n_base = blockDim.x * blockIdx.x;  // tid
    */
    for(int i = 0; i < TILE_NUM; i++) {

        // 加载 A 到 S_A, A的第m行， m 是受tid_y 影响的
        // A(m, xx) -> 转一维 A[m * N + xx], 其中 xx 是i * TILE_SIZE + tid_x
        // 加载 A 的 一个 TILE的范围， 行：m_base 开始 到 m_base+31, 列: 范围一整行
        
        // m = m_base + tid_y
        // k = n_base + tid_x
        // 这里加上 TILE_NUM，循环的话，就是一个 正方形(TILE_SIZE) 按照strip(32) 在（m_base，m_base+31) 向右滑动 0 -> N
        // 此时 A的
        int n_base = i * TILE_SIZE;
        if(m_base + tid_y < M && n_base + tid_x < N) {
            S_A[tid_y][tid_x] = A[(m_base + tid_y) * N + n_base + tid_x];
        } else {
            S_A[tid_y][tid_x] = 0.0f;
        }

        // 对于 B，就是在(k_base, k_base+31) 正方形(TILE_SIZE) Strip(32) 向下滑动 0 -> N
        // B(row, col) -> (k_base + tid_y, nbase_tid_x)
        // row * N + k

        int B_n = n_base + tid_x; // 向下滑动，由i控制， 行
        int B_k = k_base + tid_y; // 列
        if(k_base + tid_y < K && n_base + tid_x < N) {
            //S_B[tid_y][tid_x] = B[B_n * K + B_k]; // 这里不合并tid_x = 0和tid_x=1时 要跨行访问B

            // 修改为合并访问，就是在一个线程束，访问连续的 B的地址空间， tid_x 放到矩阵B的列上, 那么 B[B_k * K + B_n] 是合并的
            S_B[tid_y][tid_x] = B[B_k * K + B_n]; // B 的一个TILE 在 S_B中的存储 方向之类的都不变，就是线程加载B的时候按照行加载，实现合并访问

        } else{
            S_B[tid_y][tid_x] = 0.0f;
        }


        // 等待 TILE 加载到 shared memory
        __syncthreads();

        for(int t = 0; t < TILE_SIZE; t++) {
            // (tid_y, tid_x) 需要的是tid_y 的行 和 B的tid_x 列
            //c_reg += S_A[tid_y][t] * S_B[tid_x][t];

            //取出 A 的 行，就是列坐标变化， 取出B的列就是行坐标变化
            c_reg += S_A[tid_y][t] * S_B[t][tid_x];

        }

        __syncthreads(); // 这里是在block中， 每个线程都在计算一个元素，但是大家共用 S_A 和 S_B
    }

    if(m < M && k < K) {
        C[m * K + k] = c_reg;
    }
}

// PyTorch 绑定的矩阵乘法函数
void coalesced_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
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
    int threads_per_block = TILE_SIZE;
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
    m.def("coalesced_gemm", &coalesced_gemm, "朴素矩阵乘法 (C = A * B)");
}