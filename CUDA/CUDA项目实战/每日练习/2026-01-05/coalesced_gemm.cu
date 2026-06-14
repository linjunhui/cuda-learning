#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>
#include <torch/torch.h>



template<int TILE_SIZE>
__global__ void gemm(const float *A, const float *B, float *C, int M, int K, int N) {
    /*
        C 的形状(M, N)  M行， N列
        M: A的M行， K列
        K:
    */

    // 定义 shared memory 来存储 A 和 B的一个TILE
    __shared__ float S_A[TILE_SIZE][TILE_SIZE];
    __shared__ float S_B[TILE_SIZE][TILE_SIZE+1];

    int tile_num = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 这里 block_size == TILE_SIZE, 一个 block 对应 一个 tile

    /*
        计算一个元素时， 是固定A的一行, 行变化的慢 所以这里绑定到y上
    */
    int m_base = blockIdx.y * TILE_SIZE;
    int n_base = blockIdx.x * TILE_SIZE;

    int tid_x = threadIdx.x;  
    int tid_y = threadIdx.y; 
    float c_reg = 0;
    for(int i = 0; i < tile_num; i++) {
        /* 这里 不局限于计算单个元素的索引，从TILE的索引开始考虑，C(i, j) = .. + A(i, k) x B(k, j)
         A的列和B的行公用索引
        */


        int k_base = i * TILE_SIZE;

        int m = m_base + tid_y; // 取A 的行
        // 这里 A和B取TILE的时候，只需要对齐k_base就行了，一个TILE里面怎么处理，互不影响
        int k = k_base + tid_x;
        int B_k = k_base + tid_y;
        int n = n_base + tid_x;

        // 加载 A的tile 到共享内存
        //tile 的坐标
        if(m < M && k < K) {
            S_A[tid_y][tid_x] = A[(m) * K + k];
        } else {
            S_A[tid_y][tid_x] = 0.0f;
        }

        if(B_k < K && n < N) {
            S_B[tid_y][tid_x] = B[B_k * N + n];
        } else {
            S_B[tid_y][tid_x] = 0.0f;
        }

        __syncthreads();

        for(int t = 0; t < TILE_SIZE; t++) {
            c_reg += S_A[tid_y][t] * S_B[t][tid_x];
        }
        __syncthreads();
    }
    if(m_base + tid_y < M && n_base + tid_x < N) {
        C[(m_base + tid_y) * N + n_base + tid_x] = c_reg;
    }
}


void gemm_launch(torch::Tensor input_a, torch::Tensor input_b, torch::Tensor output) {
    const int M = input_a.size(0);
    const int K = input_a.size(1);
    const int N = input_b.size(1);
    const int TILE_SIZE = 32;

    float* A = input_a.data_ptr<float>();
    float* B = input_b.data_ptr<float>();
    float* C = output.data_ptr<float>();

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE);

    gemm<TILE_SIZE><<<grid_size, block_size>>>(A, B, C, M, K, N);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_launch, "Coalesced GEMM with Shared Memory");
}