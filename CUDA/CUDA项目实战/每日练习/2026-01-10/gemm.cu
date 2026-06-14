#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>
#include <torch/torch.h>
#include <torch/extension.h>


template<int TILE_SIZE = 32>
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int K , int N) {
    __shared__ float S_A[TILE_SIZE][TILE_SIZE];
    __shared__ float S_B[TILE_SIZE][TILE_SIZE+1];

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    float c_reg = 0.0f;

    int tile_num = (K + TILE_SIZE - 1) / TILE_SIZE;

    int m_base = blockDim.y * blockIdx.y;
    int n_base = blockDim.x * blockIdx.x;

    int n = n_base + tid_x;
    int m = m_base + tid_y;

    for(int tile_idx = 0; tile_idx < tile_num; tile_idx++) {
        
        int k_base = tile_idx * TILE_SIZE;  
        int k = k_base + tid_x;
        if(m < M && k < K) {
            S_A[tid_y][tid_x] = A[m * K + k];
        } else {
            S_A[tid_y][tid_x] = 0.0f;
        }

        // K 对B来是是B的行
        int B_k = k_base + tid_y; 
        if(B_k < K && n < N) {
            S_B[tid_x][tid_y] = B[B_k * N + n];
        } else {
            S_B[tid_x][tid_y] = 0.0f;
        }

        __syncthreads();

        /*
            计算C(m, n)
            又因为：
                int n = n_base + tid_x;
                int m = m_base + tid_y;
            所以这里 取S_A的tid_y行 和 B的tid_x 行, 这里S_B是加载B的时候进行了转置的 相当于是B的列
        */
        for(int j = 0; j < TILE_SIZE; j++) {
            c_reg += S_A[tid_y][j] * S_B[tid_x][j];
        }
    }
    if(m < M && n < N) {
        C[m * N + n] = c_reg;
    }
}

void launch_gemm(torch::Tensor input_a, torch::Tensor input_b, torch::Tensor output) {
    int M = input_a.size(0);
    int K = input_a.size(1);
    int N = input_b.size(1);

    float* A = input_a.data_ptr<float>();
    float* B = input_b.data_ptr<float>();
    float* C = output.data_ptr<float>();

    const int TILE_SIZE = 32;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);

    gemm_kernel<TILE_SIZE><<<grid_size, block_size>>>(A, B, C, M, K, N);

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &launch_gemm, "GEMM");
}