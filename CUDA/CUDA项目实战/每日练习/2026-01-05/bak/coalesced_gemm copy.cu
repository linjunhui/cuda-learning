
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>
#include <iostream>
#include <torch/torch.h>

/*
这里进行GEMM的实现
*/

template<int TILE_SIZE>
__global__ void coalesced_gemm_kernel(const float* A, const float* B, float* C, int M, int K, int N) {

    // 计算当前block中的线程index 作为 TILE的索引
    uint32_t tid_x = threadIdx.x; // col 变化的快，绑定到x  [0, TILE_SIZE)
    uint32_t tid_y = threadIdx.y; // row变化的慢，绑定到y   [0, TILE_SIZE)

    // 从 C的角度 C(i, j) 当前线程计算 C(i, j) -> (row, col)
    uint32_t col_idx = blockDim.x * blockIdx.x + tid_x; // 范围 是[0, N)
    uint32_t row_idx = blockDim.y * blockIdx.y + tid_y; // 范围 是[0, M)

    /*
     当前计算的元素索引是 (row_idx, col_idx) -->  (tid_x, tid_y) 上面的公式就能看出 要计算的元素索引 与 tid_x和tid_y的关系
    
    */






    // 划分TILE
    __shared__ float S_A[TILE_SIZE][TILE_SIZE];
    __shared__ float S_B[TILE_SIZE][TILE_SIZE + 1];

    // 计算 划分的TILE的数量
    int TILE_NUM = (K + TILE_SIZE - 1) / TILE_SIZE;
    float c_reg = 0.0f;
    
    //printf("tile num: %d\n", TILE_NUM);

    /*
     1) 这里忽略了一个大问题，就是TILE 是在 滑动
     2) 不要 局限于 当前线程 处理A的 行 或者 B的列
     3) 我们确定好一个一个TILE之后， 计算TILE 的坐标位置; BLOCK_SIZE == TILE_SIZE 就是在这时候用的
     4) 
    */
    for(int tile_idx = 0; tile_idx < TILE_NUM; tile_idx++) {
        /* 1. 首先计算 当前A的TILE 坐标,  C 是M行和N列，从block角度看就是，
            将A 按照 block(tile)切分为多个 横条，每个横条上有一个TILE方块 横向滑动，
            那么 blockIdx.x 就是 横条的index, 那么一个TILE的坐标就由 blockIdx.x 和 tile_idx 决定
        */
       // 这里是计算 一个 TILE的起始坐标
       int A_row_base = blockIdx.y * TILE_SIZE;
       int A_col_base = tile_idx * TILE_SIZE;

       // 计算 TILE中每个点的坐标， A的形状是(M, K)
       int A_row_idx = A_row_base + tid_y; // y和行都变化的慢，绑定到一起
       int A_col_idx = A_col_base + tid_x; // x和列都变化的快，绑定到一起

       if(A_row_idx < M && A_col_idx < K) {
            S_A[tid_y][tid_x] = A[A_row_idx * K + A_col_idx];
       } else {
            S_A[tid_y][tid_x] = 0.0f;
       }
        
       /*
        B的形状 (K, N)
        将N列 按照block(tile_size) 划分为若干个竖条， 每个竖条有一个TILE向下滑动
        那么每个TILE的坐标由 tile_idx 和 blockIdx.y 控制。
        补充：这里计算 C(i, j) 用A的i行和B的j列， blockIdx.x、blockIdx.y 控制 A 的行、B的列可以随意组合
       */

       int B_row_base = tile_idx * TILE_SIZE;
       int B_col_base = blockIdx.x * TILE_SIZE;

       int B_row_idx = B_row_base + tid_y; //y绑定行
       int B_col_idx = B_col_base + tid_x;

       if(B_row_idx < K && B_col_idx < N) {
            S_B[tid_x][tid_y] = B[B_row_idx * N + B_col_idx];
       } else {
            S_B[tid_x][tid_y] = 0.0f;
       }

       __syncthreads();
       for(int i = 0; i < TILE_SIZE; i++) {
            c_reg += S_A[tid_y][i] * S_B[i][tid_x];
       }

    }

    if(row_idx < M && col_idx < N) {
        C[row_idx * N + col_idx] = c_reg;
    }
    __syncthreads();
} 

/*
int main() {
    const int TILE_SIZE = 2;
    const int M = 2, K = 2, N = 1;

    float input_a[2][2] = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    float input_b[2][1] = {
        {1.0},
        {1.0}
    };

    float output[2][1] = {
        {1.0},
        {1.0}
    };

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc((void **)&d_a, 4*sizeof(float));
    cudaMalloc((void **)&d_b, 2*sizeof(float));
    cudaMalloc((void **)&d_c, 2*sizeof(float));

    cudaMemcpy(d_a, input_a, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, input_b, 2*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((M + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);

    coalesced_gemm_kernel<TILE_SIZE><<<grid_size, block_size>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_c, 2*sizeof(float), cudaMemcpyDeviceToHost);
    printf("----\n");
    // (2, 2) x (2, 1) --> (2, 1)
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf("%d %d %lf ", i, j, output[i][j]);
        }
        printf("\n");
    }
}*/


void gemm_launch(torch::Tensor input_a, torch::Tensor input_b, torch::Tensor output) {
    const int M = input_a.size(0);
    const int K = input_a.size(1);
    const int N = input_b.size(1);
    const int TILE_SIZE = 2;

    printf("M = %d K = %d N = %d\n", M, K, N);

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    // C 的 形状(M, N)
    //dim3 grid_size((M + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE);
    
    const float* A = input_a.data_ptr<float>();
    const float* B = input_b.data_ptr<float>();
    float* C = output.data_ptr<float>();

    coalesced_gemm_kernel<TILE_SIZE><<<grid_size, block_size>>>(A, B, C, M, K, N);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_launch, "Coalesced GEMM");
}