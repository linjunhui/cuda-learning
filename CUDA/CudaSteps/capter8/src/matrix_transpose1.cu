#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <cstdio>

/*
使用全局内存进行矩阵转置
A[MxN], 列数 宽度 是 N
B[i, j] = A[j, i] = A[j*N + i];

矩阵和线程的划分
将M 和 N 划分
BLOCK_SIZE_X, BLOCK_SIZE_Y


*/


const int COL_NUM = 2;
const int ROW_NUM = 3;
const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // 以output视角为主, output的形状应该是(COL_NUM, ROW_NUM)
    // 当前线程处理的元素坐标 
    // rows:  input 的行数， output的列数 
    // cols:  input 的列数， output的行数
    int tid_x = blockDim.x * blockIdx.x + threadIdx.x; // output 视角
    int tid_y = blockDim.y * blockIdx.y + threadIdx.y; // output[tid_x, tid_y] = tid_x * rows + tid_y
    
    if(tid_x < rows && tid_y < rows) {
        output[tid_x * rows + tid_y] = input[tid_y * cols + tid_x];
        printf("output[%d][%d] = input[%d][%d] = %f\n", tid_x, tid_y, tid_y, tid_x, input[tid_y * cols + tid_x]);
    }

}


// 这里 TILE 存储的就是转置后
const int TILE_SIZE_X = BLOCK_SIZE_Y;
const int TILE_SIZE_Y = BLOCK_SIZE_X;

__global__ void matrix_transpose_kernel2(const float* input, float* output, int rows, int cols) {
    // 避免bank conflict
    __shared__ float shared_tile[TILE_SIZE_X][TILE_SIZE_Y + 1];
    // 这读取 input 矩阵的内容，全局内存合并读入
    int input_col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int input_row_idx = blockDim.y * blockIdx.y + threadIdx.y;

    // 读取的时候就直接转置, 为了全局内存的合并访问，这里要从input 的视角来读入，行读入
    if(input_col_idx < cols && input_row_idx < rows) {
        shared_tile[threadIdx.y][threadIdx.x] = input[input_row_idx * cols + input_col_idx];
    } 
    __syncthreads();

    /*
        output 如何实现全局内存合并访问 行 (0, rows)
        这里就不考虑input了，从output视角，将tile 填充到output
    */
   int output_col_idx = blockIdx.y * blockDim.y + threadIdx.x;
   int output_row_idx = blockIdx.x * blockDim.x + threadIdx.y;
   if(output_col_idx < rows && output_row_idx < cols) {
        output[output_row_idx * rows + output_col_idx] = shared_tile[threadIdx.x][threadIdx.y];
   }

}


int main() {
    
    float *h_input, *h_output;
    float *d_input, *d_output;

    const int numel = COL_NUM * ROW_NUM;
    const int b_size = sizeof(float) * numel;

    h_input = (float *)malloc(b_size);
    h_output = (float *)malloc(b_size);

    cudaMalloc((void **)&d_input, b_size);
    cudaMalloc((void **)&d_output, b_size);


    // 初始化 矩阵
    for(int i = 0; i < numel; i++) {
        h_input[i] = 1.0f * i + 1.0f;
        printf("h_input[%d] = %f\t", i, h_input[i]);
    }

    cudaMemcpy(d_input, h_input, b_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const int GRID_SIZE_X = (COL_NUM + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    const int GRID_SIZE_Y = (ROW_NUM + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    dim3 blockPerGrid(GRID_SIZE_X, GRID_SIZE_Y);

    matrix_transpose_kernel2<<<blockPerGrid, threadsPerBlock>>>(d_input, d_output, ROW_NUM, COL_NUM);
    cudaError_t err = cudaGetLastError();
    printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, b_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < ROW_NUM; i++) {
        for(int j = 0; j < COL_NUM; j++) {
            printf("%d - h_input[%d][%d] = %f \t", i*COL_NUM + j, i, j, h_input[i*COL_NUM + j]);
        }
        printf("\n");
    }
    
    printf("----------\n");
    for(int i = 0; i < COL_NUM; i++) {
        for(int j = 0; j < ROW_NUM; j++) {
            printf("%f \t", h_output[i*ROW_NUM + j]);
        }
        printf("\n");
    }
}