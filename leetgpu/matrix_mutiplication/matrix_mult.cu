#include "utils.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <driver_types.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 变化快的维度

    if (row < M && col < K) {
        float value = 0.0f;
        for (int i = 0; i < N; ++i) {
            // A的第 row行和B的第i列相乘
            value += A[row * N + i] * B[i * K + col];
        }
        //printf("Thread (%d, %d) computes C[%d, %d] = %f\n", blockIdx.x, blockIdx.y, row, col, value);
        C[row * K + col] = value;
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
}


int main() {
    // 定义矩阵的尺寸
    const int M = 32; // Number of rows in A
    const int N = 1024; // Number of columns in B
    const int K = 32; // Number of columns in A and rows in B

    // 主机内存分配
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_B = (float*)malloc(N * K * sizeof(float));
    float *h_C = (float*)malloc(M * K * sizeof(float));

    // 初始化矩阵
    init_matrix(h_A, M, N);
    init_matrix(h_B, N, K);

    // 设备内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    // 将主机内存数据拷贝到设备内存
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solve(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);


    printf("Matrix A:\n");
    print_matrix(h_A, M, N);
    printf("Matrix B:\n");
    print_matrix(h_B, N, K);
    printf("Matrix C (result):\n");
    print_matrix(h_C, M, K);

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %f ms\n", elapsedTime);


    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}