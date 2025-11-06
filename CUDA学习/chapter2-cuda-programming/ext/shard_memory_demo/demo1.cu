#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include "utils.hpp"
#include "error.h"

const int M = 1025;
const int S = 200;
const int N = 1;



__global__ void matrix_mat(float *A, float *B, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float val = 0;
        for (int i = 0; i < S; i++) {
            val += A[row * S + i] * B[i * N + col];
        }
        C[row * N + col] = val;
    }
    
}

int main()
{   
    float *h_A = (float*)malloc(M*S*sizeof(float));
    float *h_B = (float*)malloc(S*N*sizeof(float));
    float *h_C = (float*)malloc(M*N*sizeof(float));

    init_matrix(h_A,M,S);
    init_matrix(h_B,S,N);

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, M*S*sizeof(float));
    cudaMalloc(&d_B, S*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A, M*S*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, S*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 结果矩阵是 M  * N 形状， 所以线程块的形状是 M * N， 一个线程计算一个元素
    dim3 block_size(N,M);

    cudaEventRecord(start);
    matrix_mat<<<1,block_size>>>(d_A, d_B, d_C);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("kernel error: %s\n", cudaGetErrorString(error));
    }

    CUDA_CHECK(error);

    cudaEventRecord(stop);

    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("cost time %f  ms\n", milliseconds);
    //print_matrix(h_C,M,N);

	//printf("Hello World!\n");
	return 0;
}