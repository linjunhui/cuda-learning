#include <stdio.h>

#define M  512
#define N  8

__global__ void mat_add(float *A, float *B, float *C) {
    // 计算线程的行索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算线程的列索引
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        // 二维转一维索引
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}


int main() { 



    return 0;
}