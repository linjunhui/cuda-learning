/*
实现 向量加法
1. 向量加法就是一维 数组

*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_idx < n) {
        c[thread_idx] = a[thread_idx] + b[thread_idx];   
        printf("thread_idx: %d, a: %f, b: %f, c: %f\n", thread_idx, a[thread_idx], b[thread_idx], c[thread_idx]);
    }   
}

int main() {
    int N = 100;
    float *a_host, *b_host, *c_host;
    float *a_device, *b_device, *c_device;

    a_host = (float *)malloc(N * sizeof(float));
    b_host = (float *)malloc(N * sizeof(float));
    c_host = (float *)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++) {
        a_host[i] = i;
        b_host[i] = i;
    }

    cudaMalloc(&a_device, N * sizeof(float));
    cudaMalloc(&b_device, N * sizeof(float));
    cudaMalloc(&c_device, N * sizeof(float));

    cudaMemcpy(a_device, a_host, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_add<<<1, N>>>(a_device, b_device, c_device, N);

    cudaMemcpy(c_host, c_device, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        printf("c[%d]: %f\n", i, c_host[i]);
    }
}