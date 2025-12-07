#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <cstdio>

__global__ void reverse_array(float* input, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N/2) {
        float tmp = input[tid];
        input[tid] = input[N - 1 - tid];
        input[N - 1 - tid] = tmp;
    }
}


int main() {
    const int arr_len = 7;
    float h_array[arr_len] = {1.0 , 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    float *d_array;
    int byte_size = sizeof(float) * arr_len;
    cudaMalloc((void **)&d_array, byte_size);

    cudaMemcpy(d_array, h_array, byte_size, cudaMemcpyHostToDevice);

    // 线程数量
    int tid_num = 7 / 2;
    int block_size = 64;
    int grid_size = (block_size + tid_num - 1) / block_size;

    reverse_array<<<grid_size, block_size>>>(d_array, arr_len);

    cudaMemcpy(h_array, d_array, byte_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    for(int i = 0; i < arr_len; i++) {
        printf("%f\t", h_array[i]);
    }
}