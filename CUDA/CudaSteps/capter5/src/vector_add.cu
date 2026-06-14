#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cstdint>


#ifndef USE_DP
typedef float real;
#else
typedef double real;
#endif

const int BLOCK_SIZE = 128;
const int NUMEL = 100000;

template<typename T>
__global__ void vector_add(T *input1, T *input2, T *output, int64_t N) {
    // 计算全局 tid
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < N) {
        output[tid] = input1[tid] + input2[tid];
        for(int i = 0; i < 100000; i++) {
            output[tid] = sqrt(output[tid]);
        }
    }
}

int main() {
    float *h_input1, *h_input2, *h_output;
    float *d_input1, *d_input2, *d_output;

    int64_t B_SIZE = sizeof(real) * NUMEL;
    h_input1 = (real *)malloc(B_SIZE);
    h_input2 = (real *)malloc(B_SIZE);
    h_output = (real *)malloc(B_SIZE);

    cudaMalloc((void **)&d_input1, B_SIZE);
    cudaMalloc((void **)&d_input2, B_SIZE);
    cudaMalloc((void **)&d_output, B_SIZE);


    // 数据初始化
    for(int i = 0; i < NUMEL; i++) {
        h_input1[i] = 1.0f * i;
        h_input2[i] = 2.0f * i;
    }

    // 初始化
    cudaMemcpy(d_input1, h_input1, B_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, B_SIZE, cudaMemcpyHostToDevice);


    dim3 block_size(BLOCK_SIZE);
    int64_t grid_size = (NUMEL + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vector_add<<<grid_size, block_size>>>(d_input1, d_input2, d_output, NUMEL);

    cudaDeviceSynchronize();
}
