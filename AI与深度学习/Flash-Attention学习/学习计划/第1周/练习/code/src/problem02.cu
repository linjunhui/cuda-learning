/*
题目 2：CUDA 程序结构
知识点：CUDA 程序基本流程
任务：将 CUDA 程序的关键步骤按正确顺序排列，包括主机/设备内存管理、数据传输、内核启动与资源释放。
提示：回忆标准流程：主机准备 → 设备准备 → 数据移动 → 内核执行 → 结果回传 → 清理。
正确顺序：4 → 2 → 5 → 1 → 3 → 6
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

void printf_error_string(char *msg, cudaError_t err) {
    printf("CUDA Operator : %s Status: %s\n", msg, cudaGetErrorString(err));
}

int main() {

    int N = 1000;
    float* h_data;
    float* d_data;

    int byte_size = sizeof(float) * N;

    // 主机准备
    h_data = (float *)malloc(byte_size);

    // 设备准备
    cudaMalloc(&d_data, byte_size);
    cudaError_t err = cudaGetLastError();
    printf_error_string("Malloc CUDA MEM", err);

    // 初始数据
    for(int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }

    // 数据传输， 数据移动
    err = cudaMemcpy(d_data, h_data, byte_size, cudaMemcpyHostToDevice);
    printf_error_string("Data Transfer from Host to Device", err);

    // 准备 cuda线程尺寸， 内核执行
    dim3 block_size(32, 1, 1);
    int grid_size_x = (N + 32 - 1) / 32;
    dim3 grid_size(grid_size_x, 1, 1);
    kernel<<<grid_size, block_size>>>(d_data, N);
    err = cudaGetLastError();
    printf_error_string("Boot Kernel", err);


    // 回传结果
    err = cudaMemcpy(h_data, d_data, byte_size, cudaMemcpyDeviceToHost);
    printf_error_string("Data Transfer from Device to Host", err);

    // 内存释放，设备同步
    cudaFree(d_data);
    err = cudaDeviceSynchronize();
    printf_error_string("Host Synchronize Device", err);


    free(h_data);

    return 0;
}