
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <cstdio>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(global_tid < width * height * 4) {
        image[global_tid] = global_tid % 4 || global_tid ==0 ? 255 - image[global_tid] : image[global_tid];
        printf("image[%d] = %d\n", global_tid, image[global_tid]);
    }
    
}

int main() {
    const int pixel_num = 256;
    unsigned char image[pixel_num] = {10, 20, 30, 255, 100, 150, 200, 255};
    int width=2, height=1;

    unsigned char *d_image;
    int byte_size = sizeof(unsigned char) * pixel_num;

    cudaMalloc((void **)&d_image, byte_size);
    cudaMemcpy(d_image, image, byte_size, cudaMemcpyHostToDevice);


    // 题目 给的是 一维
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);
    cudaError_t err = cudaGetLastError();
    printf("Cuda Err: %s\n", cudaGetErrorString(err));

    cudaMemcpy(image, d_image, byte_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0 ; i < pixel_num; i++) {
        printf("%d\t", image[i]);
    }

    return 0;
}