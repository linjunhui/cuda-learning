#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <system_error>

// 一般cuda的check都是这样写成宏
#define CUDA_CHECK(call) {                                                 \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
        printf("ERROR: %s:%d, ", __FILE__, __LINE__);                      \
        printf("CODE:%d, DETAIL:%s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                           \
    }                                                                      \
}

#define BLOCK_SIZE 16

// 初始化化 矩阵, 
void initMatrix(float* data, int size, int low, int high, int seed);
void printMat(float* data, int size);
void compareMat(float* data1, float* data2, int size);

#endif //__UTILS__HPP__