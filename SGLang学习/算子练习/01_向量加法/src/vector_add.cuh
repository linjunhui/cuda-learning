#ifndef VECTOR_ADD_H_
#define  VECTOR_ADD_H_

#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel 声明
template <typename T>
__global__ void vectorAdd(T *input1, T* input2, T *output, int64_t N);

// launch_kernel 函数声明（主机端代码）
template<typename T>
void launch_kernel(T *h_input1, T *h_input2, T *h_output, int64_t N, int64_t M_SIZE);

#endif