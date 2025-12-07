#include <cuda_runtime.h>
#include <cstdio>

#include "error_cuh"

template<typename T>
__global__ void square_add(T *input, )