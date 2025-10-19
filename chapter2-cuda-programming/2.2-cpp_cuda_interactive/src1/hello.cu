#include "hello.hpp"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel() {
  printf("Hello World!\n");
}

void hello_device() {
  hello_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}