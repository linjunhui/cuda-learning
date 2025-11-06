
#include <stdio.h>

__global__  void hello_from_gpu()
{
  printf("Hello World from GPU!\n");
}

int main()
{

    dim3 grid_size(2,2, 1);

    hello_from_gpu<<<grid_size,1>>>();
    cudaDeviceSynchronize();
    return 0;
}