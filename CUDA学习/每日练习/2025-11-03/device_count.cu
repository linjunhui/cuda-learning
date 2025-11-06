#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int count;
    cudaGetDeviceCount(&count);

    printf("device count: %d\n", count);
}
