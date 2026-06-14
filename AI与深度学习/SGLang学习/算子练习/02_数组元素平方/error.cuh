
void checkCudaError(cudaError_t error, const char *file, int line) {
    if(error != cudaSuccess) {
        printf("CUDA ERROR at %s:%d %s\n", file, line, cudaGetLastError());
        exit(1)
    }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)