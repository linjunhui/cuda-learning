#include <__clang_cuda_runtime_wrapper.h>
#include <cstdio>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int TILE_SIZE=32, int WARP_SIZE=32>
__global__ void matrix_multiply_kernel(float *matrix_a, float *matrix_b, float *output, const int M, const int K, const int N) {
    
    // 当前线程的全局索引 就是 要计算的元素 C(i, j) 的索引, 这里的BLOCK_SIZE 是 (TILE_SIZE, TILE_SIZE) 
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;


    __shared__ float matrix_a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float matrix_b_tile[TILE_SIZE][TILE_SIZE];

    // 计算 元素 C(i, j) 需要， 加载的TILE的个数
    int TILE_NUM = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 加载 数据到 shared memory
    for(int i = 0; i < TILE_NUM; i++) {


    // 加载数据到TILE
}

void launch_matrix_multiply(torch::Tensor matrix_a, torch::Tensor matrix_b, torch::Tensor output) {
    const int M = matrix_a.size(0);
    const int K = matrix_a.size(1);
    const int N = matrix_b.size(1);

    printf(" M = %d, K = %d, N = %d\n", M, K, N);

    size_t block_size_x = 16, block_size_y = 16;

    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size((M + block_size_x - 1) / block_size_x, (N + block_size_y - 1) / block_size_y);

    matrix_multiply_kernel(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), output.data_ptr<float>(), M, K, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_multiply", &launch_matrix_multiply, "Matrix Multiply");
}