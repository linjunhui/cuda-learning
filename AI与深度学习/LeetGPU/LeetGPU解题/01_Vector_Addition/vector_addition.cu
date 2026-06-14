#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/extension.h>


__global__ void vector_add(float *vector_a, float *vector_b, float *output, const int N) {
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_idx < N) {
        output[global_idx] = vector_a[global_idx] + vector_b[global_idx];
    }
}


void launch_vector_add_kernel(torch::Tensor vector_a, torch::Tensor vector_b, torch::Tensor output, const int N) {
    int block_size = 64;
    int grid_size = (N + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(vector_a.data_ptr<float>(), vector_b.data_ptr<float>(), output.data_ptr<float>(), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    m.def("vector_add", &launch_vector_add_kernel, "Vector Add");
}