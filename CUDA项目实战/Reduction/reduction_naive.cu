#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>



__global__ void reduce_naive_add_kernel(const float *input, double *sum, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N) {
        atomicAdd(sum, input[tid]);
    }
}




__global__ void reduce_block_add_kernel2(float *input, double *sum, int N) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // 当前block 的 线程base 索引
    int global_tid_base = blockIdx.x * blockDim.x * 2;
    // 处理数据的范围 是 [global_tid_base * 2, (global_tid_base + blockDim.x) * 2)

    int offset = blockDim.x;
    for(int i = offset; i > 0; i>>=1) {
        if(tid < i && global_tid + 1 < N) {
            input[global_tid * 2] += input[global_tid * 2 + i];
        }
        __syncthreads();
    }
    if(tid == 0) {
        sum[blockIdx.x] = input[global_tid_base];
    }
}

double reduce_block_add2(torch::Tensor input) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 1, "input must be 1D tensor");
    int N = input.size(0);
    int block_size = 4;
    int grid_size = (N / 2 + block_size - 1) / block_size;
    
    double *d_sum, *h_sum;

    size_t byte_size = sizeof(double) * grid_size;

    cudaMalloc((void **)&d_sum, byte_size);
    h_sum = (double *)malloc(byte_size);
    cudaMemset(d_sum, 0, byte_size);

    reduce_block_add_kernel2<<<grid_size, block_size>>>(input.data_ptr<float>(), d_sum, N);
    cudaMemcpy(h_sum, d_sum, byte_size, cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    double sum = 0.0;
    for(int i = 0; i < grid_size; i++) {
        sum += h_sum[i];
    }   
    free(h_sum);
    return sum;
}


/*
1. 在block中实现分block 归约，每个block的线程数为 block_size
*/
__global__ void reduce_block_add_kernel(float *input, double *sum, int N) {
    // 当前 全局 索引ID
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    // 当前block 的 全局ID base
    int global_tid_base = blockIdx.x * blockDim.x;

    int offset = blockDim.x >> 1;
    for(int i = offset; i > 0; i>>=1) {
        if(tid < i && global_tid + i < N) {
            input[global_tid] += input[global_tid + i];
        }
        __syncthreads(); // 等待每个线程计算完成
    }
    if(tid == 0) {
        sum[blockIdx.x] = input[global_tid_base];
    }
}

double reduce_block_add(torch::Tensor input) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 1, "input must be 1D tensor");
    int N = input.size(0);
    int block_size = 4;
    int grid_size = (N + block_size - 1) / block_size;
    
    double *d_sum, *h_sum;

    size_t byte_size = sizeof(double) * grid_size;

    cudaMalloc((void **)&d_sum, byte_size);
    h_sum = (double *)malloc(byte_size);
    cudaMemset(d_sum, 0, byte_size);

    reduce_block_add_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), d_sum, N);
    cudaMemcpy(h_sum, d_sum, byte_size, cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    double sum = 0.0;
    for(int i = 0; i < grid_size; i++) {
        sum += h_sum[i];
    }   
    free(h_sum);
    return sum;
}


double reduce_naive_add(torch::Tensor input) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 1, "input must be 1D tensor");
    int N = input.size(0);
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    double *sum, h_sum;
    cudaMalloc((void **)&sum, sizeof(double));
    cudaMemset(sum, 0, sizeof(double));
    reduce_naive_add_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), sum, N);
    cudaMemcpy(&h_sum, sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(sum);
    return h_sum;
}


float reduce_cpu(torch::Tensor input) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 1, "input must be 1D tensor");
    int N = input.size(0);
    float *h_input = input.data_ptr<float>();
    float sum = 0.0f;
    for(int i = 0; i < N; i++) {
        sum += h_input[i];
    }
    return sum;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_naive_add", &reduce_naive_add, "Reduce naive add");
    m.def("reduce_cpu", &reduce_cpu, "Reduce cpu");
    m.def("reduce_block_add", &reduce_block_add, "Reduce block add");
}