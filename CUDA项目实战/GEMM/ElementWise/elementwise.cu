#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

/*
block_size = 256
grid_size = (N + block_size - 1) / block_size
*/
__global__ void elementwise_add_f32_kernel(const float *A, const float *B, float *C, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

/*
block_size = 256 / 4
grid_size = (N + block_size - 1) / block_size
*/
__global__ void elementwise_add_f32x4_kernel(float *A, float *B, float *C, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // 线程 id
    int arr_id = 4 * tid;


    if(arr_id < N) {
        float4 reg_a = FLOAT4(A[arr_id]);
        float4 reg_b = FLOAT4(B[arr_id]);
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;

        // 内存布局什么的都没变所以可以这样写，寄存器 写到 C
        FLOAT4(C[arr_id]) = reg_c;
    }
}



// PyTorch 绑定的矩阵乘法函数
void elementwise_add_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // 检查张量类型和维度
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(A.dim() == 1, "A must be 1D tensor");
    TORCH_CHECK(B.dim() == 1, "B must be 1D tensor");
    TORCH_CHECK(C.dim() == 1, "C must be 1D tensor");
    
    TORCH_CHECK(A.size(0) == B.size(0) && B.size(0) == C.size(0), "A、B、C的维度");
    
    // 获取数据指针
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();
    

    int N = A.size(0);
    // 配置线程块和网格
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    
    // 启动核函数
    elementwise_add_f32_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}


// PyTorch 绑定的矩阵乘法函数
void elementwise_add_f32x4(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // 检查张量类型和维度
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
    TORCH_CHECK(A.dim() == 1, "A must be 1D tensor");
    TORCH_CHECK(B.dim() == 1, "B must be 1D tensor");
    TORCH_CHECK(C.dim() == 1, "C must be 1D tensor");
    
    TORCH_CHECK(A.size(0) == B.size(0) && B.size(0) == C.size(0), "A、B、C的维度");
    
    // 获取数据指针
    float* d_A = A.data_ptr<float>();
    float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();
    
    int N = A.size(0);
    // 配置线程块和网格
    int block_size = 1024 / 4;
    int grid_size = (N + block_size - 1) / block_size;
    
    // 启动核函数
    elementwise_add_f32x4_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_add_f32", &elementwise_add_f32, "Elementwise add for float32");
    m.def("elementwise_add_f32x4", &elementwise_add_f32x4, "Elementwise add for float32 vec");
}