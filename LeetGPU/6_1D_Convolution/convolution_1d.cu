#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <cstdio>


const int INPUT_LEN = 4;
const int KERNEL_LEN = 2;

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
    int input_size, int kernel_size) {
    // 首先 计算 全局 线程索引
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    //printf("output_size = %d", output_size);
    if(tid < output_size) {
        //在input的起始位置是tid
        output[tid] = 0;
        for(int i = 0; i < kernel_size; i++) {
            // 这里的 问题 在于 计算outpout的不同元素时 kernel 和 input的同一个元素会反复加载kernel_size次
            output[tid] += kernel[i] * input[tid+i]; 
        }
    }        
}


// 辅助宏：用于计算内存对齐后的地址
// 适用于：任何需要在字节级别上对齐的情况，例如划分不同数据类型的共享内存数组。
#define ALIGN(offset, alignment) \
    (((offset) + (alignment) - 1) & ~((alignment) - 1))

__global__ void convolution_1d_aligned_optimized(
    const float* input, 
    const float* kernel,      
    float* output,
    int input_size, 
    int kernel_size) 
{
    // 动态共享内存基址：使用 char* 方便进行字节级指针运算
    extern __shared__ char shared_memory_base[];
    
    // --- 1. 定义常量和变量 ---
    const int block_size = blockDim.x;
    const int tid = threadIdx.x;
    
    // sh_input 所需元素数量 (扩展加载，用于计算 block_size 个输出)
    const int sh_input_elements = block_size + kernel_size - 1;
    
    // Block 负责的全局输入起始索引
    const int input_start_index = blockIdx.x * block_size;
    const int output_index = input_start_index + tid;
    const int output_size = input_size - kernel_size + 1;
    
    // --- 2. 划分内存：Array 1 (float* sh_input) ---
    
    // 2.1 sh_input 从基址开始 (强制转换为 float*)
    float* sh_input = (float*)shared_memory_base;
    
    // 2.2 计算 sh_input 结束后的字节偏移
    size_t current_offset = sh_input_elements * sizeof(float); 
    
    // --- 3. 划分内存：Array 2 (float* sh_kernel) ---
    
    // 3.1 确定下一个数组的对齐要求 (float 是 4 字节对齐)
    size_t alignment = sizeof(float); 
    
    // 3.2 计算对齐后的起始字节偏移：确保 sh_kernel 的起始地址是对齐的
    // 注意：对于 float 对 float，aligned_offset 通常等于 current_offset，但此宏保证了正确性。
    size_t aligned_offset = ALIGN(current_offset, alignment);
    
    // 3.3 sh_kernel 从对齐后的位置开始 (基址 + 对齐后的偏移)
    float* sh_kernel = (float*)(shared_memory_base + aligned_offset);

    // --- 4. 数据加载到 Shared Memory ---

    // 4.1. 加载 sh_input (扩展加载逻辑)
    // 适用情况：当 Block Size 小于 sh_input_elements 时，一个线程可能需要加载多次。
    for(int i = tid; i < sh_input_elements; i += block_size) {
        int global_input_index = input_start_index + i;
        
        // 边界检查：处理 input 数组的末尾
        if (global_input_index < input_size) {
            sh_input[i] = input[global_input_index];
        } else {
            // 零填充 (Zero Padding)
            sh_input[i] = 0.0f; 
        }
    }
    
    // 4.2. 加载 sh_kernel
    // 适用情况：Kernel 尺寸通常较小，前 kernel_size 个线程即可加载全部数据。
    if (tid < kernel_size) {
        sh_kernel[tid] = kernel[tid];
    }

    // 确保所有线程都已完成加载
    __syncthreads();
    
    // --- 5. 计算输出 ---
    
    // 适用情况：只有负责有效输出的线程才进行计算
    if(output_index < output_size) {
        float sum = 0.0f;
        // 卷积计算循环：i 是卷积核索引
        for(int i = 0; i < kernel_size; i++) {
            // 核心优化：所有数据访问都来自 Shared Memory！
            // sh_kernel[i]：访问 kernel 数据
            // sh_input[tid + i]：访问 input 数据，利用了数据重用性
            sum += sh_kernel[i] * sh_input[tid + i]; 
        }
        output[output_index] = sum;
    }
}


int main() {
    int output_size = INPUT_LEN - KERNEL_LEN + 1;
    //float *input, *kernel, *output;
    float input[INPUT_LEN] = {2, 4, 6, 8};
    float kernel[KERNEL_LEN] = {0.5, 0.2};
    float output[output_size];

    float *d_input, *d_output, *d_kernel;
    int input_byte_size = sizeof(float) * INPUT_LEN;
    int kernel_byte_size = sizeof(float) * KERNEL_LEN;
    int output_byte_size = sizeof(float) * output_size;
    cudaMalloc((void **)&d_input, input_byte_size);
    cudaMalloc((void **)&d_kernel, kernel_byte_size);
    cudaMalloc((void **)&d_output, output_byte_size);

    cudaMemcpy(d_input, input, input_byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_byte_size, cudaMemcpyHostToDevice);

    int block_size = 64;
    int grid_size = (output_size + block_size - 1) / block_size;

    // 定义共享内存数组大小
    int shared_input_size = block_size + KERNEL_LEN - 1;
    // input block + kernel block
    int total_shared_byte = (shared_input_size + KERNEL_LEN) * sizeof(float);

    convolution_1d_kernel<<<grid_size, block_size, total_shared_byte>>>(d_input, d_kernel, d_output, INPUT_LEN, KERNEL_LEN);

    cudaMemcpy(output, d_output, output_byte_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();   

    for(int i = 0; i < output_size; i++) {
        printf("output[%d] = %f \t", i, output[i]);
    }
    printf("\n");

}