/*

Warp Max
通过一个在warp内寻找最大值的例子， 理解Warp操作

*/

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>
#include <float.h>

const int WARP_SIZE = 32;

__device__ float warp_reduce_max_f32(float val) {

    for(int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        /*
        warp内隐式同步，一个warp内的所有线程在__shfl_xor_sync指令完成前，不会执行下一条指令
        */
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}

/*

每个线程读取其对应的数据
执行Warp Reduce, 找出每个Warp内的最大值 
将结果写回输出数组

1. 将一个block处理的数组， 划分为多个Warp， 每个Warp内的线程处理 32 个数据， 共处理 N/32 个Warp，这里是求最大值
2. 每个Warp内的线程， 执行Warp Reduce, 找出每个Warp内的最大值
3. 将每个Warp内的最大值， 写回输出数组
4. 最后， 在第一个Warp内的线程， 找出所有Warp内的最大值， 写回输出数组

*/
__global__ void warp_max_kernel(float *input, float *output, int n) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    int warp_id = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;
    int warp_num = blockDim.x / WARP_SIZE;

    extern __shared__ float s_max[];

    // 1. 每个线程读取其对应的数据
    float thread_max = (global_tid < n) ? input[global_tid] : -FLT_MAX;

    // 2. 执行Warp Reduce
    thread_max = warp_reduce_max_f32(thread_max);

    // 3. 将结果写回输出数组
    if(lane_id == 0) s_max[warp_id] = thread_max;
    __syncthreads();

    if(local_tid == 0) {
        float reg_max = -FLT_MAX;
        for(int i = 0; i < warp_num; i++) {
            reg_max = fmaxf(reg_max, s_max[i]);
        }
        output[blockIdx.x] = reg_max;
        printf("blockIdx.x: %d, reg_max: %f\n", blockIdx.x, reg_max);
    }
}

int main() {
    const int N = 100;
    const int block_size = 32;
    const int grid_size = (N + block_size - 1) / block_size;
    // 每个block 输出一个最大值， 所以输出数组大小为 grid_size
    const int output_size = grid_size * sizeof(float);
    // 每个block 有 warp_num_per_block 个Warp， 所以共享内存大小为 warp_num_per_block * sizeof(float)
    const int warp_num_per_block = block_size / WARP_SIZE;
    const int shared_size = warp_num_per_block * sizeof(float);

    // 准备数据
    std::vector<float> vec(N);
    for(int i = 0; i < N; i++) vec[i] = i + 1.0f;

    std::vector<float> h_output(grid_size);

    float *d_input, *d_output;
    // 分配内存
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, grid_size * sizeof(float)); 

    // 复制数据
    cudaMemcpy(d_input, vec.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 执行核函数
    warp_max_kernel<<<grid_size, block_size, shared_size>>>(d_input, d_output, N);

    cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    // 打印结果
    float max_value = -FLT_MAX;
    for(int i = 0; i < grid_size; i++) {
        std::cout << "output[" << i << "] = " << h_output[i] << std::endl;
        max_value = fmaxf(max_value, h_output[i]);
    }
    std::cout << "Max value: " << max_value << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}