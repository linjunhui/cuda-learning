#include "timer.hpp"  // 假设你的 Timer 类定义在 Timer.h 中

// 一个简单的 CUDA kernel 示例
__global__ void dummyKernel(int *data, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        data[i] = data[i] * 2;
    }
}

int main() {
    Timer timer;

    const int N = 1 << 20; // 1 million elements
    const int bytes = N * sizeof(int);
    int *h_data = (int *)malloc(bytes);
    int *d_data;

    cudaMalloc(&d_data, bytes);

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_data[i] = i % 256;
    }

    // 数据拷贝到设备
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // 开始 GPU 计时
    timer.start_gpu();

    // 启动 kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    dummyKernel<<<numBlocks, blockSize>>>(d_data, N);

    // 结束 GPU 计时
    timer.stop_gpu();

    // 等待 GPU 执行完成
    cudaDeviceSynchronize(); // 必须同步，否则时间可能不准确

    // 输出 GPU 耗时
    timer.duration_gpu("Dummy Kernel Execution");

    // 清理资源
    cudaFree(d_data);
    free(h_data);

    return 0;
}