// 包含 CUDA 运行时 API 的头文件，提供对 CUDA 函数的支持
#include <cuda_runtime_api.h>
#include <stdio.h>  // 标准输入输出库，用于打印信息

// 定义一个 CUDA kernel 函数 VecAdd，用于在 GPU 上执行向量加法
__global__ void VecAdd(float* A, float* B, float* C)
{
    // 获取当前线程的 x 维度索引（每个线程处理一个元素）
    int i = threadIdx.x;

    // 执行向量加法：C[i] = A[i] + B[i]
    C[i] = A[i] + B[i];
}

int main()
{
    // 设置向量大小为 10 个浮点数
    int N = 10;

    // 声明三个主机（CPU）内存指针，用于存储输入和输出数据
    float *A, *B, *C;

    // 声明三个设备（GPU）内存指针，用于在 GPU 上进行计算
    float *d_A, *d_B, *d_C;

    // 在 CPU 上分配内存空间
    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));
    C = (float *)malloc(N * sizeof(float));

    // 初始化数组 A 和 B，每个元素的值等于其下标
    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i;
    }

    // 在 GPU 上分配内存空间
    cudaMalloc(&d_A, N * sizeof(float));  // 分配 A 的 GPU 内存
    cudaMalloc(&d_B, N * sizeof(float));  // 分配 B 的 GPU 内存
    cudaMalloc(&d_C, N * sizeof(float));  // 分配 C 的 GPU 内存

    // 将数据从 CPU 内存复制到 GPU 内存
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 VecAdd kernel：
    // <<<1, N>>> 表示启动 1 个线程块（block），每个块中有 N 个线程（thread）
    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    // 等待所有 GPU 计算任务完成，确保结果正确后再继续执行后续代码
    cudaDeviceSynchronize();

    // 将计算结果从 GPU 内存复制回 CPU 内存
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < N; i++)
    {
        printf("%f + %f = %f\n", A[i], B[i], C[i]);
    }

    // 清理资源：释放 CPU 和 GPU 内存
    free(A); 
    free(B); 
    free(C);
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);

    return 0;
}