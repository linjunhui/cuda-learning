#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>

/*
从C的视角出发， 就是计算  M x K 个元素， M行， K列
C = A * B，其中 A是M×N，B是N×K，C是M×K
*/
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // x 变化的快，就对应 列
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    // y变化慢 对应的是行
    int m = blockDim.y * blockIdx.y + threadIdx.y;


    if(k < K && m < M) {
        // C[m][k] = A的第m行 点乘 B的第k列
        // A是M×N，B是N×K，C是M×K
        // 做二维到一维的转换
        float c_reg = 0.0f;
        for(int n = 0; n < N; n++) {
            // A[m][n] * B[n][k]
            // A[m][n] = A[m * N + n] (A是M×N)
            // B[n][k] = B[n * K + k] (B是N×K)
            c_reg += A[m * N + n] * B[n * K + k];
        }
        // C[m][k] = C[m * K + k] (C是M×K)
        C[m * K + k] = c_reg;
    }
}

// CPU参考实现（用于验证结果）
// A是M×N，B是N×K，C是M×K
void matrix_multiply_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += A[i * N + n] * B[n * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// 验证结果
bool verify_result(const float* gpu_result, const float* cpu_result, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; i++) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        if (diff > tolerance) {
            printf("验证失败: 位置 [%d], GPU结果: %f, CPU结果: %f, 差异: %f\n", 
                   i, gpu_result[i], cpu_result[i], diff);
            return false;
        }
    }
    return true;
}

// 打印矩阵（用于小矩阵调试）
void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 初始化矩阵
void init_matrix(float* matrix, int size, int seed = 0) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f; // [-5, 5]
    }
}

// 测试用例1
// A是M×N，B是N×K，C是M×K
void test_case_1() {
    printf("========== 测试用例 1 ==========\n");
    int M = 2, N = 2, K = 3;
    
    // 初始化矩阵A (2x2) M×N
    float A[] = {1.0f, 2.0f, 4.0f, 5.0f};
    // 初始化矩阵B (2x3) N×K
    float B[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    // 期望结果 (2x3) M×K
    float expected[] = {27.0f, 30.0f, 33.0f, 78.0f, 87.0f, 96.0f};
    
    float *d_A, *d_B, *d_C;
    float *h_C = (float*)malloc(M * K * sizeof(float));
    
    // 分配设备内存
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    
    // 设置线程块和网格维度
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // 启动核函数
    matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // 复制结果回主机
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("GPU结果:\n");
    print_matrix(h_C, M, K, "C");
    printf("期望结果:\n");
    print_matrix(expected, M, K, "Expected");
    
    if (verify_result(h_C, expected, M * K)) {
        printf("✓ 测试用例1通过!\n\n");
    } else {
        printf("✗ 测试用例1失败!\n\n");
    }
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C);
}

// 测试用例2
// A是M×N，B是N×K，C是M×K
void test_case_2() {
    printf("========== 测试用例 2 ==========\n");
    int M = 3, N = 4, K = 2;
    
    // 初始化矩阵A (3x4) M×N
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    // 初始化矩阵B (4x2) N×K
    float B[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    // 期望结果 (3x2) M×K
    float expected[] = {50.0f, 60.0f,
                        114.0f, 140.0f,
                        178.0f, 220.0f};
    
    float *d_A, *d_B, *d_C;
    float *h_C = (float*)malloc(M * K * sizeof(float));
    
    // 分配设备内存
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    
    // 设置线程块和网格维度
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // 启动核函数
    matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // 复制结果回主机
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("GPU结果:\n");
    print_matrix(h_C, M, K, "C");
    printf("期望结果:\n");
    print_matrix(expected, M, K, "Expected");
    
    if (verify_result(h_C, expected, M * K)) {
        printf("✓ 测试用例2通过!\n\n");
    } else {
        printf("✗ 测试用例2失败!\n\n");
    }
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C);
}

// 随机测试用例（使用CPU验证）
// A是M×N，B是N×K，C是M×K
void test_case_random(int M, int N, int K) {
    printf("========== 随机测试用例 (M=%d, N=%d, K=%d) ==========\n", M, N, K);
    
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_B = (float*)malloc(N * K * sizeof(float));
    float *h_C_gpu = (float*)malloc(M * K * sizeof(float));
    float *h_C_cpu = (float*)malloc(M * K * sizeof(float));
    
    // 初始化矩阵
    init_matrix(h_A, M * N, 1);
    init_matrix(h_B, N * K, 2);
    
    // CPU计算
    matrix_multiply_cpu(h_A, h_B, h_C_cpu, M, N, K);
    
    // GPU计算
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    
    // 设置线程块和网格维度
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // 启动核函数
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(err));
        return;
    }
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_C_gpu, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    if (verify_result(h_C_gpu, h_C_cpu, M * K)) {
        printf("✓ 随机测试用例通过!\n");
        printf("GPU执行时间: %.3f ms\n", milliseconds);
        printf("性能: %.2f GFLOPS\n", 
               (2.0f * M * N * K) / (milliseconds * 1e6f));
    } else {
        printf("✗ 随机测试用例失败!\n");
    }
    
    printf("\n");
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 性能测试
// A是M×N，B是N×K，C是M×K
void performance_test() {
    printf("========== 性能测试 (M=8192, N=6144, K=4096) ==========\n");
    int M = 8192, N = 6144, K = 4096;
    
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_B = (float*)malloc(N * K * sizeof(float));
    float *h_C = (float*)malloc(M * K * sizeof(float));
    
    // 初始化矩阵
    init_matrix(h_A, M * N, 1);
    init_matrix(h_B, N * K, 2);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    
    // 设置线程块和网格维度
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // 预热
    matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int num_iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        matrix_multiplication_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / num_iterations;
    
    // 计算GFLOPS
    // 矩阵乘法: C = A * B
    // A是M×N，B是N×K，C是M×K
    // 每个C[i][j]需要N次乘法和N次加法，共2*N次浮点运算
    // 总运算量: M * K * 2 * N
    float gflops = (2.0f * M * N * K) / (avg_time * 1e6f);
    
    printf("平均执行时间: %.3f ms\n", avg_time);
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("吞吐量: %.2f GB/s (读取A+B, 写入C)\n", 
           (M * N + N * K + M * K) * sizeof(float) / (avg_time * 1e6f));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    printf("矩阵乘法 CUDA 测试程序\n");
    printf("======================\n\n");
    
    // 运行测试用例
    test_case_1();
    test_case_2();
    test_case_random(100, 100, 100);
    test_case_random(512, 512, 512);
    
    // 性能测试
    performance_test();
    
    printf("所有测试完成!\n");
    return 0;
}
