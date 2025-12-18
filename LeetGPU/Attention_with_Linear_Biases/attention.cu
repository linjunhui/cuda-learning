#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <cstdio>
#include <vector>
#include <cmath>

/*
朴素的方法，没法一次性在GPU完成
*/


const int TILE_SIZE = 32;

/*
一个线程计算 一个C(m, k)
*/
__global__ void naive_gemm(const float *A, const float *B, float *C, int M, int N, int K, float scale_factor) {
    int m = blockDim.y * blockIdx.y + threadIdx.y; 
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    __shared__ float S_A[TILE_SIZE][TILE_SIZE];
    __shared__ float S_B[TILE_SIZE][TILE_SIZE + 1];

    int TILE_NUM = (N + TILE_SIZE - 1) / TILE_SIZE;

    float c_reg = 0.0f;

    for(int i = 0; i < TILE_NUM; i++) {
        // 加载 A到TILE, 加载一个方块
        int n_base = i * TILE_SIZE;
        // 这里希望 A的加载合并访问，这里让 a_col 随tid_x变化
        int a_col = n_base + tid_x;
        if(m < M && a_col < N) {
            S_A[tid_y][tid_x] = A[m * N + a_col];
        } else{
            S_A[tid_y][tid_x] = 0.0f;
        }

        int b_row = n_base + tid_y; // B 的行坐标随 tid_y变化，让列坐标随tid_x变化，实现合并访问
        if(b_row < N && k < K) {
            //S_B[tid_y][tid_x] = B[b_row * K + k]; 
            S_B[tid_x][tid_y] = B[b_row * K + k]; // 这种写法 可以实现 B的转置加载，但是要注意 bank conflict
        } else {
            S_B[tid_x][tid_y] = 0.0f;
            //S_B[tid_y][tid_x] = 0.0f;
        }

        __syncthreads(); // 等待TILE加载完成

        for(int j = 0; j < TILE_SIZE; j++) {
            c_reg += S_A[tid_y][j] * S_B[tid_x][j];
            //c_reg += S_A[tid_y][j] * S_B[j][tid_x];
        }

        __syncthreads();
    }

    if(m < M && k < K) {
        C[m * K + k] = c_reg * scale_factor;
    }
}

__global__ void transpose_matrix(const float *A, float *B, const int rows, const int cols) {
    __shared__ float S[TILE_SIZE][TILE_SIZE+1];

    /*
        每个Block转置一个 TILE方块
    */

    // 矩阵A的全局索引
    int input_col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int input_row_idx = blockDim.y * blockIdx.y + threadIdx.y;

    int tid_y = threadIdx.y;
    int tid_x = threadIdx.x;

    if(input_col_idx < cols && input_row_idx < rows) {
        // A 是行主序， 按行读 实现合并
        S[tid_y][tid_x] = A[input_row_idx * cols + input_col_idx];
    } // 这里是转置，没有设置 超出边界时 S[tid_y][tid_x] = 0.0f;

    __syncthreads();

    // col 随 threadIdx.x 变化， 让B的写入是合并的
    /*
        (blockIdx.x * blockDim.x, blockIdx.y * blockDim.y ) 决定的是block的位置，所以这里不变
        threadIdx.x 变化的块，绑定到 ouput_col_idx, 按行写B
        block 是一个正方形，所以这里交换threadId.x和threadIdx.y 没影响
    */
    int output_col_idx = blockIdx.y * blockDim.y + threadIdx.x;
    int output_row_idx = blockIdx.x * blockDim.x + threadIdx.y;

    if(output_row_idx < cols && output_col_idx < rows) {
        // 按行写入, 这里是转置 rows 就是B的列数, 这里tid_x = (0, 1, 2..)的时候都是读取的S的同一列，会出现bank conflict
        B[output_row_idx * rows + output_col_idx] = S[tid_x][tid_y];
    }
}


__global__ void cal_position_bias(float *matrix, const float alpha, const int M, const int N) {

    // score_matrix(i, j) 

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x; // j tid_x 是列坐标
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y; // i tid_y 是行坐标

    if(tid_y < M && tid_x < N) {
        matrix[tid_y * N + tid_x] += alpha * (tid_y - tid_x);
    }
}

/*
返回 device上的指针, 复用，避免反复在CPU 和 Device 拷贝数据
*/

void naive_gemm_wrapper(const float *A, const float *B, float *output, int M, int N, int K, float scale_factor) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((K + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE);

    naive_gemm<<<grid_size, block_size>>>(A, B, output, M, N, K, scale_factor);
    cudaDeviceSynchronize();
}

void transpose_matrix_wrapper(const float *A, float *output, int rows, int cols) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((rows + TILE_SIZE - 1)/TILE_SIZE, (cols + TILE_SIZE - 1)/TILE_SIZE);

    transpose_matrix<<<grid_size, block_size>>>(A, output, rows, cols);
    cudaDeviceSynchronize();
}

void cal_position_bias_wrapper(float *matrix, float alpha, int rows, int cols) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((rows + TILE_SIZE - 1)/TILE_SIZE, (cols + TILE_SIZE - 1)/TILE_SIZE);
    cal_position_bias<<<grid_size, block_size>>>(matrix, alpha, rows, cols);
    cudaDeviceSynchronize();
}


/* softmax 计算 */

__global__ void max_reduction_kernel(const float *arr, float *output, int N) {
    extern __shared__ float s_max[]; // 

    int block_idx = blockIdx.x;
    int global_tid = block_idx * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if(tid < N) {
        s_max[tid] = arr[global_tid];
    } else {
        s_max[tid] = -FLT_MAX;
    }
    __syncthreads();
    // reduce 计算 max
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        // 比较 arr[tid] 与 arr[tid+offset]
        if(global_tid < N && global_tid + offset < N) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid+offset]);
        }
        __syncthreads();
    }
    if(tid == 0) {
        output[block_idx] = s_max[0];
    }
}


__global__ void softmax_sum_reduction_kernel(const float *arr, float *output_sum, const float g_max, int N) {
    // 在每个block 计算指数 再求和
    extern __shared__ float s_sum[];

    // 加载数据到shared memory
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_x = threadIdx.x;

    if(global_idx < N) {
        //s_sum[tid_x] = arr[global_idx];
        s_sum[tid_x] = __expf(arr[global_idx] - g_max);
    } else {
        s_sum[tid_x] = 0.0f; // 参与求和，但是不影响
    }


    
    __syncthreads();

    // reduction 求和
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if(tid_x < offset) { // 保证这个条件 tid_x + offset 不会越界  blockDim.x
            s_sum[tid_x] += s_sum[tid_x + offset];
        }
        __syncthreads();
    }
    if(tid_x == 0) {
        output_sum[blockIdx.x] = s_sum[0];
    }
}

__global__ void softmax_kernel(const float *arr, float *output, const float g_max, const float S_factor, int N) {
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(global_idx < N) {
        output[global_idx] = __expf(arr[global_idx] - g_max) / S_factor;
    }
}

float max_reduction_wrapper(const float *d_arr, const int N) {

    // 定义一个数组接收每个block的最大值
    float *d_block_max;
    float *h_block_max;

    int block_size = 4;
    int grid_size = (N + block_size - 1) / block_size;
    size_t shared_bytes_size = block_size * sizeof(float);


    h_block_max = (float *)malloc(grid_size * sizeof(float));
    cudaMalloc((void **)&d_block_max, grid_size * sizeof(float));
    max_reduction_kernel<<<grid_size, block_size, shared_bytes_size>>>(d_arr, d_block_max, N);
    cudaMemcpy(h_block_max, d_block_max, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 这里 grid_size 不会特别大，直接求最大值了
    float rlt_max = -FLT_MAX;
    for(int i = 0; i < grid_size; i++) {
        rlt_max = fmaxf(rlt_max, h_block_max[i]);
    }

    return rlt_max;
}

float softmax_sum_reduction_wrapper(const float *d_arr, const float g_max, const int N) {
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;

    // 需要一个数组存储每个block的和
    float *h_block_sum;
    float *d_block_sum;
    h_block_sum = (float *)malloc(grid_size * sizeof(float));
    cudaMalloc((void **)&d_block_sum, grid_size * sizeof(float));

    softmax_sum_reduction_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(d_arr, d_block_sum, g_max, N);
    cudaMemcpy(h_block_sum, d_block_sum, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    float rlt_sum = 0.0f;
    for(int i = 0; i < grid_size; i++) {
        rlt_sum += h_block_sum[i];
    }

    return rlt_sum;
}

void softmax_wrapper(const float *arr, float *output, float g_max, float S_factor, int N) {
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    
    softmax_kernel<<<grid_size, block_size>>>(arr, output, g_max, S_factor, N);
    cudaDeviceSynchronize();
}

void softmax(const float *d_input, float *d_output, int N) {

    // 计算 x_max
    float g_x_max = max_reduction_wrapper(d_arr, N);
    printf("g_x_max: %f\n", g_x_max);

    // 计算 softmax分母
    float S_factor = softmax_sum_reduction_wrapper(d_arr, g_x_max, N);
    printf("S_factor: %f\n", S_factor);

    softmax_wrapper(d_arr, d_output, g_x_max, S_factor, N);
}


void print_matrix(const float *matrix, int rows, int cols) {
    float *h_matrix = (float*)malloc(rows * cols * sizeof(float));
    cudaMemcpy(h_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%lf ", h_matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("Matrix %d x %d\n", rows, cols);
}

/*
题目说明：Q K V 这Q的维度和K V并不相同
*/
void attention_with_linear_bias(const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha) {
    // 0. 数据都先移动到显存
    float *d_Q, *d_K, *d_V;
    int q_size = M * d;
    int k_v_size = N * d;

    cudaMalloc((void **)&d_Q, q_size * sizeof(float));
    cudaMalloc((void **)&d_K, k_v_size * sizeof(float));
    cudaMalloc((void **)&d_V, k_v_size * sizeof(float));
    cudaMemcpy(d_Q, Q, q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, k_v_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, k_v_size * sizeof(float), cudaMemcpyHostToDevice);

    // 1. 计算 K^T
    float* d_K_T;
    cudaMalloc((void **)&d_K_T, k_v_size * sizeof(float));
    transpose_matrix_wrapper(d_K, d_K_T, N, d);
    print_matrix(d_K_T, d, N);

    
    // 2. 计算 GEMM  Q*K^T/sqrt(d)
    float scale_factor = 1.0 / (float) sqrt(d);
    float* score_matrix; // M * N
    cudaMalloc((void **)&score_matrix, M * N * sizeof(float));
    naive_gemm_wrapper(d_Q, d_K_T, score_matrix, M, d, N, scale_factor);
    print_matrix(score_matrix, M, N);

    // 3. 使用相对位置计算偏置
    cal_position_bias_wrapper(score_matrix, alpha,M, N);
    print_matrix(score_matrix, M, N);

    // 4. 计算softmax
    float* d_softmax_output;
}


int main() {

    int M = 2, N = 2, d = 4;
   
    std::vector<float> Q(M * d);
    std::vector<float> K(N * d);
    std::vector<float> V(M * d);

    for(int i = 0; i < M*d; i++) Q[i] = static_cast<float>(i + 1.0);
    for(int i = 0; i < N*d; i++) K[i] = static_cast<float>(i + 1.0);
    for(int i = 0; i < N*d; i++) V[i] = static_cast<float>(i + 1.0);

    //naive_gemm_wrapper(A.data(), B.data(), C.data(), M, N, K, 1.0f);
    float alpha = 0.1;
    attention_with_linear_bias(Q.data(), K.data(), V.data(), NULL,M, N, d, alpha);

    return 0;
}