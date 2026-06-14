#include<cstdio>
#include<cuda_runtime.h>


#ifdef DB
typedef double real;
#else
typedef float real;
#endif

const int  BLOCK_SIZE = 128;

__global__ void reduce_global(real *arr, real *sum) {
    // 在当前 block 中求和
    // 折半规约
    // 先获取 block_size
    int block_size = blockDim.x;
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;

    // 当前 block 控制的 数组范围 
    real *arr_curr = arr + block_idx * block_size;

    for(int offset = block_size / 2; offset > 0; offset = offset >> 1) {
        if(tid < offset) {
            arr_curr[tid] += arr_curr[tid+offset]; // 这里每次都要从 全局内存读取，再写入全局内存
        }
        __syncthreads();
    }

    if(tid == 0) {
        sum[block_idx] = arr_curr[0];
    }
}

__global__ void reduce_static(real *arr, real *sum, int N) {
    const int block_size = blockDim.x;
    int tid = threadIdx.x;
    int block_idx = blockIdx.x;
    int global_idx = block_idx * block_size + tid;

    __shared__ real shared_arr[BLOCK_SIZE];
    // if(global_idx < N) shared_arr[tid] = arr[global_idx]; else shared_arr[tid] = 0.0; // + 0 不影响最终结果
    shared_arr[tid] = (global_idx < N) ? arr[global_idx] : 0.0;

    for(int offset = block_size / 2; offset > 0; offset = offset/2) {
        if(tid < offset) {
            shared_arr[tid] += shared_arr[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0) { // 当前block 的第一个线程
        sum[block_idx] = shared_arr[0];
    }

}


int main() {
    const int N = 16;

    real h_arr[N];
    real *d_arr;
    for(int i = 0; i < N; i++) {
        h_arr[i] = (i+1) * 1.0;
    }

    int byte_size = sizeof(real) * N;
    int block_size = 4;
    int grid_size = (N + block_size - 1) / block_size;
    real h_sum[grid_size];
    real *d_sum;


    cudaMalloc((void**)&d_arr, byte_size);
    cudaMalloc((void **)&d_sum, grid_size * sizeof(real));
    cudaMemcpy(d_arr, h_arr, byte_size, cudaMemcpyHostToDevice);

    //reduce_global<<<grid_size, block_size>>>(d_arr, d_sum);

    reduce_static<<<grid_size, block_size>>>(d_arr, d_sum, N);

    cudaMemcpy(h_sum, d_sum, grid_size * sizeof(real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    
    for(int i = 0; i < grid_size; i++) {
        printf("sum[%d] = %f\n", i, h_sum[i]);
    }

}