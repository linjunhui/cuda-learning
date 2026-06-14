#include <cmath>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include "common.h"



template<int BLOCK_SIZE=1024, int WARP_SIZE=32>
__global__ void online_safe_softmax_kernel(const float *x, float *output, int D) {
    int row_idx = blockIdx.x;
    int element_idx_in_row = threadIdx.x;
    int lane_idx = element_idx_in_row % WARP_SIZE;
    int warp_idx = element_idx_in_row / WARP_SIZE;

    const int WARP_NUM = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ MD shared_warp_md[WARP_NUM];

    MD thread_md = {0, -FLT_MAX};
#pragma unroll
    for(int i = element_idx_in_row; i < D; i += BLOCK_SIZE) {
        /*
            流程：
                1. 找max_value
        */
       int element_idx_in_x = row_idx * D + i;
       float new_max = thread_md.max_value > x[element_idx_in_x] ? thread_md.max_value : x[element_idx_in_x];
       MD current_md = {1, x[element_idx_in_x]};
       
       
       //thread_md.sum_exp = thread_md.sum_exp * __expf(thread_md.max_value - new_max) + __expf(x[i] - new_max);
       thread_md.sum_exp = thread_md.sum_exp * __expf(thread_md.max_value - new_max) + current_md.sum_exp * __expf(current_md.max_value - new_max);     
       
       thread_md.max_value = new_max; // 先计算再更新
    }

    // 所有线程都要 做warp reduce 但是只取lane id = 0的结果
    MD warp_md = warp_reduce_kernel(thread_md);
    if(lane_idx == 0) {
        shared_warp_md[warp_idx] = warp_md;
    }
    __syncthreads();

    /* 
        每个warp计算完毕， 对warp结果汇总计算
        边界问题：warp_num 不一定大于 是可能小于32的，取决于 D和Block_SIZE
        一个 WARP reduce是32个线程一起操作(除非改mask)，所以这里要考虑超出WARP_NUM的数据如何处理

    */

    MD final_lane_md = {0.0f, -FLT_MAX};

    if(warp_idx == 0) {
        final_lane_md = lane_idx < WARP_NUM ? shared_warp_md[lane_idx] : final_lane_md;
        final_lane_md = warp_reduce_kernel(final_lane_md);
        if(lane_idx == 0) {
            shared_warp_md[0] = final_lane_md;
        }
    }
    
    __syncthreads(); // 这里要做一个同步，否则 final_md 是拿到的还没更新的 shared_warp_md[0]
    MD final_md = shared_warp_md[0];

    // 开始计算每个元素的 softmax
    float inv_sum = 1.0f / final_md.sum_exp;
    float max_value = final_md.max_value;
#pragma unroll
    for(int i = element_idx_in_row; i < D; i += BLOCK_SIZE) {
        int element_idx_in_x = row_idx * D + i;
        output[element_idx_in_x] = inv_sum * __expf(x[element_idx_in_x] - max_value);
    }

}

torch::Tensor dispatch_softmax(torch::Tensor input) {

    auto input_contig = input.contiguous();
    // 获取维度信息
    const int N = input_contig.size(0);
    const int D = input_contig.size(1);

    // 使用torch来初始化，不用自己写cudaMalloc
    auto output = torch::empty_like(input_contig);

    int block_size = 1024;
    int grid_size = N;

    online_safe_softmax_kernel<1024, 32><<<grid_size, block_size>>>(input_contig.data_ptr<float>(), output.data_ptr<float>(), D);

    return output;
}