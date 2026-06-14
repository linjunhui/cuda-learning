#include <cmath>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <vector_types.h>
#include "common.h"

template<int BLOCK_SIZE=256, int WARP_SIZE=32>
__global__ void online_safe_softmax_vec_kernel(const float4 *input, float4 *output, int D) {
    int row_idx = blockIdx.x;
    // 向量化版本：每个线程处理 4 个连续元素（1 个 float4）
    // block_size 从 1024 减少到 256，但每个线程处理的数据量不变（都是 D 个 float）
    int vec_idx = threadIdx.x;  // 0 到 255，表示 float4 的索引
    int lane_idx = vec_idx % WARP_SIZE;
    int warp_idx = vec_idx / WARP_SIZE;

    const int WARP_NUM = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ MD shared_warp_md[WARP_NUM];

    MD thread_md = {0, -FLT_MAX};
#pragma unroll
    for(int i = vec_idx; i < D; i += BLOCK_SIZE) {
        /*
            流程：
                1. 找max_value, 但是现在有4个元素如何处理 x, y, z, w
            优化点：
                - 线程 0 访问 float4[0]，线程 1 访问 float4[1]，实现合并访问
                - 每个线程处理 4 个连续元素，减少内存事务
        */
        
       int element_idx_in_x = row_idx * D + i;

       // 将 float4 的4个元素作为 一个区间，计算 分母 和 max_value
       float4 current_input = input[element_idx_in_x];
       // thread_md 的 max_value 与 当前 元素的current_input 的 4个元素比较，找到最大值
       float new_max = thread_md.max_value > current_input.x ? thread_md.max_value : current_input.x;
       new_max = new_max > current_input.y ? new_max : current_input.y;
       new_max = new_max > current_input.z ? new_max : current_input.z;
       new_max = new_max > current_input.w ? new_max : current_input.w;

       // 当前元素的 4个元素 与 thread_md的sum_exp 一起来计算 sum_exp
       float sum_exp = __expf(current_input.x - new_max) + __expf(current_input.y - new_max) + __expf(current_input.z - new_max) + __expf(current_input.w - new_max);
       
       thread_md.sum_exp = thread_md.sum_exp * __expf(thread_md.max_value - new_max) + sum_exp;     
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
    for(int i = vec_idx; i < D; i += BLOCK_SIZE) {
        int element_idx_in_x = row_idx * D + i;
        float4 current_input = input[element_idx_in_x];
        float4 current_output;
        current_output.x = inv_sum * __expf(current_input.x - max_value);
        current_output.y = inv_sum * __expf(current_input.y - max_value);
        current_output.z = inv_sum * __expf(current_input.z - max_value);
        current_output.w = inv_sum * __expf(current_input.w - max_value);
        output[element_idx_in_x] = current_output;
    }

}

torch::Tensor dispatch_softmax_vec(torch::Tensor input) {

    auto input_contig = input.contiguous();
    // 获取维度信息
    const int N = input_contig.size(0);
    const int D = input_contig.size(1);

    // 判断 D 是不是 4的倍数
    TORCH_CHECK(D % 4 == 0, "softmax float4实现， D必须是4的倍数， 但实际维度是 ", D);
    // 使用torch来初始化，不用自己写cudaMalloc
    auto output = torch::empty_like(input_contig);

    const float4 * vec_input = reinterpret_cast<const float4 *>(input_contig.data_ptr<float>());
    float4 * vec_output = reinterpret_cast<float4 *>(output.data_ptr<float>());

    // 向量化优化：block_size 从 1024 减少到 256
    // 因为每个线程现在处理 4 个连续元素（1 个 float4），线程数减少 4 倍
    // 但每个线程处理的数据量不变（都是 D 个 float），所以总工作量相同
    int block_size = 256;
    int grid_size = N;
    int D_vec4 = D / 4;  // float4 的数量

    online_safe_softmax_vec_kernel<256, 32><<<grid_size, block_size>>>(vec_input, vec_output, D_vec4);

    return output;
}