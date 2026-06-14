#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

struct __align__(8) MD {
    float m;    // 最大值
    float d;    // 归一化因子(分母)
};

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
    // 32个bit位，为1表示改线程参与 shuffle
    unsigned int mask = 0xffffffff;

#pragma unroll // 循环展开优化
    for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        // 暂存来自其它线程的 m和d
        /*
            逐步合并两个区间 计算safe softmax
            1. 两个区间都有一个max, 要向bigger的区间进行缩放
            2. 原来是max1计算的o =  (e^(x-max1)), 现在合并区间max是max的话，t=(e^(x-max2)), 那么要有一个缩放因子(e^(max1-max2))
        */

        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value;

        value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
        value.m = bigger_m.m;
    } 
    
    /* 
    返回当前 warp 32个线程中的最大值 和 归一化的分母
    当前warp的最大值作用：
        1. 可以与别的warp的区间进行合并时，用于缩放 
        2. 计算全局最大值
    */
    return value; 
}

/*
@param x: 输入数据指针
@param y: 输出数据指针
@param N: 当前 token 的元素数量
*/
template<const int NUM_THREADS = 256>
__global__ void online_safe_softmax_f32_per_token_kernel(const float *x, float *y, int N) {
    int local_tid = threadIdx.x;
    // 每个block的线程数量，就是
    int  global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;

    const int WARP_NUM = NUM_THREADS / WARP_SIZE;
    int warp_id = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;

    MD val;
    // 准备 好数据，进行warp reduce
    val.m = global_tid < N ? x[global_tid] : -FLT_MAX;
    val.d = global_tid < N ? 1.0f : 0.0; // 在范围内的设置为1.0 后面要进行缩放，不影响分母的值；不在范围内的设置为0， 怎么计算都是还是0 不影响分母

    // 定义一个 shared memory 存储每个 warp计算的值
    __shared__ MD shared[WARP_NUM];

    // 第一级 Warp Reduce
    MD res = warp_reduce_md_op(val);

    // 每个warp中第0个线程的结果才是我们想要的
    if(lane_id == 0) {
        shared[warp_id] = res;
    }
    __syncthreads();

    // 第二级 Block Reduce，对每个warp的值规约
    // 在第一个warp 内对共享内存中的MD 值进行规约, 前面每个warp算的值
    /*
        warp_reduce_md_op 是 32个线程算
        shared 元素个数是 warp的个数，且必定小于等于32， 一个block 最多1024个线程
        1. 主要要考虑  元素个数小于32个情况, lane_id 超过 WARP_NUM时，设置一个不影响softmax的默认值{-FLT_MAX, 0.0f}
        2. 不必考虑元素个数大于32的情况
    */
    if(local_tid < WARP_SIZE) {
        MD block_res = (lane_id < WARP_NUM) ? shared[lane_id] : MD{-FLT_MAX, 0.0f};
        block_res = warp_reduce_md_op<WARP_NUM>(block_res);

        if(lane_id == 0) {
            // 把 当前block的结果存储到 shared[0] 中， 作为最终结果
            shared[0] = block_res;
        }
    }

    __syncthreads();

    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);

    if(global_tid < N) {
        y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
    }
}