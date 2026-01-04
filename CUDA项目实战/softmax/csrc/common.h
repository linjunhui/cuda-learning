#ifndef COMMON_H
#define COMMON_H

const int WARP_SIZE = 32;

struct __align__(8) MD {
    float sum_exp;
    float max_value;
};


static __device__ __forceinline__  MD warp_reduce_kernel(MD md) {
    // 1. mask， 第一个控制参与的线程, 一个warp 32个线程，这里的mask 一个bit位表示一个线程
    unsigned int mask = 0xffffffff;
    /*
        当前线程 id: 0x0
        offset: 0b00010000 = 0x0010  16

    */
    MD other;
#pragma unroll
    for(int offset = 0x10; offset > 0; offset >>= 1) {
        other.sum_exp = __shfl_xor_sync(mask, md.sum_exp, offset);
        other.max_value = __shfl_xor_sync(mask, md.max_value, offset);

        float new_max = other.max_value > md.max_value ? other.max_value : md.max_value;

        md.sum_exp = md.sum_exp * __expf(md.max_value - new_max) + other.sum_exp * __expf(other.max_value - new_max);
        md.max_value = new_max;
    }
    return md;
}

#endif