
/*
Fill算子是一个非常简单的算子，它的功能是使用指定的标量值填充整个张量。
Fill算子的实现非常简单，只需要一个核函数，这个核函数会遍历整个张量，将每个元素设置为指定的标量值。

1. 支持多种数据类型（float32, int64, complex 等）

*/

template<typename T>
struct FillFunctor {
    T value;
    FillFunctor(T val): value(val){}

    __device__ __forceinline__ T operator() () {
        return value;
    }
};

/*
1. 用functor 填充 vec
2. grid-stride loop
*/
template<typename T>
__global__ void fill_kernel(T *vec, int64_t N, FillFunctor<T> functor) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; // 全局 线程 id
    // 总线程数：grid 有 gridDim.x 个block, 每个block有 blockDim.x 个线程
    int64_t stride = blockDim.x * gridDim.x;

    for(int64_t i = idx; i < N; i+=stride) {
        vec[i] = functor();
    }
    
}

