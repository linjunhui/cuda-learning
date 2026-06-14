#include <cstdio>
#include <cstdlib>
#include "vector_add.cuh"

int main() {
    float *h_input1, *h_input2, *h_output;
    int64_t N = 1000;
    int64_t M_SIZE = sizeof(int) * N;

    h_input1 = (float *)malloc(M_SIZE);
    h_input2 = (float *)malloc(M_SIZE);
    h_output = (float *)malloc(M_SIZE);

    // 数据初始化
    for(int i = 0; i < N; i++) {
        h_input1[i] = 1.0f * i;
        h_input2[i] = 2.0f * i;
    }

    launch_kernel(h_input1, h_input2, h_output, N, M_SIZE);

    for(int i = 0; i < 100; i++) {
        printf("h_output[%d] = %lf\n", i, h_output[i]);
    }

}