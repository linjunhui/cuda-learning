/*
题目 7：内存层次结构
知识点：GPU 内存层次

题目描述：
1. 求解 一组 向量 元素的平方和
*/

#include<cstdio>


__global__ reduceGlobalMem() {
    
}

int main() {
    float arr[10] = {1.0, 2.0, };

    float sum = cpuCal(arr);
    printf("sum = %f\n", sum);

    return 0;
}