#include <cstdio>
#include <chrono>  // 👈 新增头文件

template<size_t N>
float cpuCal(float (&vec)[N]) {
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        sum += vec[i] * vec[i];
    }
    return sum;
}

int main() {
    const int N = 1000000;
    float arr[N];
    for(int i = 0; i < N; i++) {
        arr[i] = i * 1.0f;
    }

    // 👇 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    float sum = cpuCal(arr);

    // 👇 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时（纳秒）
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double time_ms = duration.count() / 1e6; // 转为毫秒

    printf("sum = %f\n", sum);
    printf("CPU time: %.6f ms\n", time_ms);

    return 0;
}