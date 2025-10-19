#include <chrono>           // C++ 标准库中的高精度时间测量功能
#include <driver_types.h>   // CUDA 驱动类型定义（如 cudaEvent_t）
#include <string>           // 使用 std::string 字符串类
#include <cuda_runtime.h>   // CUDA 运行时 API（用于 GPU 时间测量）

// 定义一个 Timer 类，用于测量 CPU 和 GPU 的执行时间
class Timer {
public:
    // 使用 using 给常用的单位比例起别名，用于 CPU 时间的模板参数
    using s  = std::ratio<1, 1>;            // 秒（second）
    using ms = std::ratio<1, 1000>;         // 毫秒（millisecond）
    using us = std::ratio<1, 1000000>;      // 微秒（microsecond）
    using ns = std::ratio<1, 1000000000>;   // 纳秒（nanosecond）

public:
    // 构造函数和析构函数声明
    Timer();
    ~Timer();

public:
    // 成员变量：记录 CPU 开始和结束时间点
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStart; // CPU 开始时间点
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;  // CPU 结束时间点

    // 成员变量：记录 GPU 开始和结束事件
    cudaEvent_t _cudaStart, _cudaStop;  // 用于 GPU 时间测量的两个 CUDA 事件

    float _timeElapsed;  // 存储 GPU 耗时（毫秒）

public:
    // 启动/停止 CPU 计时器
    void start_cpu();
    void stop_cpu();

    // 启动/停止 GPU 计时器
    void start_gpu();
    void stop_gpu();

    // 模板方法：输出 CPU 耗时（支持不同单位）
    template <typename span>
    void duration_cpu(std::string msg);

    // 输出 GPU 耗时（固定为毫秒）
    void duration_gpu(std::string msg);
};

// 构造函数实现：初始化 CPU 时间点和 GPU 事件
Timer::Timer() {
    _timeElapsed = 0.0f;

    // 初始化 CPU 时间点为当前时刻
    _cStart = std::chrono::high_resolution_clock::now();
    _cStop = std::chrono::high_resolution_clock::now();

    // 创建两个 CUDA 事件，用于 GPU 时间测量
    cudaEventCreate(&_cudaStart);
    cudaEventCreate(&_cudaStop);
}

// 析构函数实现：目前为空，但可以在这里释放资源
Timer::~Timer() {
}

// 开始 CPU 计时：记录当前时间点
void Timer::start_cpu() {
    _cStart = std::chrono::high_resolution_clock::now();
}

// 停止 CPU 计时：记录当前时间点
void Timer::stop_cpu() {
    _cStop = std::chrono::high_resolution_clock::now();
}

// 模板函数 duration_cpu：
// 功能：根据模板参数 span（单位）计算并输出 CPU 耗时
template <typename span>
void Timer::duration_cpu(std::string msg) {
    std::string str;

    // 判断模板参数类型，并设置对应的单位字符串
    if (std::is_same<span, s>::value) {
        str = "s";  // 秒
    } else if (std::is_same<span, ms>::value) {
        str = "ms"; // 毫秒
    } else if (std::is_same<span, us>::value) {
        str = "us"; // 微秒
    } else if (std::is_same<span, ns>::value) {
        str = "ns"; // 纳秒
    }

    // 将开始和结束时间点之间的差值转换为指定单位的时间间隔
    std::chrono::duration<double, span> elapsed = _cStop - _cStart;

    // 打印格式化结果：消息 + 时间 + 单位
    printf("%-40s uses %.6lf %s\n", msg.c_str(), elapsed.count(), str.c_str());
}

// 开始 GPU 计时：在默认流中记录开始事件
void Timer::start_gpu() {
    cudaEventRecord(_cudaStart, 0);  // 0 表示默认流（null stream）
}

// 停止 GPU 计时：在默认流中记录结束事件
void Timer::stop_gpu() {
    cudaEventRecord(_cudaStop, 0);   // 0 表示默认流
}

// 输出 GPU 耗时：使用 cudaEventElapsedTime 获取两个事件之间经过的时间（单位：毫秒）
void Timer::duration_gpu(std::string msg) {
    // 计算两个事件之间的耗时（单位：毫秒）
    cudaEventElapsedTime(&_timeElapsed, _cudaStart, _cudaStop);

    // 打印格式化结果：消息 + 时间（毫秒）
    printf("%-40s uses %.6lf ms\n", msg.c_str(), _timeElapsed);
}