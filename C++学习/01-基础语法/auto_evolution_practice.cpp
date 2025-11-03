/*
 * auto关键字演进实践代码
 * 演示从C++11到C++20的auto语法变化
 * 
 * 编译选项：
 * C++11: g++ -std=c++11 auto_evolution_practice.cpp
 * C++14: g++ -std=c++14 auto_evolution_practice.cpp  
 * C++17: g++ -std=c++17 auto_evolution_practice.cpp
 * C++20: g++ -std=c++20 auto_evolution_practice.cpp
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <utility>

// ============================================================================
// C++11: 变量类型推导 + 尾置返回类型
// ============================================================================

// C++11: 尾置返回类型
template<typename T, typename U>
auto add_cpp11(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++11: 具体类型函数的尾置返回类型
auto square_cpp11(int x) -> int {
    return x * x;
}

void demonstrate_cpp11() {
    std::cout << "=== C++11 特性演示 ===" << std::endl;
    
    // 变量类型推导
    auto x = 42;                    // int
    auto y = 3.14;                  // double
    auto name = std::string("Hello"); // std::string
    auto numbers = std::vector<int>{1, 2, 3, 4, 5}; // std::vector<int>
    
    // 复杂类型推导
    auto data = std::map<std::string, int>{
        {"apple", 5}, {"banana", 3}, {"orange", 8}
    };
    
    std::cout << "变量推导：" << std::endl;
    std::cout << "x = " << x << " (int)" << std::endl;
    std::cout << "y = " << y << " (double)" << std::endl;
    std::cout << "name = " << name << " (std::string)" << std::endl;
    std::cout << "numbers size = " << numbers.size() << " (std::vector<int>)" << std::endl;
    std::cout << "data size = " << data.size() << " (std::map<std::string, int>)" << std::endl;
    
    // 函数调用
    auto result1 = add_cpp11(5, 3.14);     // double
    auto result2 = square_cpp11(7);         // int
    
    std::cout << "函数调用：" << std::endl;
    std::cout << "add_cpp11(5, 3.14) = " << result1 << std::endl;
    std::cout << "square_cpp11(7) = " << result2 << std::endl;
    
    std::cout << std::endl;
}

// ============================================================================
// C++14: 省略尾置返回类型
// ============================================================================

// C++14: 省略返回类型
template<typename T, typename U>
auto multiply_cpp14(T a, U b) {
    return a * b;
}

// C++14: 具体类型函数省略返回类型
auto cube_cpp14(int x) {
    return x * x * x;
}

// C++14: 复杂返回类型推导
auto create_vector_cpp14() {
    return std::vector<int>{1, 2, 3, 4, 5};
}

void demonstrate_cpp14() {
    std::cout << "=== C++14 特性演示 ===" << std::endl;
    
    // 函数调用
    auto result1 = multiply_cpp14(4, 2.5);  // double
    auto result2 = cube_cpp14(3);            // int
    auto numbers = create_vector_cpp14();    // std::vector<int>
    
    std::cout << "函数调用：" << std::endl;
    std::cout << "multiply_cpp14(4, 2.5) = " << result1 << std::endl;
    std::cout << "cube_cpp14(3) = " << result2 << std::endl;
    std::cout << "create_vector_cpp14() size = " << numbers.size() << std::endl;
    
    std::cout << std::endl;
}

// ============================================================================
// C++17: 结构化绑定
// ============================================================================

// C++17: 结构化绑定
std::pair<int, std::string> get_pair_cpp17() {
    return {42, "hello"};
}

std::tuple<int, double, std::string> get_tuple_cpp17() {
    return {10, 3.14, "world"};
}

struct Point {
    int x, y;
    Point(int x, int y) : x(x), y(y) {}
};

Point get_point_cpp17() {
    return {5, 10};
}

void demonstrate_cpp17() {
    std::cout << "=== C++17 特性演示 ===" << std::endl;
    
    // 结构化绑定
    auto [number, text] = get_pair_cpp17();
    auto [a, b, c] = get_tuple_cpp17();
    auto [x, y] = get_point_cpp17();
    
    std::cout << "结构化绑定：" << std::endl;
    std::cout << "pair: " << number << ", " << text << std::endl;
    std::cout << "tuple: " << a << ", " << b << ", " << c << std::endl;
    std::cout << "point: (" << x << ", " << y << ")" << std::endl;
    
    // 数组解构
    int arr[] = {1, 2, 3, 4, 5};
    auto [first, second, third, fourth, fifth] = arr;
    
    std::cout << "数组解构：" << std::endl;
    std::cout << "arr[0-4]: " << first << ", " << second << ", " 
              << third << ", " << fourth << ", " << fifth << std::endl;
    
    std::cout << std::endl;
}

// ============================================================================
// C++20: 函数参数中的 auto
// ============================================================================

// C++20: 参数中的 auto
auto add_cpp20(auto a, auto b) {
    return a + b;
}

// C++20: 混合使用
auto process_cpp20(auto data, int count) {
    return data * count;
}

// C++20: 复杂函数
auto create_map_cpp20(auto key, auto value) {
    return std::map<decltype(key), decltype(value)>{{key, value}};
}

void demonstrate_cpp20() {
    std::cout << "=== C++20 特性演示 ===" << std::endl;
    
    // 函数调用
    auto result1 = add_cpp20(5, 3.14);      // double
    auto result2 = add_cpp20(std::string("Hello"), std::string(" World")); // std::string
    auto result3 = process_cpp20(2.5, 4);   // double
    
    std::cout << "函数调用：" << std::endl;
    std::cout << "add_cpp20(5, 3.14) = " << result1 << std::endl;
    std::cout << "add_cpp20(string, string) = " << result2 << std::endl;
    std::cout << "process_cpp20(2.5, 4) = " << result3 << std::endl;
    
    // 复杂类型推导
    auto data_map = create_map_cpp20(std::string("key"), 42);
    std::cout << "create_map_cpp20 size = " << data_map.size() << std::endl;
    
    std::cout << std::endl;
}

// ============================================================================
// 综合对比演示
// ============================================================================

void demonstrate_comparison() {
    std::cout << "=== 语法对比演示 ===" << std::endl;
    
    // 相同的功能，不同的语法
    int a = 5, b = 3;
    
    // C++11 风格
    auto result11 = add_cpp11(a, b);
    
    // C++14 风格  
    auto result14 = multiply_cpp14(a, b);
    
    // C++20 风格
    auto result20 = add_cpp20(a, b);
    
    std::cout << "相同功能的不同语法：" << std::endl;
    std::cout << "C++11: " << result11 << std::endl;
    std::cout << "C++14: " << result14 << std::endl;
    std::cout << "C++20: " << result20 << std::endl;
    
    std::cout << std::endl;
}

// ============================================================================
// 性能测试
// ============================================================================

#include <chrono>

auto benchmark_auto_vs_explicit() {
    std::cout << "=== 性能对比测试 ===" << std::endl;
    
    const int iterations = 1000000;
    std::vector<int> data(iterations, 42);
    
    // 使用 auto
    auto start = std::chrono::high_resolution_clock::now();
    auto sum_auto = 0;
    for (auto item : data) {
        sum_auto += item;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto auto_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 使用显式类型
    start = std::chrono::high_resolution_clock::now();
    int sum_explicit = 0;
    for (int item : data) {
        sum_explicit += item;
    }
    end = std::chrono::high_resolution_clock::now();
    auto explicit_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "auto 版本耗时: " << auto_time.count() << " 微秒" << std::endl;
    std::cout << "显式类型版本耗时: " << explicit_time.count() << " 微秒" << std::endl;
    std::cout << "性能差异: " << (auto_time.count() - explicit_time.count()) << " 微秒" << std::endl;
    
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "C++ auto 关键字演进实践演示" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << std::endl;
    
    // 演示各个版本的特性
    demonstrate_cpp11();
    demonstrate_cpp14();
    demonstrate_cpp17();
    demonstrate_cpp20();
    
    // 综合对比
    demonstrate_comparison();
    
    // 性能测试
    benchmark_auto_vs_explicit();
    
    std::cout << "=== 总结 ===" << std::endl;
    std::cout << "C++11: 引入 auto 变量推导和尾置返回类型" << std::endl;
    std::cout << "C++14: 允许省略尾置返回类型" << std::endl;
    std::cout << "C++17: 支持结构化绑定" << std::endl;
    std::cout << "C++20: 支持函数参数中的 auto" << std::endl;
    
    return 0;
}






















