#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>
#include <chrono>

/*
演示auto的编译时类型推导机制
证明auto不影响运行时性能
*/

// 函数用于测试类型推导
template<typename T>
void printTypeInfo(const T& value, const std::string& description) {
    std::cout << description << ": " 
              << "类型=" << typeid(T).name() 
              << ", 大小=" << sizeof(T) << "字节"
              << ", 值=" << value << std::endl;
}

int main() {
    std::cout << "=== auto编译时类型推导演示 ===" << std::endl;
    
    // 1. 基本类型推导
    std::cout << "\n1. 基本类型推导:" << std::endl;
    
    auto a = 42;           // 推导为 int
    auto b = 3.14;         // 推导为 double
    auto c = 'A';          // 推导为 char
    auto d = true;         // 推导为 bool
    
    printTypeInfo(a, "auto a = 42");
    printTypeInfo(b, "auto b = 3.14");
    printTypeInfo(c, "auto c = 'A'");
    printTypeInfo(d, "auto d = true");
    
    // 2. 显式类型声明对比
    std::cout << "\n2. 显式类型声明对比:" << std::endl;
    
    int e = 42;
    double f = 3.14;
    char g = 'A';
    bool h = true;
    
    printTypeInfo(e, "int e = 42");
    printTypeInfo(f, "double f = 3.14");
    printTypeInfo(g, "char g = 'A'");
    printTypeInfo(h, "bool h = true");
    
    // 3. 容器类型推导
    std::cout << "\n3. 容器类型推导:" << std::endl;
    
    auto vec = std::vector<int>{1, 2, 3, 4, 5};
    auto str = std::string("Hello");
    
    std::cout << "auto vec: " << typeid(vec).name() << std::endl;
    std::cout << "auto str: " << typeid(str).name() << std::endl;
    
    // 4. for循环中的类型推导
    std::cout << "\n4. for循环中的类型推导:" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    // auto推导 - 编译时确定类型
    std::cout << "auto推导的for循环:" << std::endl;
    for(auto num : numbers) {
        std::cout << "  类型: " << typeid(num).name() 
                  << ", 大小: " << sizeof(num) << "字节"
                  << ", 值: " << num << std::endl;
    }
    
    // 显式类型 - 运行时行为完全相同
    std::cout << "显式类型的for循环:" << std::endl;
    for(int num : numbers) {
        std::cout << "  类型: " << typeid(num).name() 
                  << ", 大小: " << sizeof(num) << "字节"
                  << ", 值: " << num << std::endl;
    }
    
    // 5. 引用类型推导
    std::cout << "\n5. 引用类型推导:" << std::endl;
    
    for(auto& ref : numbers) {
        std::cout << "auto& 类型: " << typeid(ref).name() 
                  << ", 大小: " << sizeof(ref) << "字节" << std::endl;
        break; // 只显示第一个
    }
    
    for(int& ref : numbers) {
        std::cout << "int& 类型: " << typeid(ref).name() 
                  << ", 大小: " << sizeof(ref) << "字节" << std::endl;
        break; // 只显示第一个
    }
    
    // 6. 复杂类型推导
    std::cout << "\n6. 复杂类型推导:" << std::endl;
    
    auto complex_vec = std::vector<std::string>{"hello", "world"};
    auto& first_element = complex_vec[0];
    
    std::cout << "complex_vec 类型: " << typeid(complex_vec).name() << std::endl;
    std::cout << "first_element 类型: " << typeid(first_element).name() << std::endl;
    
    // 7. 性能测试 - 证明运行时性能相同
    std::cout << "\n7. 性能对比测试:" << std::endl;
    
    const int SIZE = 1000000;
    std::vector<int> test_vec(SIZE, 42);
    
    // 测试auto循环性能
    auto start = std::chrono::high_resolution_clock::now();
    int sum1 = 0;
    for(auto num : test_vec) {
        sum1 += num;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto auto_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试显式类型循环性能
    start = std::chrono::high_resolution_clock::now();
    int sum2 = 0;
    for(int num : test_vec) {
        sum2 += num;
    }
    end = std::chrono::high_resolution_clock::now();
    auto explicit_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "auto循环时间: " << auto_time.count() << " 微秒" << std::endl;
    std::cout << "显式类型循环时间: " << explicit_time.count() << " 微秒" << std::endl;
    std::cout << "结果相同: " << (sum1 == sum2 ? "是" : "否") << std::endl;
    
    return 0;
}
