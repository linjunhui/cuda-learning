#include <iostream>
#include <string>

/*
const vs constexpr 简洁对比
*/

// constexpr 函数
constexpr int square(int x) {
    return x * x;
}

constexpr double circleArea(double radius) {
    constexpr double pi = 3.14159;
    return pi * radius * radius;
}

// const 函数
int getRuntimeValue() {
    return 42;
}

// constexpr 类
class MathConstants {
public:
    static constexpr double PI = 3.14159;
    static constexpr double E = 2.71828;
    static constexpr int MAX_SIZE = 1000;
};

// const 类
class AppConfig {
private:
    const std::string name;
    const int version;
    
public:
    AppConfig(const std::string& n, int v) : name(n), version(v) {}
    
    const std::string& getName() const { return name; }
    int getVersion() const { return version; }
};

int main() {
    std::cout << "=== const vs constexpr 对比 ===" << std::endl;
    
    // 1. 变量对比
    std::cout << "\n1. 变量对比:" << std::endl;
    
    // constexpr 变量（编译时确定）
    constexpr int compileTimeValue = 42;
    constexpr double pi = 3.14159;
    
    // const 变量（运行时确定）
    const int runtimeValue = getRuntimeValue();
    const std::string appName = "MyApp";
    
    std::cout << "constexpr 值: " << compileTimeValue << std::endl;
    std::cout << "const 值: " << runtimeValue << std::endl;
    
    // 2. 函数对比
    std::cout << "\n2. 函数对比:" << std::endl;
    
    // constexpr 函数（编译时计算）
    constexpr int result1 = square(5);
    constexpr double area1 = circleArea(3.0);
    
    // const 函数（运行时计算）
    int result2 = square(5);
    double area2 = circleArea(3.0);
    
    std::cout << "constexpr 计算: " << result1 << ", " << area1 << std::endl;
    std::cout << "const 计算: " << result2 << ", " << area2 << std::endl;
    
    // 3. 类对比
    std::cout << "\n3. 类对比:" << std::endl;
    
    // constexpr 类
    std::cout << "MathConstants::PI: " << MathConstants::PI << std::endl;
    std::cout << "MathConstants::E: " << MathConstants::E << std::endl;
    std::cout << "MathConstants::MAX_SIZE: " << MathConstants::MAX_SIZE << std::endl;
    
    // const 类
    AppConfig config("MyApp", 1);
    std::cout << "AppConfig name: " << config.getName() << std::endl;
    std::cout << "AppConfig version: " << config.getVersion() << std::endl;
    
    // 4. 数组大小
    std::cout << "\n4. 数组大小:" << std::endl;
    
    // constexpr 可以用作数组大小
    constexpr int ARRAY_SIZE = 10;
    int array[ARRAY_SIZE];
    std::cout << "数组大小: " << sizeof(array) / sizeof(array[0]) << std::endl;
    
    // const 不能用作数组大小（除非是编译时常量）
    const int size = 5;
    int array2[size];  // 在某些编译器中可能工作，但不是标准
    std::cout << "数组2大小: " << sizeof(array2) / sizeof(array2[0]) << std::endl;
    
    // 5. 性能对比
    std::cout << "\n5. 性能对比:" << std::endl;
    
    // 编译时计算（constexpr）
    constexpr int fact5 = square(5);
    std::cout << "编译时计算 5²: " << fact5 << std::endl;
    
    // 运行时计算（const）
    int fact5_runtime = square(5);
    std::cout << "运行时计算 5²: " << fact5_runtime << std::endl;
    
    // 6. 使用建议
    std::cout << "\n6. 使用建议:" << std::endl;
    std::cout << "✅ 数学常量 → constexpr" << std::endl;
    std::cout << "✅ 数组大小 → constexpr" << std::endl;
    std::cout << "✅ 编译时计算 → constexpr" << std::endl;
    std::cout << "✅ 函数参数 → const" << std::endl;
    std::cout << "✅ 成员函数 → const" << std::endl;
    std::cout << "✅ 运行时常量 → const" << std::endl;
    
    return 0;
}





















