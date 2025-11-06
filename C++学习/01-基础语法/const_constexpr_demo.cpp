#include <iostream>
#include <string>
#include <array>

/*
演示const和constexpr的使用
*/

// constexpr 函数：编译时计算
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr double circleArea(double radius) {
    constexpr double pi = 3.14159265359;
    return pi * radius * radius;
}

constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// constexpr 类
class Point {
private:
    int x, y;
    
public:
    constexpr Point(int x, int y) : x(x), y(y) {}
    
    constexpr int getX() const { return x; }
    constexpr int getY() const { return y; }
    constexpr int getDistanceSquared() const { return x * x + y * y; }
};

// const 类
class Config {
private:
    const std::string appName;
    const int version;
    mutable int accessCount;  // mutable 可以在 const 函数中修改
    
public:
    Config(const std::string& name, int ver) 
        : appName(name), version(ver), accessCount(0) {}
    
    // const 成员函数
    const std::string& getAppName() const { 
        accessCount++;  // mutable 成员可以修改
        return appName; 
    }
    
    int getVersion() const { return version; }
    int getAccessCount() const { return accessCount; }
};

// 演示 const 指针
void demonstrateConstPointers() {
    int value = 42;
    int anotherValue = 100;
    
    // 指向常量的指针
    const int* ptr1 = &value;
    std::cout << "指向常量的指针: " << *ptr1 << std::endl;
    // *ptr1 = 50;  // 错误！不能修改指向的值
    ptr1 = &anotherValue;  // 可以改变指针指向
    
    // 常量指针
    int* const ptr2 = &value;
    std::cout << "常量指针: " << *ptr2 << std::endl;
    *ptr2 = 50;  // 可以修改指向的值
    // ptr2 = &anotherValue;  // 错误！不能改变指针指向
    
    // 指向常量的常量指针
    const int* const ptr3 = &value;
    std::cout << "指向常量的常量指针: " << *ptr3 << std::endl;
    // *ptr3 = 60;  // 错误！不能修改指向的值
    // ptr3 = &anotherValue;  // 错误！不能改变指针指向
}

int main() {
    std::cout << "=== const 和 constexpr 演示 ===" << std::endl;
    
    // 1. constexpr 变量
    std::cout << "\n1. constexpr 变量:" << std::endl;
    constexpr int MAX_SIZE = 100;
    constexpr double PI = 3.14159265359;
    constexpr int ARRAY_SIZE = 10;
    
    std::cout << "MAX_SIZE: " << MAX_SIZE << std::endl;
    std::cout << "PI: " << PI << std::endl;
    std::cout << "ARRAY_SIZE: " << ARRAY_SIZE << std::endl;
    
    // 使用 constexpr 变量作为数组大小
    std::array<int, ARRAY_SIZE> arr;
    std::cout << "数组大小: " << arr.size() << std::endl;
    
    // 2. constexpr 函数
    std::cout << "\n2. constexpr 函数:" << std::endl;
    constexpr int fact5 = factorial(5);
    constexpr double area = circleArea(5.0);
    constexpr int fib10 = fibonacci(10);
    
    std::cout << "5! = " << fact5 << std::endl;
    std::cout << "半径5的圆面积: " << area << std::endl;
    std::cout << "斐波那契数列第10项: " << fib10 << std::endl;
    
    // 运行时计算
    int n;
    std::cout << "输入一个数字计算阶乘: ";
    std::cin >> n;
    int runtimeFact = factorial(n);
    std::cout << n << "! = " << runtimeFact << std::endl;
    
    // 3. constexpr 类
    std::cout << "\n3. constexpr 类:" << std::endl;
    constexpr Point p(3, 4);
    constexpr int distance = p.getDistanceSquared();
    
    std::cout << "点坐标: (" << p.getX() << ", " << p.getY() << ")" << std::endl;
    std::cout << "距离平方: " << distance << std::endl;
    
    // 4. const 类
    std::cout << "\n4. const 类:" << std::endl;
    Config config("MyApp", 1);
    std::cout << "应用名称: " << config.getAppName() << std::endl;
    std::cout << "版本: " << config.getVersion() << std::endl;
    std::cout << "访问次数: " << config.getAccessCount() << std::endl;
    
    // 5. const 指针演示
    std::cout << "\n5. const 指针演示:" << std::endl;
    demonstrateConstPointers();
    
    // 6. const 函数参数
    std::cout << "\n6. const 函数参数:" << std::endl;
    auto printValue = [](const int& value) {
        std::cout << "值: " << value << std::endl;
    };
    
    printValue(42);
    printValue(100);
    
    // 7. 性能对比
    std::cout << "\n7. 性能对比:" << std::endl;
    
    // 编译时计算（constexpr）
    constexpr int compileTimeResult = factorial(10);
    std::cout << "编译时计算 10!: " << compileTimeResult << std::endl;
    
    // 运行时计算（const）
    int runtimeResult = factorial(10);
    std::cout << "运行时计算 10!: " << runtimeResult << std::endl;
    
    // 8. 类型信息
    std::cout << "\n8. 类型信息:" << std::endl;
    std::cout << "const int 类型: " << typeid(const int).name() << std::endl;
    std::cout << "int 类型: " << typeid(int).name() << std::endl;
    std::cout << "Point 类型: " << typeid(Point).name() << std::endl;
    std::cout << "const Point 类型: " << typeid(const Point).name() << std::endl;
    
    // 9. 现代C++特性（C++17）
    std::cout << "\n9. 现代C++特性:" << std::endl;
    
    // if constexpr 示例
    auto processValue = [](auto value) {
        if constexpr (std::is_integral_v<decltype(value)>) {
            std::cout << "整数类型: " << value << std::endl;
        } else if constexpr (std::is_floating_point_v<decltype(value)>) {
            std::cout << "浮点类型: " << value << std::endl;
        } else {
            std::cout << "其他类型" << std::endl;
        }
    };
    
    processValue(42);
    processValue(3.14);
    processValue("Hello");
    
    std::cout << "\n演示完成！" << std::endl;
    
    return 0;
}






















