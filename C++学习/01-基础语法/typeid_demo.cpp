#include <iostream>
#include <typeinfo>
#include <string>

auto add(auto a, auto b) -> decltype(a + b) {
    return a + b;
}

int main() {
    std::cout << "=== typeid().name() 输出演示 ===" << std::endl;
    
    // 基本类型
    auto a = 1;           // int
    auto b = 2.0;         // double
    auto c = 3.14f;       // float
    auto d = 'A';         // char
    auto e = true;        // bool
    
    std::cout << "基本类型演示：" << std::endl;
    std::cout << "auto a = 1;           // typeid: " << typeid(a).name() << std::endl;
    std::cout << "auto b = 2.0;         // typeid: " << typeid(b).name() << std::endl;
    std::cout << "auto c = 3.14f;       // typeid: " << typeid(c).name() << std::endl;
    std::cout << "auto d = 'A';         // typeid: " << typeid(d).name() << std::endl;
    std::cout << "auto e = true;        // typeid: " << typeid(e).name() << std::endl;
    
    std::cout << "\n类型推导演示：" << std::endl;
    
    // 类型推导示例
    auto result1 = add(1, 2);        // int + int = int
    auto result2 = add(1, 2.0);      // int + double = double
    auto result3 = add(1.0, 2.0);    // double + double = double
    auto result4 = add(1.0f, 2.0f);  // float + float = float
    
    std::cout << "add(1, 2) = " << result1 << "     // typeid: " << typeid(result1).name() << std::endl;
    std::cout << "add(1, 2.0) = " << result2 << "   // typeid: " << typeid(result2).name() << std::endl;
    std::cout << "add(1.0, 2.0) = " << result3 << " // typeid: " << typeid(result3).name() << std::endl;
    std::cout << "add(1.0f, 2.0f) = " << result4 << " // typeid: " << typeid(result4).name() << std::endl;
    
    std::cout << "\n类型转换规则演示：" << std::endl;
    
    // 展示类型转换规则
    std::cout << "int + int = " << typeid(decltype(1 + 2)).name() << std::endl;
    std::cout << "int + double = " << typeid(decltype(1 + 2.0)).name() << std::endl;
    std::cout << "double + double = " << typeid(decltype(1.0 + 2.0)).name() << std::endl;
    std::cout << "float + float = " << typeid(decltype(1.0f + 2.0f)).name() << std::endl;
    std::cout << "int + float = " << typeid(decltype(1 + 2.0f)).name() << std::endl;
    
    std::cout << "\n=== 解释 ===" << std::endl;
    std::cout << "字母 'd' 表示 double 类型" << std::endl;
    std::cout << "这是因为 int + double 的结果类型是 double" << std::endl;
    std::cout << "C++的类型转换规则：较小精度类型会转换为较大精度类型" << std::endl;
    
    return 0;
}























