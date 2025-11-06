#include <iostream>

/*
简单证明auto编译时类型推导的程序
*/

int main() {
    // 1. 基本类型推导
    auto a = 42;        // 编译时推导为 int
    int b = 42;         // 显式声明为 int
    
    // 2. 验证类型相同
    std::cout << "auto a 的类型: " << typeid(a).name() << std::endl;
    std::cout << "int b 的类型: " << typeid(b).name() << std::endl;
    std::cout << "类型相同: " << (typeid(a) == typeid(b) ? "是" : "否") << std::endl;
    
    // 3. 验证大小相同
    std::cout << "auto a 的大小: " << sizeof(a) << " 字节" << std::endl;
    std::cout << "int b 的大小: " << sizeof(b) << " 字节" << std::endl;
    
    // 4. 验证地址和值
    std::cout << "auto a 的值: " << a << std::endl;
    std::cout << "int b 的值: " << b << std::endl;
    
    // 5. 简单性能测试
    const int SIZE = 1000000;
    int sum1 = 0, sum2 = 0;
    
    // auto版本
    for(auto i = 0; i < SIZE; ++i) {
        sum1 += i;
    }
    
    // 显式类型版本
    for(int i = 0; i < SIZE; ++i) {
        sum2 += i;
    }
    
    std::cout << "auto版本结果: " << sum1 << std::endl;
    std::cout << "显式版本结果: " << sum2 << std::endl;
    std::cout << "结果相同: " << (sum1 == sum2 ? "是" : "否") << std::endl;
    
    return 0;
}























