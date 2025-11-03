#include <iostream>

/*
编写程序实现以下功能：
1. 使用auto关键字声明不同类型的变量
2. 使用auto推导函数返回类型
3. 使用decltype推导表达式类型
*/

auto add(auto a, auto b) -> decltype(a + b) {
    return a + b;
}

int main() {
    auto a = 1;
    auto b = 2.0;
    auto c = add(a, b);
    std::cout << c << std::endl;
    // 打印c的类型
    std::cout << "c的类型是: " << typeid(c).name() << std::endl;
    std::cout << "a的类型是: " << typeid(a).name() << std::endl;
    std::cout << "b的类型是: " << typeid(b).name() << std::endl;
    std::cout << "add函数的返回类型是: " << typeid(add(a, b)).name() << std::endl;
    
    // 正确的方式：使用typeid获取decltype推导的类型
    decltype(add(a, b)) result = add(a, b);
    std::cout << "使用decltype推导的类型: " << typeid(result).name() << std::endl;
    std::cout << "decltype推导的结果值: " << result << std::endl;
    return 0;
}