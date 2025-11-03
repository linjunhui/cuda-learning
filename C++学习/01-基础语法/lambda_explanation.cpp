#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

/*
演示lambda表达式与函数的关系
lambda本质上就是匿名函数的定义
*/

// 传统函数定义
int square(int x) {
    return x * x;
}

bool isEven(int x) {
    return x % 2 == 0;
}

void printNumber(int x) {
    std::cout << x << " ";
}

int main() {
    std::cout << "=== Lambda表达式与函数的关系 ===" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 1. 传统函数 vs Lambda表达式
    std::cout << "\n1. 传统函数 vs Lambda表达式对比:" << std::endl;
    
    // 传统函数调用
    std::cout << "传统函数 square(5) = " << square(5) << std::endl;
    
    // Lambda表达式 - 相当于定义了一个匿名函数
    auto lambdaSquare = [](int x) { return x * x; };
    std::cout << "Lambda square(5) = " << lambdaSquare(5) << std::endl;
    
    // 2. Lambda表达式的完整语法
    std::cout << "\n2. Lambda表达式语法分析:" << std::endl;
    
    // [捕获列表](参数列表) -> 返回类型 { 函数体 }
    auto lambda1 = [](int x) -> int { return x * x; };  // 显式返回类型
    auto lambda2 = [](int x) { return x * x; };         // 自动推导返回类型
    auto lambda3 = [](auto x) { return x * x; };        // 泛型lambda (C++14)
    
    std::cout << "lambda1(3) = " << lambda1(3) << std::endl;
    std::cout << "lambda2(3) = " << lambda2(3) << std::endl;
    std::cout << "lambda3(3) = " << lambda3(3) << std::endl;
    
    // 3. 捕获列表 - lambda可以访问外部变量
    std::cout << "\n3. 捕获列表演示:" << std::endl;
    
    int multiplier = 10;
    
    // 值捕获 - 复制外部变量
    auto lambdaByValue = [multiplier](int x) { return x * multiplier; };
    
    // 引用捕获 - 直接访问外部变量
    auto lambdaByRef = [&multiplier](int x) { return x * multiplier; };
    
    std::cout << "值捕获: lambdaByValue(5) = " << lambdaByValue(5) << std::endl;
    std::cout << "引用捕获: lambdaByRef(5) = " << lambdaByRef(5) << std::endl;
    
    // 修改外部变量
    multiplier = 20;
    std::cout << "修改multiplier后:" << std::endl;
    std::cout << "值捕获: lambdaByValue(5) = " << lambdaByValue(5) << " (不变)" << std::endl;
    std::cout << "引用捕获: lambdaByRef(5) = " << lambdaByRef(5) << " (改变)" << std::endl;
    
    // 4. 与STL算法结合使用
    std::cout << "\n4. 与STL算法结合使用:" << std::endl;
    
    // 传统方式 - 使用函数指针
    std::cout << "使用传统函数:" << std::endl;
    std::for_each(numbers.begin(), numbers.end(), printNumber);
    std::cout << std::endl;
    
    // Lambda方式 - 内联定义函数
    std::cout << "使用Lambda表达式:" << std::endl;
    std::for_each(numbers.begin(), numbers.end(), [](int x) { std::cout << x << " "; });
    std::cout << std::endl;
    
    // 5. 复杂Lambda示例
    std::cout << "\n5. 复杂Lambda示例:" << std::endl;
    
    // 计算平方并过滤偶数
    std::vector<int> squares;
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(squares),
                   [](int x) { return x * x; });  // 计算平方的lambda
    
    std::cout << "平方数: ";
    for(int x : squares) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    // 过滤偶数
    std::vector<int> evens;
    std::copy_if(numbers.begin(), numbers.end(), std::back_inserter(evens),
                 [](int x) { return x % 2 == 0; });  // 判断偶数的lambda
    
    std::cout << "偶数: ";
    for(int x : evens) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    // 6. Lambda的多种形式
    std::cout << "\n6. Lambda的多种形式:" << std::endl;
    
    // 无参数lambda
    auto getCurrentTime = []() { return time(nullptr); };
    std::cout << "当前时间: " << getCurrentTime() << std::endl;
    
    // 多参数lambda
    auto add = [](int a, int b) { return a + b; };
    std::cout << "add(3, 4) = " << add(3, 4) << std::endl;
    
    // 捕获多个变量
    int a = 10, b = 20;
    auto complexLambda = [a, &b](int x) { 
        b = a + x;  // 修改b
        return a * x; 
    };
    std::cout << "complexLambda(5) = " << complexLambda(5) << std::endl;
    std::cout << "修改后的b = " << b << std::endl;
    
    // 7. Lambda与函数指针的等价性
    std::cout << "\n7. Lambda与函数指针的等价性:" << std::endl;
    
    // 函数指针
    int (*funcPtr)(int) = square;
    std::cout << "函数指针: funcPtr(6) = " << funcPtr(6) << std::endl;
    
    // Lambda（可以赋值给函数指针，如果捕获列表为空）
    int (*lambdaPtr)(int) = [](int x) { return x * x; };
    std::cout << "Lambda指针: lambdaPtr(6) = " << lambdaPtr(6) << std::endl;
    
    return 0;
}






















