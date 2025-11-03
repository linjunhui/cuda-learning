#include <iostream>
#include <vector>
#include <algorithm>

/*
直观对比Lambda表达式与传统函数的等价性
*/

// 传统函数定义
int multiplyByTwo(int x) {
    return x * 2;
}

bool isGreaterThanFive(int x) {
    return x > 5;
}

void printWithPrefix(int x) {
    std::cout << "Number: " << x << std::endl;
}

int main() {
    std::cout << "=== Lambda表达式就是匿名函数 ===" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 1. 完全等价的两种写法
    std::cout << "\n1. 完全等价的两种写法:" << std::endl;
    
    // 方式1：使用传统函数
    std::cout << "使用传统函数:" << std::endl;
    std::cout << "multiplyByTwo(5) = " << multiplyByTwo(5) << std::endl;
    
    // 方式2：使用Lambda表达式（匿名函数）
    std::cout << "使用Lambda表达式:" << std::endl;
    std::cout << "[](int x) { return x * 2; }(5) = " 
              << [](int x) { return x * 2; }(5) << std::endl;
    
    // 2. Lambda表达式可以赋值给变量，就像函数一样
    std::cout << "\n2. Lambda可以像函数一样使用:" << std::endl;
    
    // 定义lambda并赋值给变量
    auto lambdaMultiply = [](int x) { return x * 2; };
    auto lambdaIsGreater = [](int x) { return x > 5; };
    auto lambdaPrint = [](int x) { std::cout << "Number: " << x << std::endl; };
    
    // 像调用函数一样调用lambda
    std::cout << "lambdaMultiply(7) = " << lambdaMultiply(7) << std::endl;
    std::cout << "lambdaIsGreater(8) = " << lambdaIsGreater(8) << std::endl;
    std::cout << "lambdaPrint(42): ";
    lambdaPrint(42);
    
    // 3. 在算法中使用 - 两种方式完全等价
    std::cout << "\n3. 在STL算法中使用:" << std::endl;
    
    // 使用传统函数
    std::cout << "使用传统函数遍历:" << std::endl;
    std::for_each(numbers.begin(), numbers.end(), printWithPrefix);
    
    // 使用Lambda表达式
    std::cout << "使用Lambda表达式遍历:" << std::endl;
    std::for_each(numbers.begin(), numbers.end(), 
                  [](int x) { std::cout << "Number: " << x << std::endl; });
    
    // 4. 过滤操作对比
    std::cout << "\n4. 过滤操作对比:" << std::endl;
    
    std::vector<int> result1, result2;
    
    // 传统方式
    std::copy_if(numbers.begin(), numbers.end(), std::back_inserter(result1), isGreaterThanFive);
    
    // Lambda方式
    std::copy_if(numbers.begin(), numbers.end(), std::back_inserter(result2),
                 [](int x) { return x > 5; });
    
    std::cout << "传统函数过滤结果: ";
    for(int x : result1) std::cout << x << " ";
    std::cout << std::endl;
    
    std::cout << "Lambda过滤结果: ";
    for(int x : result2) std::cout << x << " ";
    std::cout << std::endl;
    
    // 5. 变换操作对比
    std::cout << "\n5. 变换操作对比:" << std::endl;
    
    std::vector<int> transformed1, transformed2;
    
    // 传统方式
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(transformed1), multiplyByTwo);
    
    // Lambda方式
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(transformed2),
                   [](int x) { return x * 2; });
    
    std::cout << "传统函数变换结果: ";
    for(int x : transformed1) std::cout << x << " ";
    std::cout << std::endl;
    
    std::cout << "Lambda变换结果: ";
    for(int x : transformed2) std::cout << x << " ";
    std::cout << std::endl;
    
    // 6. 复杂Lambda示例 - 相当于复杂函数
    std::cout << "\n6. 复杂Lambda示例:" << std::endl;
    
    // 这个lambda相当于一个复杂的函数
    auto complexLambda = [](int x) {
        if (x % 2 == 0) {
            return x * x;  // 偶数返回平方
        } else {
            return x * 3;  // 奇数返回3倍
        }
    };
    
    std::cout << "复杂Lambda处理结果:" << std::endl;
    for(int x : numbers) {
        std::cout << x << " -> " << complexLambda(x) << std::endl;
    }
    
    // 7. Lambda的语法分解
    std::cout << "\n7. Lambda语法分解:" << std::endl;
    std::cout << "Lambda语法: [捕获列表](参数列表) -> 返回类型 { 函数体 }" << std::endl;
    std::cout << "例如: [](int x) -> int { return x * 2; }" << std::endl;
    std::cout << "  []         - 捕获列表（空，不捕获外部变量）" << std::endl;
    std::cout << "  (int x)    - 参数列表" << std::endl;
    std::cout << "  -> int     - 返回类型（可省略，自动推导）" << std::endl;
    std::cout << "  { return x * 2; } - 函数体" << std::endl;
    
    return 0;
}






















