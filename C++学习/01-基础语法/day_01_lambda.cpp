#include <iostream>

/*
编写程序使用lambda表达式实现以下功能：
1. 计算平方
2. 判断奇偶性
3. 字符串比较
*/

int main() {
    int arr[] = {1, 2, 3, 4, 5};

    // 这里不能是int， 这里是auto，返回值是一个函数 closure type 类型
    auto square = [](int x) {return x * x;};

    std::cout <<square(2) <<std::endl;

    // 定义因子
    int factorial = 20;

    auto factorial_lambda = [factorial](int x) {return factorial * x;};
    std::cout <<factorial_lambda(5) <<std::endl;

    return 0;
}