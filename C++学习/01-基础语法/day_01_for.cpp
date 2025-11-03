
#include <iostream>
#include <vector>

/*
编写程序使用范围for循环完成以下任务：
1. 遍历数组并计算总和
2. 遍历字符串并统计字符
3. 遍历容器并查找特定元素
*/

int main() {
    // 1. 遍历数组
    int arr[] = {1, 2, 3, 4, 5};

    // 引用传递：i是数组元素的引用，更高效
    std::cout << "引用传递遍历: ";
    for(auto &i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // 值传递：i是数组元素的副本，每次都要拷贝
    std::cout << "值传递遍历: ";
    for(auto i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // 演示修改效果
    std::cout << "\n演示修改效果:" << std::endl;
    
    // 使用引用修改原数组
    std::cout << "修改前: ";
    for(auto &i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    for(auto &i : arr) {
        i *= 2;  // 修改原数组
    }
    
    std::cout << "引用修改后: ";
    for(auto &i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // 重置数组
    int arr2[] = {1, 2, 3, 4, 5};
    std::cout << "重置后: ";
    for(auto &i : arr2) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    // 使用值传递不会修改原数组
    for(auto i : arr2) {
        i *= 2;  // 只修改副本，不影响原数组
    }
    
    std::cout << "值传递修改后: ";
    for(auto &i : arr2) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::string str = "Hello World";

    for(char c : str) {
        std::cout << c << " ";
    }

    std::vector<int> vec = {1, 2, 3, 4, 5};
    for(auto i : vec) {
        std::cout << i << " ";
    }

    std::cout << std::endl;

    for(auto &i : vec) {
        std::cout << i << " ";
    }

    std::cout << std::endl;
}