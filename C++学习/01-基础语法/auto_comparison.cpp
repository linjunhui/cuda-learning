#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <typeinfo>

/*
演示auto在for循环中的使用场景和最佳实践
*/

int main() {
    std::cout << "=== auto在for循环中的使用分析 ===" << std::endl;
    
    // 1. 基本数组遍历
    std::cout << "\n1. 基本数组遍历:" << std::endl;
    int arr[] = {1, 2, 3, 4, 5};
    
    // 传统写法
    std::cout << "传统写法: ";
    for(int i = 0; i < 5; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    
    // auto写法
    std::cout << "auto写法: ";
    for(auto i : arr) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    // 2. 容器遍历 - 显示auto的优势
    std::cout << "\n2. 容器遍历对比:" << std::endl;
    std::vector<std::string> words = {"hello", "world", "c++", "programming"};
    
    // 传统迭代器写法 - 冗长
    std::cout << "传统迭代器写法: ";
    for(std::vector<std::string>::iterator it = words.begin(); it != words.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // auto迭代器写法 - 简洁
    std::cout << "auto迭代器写法: ";
    for(auto it = words.begin(); it != words.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // 范围for循环 - 最简洁
    std::cout << "范围for循环: ";
    for(const auto& word : words) {
        std::cout << word << " ";
    }
    std::cout << std::endl;
    
    // 3. 复杂类型 - auto的威力
    std::cout << "\n3. 复杂类型遍历:" << std::endl;
    std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 87}, {"Charlie", 92}};
    
    // 传统写法 - 非常冗长
    std::cout << "传统写法 (非常冗长):" << std::endl;
    for(std::map<std::string, int>::iterator it = scores.begin(); it != scores.end(); ++it) {
        std::cout << it->first << ": " << it->second << std::endl;
    }
    
    // auto写法 - 简洁明了
    std::cout << "auto写法 (简洁):" << std::endl;
    for(const auto& pair : scores) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    // 4. 性能对比演示
    std::cout << "\n4. 性能对比演示:" << std::endl;
    std::vector<std::string> large_vec;
    for(int i = 0; i < 1000; ++i) {
        large_vec.push_back("String_" + std::to_string(i));
    }
    
    // 值传递 - 会拷贝字符串
    std::cout << "值传递 (auto str): 会拷贝字符串" << std::endl;
    int count1 = 0;
    for(auto str : large_vec) {
        count1++;
        if(count1 <= 3) {
            std::cout << "处理: " << str << std::endl;
        }
    }
    
    // 引用传递 - 不拷贝
    std::cout << "引用传递 (const auto& str): 不拷贝字符串" << std::endl;
    int count2 = 0;
    for(const auto& str : large_vec) {
        count2++;
        if(count2 <= 3) {
            std::cout << "处理: " << str << std::endl;
        }
    }
    
    // 5. auto vs 显式类型声明
    std::cout << "\n5. auto vs 显式类型声明:" << std::endl;
    
    // 简单类型 - 显式声明可能更清晰
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    std::cout << "显式类型 (int): ";
    for(int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    std::cout << "auto类型: ";
    for(auto num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // 6. 类型推导演示
    std::cout << "\n6. 类型推导演示:" << std::endl;
    std::vector<double> doubles = {1.1, 2.2, 3.3, 4.4, 5.5};
    
    for(auto& d : doubles) {
        std::cout << "类型: " << typeid(d).name() << ", 值: " << d << std::endl;
        d *= 2;  // 修改原值
    }
    
    std::cout << "修改后的值: ";
    for(const auto& d : doubles) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    
    return 0;
}























