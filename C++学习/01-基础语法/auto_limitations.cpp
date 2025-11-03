#include <iostream>
#include <vector>
#include <type_traits>

/*
演示auto的限制和正确的使用方式
*/

int main() {
    std::cout << "=== auto的限制和正确使用 ===" << std::endl;
    
    // 1. 错误用法 - auto不能用作模板参数
    std::cout << "\n1. auto的限制:" << std::endl;
    
    // ❌ 错误：auto不能用作模板参数
    // std::vector<auto> vec = {1, 2, 3, 4, 5};  // 编译错误！
    
    std::cout << "❌ std::vector<auto> 是无效的 - auto不能用作模板参数" << std::endl;
    
    // 2. 正确的用法
    std::cout << "\n2. 正确的用法:" << std::endl;
    
    // ✅ 正确：显式指定类型
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::cout << "✅ std::vector<int> vec1 = {1, 2, 3, 4, 5};" << std::endl;
    
    // ✅ 正确：使用auto推导整个对象
    auto vec2 = std::vector<int>{1, 2, 3, 4, 5};
    std::cout << "✅ auto vec2 = std::vector<int>{1, 2, 3, 4, 5};" << std::endl;
    
    // ✅ 正确：C++17的类模板参数推导
    std::vector vec3 = {1, 2, 3, 4, 5};  // C++17特性
    std::cout << "✅ std::vector vec3 = {1, 2, 3, 4, 5};  // C++17" << std::endl;
    
    // 3. auto可以推导容器类型
    std::cout << "\n3. auto推导容器类型:" << std::endl;
    
    // 推导为 std::vector<int>
    auto numbers = std::vector{1, 2, 3, 4, 5};
    std::cout << "numbers 类型: " << typeid(numbers).name() << std::endl;
    
    // 推导为 std::vector<std::string>
    auto words = std::vector{std::string("hello"), std::string("world")};
    std::cout << "words 类型: " << typeid(words).name() << std::endl;
    
    // 4. 在for循环中正确使用auto
    std::cout << "\n4. for循环中的正确使用:" << std::endl;
    
    // 遍历容器
    std::cout << "遍历numbers: ";
    for(const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // 遍历字符串
    std::string str = "Hello";
    std::cout << "遍历字符串: ";
    for(auto c : str) {
        std::cout << c << " ";
    }
    std::cout << std::endl;
    
    // 5. auto的其他限制
    std::cout << "\n5. auto的其他限制:" << std::endl;
    
    // ❌ 错误：auto不能用于函数参数
    // void func(auto x) { }  // 编译错误！
    std::cout << "❌ auto不能用于函数参数" << std::endl;
    
    // ❌ 错误：auto不能用于类成员
    // class MyClass { auto member; };  // 编译错误！
    std::cout << "❌ auto不能用于类成员变量" << std::endl;
    
    // ❌ 错误：auto不能用于数组声明
    // auto arr[5] = {1, 2, 3, 4, 5};  // 编译错误！
    std::cout << "❌ auto不能用于数组声明" << std::endl;
    
    // 6. 正确的替代方案
    std::cout << "\n6. 正确的替代方案:" << std::endl;
    
    // 使用显式类型
    std::vector<int> vec_int = {1, 2, 3, 4, 5};
    std::vector<std::string> vec_str = {"hello", "world"};
    
    // 使用auto推导整个对象
    auto vec_auto = std::vector<int>{1, 2, 3, 4, 5};
    
    // 使用decltype推导类型
    auto vec_decltype = std::vector<decltype(42)>{1, 2, 3, 4, 5};
    
    std::cout << "✅ 使用显式类型: std::vector<int>" << std::endl;
    std::cout << "✅ 使用auto推导对象: auto vec = std::vector<int>{...}" << std::endl;
    std::cout << "✅ 使用decltype推导类型: std::vector<decltype(42)>" << std::endl;
    
    // 7. 实际应用示例
    std::cout << "\n7. 实际应用示例:" << std::endl;
    
    // 计算数组元素的总和
    auto sum = 0;
    for(const auto& num : numbers) {
        sum += num;
    }
    std::cout << "数组元素总和: " << sum << std::endl;
    
    // 查找最大元素
    auto max_val = numbers[0];
    for(const auto& num : numbers) {
        if(num > max_val) {
            max_val = num;
        }
    }
    std::cout << "最大元素: " << max_val << std::endl;
    
    return 0;
}






















