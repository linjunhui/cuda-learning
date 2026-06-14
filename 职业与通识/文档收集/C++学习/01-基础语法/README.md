# C++基础语法学习

## 学习目标

熟练掌握C++基本语法和特性，为后续学习打下坚实基础。

## 学习时间

1个月（30天）

## 学习内容

### 第1周：C++核心语法
- C++11/14/17/20新特性
- 变量和数据类型
- 控制流语句
- 函数定义和调用
- 类和对象基础

### 第2周：模板编程基础
- 函数模板
- 类模板
- 模板特化
- 模板参数推导

### 第3周：STL标准库
- 容器（vector, list, map, set等）
- 迭代器
- 算法库
- 函数对象

### 第4周：内存管理和异常处理
- 智能指针（unique_ptr, shared_ptr, weak_ptr）
- RAII原则
- 异常处理机制
- 资源管理最佳实践

## 实践项目

### 项目1：简单容器类（第1周）
实现一个简单的动态数组类，包含以下功能：
- 动态内存分配
- 元素访问和修改
- 大小管理
- 拷贝构造和赋值操作

### 项目2：STL算法应用（第2-3周）
使用STL完成以下任务：
- 数据排序和搜索
- 容器操作
- 算法组合应用

### 项目3：内存安全程序（第4周）
编写一个内存安全的C++程序：
- 使用智能指针管理资源
- 异常安全编程
- 资源泄漏检测

## 学习资源

### 书籍
- 《C++ Primer》第5版（第1-8章）
- 《Effective C++》第1-10条

### 在线资源
- C++官方文档
- cppreference.com
- Stack Overflow

### 练习平台
- LeetCode（基础题目）
- HackerRank
- CodeChef

## 每日学习计划

### 工作日（2-3小时）
- 理论学习：1小时
- 编程实践：1-2小时

### 周末（4-6小时）
- 理论学习：2小时
- 编程实践：2-4小时

## 检查点

### 第1周检查点
- [ ] 掌握C++基本语法
- [ ] 能够编写简单的C++程序
- [ ] 理解类和对象概念

### 第2周检查点
- [ ] 掌握模板编程基础
- [ ] 能够编写函数模板和类模板
- [ ] 完成项目1

### 第3周检查点
- [ ] 熟练使用STL容器和算法
- [ ] 能够选择合适的容器和算法
- [ ] 完成项目2

### 第4周检查点
- [ ] 掌握智能指针使用
- [ ] 理解异常处理机制
- [ ] 完成项目3

## 代码示例

### 智能指针使用示例
```cpp
#include <memory>
#include <iostream>

class Resource {
public:
    Resource() { std::cout << "Resource created\n"; }
    ~Resource() { std::cout << "Resource destroyed\n"; }
    void use() { std::cout << "Resource used\n"; }
};

int main() {
    // 使用unique_ptr
    std::unique_ptr<Resource> ptr = std::make_unique<Resource>();
    ptr->use();
    
    // 使用shared_ptr
    std::shared_ptr<Resource> shared = std::make_shared<Resource>();
    shared->use();
    
    return 0;
}
```

### STL容器使用示例
```cpp
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3};
    
    // 排序
    std::sort(numbers.begin(), numbers.end());
    
    // 查找
    auto it = std::find(numbers.begin(), numbers.end(), 5);
    if (it != numbers.end()) {
        std::cout << "Found 5 at position: " << std::distance(numbers.begin(), it) << std::endl;
    }
    
    return 0;
}
```

## 常见问题

### Q: 什么时候使用智能指针？
A: 当需要动态分配内存时，优先使用智能指针而不是原始指针，可以自动管理内存生命周期。

### Q: 如何选择合适的STL容器？
A: 根据使用场景选择：
- vector：随机访问频繁
- list：频繁插入删除
- map：需要键值对
- set：需要去重和排序

### Q: 异常处理的最佳实践是什么？
A: 使用RAII原则，在析构函数中清理资源，避免在析构函数中抛出异常。

---

**学习开始时间**：2024-01-15  
**预计完成时间**：2024-02-15
