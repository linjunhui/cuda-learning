# 第3周：STL标准库

## 学习目标

熟练掌握STL标准库的使用，包括容器、迭代器、算法库和函数对象。

## 学习内容

### 1. 容器（Containers）

#### 1.1 序列容器

##### vector - 动态数组
```cpp
#include <vector>
#include <iostream>

int main() {
    // 创建vector
    std::vector<int> vec;
    std::vector<int> vec2(10);        // 10个默认值
    std::vector<int> vec3(10, 5);     // 10个5
    std::vector<int> vec4{1, 2, 3, 4, 5}; // 初始化列表
    
    // 添加元素
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    vec.emplace_back(4);              // C++11，更高效
    
    // 访问元素
    std::cout << "First element: " << vec[0] << std::endl;
    std::cout << "Last element: " << vec.back() << std::endl;
    std::cout << "Size: " << vec.size() << std::endl;
    std::cout << "Capacity: " << vec.capacity() << std::endl;
    
    // 遍历
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
    
    // 范围for循环
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // 迭代器遍历
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

##### list - 双向链表
```cpp
#include <list>
#include <iostream>

int main() {
    std::list<int> lst{1, 2, 3, 4, 5};
    
    // 插入元素
    lst.push_front(0);                // 头部插入
    lst.push_back(6);                 // 尾部插入
    
    // 在指定位置插入
    auto it = lst.begin();
    std::advance(it, 3);              // 移动到第4个位置
    lst.insert(it, 99);               // 在位置3插入99
    
    // 删除元素
    lst.pop_front();                  // 删除头部元素
    lst.pop_back();                   // 删除尾部元素
    
    // 删除指定元素
    lst.remove(3);                    // 删除所有值为3的元素
    
    // 遍历
    for (const auto& element : lst) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

##### deque - 双端队列
```cpp
#include <deque>
#include <iostream>

int main() {
    std::deque<int> dq{1, 2, 3, 4, 5};
    
    // 双端操作
    dq.push_front(0);                 // 头部插入
    dq.push_back(6);                  // 尾部插入
    dq.pop_front();                   // 头部删除
    dq.pop_back();                    // 尾部删除
    
    // 随机访问
    std::cout << "Element at index 2: " << dq[2] << std::endl;
    
    // 遍历
    for (const auto& element : dq) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### 1.2 关联容器

##### map - 键值对容器
```cpp
#include <map>
#include <string>
#include <iostream>

int main() {
    std::map<std::string, int> scores;
    
    // 插入元素
    scores["Alice"] = 95;
    scores["Bob"] = 87;
    scores["Charlie"] = 92;
    
    // 使用insert
    scores.insert({"David", 88});
    scores.insert(std::make_pair("Eve", 90));
    
    // 查找元素
    auto it = scores.find("Alice");
    if (it != scores.end()) {
        std::cout << "Alice's score: " << it->second << std::endl;
    }
    
    // 使用[]操作符
    std::cout << "Bob's score: " << scores["Bob"] << std::endl;
    
    // 遍历
    for (const auto& pair : scores) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    // 删除元素
    scores.erase("Charlie");
    
    return 0;
}
```

##### set - 集合
```cpp
#include <set>
#include <iostream>

int main() {
    std::set<int> s{5, 2, 8, 1, 9, 3};
    
    // 插入元素
    s.insert(4);
    s.insert(7);
    
    // 查找元素
    auto it = s.find(5);
    if (it != s.end()) {
        std::cout << "Found: " << *it << std::endl;
    }
    
    // 遍历（自动排序）
    for (const auto& element : s) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // 删除元素
    s.erase(8);
    
    // 集合操作
    std::set<int> s2{2, 4, 6, 8, 10};
    std::set<int> intersection;
    std::set_intersection(s.begin(), s.end(),
                         s2.begin(), s2.end(),
                         std::inserter(intersection, intersection.begin()));
    
    std::cout << "Intersection: ";
    for (const auto& element : intersection) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

##### unordered_map - 哈希表
```cpp
#include <unordered_map>
#include <string>
#include <iostream>

int main() {
    std::unordered_map<std::string, int> hashMap;
    
    // 插入元素
    hashMap["apple"] = 1;
    hashMap["banana"] = 2;
    hashMap["cherry"] = 3;
    
    // 查找元素
    if (hashMap.find("apple") != hashMap.end()) {
        std::cout << "Found apple: " << hashMap["apple"] << std::endl;
    }
    
    // 遍历
    for (const auto& pair : hashMap) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    // 统计信息
    std::cout << "Bucket count: " << hashMap.bucket_count() << std::endl;
    std::cout << "Load factor: " << hashMap.load_factor() << std::endl;
    
    return 0;
}
```

### 2. 迭代器（Iterators）

#### 2.1 迭代器类型
```cpp
#include <vector>
#include <list>
#include <iostream>

int main() {
    std::vector<int> vec{1, 2, 3, 4, 5};
    std::list<int> lst{1, 2, 3, 4, 5};
    
    // 前向迭代器
    std::cout << "Vector traversal: ";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // 反向迭代器
    std::cout << "Reverse traversal: ";
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // 常量迭代器
    std::cout << "Const traversal: ";
    for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // 双向迭代器（list）
    auto it = lst.begin();
    ++it;                           // 前向移动
    --it;                           // 后向移动
    
    return 0;
}
```

#### 2.2 迭代器操作
```cpp
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 迭代器算术
    auto it1 = vec.begin();
    auto it2 = it1 + 3;             // 前进3个位置
    auto it3 = it2 - 1;             // 后退1个位置
    auto distance = it2 - it1;      // 计算距离
    
    std::cout << "Distance: " << distance << std::endl;
    std::cout << "Element at it2: " << *it2 << std::endl;
    
    // 迭代器比较
    if (it1 < it2) {
        std::cout << "it1 comes before it2" << std::endl;
    }
    
    // 使用迭代器修改元素
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        *it *= 2;                   // 每个元素乘以2
    }
    
    // 输出结果
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### 3. 算法库（Algorithms）

#### 3.1 查找算法
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec{1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
    
    // find - 查找元素
    auto it = std::find(vec.begin(), vec.end(), 5);
    if (it != vec.end()) {
        std::cout << "Found 5 at position: " << std::distance(vec.begin(), it) << std::endl;
    }
    
    // find_if - 条件查找
    auto it2 = std::find_if(vec.begin(), vec.end(), 
                           [](int x) { return x > 7; });
    if (it2 != vec.end()) {
        std::cout << "First element > 7: " << *it2 << std::endl;
    }
    
    // binary_search - 二分查找（需要排序）
    std::sort(vec.begin(), vec.end());
    bool found = std::binary_search(vec.begin(), vec.end(), 5);
    std::cout << "5 found: " << (found ? "Yes" : "No") << std::endl;
    
    // lower_bound - 查找第一个不小于给定值的元素
    auto it3 = std::lower_bound(vec.begin(), vec.end(), 6);
    std::cout << "First element >= 6: " << *it3 << std::endl;
    
    return 0;
}
```

#### 3.2 排序算法
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec{5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    std::cout << "Original: ";
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // sort - 排序
    std::sort(vec.begin(), vec.end());
    std::cout << "Sorted: ";
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // 自定义比较函数
    std::sort(vec.begin(), vec.end(), std::greater<int>());
    std::cout << "Descending: ";
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // 使用lambda表达式
    std::sort(vec.begin(), vec.end(), 
              [](int a, int b) { return a < b; });
    std::cout << "Ascending (lambda): ";
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### 3.3 变换算法
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec1{1, 2, 3, 4, 5};
    std::vector<int> vec2(vec1.size());
    
    // transform - 变换
    std::transform(vec1.begin(), vec1.end(), vec2.begin(),
                   [](int x) { return x * x; });
    
    std::cout << "Original: ";
    for (const auto& element : vec1) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Squared: ";
    for (const auto& element : vec2) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // for_each - 对每个元素执行操作
    std::cout << "For each: ";
    std::for_each(vec1.begin(), vec1.end(),
                  [](int x) { std::cout << x * 2 << " "; });
    std::cout << std::endl;
    
    return 0;
}
```

#### 3.4 数值算法
```cpp
#include <numeric>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec{1, 2, 3, 4, 5};
    
    // accumulate - 累加
    int sum = std::accumulate(vec.begin(), vec.end(), 0);
    std::cout << "Sum: " << sum << std::endl;
    
    // 自定义累加操作
    int product = std::accumulate(vec.begin(), vec.end(), 1,
                                 [](int a, int b) { return a * b; });
    std::cout << "Product: " << product << std::endl;
    
    // partial_sum - 部分和
    std::vector<int> partialSums(vec.size());
    std::partial_sum(vec.begin(), vec.end(), partialSums.begin());
    
    std::cout << "Partial sums: ";
    for (const auto& element : partialSums) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // inner_product - 内积
    std::vector<int> vec2{2, 3, 4, 5, 6};
    int innerProduct = std::inner_product(vec.begin(), vec.end(),
                                         vec2.begin(), 0);
    std::cout << "Inner product: " << innerProduct << std::endl;
    
    return 0;
}
```

### 4. 函数对象（Function Objects）

#### 4.1 预定义函数对象
```cpp
#include <functional>
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec{5, 2, 8, 1, 9, 3};
    
    // 使用预定义函数对象
    std::sort(vec.begin(), vec.end(), std::greater<int>());
    std::cout << "Descending order: ";
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // 使用less
    std::sort(vec.begin(), vec.end(), std::less<int>());
    std::cout << "Ascending order: ";
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // 使用equal_to
    auto it = std::find_if(vec.begin(), vec.end(),
                          std::bind(std::equal_to<int>(), std::placeholders::_1, 5));
    if (it != vec.end()) {
        std::cout << "Found 5" << std::endl;
    }
    
    return 0;
}
```

#### 4.2 自定义函数对象
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

// 自定义函数对象
class Square {
public:
    int operator()(int x) const {
        return x * x;
    }
};

class IsEven {
public:
    bool operator()(int x) const {
        return x % 2 == 0;
    }
};

int main() {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 使用自定义函数对象
    Square square;
    std::cout << "Square of 5: " << square(5) << std::endl;
    
    // 使用transform
    std::vector<int> squares(vec.size());
    std::transform(vec.begin(), vec.end(), squares.begin(), square);
    
    std::cout << "Squares: ";
    for (const auto& element : squares) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // 使用find_if
    IsEven isEven;
    auto it = std::find_if(vec.begin(), vec.end(), isEven);
    if (it != vec.end()) {
        std::cout << "First even number: " << *it << std::endl;
    }
    
    return 0;
}
```

#### 4.3 Lambda表达式
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 基本lambda
    auto square = [](int x) { return x * x; };
    std::cout << "Square of 5: " << square(5) << std::endl;
    
    // 捕获变量
    int multiplier = 3;
    auto multiply = [multiplier](int x) { return x * multiplier; };
    
    std::cout << "5 * 3 = " << multiply(5) << std::endl;
    
    // 使用lambda与算法
    std::vector<int> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](int x) { return x * x; });
    
    std::cout << "Squares: ";
    for (const auto& element : result) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    
    // 条件查找
    auto it = std::find_if(vec.begin(), vec.end(),
                          [](int x) { return x > 7; });
    if (it != vec.end()) {
        std::cout << "First element > 7: " << *it << std::endl;
    }
    
    return 0;
}
```

## 实践练习

### 练习1：容器操作
编写程序实现以下功能：
1. 使用vector存储学生成绩，计算平均分
2. 使用map存储单词及其出现次数
3. 使用set存储不重复的随机数
4. 使用deque实现一个简单的队列

### 练习2：算法应用
使用STL算法实现以下功能：
1. 查找数组中的最大值和最小值
2. 统计数组中满足条件的元素个数
3. 将两个数组合并并排序
4. 删除数组中所有重复元素

### 练习3：迭代器操作
编写程序演示：
1. 不同容器的迭代器特性
2. 迭代器的算术运算
3. 使用迭代器修改容器元素
4. 反向迭代器的使用

## 每日学习任务

### 第1天：序列容器
- 学习vector、list、deque的使用
- 理解不同容器的特点和适用场景
- 练习基本的容器操作

### 第2天：关联容器
- 学习map、set、unordered_map的使用
- 理解有序容器和无序容器的区别
- 练习关联容器的查找和插入操作

### 第3天：迭代器
- 学习各种迭代器类型
- 理解迭代器的操作和限制
- 练习使用迭代器遍历容器

### 第4天：查找和排序算法
- 学习find、sort等基本算法
- 理解算法的复杂度
- 练习使用算法解决实际问题

### 第5天：变换和数值算法
- 学习transform、accumulate等算法
- 理解函数式编程思想
- 练习算法的组合使用

### 第6天：函数对象
- 学习预定义函数对象
- 掌握lambda表达式的使用
- 练习自定义函数对象

### 第7天：综合练习
- 完成所有实践练习
- 综合运用STL解决复杂问题
- 准备下周学习内容

## 项目2：STL算法应用

### 项目要求
使用STL实现一个学生管理系统，包含以下功能：

```cpp
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <iostream>

struct Student {
    std::string name;
    int id;
    std::vector<int> scores;
    double average;
    
    Student(const std::string& n, int i) : name(n), id(i) {}
};

class StudentManager {
private:
    std::vector<Student> students;
    
public:
    // 添加学生
    void addStudent(const Student& student);
    
    // 删除学生
    void removeStudent(int id);
    
    // 查找学生
    Student* findStudent(int id);
    
    // 计算平均分
    void calculateAverages();
    
    // 排序学生（按平均分）
    void sortByAverage();
    
    // 统计信息
    void printStatistics();
    
    // 显示所有学生
    void displayAll();
};
```

## 检查点

### 第3周结束时的能力要求
- [ ] 熟练使用各种STL容器
- [ ] 理解不同容器的特点和适用场景
- [ ] 能够使用迭代器遍历和操作容器
- [ ] 掌握常用STL算法的使用
- [ ] 能够编写和使用函数对象
- [ ] 熟练使用lambda表达式
- [ ] 能够选择合适的容器和算法
- [ ] 完成项目2的主要功能

## 常见问题解答

### Q: 如何选择合适的容器？
A: 根据使用场景选择：
- vector：需要随机访问，频繁在末尾操作
- list：需要频繁插入删除，不需要随机访问
- map：需要键值对，需要排序
- unordered_map：需要键值对，不需要排序，要求高性能

### Q: 迭代器失效的原因？
A: 迭代器失效通常发生在容器结构改变时，如插入、删除操作。vector的插入删除可能导致所有迭代器失效，list的插入删除只影响被操作的迭代器。

### Q: 如何提高STL算法性能？
A: 
- 选择合适的容器
- 使用移动语义
- 避免不必要的拷贝
- 使用reserve预分配内存
- 选择合适的算法复杂度

### Q: lambda表达式的捕获方式？
A: 
- `[=]`：按值捕获所有变量
- `[&]`：按引用捕获所有变量
- `[x, &y]`：按值捕获x，按引用捕获y
- `[=, &x]`：按值捕获所有变量，但x按引用捕获

---

**学习时间**：第3周  
**预计完成时间**：2024-02-05



