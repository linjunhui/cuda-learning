# C++知识库

## 工程说明

本知识库基于 `cuda-learning` 工程中的 `C++学习/01-基础语法/` 目录下的实际学习材料汇总而成。工程采用系统性的学习方式，当前已完成基础语法阶段的学习，包含详细的学习讲义、实践代码和技术文档。知识库会随着工程内容的持续更新而保持同步。

## 基础知识汇总

### auto关键字的演进详解

#### C++11：变量类型推导和尾置返回类型

auto关键字是C++11引入的重要特性，主要用于变量类型推导和尾置返回类型。auto可以自动推导变量的类型，大大简化了代码，特别是在处理复杂类型时。

**变量类型推导示例**（来自工程中的 `day_01_auto.cpp`）：

```cpp
auto a = 1;           // 推导为 int
auto b = 2.0;          // 推导为 double
auto c = add(a, b);    // 推导为函数返回类型
```

auto推导类型时遵循初始化表达式的类型，不会进行类型提升或转换。这意味着 `auto x = 42` 推导为 `int`，而 `auto y = 42.0` 推导为 `double`。

**尾置返回类型**（来自工程中的讲义）：

```cpp
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}
```

C++11中使用 `auto` 和 `decltype` 配合实现尾置返回类型，这对于模板函数特别有用，因为返回类型可能依赖于模板参数。

#### C++14：简化返回类型推导

C++14允许省略尾置返回类型，让编译器自动推导函数返回类型：

```cpp
// C++14 简化写法
template<typename T, typename U>
auto add(T a, U b) {  // 返回类型自动推导
    return a * b;
}
```

编译器根据函数体的return语句自动推导返回类型。如果函数有多个return语句，它们必须返回相同的类型。

#### C++17：结构化绑定支持

C++17引入了结构化绑定，auto可以用于解构：

```cpp
std::pair<int, std::string> get_pair() {
    return {42, "hello"};
}

auto [number, text] = get_pair();  // 解构赋值
```

结构化绑定使得从复合类型中提取成员变得更加简洁，特别适合处理 `pair`、`tuple` 等类型。

#### C++20：函数参数中的auto

C++20最重要的变化是允许在函数参数中使用auto：

```cpp
// C++20: 函数参数中的 auto
auto add(auto a, auto b) {
    return a + b;
}
```

这等价于传统模板写法，但语法更简洁。auto参数会为每个不同的类型组合生成对应的函数实例。

#### 实际应用和注意事项

**使用auto的优势**：
1. **简化复杂类型**：特别是STL容器的迭代器类型，`auto it = vec.begin()` 比 `std::vector<int>::iterator it = vec.begin()` 简洁得多
2. **类型安全**：编译器在编译时推导类型，避免类型不匹配
3. **代码维护性**：当函数返回类型改变时，使用auto的代码无需修改

**使用auto的注意事项**：
1. **可读性平衡**：过度使用auto可能导致代码难以理解，需要权衡
2. **类型推导规则**：理解auto的类型推导规则，特别是引用和const的处理
3. **兼容性考虑**：不同C++标准的auto支持程度不同，需要注意编译选项

### Lambda表达式详解

#### Lambda表达式本质

Lambda表达式是C++11引入的匿名函数实现方式，本质上就是定义函数。根据工程中的 `lambda_summary.md` 和实际代码 `day_01_lambda.cpp`，lambda表达式提供了一种简洁的方式来定义和使用函数。

**基本语法**：

```cpp
[](int x) { return x * x; }  // 最简单的lambda
```

完整语法是：`[捕获列表](参数列表) -> 返回类型 { 函数体 }`

**与传统函数的对比**（来自工程讲义）：

```cpp
// 传统函数定义
int square(int x) {
    return x * x;
}

// Lambda表达式（匿名函数）
auto lambdaSquare = [](int x) { return x * x; };

// 两种方式调用结果完全相同
square(5);        // 返回 25
lambdaSquare(5);  // 返回 25
```

#### 捕获列表详解

Lambda表达式的独特之处在于可以捕获外部变量，这在传统函数中是无法做到的。

**值捕获**（来自 `day_01_lambda.cpp`）：

```cpp
int factorial = 20;
auto factorial_lambda = [factorial](int x) { return factorial * x; };
```

值捕获会复制变量的值到lambda中，lambda内部对变量的修改不会影响外部变量。

**引用捕获**：

```cpp
int counter = 0;
auto increment = [&counter](int x) { counter += x; };
```

引用捕获使用 `&` 符号，lambda内部对变量的修改会影响外部变量。

**捕获所有变量**：

```cpp
auto lambda1 = [=](int x) { return x * multiplier; };  // 值捕获所有
auto lambda2 = [&](int x) { return x * multiplier; };  // 引用捕获所有
```

#### Lambda表达式的应用场景

**与STL算法配合使用**（来自工程讲义）：

```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5};

// Lambda方式：内联定义，更简洁
std::copy_if(numbers.begin(), numbers.end(), std::back_inserter(evens),
             [](int x) { return x % 2 == 0; });
```

**赋值给变量使用**（来自实际代码）：

```cpp
auto square = [](int x) { return x * x; };  // 注意这里是auto，返回值是closure type类型
std::cout << square(2) << std::endl;
```

#### C++14和C++17的改进

C++14引入了泛型lambda，允许lambda的参数类型使用auto：

```cpp
auto generic = [](auto x) { return x * 2; };
```

这使得lambda可以处理不同类型的参数，提高了灵活性。

C++17进一步改进了lambda，允许在捕获列表中使用结构化绑定等新特性。

#### Lambda的优势和使用建议

**Lambda的优势**：
1. **简洁性**：不需要单独定义函数，代码更紧凑
2. **灵活性**：可以捕获外部变量，局部作用域
3. **性能**：编译器可以内联优化，没有函数调用开销
4. **现代C++风格**：函数式编程风格，与STL算法完美配合

**使用建议**：
- ✅ **推荐使用Lambda的场景**：STL算法、事件处理、排序比较、临时函数
- ❌ **不推荐使用Lambda的场景**：复杂逻辑（函数体过长）、多处复用（应定义普通函数）、需要单独调试的函数

### 移动语义详解

#### 移动语义的基本原理

移动语义是C++11引入的重要特性，允许将资源从一个对象"移动"到另一个对象，而不是复制。移动语义解决了传统拷贝操作在处理大型对象时的性能问题。

**传统拷贝的问题**（来自工程中的 `移动语义详解讲义.md`）：

```cpp
class LargeObject {
private:
    std::vector<int> data;  // 可能包含大量数据
    std::string name;
    
public:
    // 拷贝构造函数 - 性能问题！
    LargeObject(const LargeObject& other) 
        : name(other.name), data(other.data) {
        // 需要复制所有data，时间复杂度O(n)
    }
};
```

当LargeObject包含大量数据时，拷贝操作会复制所有数据，造成性能开销。

**移动语义的优势**：

```cpp
// 移动构造函数 - 高效！
LargeObject(LargeObject&& other) noexcept 
    : name(std::move(other.name)), data(std::move(other.data)) {
    // 只是转移指针，时间复杂度O(1)
    other.data.clear();  // 将源对象置于有效但未指定状态
}
```

移动构造函数接受右值引用参数，将资源从源对象转移到目标对象，然后源对象置于有效但未指定的状态。

#### 左值和右值理解

**左值（lvalue）**：可以取地址的表达式，通常有名字，可以出现在赋值号左边

```cpp
int x = 42;    // x是左值
int* p = &x;   // 可以取地址
```

**右值（rvalue）**：不能取地址的表达式，通常是临时对象，不能出现在赋值号左边

```cpp
int x = 42;           // 42是右值
int y = x + 1;        // x + 1是右值（临时对象）
```

**右值引用**：用 `&&` 表示，可以绑定到右值

```cpp
int&& rref = 42;      // 右值引用绑定到右值
```

#### std::move函数

`std::move` 函数将左值转换为右值引用，告诉编译器可以使用移动语义：

```cpp
LargeObject obj1("Object1", 1000000);
LargeObject obj2 = std::move(obj1);  // 调用移动构造函数
```

**重要理解**：`std::move` 实际上并不移动任何东西，它只是进行类型转换，将左值转换为右值引用。真正的移动发生在移动构造函数或移动赋值操作符中。

**移动后的状态**：移动后，源对象应该处于有效但未指定的状态，不应该再使用，除非重新赋值。

```cpp
obj1 = LargeObject("New", 100);  // 可以重新赋值使用
```

#### 移动语义在智能指针中的应用

移动语义在智能指针中特别重要。`unique_ptr` 使用移动语义实现所有权的转移：

```cpp
std::unique_ptr<LargeObject> ptr1 = std::make_unique<LargeObject>(...);
std::unique_ptr<LargeObject> ptr2 = std::move(ptr1);  // 移动后ptr1变为nullptr
```

移动后原指针变为空，这是 `unique_ptr` 独占所有权的体现。

`shared_ptr` 也支持移动，移动比拷贝更高效，因为它只需要转移指针，而不需要增加引用计数：

```cpp
std::shared_ptr<LargeObject> ptr1 = std::make_shared<LargeObject>(...);
std::shared_ptr<LargeObject> ptr2 = std::move(ptr1);  // 移动，不增加引用计数
```

#### 移动语义在容器中的应用

容器类如 `vector`、`string` 等都实现了移动语义，可以在传递大型对象时避免不必要的拷贝：

```cpp
std::vector<int> large_vec(1000000, 42);
std::vector<int> vec2 = std::move(large_vec);  // 移动，不复制数据
```

移动操作的时间复杂度通常是O(1)，而拷贝操作可能是O(n)。正确使用移动语义可以显著提高程序性能。

#### noexcept标记的重要性

移动操作通常标记为 `noexcept`，表示不会抛出异常：

```cpp
LargeObject(LargeObject&& other) noexcept {
    // 移动实现
}
```

这对于容器等需要异常安全保证的场景很重要。如果移动操作可能抛出异常，容器可能会回退到拷贝操作，影响性能。

### const和constexpr详解

#### const关键字

`const` 表示"常量"，表示值在初始化后不能被修改。`const` 是运行时常量，值在运行时确定。

**const变量的使用**（来自工程中的 `const_constexpr_详解.md`）：

```cpp
const int MAX_SIZE = 100;           // 编译时常量
const std::string APP_NAME = "MyApp"; // 运行时常量
```

**const函数参数**：

```cpp
void printValue(const int value) {
    // value 不能被修改
    std::cout << value << std::endl;
}

void processString(const std::string& str) {
    // str 不能被修改，但可以读取
    std::cout << str.length() << std::endl;
}
```

**const成员函数**：

```cpp
class Rectangle {
private:
    double width, height;
    
public:
    // const 成员函数：不能修改成员变量
    double getArea() const {
        return width * height;
    }
    
    // 非 const 成员函数：可以修改成员变量
    void setWidth(double w) {
        width = w;
    }
};
```

#### constexpr关键字

`constexpr` 表示"常量表达式"，要求值在编译时就能确定。`constexpr` 是编译时常量，编译器可以在编译时计算。

**constexpr变量**：

```cpp
constexpr int MAX_SIZE = 100;      // 编译时常量
constexpr double PI = 3.14159;     // 编译时常量
constexpr int ARRAY_SIZE = 10;     // 可以用作数组大小
```

**constexpr函数**：

```cpp
// constexpr 函数：可以在编译时计算
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int fact_5 = factorial(5);  // 编译时计算
```

#### const vs constexpr 对比

| 特性 | const | constexpr |
|------|-------|-----------|
| **确定时机** | 运行时 | 编译时 |
| **初始化** | 可以延迟 | 必须立即 |
| **函数** | 运行时常量 | 编译时常量 |
| **性能** | 一般 | 更好（编译时优化） |
| **使用场景** | 通用常量 | 编译时常量 |

**实际应用场景**：

```cpp
// 数组大小定义 - 推荐使用 constexpr
constexpr int ARRAY_SIZE = 100;
int array[ARRAY_SIZE];  // 编译时确定大小

// const 可能有问题
const int size = getSize();  // 运行时确定
// int array[size];  // 错误！数组大小必须是编译时常量
```

#### 最佳实践

**选择原则**：
- **使用 constexpr 的情况**：数学常量（π、e等）、数组大小、模板参数、编译时计算、性能敏感的常量
- **使用 const 的情况**：运行时常量、函数参数、成员函数、通用常量

### new操作符返回指针详解

#### 核心理解

**new操作符返回的是指针**，这是理解动态内存分配的关键。根据工程中的 `new_pointer_summary.md`，`new` 操作符返回指向动态分配对象的指针。

**基本用法对比**：

```cpp
// new返回指针
Student* ptr = new Student("Alice", 20);

// 类型对比
Student obj("Bob", 21);        // obj是Student类型
Student* ptr = new Student("Alice", 20);  // ptr是Student*类型（指针）
```

#### 指针与普通对象的区别

| 特性 | 普通对象 | new对象（指针） |
|------|----------|----------------|
| **存储位置** | 栈上 | 堆上 |
| **内存管理** | 自动 | 手动 |
| **访问方式** | `obj.method()` | `ptr->method()` |
| **类型** | `Student` | `Student*` |
| **大小** | 对象大小 | 指针大小（通常8字节） |

**访问方式对比**：

```cpp
// 普通对象
SimpleClass obj("StackObject", 100);
obj.show();                    // 直接访问

// 指针对象
SimpleClass* ptr = new SimpleClass("HeapObject", 200);
ptr->show();                   // 通过指针访问
```

#### 内存管理

**普通对象（栈上）**：
- **自动管理**：作用域结束时自动析构
- **生命周期**：与作用域绑定
- **效率**：高（无需动态分配）

**new对象（堆上）**：
- **手动管理**：必须使用 `delete` 释放
- **生命周期**：直到手动释放
- **效率**：较低（需要动态分配）

#### 现代C++建议

**推荐使用智能指针**（来自工程讲义）：

```cpp
#include <memory>

// 使用unique_ptr
std::unique_ptr<Student> student = std::make_unique<Student>("Alice", 20);
// 自动管理内存

// 使用shared_ptr
std::shared_ptr<Student> student = std::make_shared<Student>("Alice", 20);
// 引用计数管理
```

#### 常见错误

**忘记释放内存**：

```cpp
Student* ptr = new Student("Alice", 20);
// 忘记delete ptr;  // 内存泄漏！
```

**重复释放**：

```cpp
Student* ptr = new Student("Alice", 20);
delete ptr;
delete ptr;  // 错误！重复释放
```

**使用已释放的指针**：

```cpp
Student* ptr = new Student("Alice", 20);
delete ptr;
ptr->show();  // 错误！使用已释放的指针
```

## 面试要点总结

### auto关键字面试要点

1. **auto的演进过程**：从C++11的变量推导和尾置返回类型，到C++20的函数参数auto，每个版本的特点要清楚
2. **类型推导规则**：理解auto如何推导类型，特别是引用、const、指针的处理
3. **使用场景**：何时使用auto，何时避免使用auto
4. **与decltype的区别**：auto推导初始化表达式类型，decltype推导表达式类型

### Lambda表达式面试要点

1. **Lambda的本质**：Lambda表达式就是匿名函数，与传统函数功能等价
2. **捕获列表**：值捕获vs引用捕获，捕获所有vs捕获特定变量
3. **应用场景**：与STL算法配合、事件处理、排序比较等
4. **性能考虑**：编译器内联优化，与传统函数性能对比

### 移动语义面试要点

1. **移动语义的原理**：资源转移，避免不必要的拷贝
2. **左值右值理解**：能够区分左值和右值，理解右值引用的作用
3. **std::move的作用**：类型转换，不是真正的移动
4. **移动后的状态**：源对象处于有效但未指定状态
5. **应用场景**：智能指针、容器、大型对象的传递

### const和constexpr面试要点

1. **根本区别**：const是运行时常量，constexpr是编译时常量
2. **使用场景**：数组大小用constexpr，函数参数用const
3. **const成员函数**：不能修改成员变量，提高代码安全性
4. **constexpr函数**：编译时计算，性能优化

### new和指针面试要点

1. **new返回指针**：new操作符返回的是指针类型，不是对象类型
2. **栈vs堆**：普通对象在栈上，new对象在堆上
3. **内存管理**：栈对象自动管理，堆对象手动管理
4. **现代C++实践**：优先使用智能指针，避免手动管理内存

## 引导面试者深入理解

### 理解C++的演进方向

C++从C++98到C++20的演进体现了语言向更简洁、更安全、更高效的方向发展。auto关键字、lambda表达式、移动语义等特性都是为了简化代码、提高性能和增强类型安全。

### 掌握现代C++编程风格

现代C++强调：
1. **类型安全**：使用auto推导复杂类型，避免类型错误
2. **函数式编程**：使用lambda表达式，配合STL算法
3. **资源管理**：使用智能指针，遵循RAII原则
4. **性能优化**：使用移动语义，避免不必要的拷贝

### 理论与实践结合

理解这些特性的原理很重要，但更重要的是能够在实际编程中正确使用。建议：
1. **编写代码验证**：通过实际代码验证理论理解
2. **性能对比**：比较不同实现方式的性能差异
3. **错误案例分析**：学习常见错误，避免在实际项目中犯错

### 持续学习

C++标准在不断更新，新特性不断引入。要保持学习习惯，关注新技术，同时也要回顾和总结已学知识，形成完整的知识体系。
