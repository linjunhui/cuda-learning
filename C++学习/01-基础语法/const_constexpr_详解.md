# C++ const 和 constexpr 详解

## 概述

`const`和`constexpr`是C++中用于定义常量的关键字，但它们有不同的含义和使用场景。

## 1. const 关键字

### 1.1 基本概念
`const`表示"常量"，表示值在初始化后不能被修改。

### 1.2 const 的使用场景

#### 1.2.1 const 变量
```cpp
const int MAX_SIZE = 100;           // 编译时常量
const double PI = 3.14159;         // 编译时常量
const std::string APP_NAME = "MyApp"; // 运行时常量
```

#### 1.2.2 const 函数参数
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

#### 1.2.3 const 成员函数
```cpp
class Rectangle {
private:
    double width, height;
    
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
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

#### 1.2.4 const 指针
```cpp
int value = 42;

// 指向常量的指针
const int* ptr1 = &value;          // 指针可以改变，但指向的值不能改变
int const* ptr2 = &value;          // 同上

// 常量指针
int* const ptr3 = &value;          // 指针不能改变，但指向的值可以改变

// 指向常量的常量指针
const int* const ptr4 = &value;    // 指针和指向的值都不能改变
```

### 1.3 const 的特点
- **运行时常量**：值在运行时确定
- **可以延迟初始化**：不一定在编译时就知道值
- **类型安全**：防止意外修改

## 2. constexpr 关键字

### 2.1 基本概念
`constexpr`表示"常量表达式"，要求值在编译时就能确定。

### 2.2 constexpr 的使用场景

#### 2.2.1 constexpr 变量
```cpp
constexpr int MAX_SIZE = 100;      // 编译时常量
constexpr double PI = 3.14159;    // 编译时常量
constexpr int ARRAY_SIZE = 10;    // 可以用作数组大小
```

#### 2.2.2 constexpr 函数
```cpp
// constexpr 函数：可以在编译时计算
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr double circleArea(double radius) {
    constexpr double pi = 3.14159265359;
    return pi * radius * radius;
}
```

#### 2.2.3 constexpr 构造函数
```cpp
class Point {
private:
    int x, y;
    
public:
    // constexpr 构造函数
    constexpr Point(int x, int y) : x(x), y(y) {}
    
    constexpr int getX() const { return x; }
    constexpr int getY() const { return y; }
    constexpr int getDistance() const { return x * x + y * y; }
};
```

### 2.3 constexpr 的特点
- **编译时常量**：值在编译时确定
- **性能优化**：编译器可以内联和优化
- **类型安全**：编译时类型检查

## 3. const vs constexpr 对比

### 3.1 基本区别

| 特性 | const | constexpr |
|------|-------|-----------|
| **确定时机** | 运行时 | 编译时 |
| **初始化** | 可以延迟 | 必须立即 |
| **函数** | 运行时常量 | 编译时常量 |
| **性能** | 一般 | 更好（编译时优化） |
| **使用场景** | 通用常量 | 编译时常量 |

### 3.2 使用示例对比

```cpp
// const 示例
const int runtimeValue = getValue();        // 运行时确定
const std::string name = "Hello";           // 运行时确定

// constexpr 示例
constexpr int compileTimeValue = 42;        // 编译时确定
constexpr double pi = 3.14159;              // 编译时确定
```

## 4. 实际应用场景

### 4.1 数组大小定义
```cpp
// 推荐使用 constexpr
constexpr int ARRAY_SIZE = 100;
int array[ARRAY_SIZE];  // 编译时确定大小

// const 可能有问题
const int size = getSize();  // 运行时确定
// int array[size];  // 错误！数组大小必须是编译时常量
```

### 4.2 数学常量
```cpp
constexpr double PI = 3.14159265359;
constexpr double E = 2.71828182846;
constexpr int MAX_INT = 2147483647;
```

### 4.3 配置常量
```cpp
constexpr int MAX_CONNECTIONS = 1000;
constexpr int TIMEOUT_SECONDS = 30;
constexpr size_t BUFFER_SIZE = 4096;
```

### 4.4 模板参数
```cpp
template<int N>
class Array {
    // N 必须是编译时常量
};

Array<10> arr;  // 10 必须是 constexpr
```

## 5. 最佳实践

### 5.1 选择原则

#### 使用 constexpr 的情况
- 数学常量（π、e等）
- 数组大小
- 模板参数
- 编译时计算
- 性能敏感的常量

#### 使用 const 的情况
- 运行时常量
- 函数参数
- 成员函数
- 通用常量

### 5.2 代码示例

```cpp
class MathUtils {
public:
    // 编译时常量
    static constexpr double PI = 3.14159265359;
    static constexpr double E = 2.71828182846;
    
    // 编译时函数
    static constexpr double circleArea(double radius) {
        return PI * radius * radius;
    }
    
    static constexpr int factorial(int n) {
        return (n <= 1) ? 1 : n * factorial(n - 1);
    }
};

class Config {
private:
    // 运行时常量
    const std::string appName;
    const int version;
    
public:
    Config(const std::string& name, int ver) 
        : appName(name), version(ver) {}
    
    // const 成员函数
    const std::string& getAppName() const { return appName; }
    int getVersion() const { return version; }
};
```

## 6. 常见错误和注意事项

### 6.1 常见错误

#### 错误1：constexpr 函数中使用非 constexpr 操作
```cpp
// 错误
constexpr int badFunction(int x) {
    std::cout << x << std::endl;  // 错误！cout 不是 constexpr
    return x * 2;
}
```

#### 错误2：constexpr 变量使用运行时值
```cpp
int getValue() { return 42; }

// 错误
constexpr int value = getValue();  // 错误！getValue() 不是 constexpr
```

#### 错误3：const 成员函数修改成员变量
```cpp
class BadClass {
private:
    int value;
    
public:
    // 错误
    int getValue() const {
        value = 42;  // 错误！const 函数不能修改成员变量
        return value;
    }
};
```

### 6.2 注意事项

1. **constexpr 函数**：所有参数和返回值都必须是字面量类型
2. **const 成员函数**：不能修改成员变量，但可以修改 mutable 成员
3. **const 指针**：注意指针本身和指向内容的区别
4. **性能考虑**：constexpr 通常比 const 性能更好

## 7. 现代C++特性

### 7.1 C++11 特性
- `constexpr` 函数和变量
- `constexpr` 构造函数

### 7.2 C++14 特性
- `constexpr` 函数可以包含更多语句
- `constexpr` 函数可以修改局部变量

### 7.3 C++17 特性
- `if constexpr` 编译时条件
- `constexpr` lambda 表达式

```cpp
// C++17 示例
template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        // 编译时分支
        std::cout << "Integer: " << value << std::endl;
    } else {
        std::cout << "Other type" << std::endl;
    }
}
```

## 8. 总结

### 8.1 关键要点
- **const**：运行时常量，通用常量
- **constexpr**：编译时常量，性能优化
- **选择原则**：编译时能确定用 constexpr，否则用 const
- **性能**：constexpr 通常性能更好
- **类型安全**：两者都提供类型安全

### 8.2 使用建议
1. **优先使用 constexpr**：对于编译时常量
2. **合理使用 const**：对于运行时常量和函数参数
3. **注意区别**：编译时 vs 运行时
4. **性能考虑**：constexpr 提供更好的优化机会
5. **现代C++**：充分利用 constexpr 的编译时计算能力

### 8.3 记忆口诀
- **const 运行时常量，constexpr 编译时常量**
- **数组大小用 constexpr，函数参数用 const**
- **性能优先 constexpr，通用常量用 const**
- **编译时确定用 constexpr，运行时确定用 const**
