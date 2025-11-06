#include <iostream>
#include <string>
#include <vector>

/*
专门演示初始化列表的使用和优势
*/

class InitializationListDemo {
private:
    const int id;                    // const成员必须用初始化列表
    std::string& name;               // 引用成员必须用初始化列表
    std::vector<int> data;           // 复杂对象建议用初始化列表
    int* dynamicArray;              // 指针成员
    size_t arraySize;                // 数组大小
    
public:
    // ✅ 正确：使用初始化列表
    InitializationListDemo(int i, std::string& n, size_t size) 
        : id(i), name(n), data(size, 0), arraySize(size) {
        // 在构造函数体中分配动态内存
        dynamicArray = new int[arraySize];
        for(size_t i = 0; i < arraySize; ++i) {
            dynamicArray[i] = i;
        }
        std::cout << "使用初始化列表构造: ID=" << id << ", Name=" << name << std::endl;
    }
    
    // ❌ 错误：尝试在构造函数体中初始化const和引用成员
    /*
    InitializationListDemo(int i, std::string& n, size_t size) {
        // id = i;        // 错误：const成员不能赋值
        // name = n;      // 错误：引用成员不能重新绑定
        // data = std::vector<int>(size); // 效率低：先默认构造再赋值
        arraySize = size;
        dynamicArray = new int[arraySize];
    }
    */
    
    // 析构函数
    ~InitializationListDemo() {
        delete[] dynamicArray;
        std::cout << "析构函数调用: ID=" << id << std::endl;
    }
    
    // 显示信息
    void display() const {
        std::cout << "ID: " << id << ", Name: " << name 
                  << ", Data size: " << data.size() 
                  << ", Array size: " << arraySize << std::endl;
    }
    
    // 访问器
    int getId() const { return id; }
    const std::string& getName() const { return name; }
    const std::vector<int>& getData() const { return data; }
};

// 演示const成员的使用
class ConstMemberDemo {
private:
    const int value;
    const std::string name;
    
public:
    // const成员必须用初始化列表
    ConstMemberDemo(int v, const std::string& n) : value(v), name(n) {
        std::cout << "ConstMemberDemo构造: value=" << value << ", name=" << name << std::endl;
    }
    
    // 显示信息
    void display() const {
        std::cout << "Value: " << value << ", Name: " << name << std::endl;
    }
};

// 演示引用成员的使用
class ReferenceMemberDemo {
private:
    std::string& name;
    int& counter;
    
public:
    // 引用成员必须用初始化列表
    ReferenceMemberDemo(std::string& n, int& c) : name(n), counter(c) {
        counter++;  // 通过引用修改外部变量
        std::cout << "ReferenceMemberDemo构造: name=" << name << ", counter=" << counter << std::endl;
    }
    
    // 显示信息
    void display() const {
        std::cout << "Name: " << name << ", Counter: " << counter << std::endl;
    }
};

// 演示初始化列表的性能优势
class PerformanceDemo {
private:
    std::vector<int> data;
    std::string name;
    
public:
    // ✅ 高效：使用初始化列表
    PerformanceDemo(size_t size, const std::string& n) : data(size, 42), name(n) {
        std::cout << "PerformanceDemo构造（初始化列表）: " << name 
                  << ", data size: " << data.size() << std::endl;
    }
    
    // ❌ 低效：在构造函数体中赋值
    PerformanceDemo(size_t size, const std::string& n, bool useAssignment) : name(n) {
        // 先默认构造data，再赋值
        data = std::vector<int>(size, 42);
        std::cout << "PerformanceDemo构造（赋值方式）: " << name 
                  << ", data size: " << data.size() << std::endl;
    }
    
    void display() const {
        std::cout << "Name: " << name << ", Data size: " << data.size() << std::endl;
    }
};

int main() {
    std::cout << "=== 初始化列表详解演示 ===" << std::endl;
    
    // 1. const成员演示
    std::cout << "\n1. const成员演示:" << std::endl;
    ConstMemberDemo constDemo(100, "ConstDemo");
    constDemo.display();
    
    // 2. 引用成员演示
    std::cout << "\n2. 引用成员演示:" << std::endl;
    std::string refName = "RefDemo";
    int counter = 0;
    std::cout << "创建前 counter = " << counter << std::endl;
    
    ReferenceMemberDemo refDemo(refName, counter);
    std::cout << "创建后 counter = " << counter << std::endl;
    refDemo.display();
    
    // 3. 复杂对象初始化
    std::cout << "\n3. 复杂对象初始化:" << std::endl;
    std::string complexName = "ComplexDemo";
    InitializationListDemo complexDemo(200, complexName, 5);
    complexDemo.display();
    
    // 4. 性能对比
    std::cout << "\n4. 性能对比:" << std::endl;
    PerformanceDemo perf1(1000, "InitList");
    PerformanceDemo perf2(1000, "Assignment", true);
    
    // 5. 初始化列表的多种形式
    std::cout << "\n5. 初始化列表的多种形式:" << std::endl;
    
    // 基本类型初始化
    class BasicInit {
    private:
        int a, b, c;
    public:
        BasicInit(int x, int y, int z) : a(x), b(y), c(z) {
            std::cout << "BasicInit: a=" << a << ", b=" << b << ", c=" << c << std::endl;
        }
    };
    
    BasicInit basic(1, 2, 3);
    
    // 表达式初始化
    class ExpressionInit {
    private:
        int sum, product;
    public:
        ExpressionInit(int x, int y) : sum(x + y), product(x * y) {
            std::cout << "ExpressionInit: sum=" << sum << ", product=" << product << std::endl;
        }
    };
    
    ExpressionInit expr(4, 5);
    
    // 函数调用初始化
    class FunctionInit {
    private:
        int value;
        std::string name;
    public:
        FunctionInit(int x) : value(x), name("Item_" + std::to_string(x)) {
            std::cout << "FunctionInit: value=" << value << ", name=" << name << std::endl;
        }
    };
    
    FunctionInit func(42);
    
    std::cout << "\n作用域结束，析构函数调用:" << std::endl;
    
    return 0;
}






















