#include <iostream>
#include <string>

/*
演示构造函数形参与成员变量同名的情况
*/

class SameNameDemo {
private:
    std::string name;
    int age;
    double salary;
    
public:
    // 方法1：使用this指针区分
    SameNameDemo(const std::string& name, int age, double salary) {
        this->name = name;      // this->name 指向成员变量
        this->age = age;        // this->age 指向成员变量
        this->salary = salary;  // this->salary 指向成员变量
        std::cout << "使用this指针构造: " << this->name << std::endl;
    }
    
    // 方法2：使用初始化列表（推荐）
    SameNameDemo(const std::string& n, int a, double s) 
        : name(n), age(a), salary(s) {
        std::cout << "使用初始化列表构造: " << name << std::endl;
    }
    
    // 方法3：形参与成员变量同名，使用初始化列表
    SameNameDemo(const std::string& name, int age, double salary, bool useInitList) 
        : name(name), age(age), salary(salary) {
        // 在初始化列表中，左边的name是成员变量，右边的name是参数
        std::cout << "形参与成员变量同名，使用初始化列表: " << name << std::endl;
    }
    
    // 显示信息
    void display() const {
        std::cout << "Name: " << name << ", Age: " << age << ", Salary: " << salary << std::endl;
    }
    
    // 设置函数演示
    void setName(const std::string& name) {
        this->name = name;  // 必须使用this指针
    }
    
    void setAge(int age) {
        this->age = age;   // 必须使用this指针
    }
};

// 演示错误的情况
class ErrorDemo {
private:
    std::string name;
    int age;
    
public:
    // ❌ 错误：形参与成员变量同名，没有使用this指针
    ErrorDemo(const std::string& name, int age) {
        // name = name;  // 错误：参数赋值给自己，成员变量没有被赋值
        // age = age;    // 错误：参数赋值给自己，成员变量没有被赋值
        
        // 正确的做法：
        this->name = name;
        this->age = age;
        std::cout << "ErrorDemo构造: " << this->name << std::endl;
    }
    
    void display() const {
        std::cout << "Name: " << name << ", Age: " << age << std::endl;
    }
};

// 演示最佳实践
class BestPracticeDemo {
private:
    std::string name;
    int age;
    double salary;
    
public:
    // ✅ 最佳实践1：使用初始化列表
    BestPracticeDemo(const std::string& name, int age, double salary) 
        : name(name), age(age), salary(salary) {
        std::cout << "最佳实践：初始化列表，形参与成员变量同名" << std::endl;
    }
    
    // ✅ 最佳实践2：使用不同的参数名
    BestPracticeDemo(const std::string& n, int a, double s, bool alternative) 
        : name(n), age(a), salary(s) {
        std::cout << "最佳实践：不同参数名" << std::endl;
    }
    
    // ✅ 最佳实践3：使用this指针（在构造函数体中）
    BestPracticeDemo(const std::string& name, int age, double salary, int method) {
        this->name = name;
        this->age = age;
        this->salary = salary;
        std::cout << "最佳实践：使用this指针" << std::endl;
    }
    
    void display() const {
        std::cout << "Name: " << name << ", Age: " << age << ", Salary: " << salary << std::endl;
    }
    
    // 成员函数中的同名参数处理
    void updateInfo(const std::string& name, int age) {
        this->name = name;  // 必须使用this指针
        this->age = age;    // 必须使用this指针
        std::cout << "更新信息: " << this->name << std::endl;
    }
};

// 演示复杂情况
class ComplexDemo {
private:
    std::string name;
    int* data;
    size_t size;
    
public:
    // 形参与成员变量同名，使用初始化列表
    ComplexDemo(const std::string& name, int* data, size_t size) 
        : name(name), data(new int[size]), size(size) {
        // 复制数据
        for(size_t i = 0; i < size; ++i) {
            this->data[i] = data[i];
        }
        std::cout << "ComplexDemo构造: " << this->name << std::endl;
    }
    
    // 析构函数
    ~ComplexDemo() {
        delete[] data;
        std::cout << "ComplexDemo析构: " << name << std::endl;
    }
    
    // 赋值操作符
    ComplexDemo& operator=(const ComplexDemo& other) {
        if(this != &other) {
            name = other.name;
            size = other.size;
            
            delete[] data;
            data = new int[size];
            for(size_t i = 0; i < size; ++i) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }
    
    void display() const {
        std::cout << "Name: " << name << ", Size: " << size << ", Data: ";
        for(size_t i = 0; i < size; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    std::cout << "=== 构造函数形参与成员变量同名演示 ===" << std::endl;
    
    // 1. 使用this指针
    std::cout << "\n1. 使用this指针:" << std::endl;
    SameNameDemo demo1("Alice", 25, 50000.0);
    demo1.display();
    
    // 2. 使用初始化列表
    std::cout << "\n2. 使用初始化列表:" << std::endl;
    SameNameDemo demo2("Bob", 30, 60000.0);
    demo2.display();
    
    // 3. 形参与成员变量同名，使用初始化列表
    std::cout << "\n3. 形参与成员变量同名，使用初始化列表:" << std::endl;
    SameNameDemo demo3("Charlie", 35, 70000.0, true);
    demo3.display();
    
    // 4. 错误演示
    std::cout << "\n4. 错误演示（已修正）:" << std::endl;
    ErrorDemo error("David", 28);
    error.display();
    
    // 5. 最佳实践演示
    std::cout << "\n5. 最佳实践演示:" << std::endl;
    BestPracticeDemo best1("Eve", 32, 80000.0);
    best1.display();
    
    BestPracticeDemo best2("Frank", 29, 75000.0, true);
    best2.display();
    
    BestPracticeDemo best3("Grace", 31, 85000.0, 1);
    best3.display();
    
    // 6. 复杂情况演示
    std::cout << "\n6. 复杂情况演示:" << std::endl;
    int testData[] = {1, 2, 3, 4, 5};
    ComplexDemo complex("Complex", testData, 5);
    complex.display();
    
    // 7. 成员函数中的同名参数
    std::cout << "\n7. 成员函数中的同名参数:" << std::endl;
    best1.updateInfo("UpdatedEve", 33);
    best1.display();
    
    std::cout << "\n作用域结束，析构函数调用:" << std::endl;
    
    return 0;
}






















