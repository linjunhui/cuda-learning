#include <iostream>
#include <string>

// 方法1：使用初始化列表（推荐）
class Student1 {
private:
    std::string name;
    int age;
    
public:
    Student1(const std::string& name, int age) 
        : name(name), age(age) {
        std::cout << "方法1-初始化列表: " << this->name << ", " << this->age << std::endl;
    }
    
    void display() const {
        std::cout << "Student1: " << name << ", " << age << std::endl;
    }
};

// 方法2：使用this指针
class Student2 {
private:
    std::string name;
    int age;
    
public:
    Student2(const std::string& name, int age) {
        this->name = name;
        this->age = age;
        std::cout << "方法2-this指针: " << this->name << ", " << this->age << std::endl;
    }
    
    void display() const {
        std::cout << "Student2: " << name << ", " << age << std::endl;
    }
};

// 方法3：使用不同的参数名
class Student3 {
private:
    std::string name;
    int age;
    
public:
    Student3(const std::string& n, int a) 
        : name(n), age(a) {
        std::cout << "方法3-不同参数名: " << name << ", " << age << std::endl;
    }
    
    void display() const {
        std::cout << "Student3: " << name << ", " << age << std::endl;
    }
};

int main() {
    std::cout << "=== 构造函数形参与成员变量同名演示 ===" << std::endl;
    
    std::cout << "\n创建对象:" << std::endl;
    Student1 s1("Alice", 20);
    Student2 s2("Bob", 21);
    Student3 s3("Charlie", 22);
    
    std::cout << "\n显示信息:" << std::endl;
    s1.display();
    s2.display();
    s3.display();
    
    std::cout << "\n三种方法都正确，推荐使用初始化列表！" << std::endl;
    
    return 0;
}























