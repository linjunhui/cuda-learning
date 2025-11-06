#include <iostream>
#include <string>

/*
演示new操作符返回指针的特性
*/

class Student {
private:
    std::string name;
    int age;
    
public:
    Student(const std::string& name, int age) : name(name), age(age) {
        std::cout << "Student构造函数调用: " << name << std::endl;
    }
    
    ~Student() {
        std::cout << "Student析构函数调用: " << name << std::endl;
    }
    
    void display() const {
        std::cout << "Name: " << name << ", Age: " << age << std::endl;
    }
    
    std::string getName() const { return name; }
    int getAge() const { return age; }
};

int main() {
    std::cout << "=== new操作符返回指针演示 ===" << std::endl;
    
    // 1. new操作符返回指针
    std::cout << "\n1. new操作符返回指针:" << std::endl;
    
    // new返回Student*类型的指针
    Student* ptr1 = new Student("Alice", 20);
    std::cout << "ptr1的类型: " << typeid(ptr1).name() << std::endl;
    std::cout << "ptr1的值(地址): " << ptr1 << std::endl;
    
    // 使用指针访问对象
    ptr1->display();
    std::cout << "通过指针访问: " << ptr1->getName() << std::endl;
    
    // 2. 指针与普通对象的区别
    std::cout << "\n2. 指针与普通对象的区别:" << std::endl;
    
    // 普通对象（栈上）
    Student obj("Bob", 21);
    std::cout << "obj的类型: " << typeid(obj).name() << std::endl;
    std::cout << "obj的地址: " << &obj << std::endl;
    obj.display();
    
    // 指针对象（堆上）
    Student* ptr2 = new Student("Charlie", 22);
    std::cout << "ptr2的类型: " << typeid(ptr2).name() << std::endl;
    std::cout << "ptr2的值(地址): " << ptr2 << std::endl;
    ptr2->display();
    
    // 3. 指针的大小
    std::cout << "\n3. 指针的大小:" << std::endl;
    std::cout << "Student对象大小: " << sizeof(Student) << " 字节" << std::endl;
    std::cout << "Student指针大小: " << sizeof(Student*) << " 字节" << std::endl;
    std::cout << "指针大小(通常): " << sizeof(void*) << " 字节" << std::endl;
    
    // 4. 动态分配数组
    std::cout << "\n4. 动态分配数组:" << std::endl;
    
    // new[]也返回指针
    Student* students = new Student[3]{
        Student("David", 23),
        Student("Eve", 24),
        Student("Frank", 25)
    };
    
    std::cout << "students数组指针: " << students << std::endl;
    
    // 遍历数组
    for(int i = 0; i < 3; ++i) {
        std::cout << "students[" << i << "]: ";
        students[i].display();
    }
    
    // 5. 指针的算术运算
    std::cout << "\n5. 指针的算术运算:" << std::endl;
    std::cout << "students[0]地址: " << &students[0] << std::endl;
    std::cout << "students[1]地址: " << &students[1] << std::endl;
    std::cout << "地址差: " << &students[1] - &students[0] << " 个Student对象" << std::endl;
    
    // 6. 内存管理
    std::cout << "\n6. 内存管理:" << std::endl;
    
    // 必须手动释放内存
    std::cout << "释放单个对象内存..." << std::endl;
    delete ptr1;  // 释放单个对象
    delete ptr2;  // 释放单个对象
    
    std::cout << "释放数组内存..." << std::endl;
    delete[] students;  // 释放数组
    
    // 7. 空指针检查
    std::cout << "\n7. 空指针检查:" << std::endl;
    
    Student* nullPtr = nullptr;
    std::cout << "nullPtr是否为null: " << (nullPtr == nullptr ? "是" : "否") << std::endl;
    
    // 安全的使用方式
    if(nullPtr != nullptr) {
        nullPtr->display();
    } else {
        std::cout << "指针为空，无法访问对象" << std::endl;
    }
    
    // 8. 现代C++的替代方案
    std::cout << "\n8. 现代C++的替代方案:" << std::endl;
    
    // 使用智能指针（C++11）
    std::cout << "推荐使用智能指针而不是裸指针" << std::endl;
    std::cout << "例如: std::unique_ptr<Student>, std::shared_ptr<Student>" << std::endl;
    
    std::cout << "\n作用域结束，栈上对象自动析构:" << std::endl;
    
    return 0;
}






















