#include <iostream>
#include <string>

class SimpleClass {
private:
    std::string name;
    int value;
    
public:
    SimpleClass(const std::string& name, int value) : name(name), value(value) {
        std::cout << "构造: " << name << std::endl;
    }
    
    ~SimpleClass() {
        std::cout << "析构: " << name << std::endl;
    }
    
    void show() const {
        std::cout << name << " = " << value << std::endl;
    }
};

int main() {
    std::cout << "=== new返回指针 vs 普通对象 ===" << std::endl;
    
    // 1. 普通对象（栈上）
    std::cout << "\n1. 普通对象（栈上）:" << std::endl;
    SimpleClass obj("StackObject", 100);
    std::cout << "obj类型: " << typeid(obj).name() << std::endl;
    std::cout << "obj地址: " << &obj << std::endl;
    obj.show();
    
    // 2. new对象（堆上）
    std::cout << "\n2. new对象（堆上）:" << std::endl;
    SimpleClass* ptr = new SimpleClass("HeapObject", 200);
    std::cout << "ptr类型: " << typeid(ptr).name() << std::endl;
    std::cout << "ptr值(地址): " << ptr << std::endl;
    ptr->show();
    
    // 3. 访问方式对比
    std::cout << "\n3. 访问方式对比:" << std::endl;
    std::cout << "普通对象访问: obj.show()" << std::endl;
    obj.show();
    
    std::cout << "指针对象访问: ptr->show()" << std::endl;
    ptr->show();
    
    // 4. 内存管理
    std::cout << "\n4. 内存管理:" << std::endl;
    std::cout << "普通对象: 自动管理（栈上）" << std::endl;
    std::cout << "new对象: 手动管理（堆上）" << std::endl;
    
    // 手动释放new的对象
    std::cout << "手动释放new的对象..." << std::endl;
    delete ptr;
    
    std::cout << "\n作用域结束，栈上对象自动析构:" << std::endl;
    
    return 0;
}






















