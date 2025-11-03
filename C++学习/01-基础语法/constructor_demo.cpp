#include <iostream>
#include <string>
#include <vector>

/*
演示构造函数和初始化列表的完整使用
*/

class Student {
private:
    std::string name;
    int age;
    std::vector<double> scores;
    static int totalStudents;
    
public:
    // 默认构造函数
    Student() : name("Unknown"), age(0) {
        scores.resize(3, 0.0);
        totalStudents++;
        std::cout << "默认构造函数调用: " << name << std::endl;
    }
    
    // 参数化构造函数 - 使用初始化列表
    Student(const std::string& n, int a) : name(n), age(a) {
        scores.resize(3, 0.0);
        totalStudents++;
        std::cout << "参数化构造函数调用: " << name << std::endl;
    }
    
    // 复制构造函数
    Student(const Student& other) : name(other.name + "_copy"), age(other.age) {
        scores = other.scores;  // vector的复制构造函数
        totalStudents++;
        std::cout << "复制构造函数调用: " << name << std::endl;
    }
    
    // 移动构造函数 (C++11)
    Student(Student&& other) noexcept 
        : name(std::move(other.name)), age(other.age), scores(std::move(other.scores)) {
        other.age = 0;
        other.name = "Moved";
        totalStudents++;
        std::cout << "移动构造函数调用: " << name << std::endl;
    }
    
    // 析构函数
    ~Student() {
        totalStudents--;
        std::cout << "析构函数调用: " << name << std::endl;
    }
    
    // 赋值操作符
    Student& operator=(const Student& other) {
        if(this != &other) {
            name = other.name + "_assigned";
            age = other.age;
            scores = other.scores;
            std::cout << "赋值操作符调用: " << name << std::endl;
        }
        return *this;
    }
    
    // 移动赋值操作符
    Student& operator=(Student&& other) noexcept {
        if(this != &other) {
            name = std::move(other.name);
            age = other.age;
            scores = std::move(other.scores);
            other.age = 0;
            other.name = "Moved";
            std::cout << "移动赋值操作符调用: " << name << std::endl;
        }
        return *this;
    }
    
    // 成员函数
    void addScore(double score) {
        if(scores.size() < 3) {
            scores.push_back(score);
        } else {
            scores[0] = scores[1];
            scores[1] = scores[2];
            scores[2] = score;
        }
    }
    
    double getAverage() const {
        if(scores.empty()) return 0.0;
        double sum = 0.0;
        for(double score : scores) {
            sum += score;
        }
        return sum / scores.size();
    }
    
    // 访问器函数
    const std::string& getName() const { return name; }
    int getAge() const { return age; }
    const std::vector<double>& getScores() const { return scores; }
    
    // 静态成员函数
    static int getTotalStudents() {
        return totalStudents;
    }
    
    // 显示信息
    void display() const {
        std::cout << "学生: " << name << ", 年龄: " << age 
                  << ", 平均分: " << getAverage() << std::endl;
    }
};

// 静态成员变量定义
int Student::totalStudents = 0;

// 演示函数
Student createStudent(const std::string& name, int age) {
    std::cout << "创建临时学生对象..." << std::endl;
    return Student(name, age);
}

void processStudent(Student s) {
    std::cout << "处理学生: " << s.getName() << std::endl;
}

int main() {
    std::cout << "=== 构造函数和初始化列表演示 ===" << std::endl;
    
    // 1. 默认构造函数
    std::cout << "\n1. 默认构造函数:" << std::endl;
    Student s1;
    s1.display();
    
    // 2. 参数化构造函数
    std::cout << "\n2. 参数化构造函数:" << std::endl;
    Student s2("Alice", 20);
    s2.addScore(85.5);
    s2.addScore(92.0);
    s2.addScore(78.5);
    s2.display();
    
    // 3. 复制构造函数
    std::cout << "\n3. 复制构造函数:" << std::endl;
    Student s3 = s2;  // 调用复制构造函数
    s3.display();
    
    // 4. 移动构造函数
    std::cout << "\n4. 移动构造函数:" << std::endl;
    Student s4 = createStudent("Bob", 21);  // 临时对象，调用移动构造函数
    s4.display();
    
    // 5. 赋值操作符
    std::cout << "\n5. 赋值操作符:" << std::endl;
    Student s5("Charlie", 22);
    s5 = s2;  // 调用赋值操作符
    s5.display();
    
    // 6. 移动赋值操作符
    std::cout << "\n6. 移动赋值操作符:" << std::endl;
    Student s6("David", 23);
    s6 = createStudent("Eve", 24);  // 临时对象，调用移动赋值操作符
    s6.display();
    
    // 7. 函数参数传递
    std::cout << "\n7. 函数参数传递:" << std::endl;
    processStudent(s2);  // 按值传递，调用复制构造函数
    
    // 8. 静态成员函数
    std::cout << "\n8. 静态成员函数:" << std::endl;
    std::cout << "当前学生总数: " << Student::getTotalStudents() << std::endl;
    
    // 9. 作用域结束
    std::cout << "\n9. 作用域结束，析构函数调用:" << std::endl;
    
    return 0;
}





















