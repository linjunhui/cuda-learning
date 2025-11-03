#include <iostream>

class Rectangle {
    private:
        int length;
        int width;
    public:
        Rectangle(int length, int width) {
            this->length = length;
            this->width = width;
        }

        int area() {
            return this->length * this->width;
        }
};

class Cat {
    private:
        std::string name;
        int age;
    public:
        Cat(const std::string& name, int age):name(name), age(age) {

        }
        
        std::string getName() {
            return this->name;
        }

};

int main() {
    auto rect = Rectangle(10, 100);

    std::cout << rect.area() << std::endl;

    auto cat = new Cat("hi", 100);
    std::cout << cat->getName() << std::endl;
    return 0;
}