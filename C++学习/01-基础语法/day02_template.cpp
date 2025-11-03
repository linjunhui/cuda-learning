#include <iostream>

template<typename T>
T add(T a, T b) {
    return a + b;
}


template<typename T, typename U>
T multipy(T a, U b) {
    return a * b;
}

int main() {

    int a = 10, b = 20;
    int c = add(a, b);

    std::cout << c << std::endl;

    std::cout << multipy(10, 0.5) << std::endl;
    return 0;
}