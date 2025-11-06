#include <iostream>

template<typename T>
void init_matrix(T* matrix, int M, int N) {
    for(int i = 0; i < M*N; i++) {
        matrix[i] = i;
    }
}

template<typename T>
void print_matrix(T* matrix, int M, int N) {
    for(int i = 0; i < M*N; i++) {
        std::cout << matrix[i] << " ";
        if((i+1) % N == 0) std::cout << std::endl;
    }
}