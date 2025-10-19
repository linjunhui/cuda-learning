#include <cstdlib>
#include <iostream>

// 初始化矩阵
void init_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = i+1; // 随机初始化为0-99之间的数
        }
    }
}


void compare_matrix(float* matrix1, float* matrix2, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (matrix1[i * cols + j] != matrix2[i * cols + j]) {
                std::cout << "Matrices are not equal at (" << i << ", " << j << "): "
                          << matrix1[i * cols + j] << " != " << matrix2[i * cols + j] << std::endl;
                return;
            }
        }
    }
    std::cout << "Matrices are equal." << std::endl;
}

// 打印矩阵
void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}