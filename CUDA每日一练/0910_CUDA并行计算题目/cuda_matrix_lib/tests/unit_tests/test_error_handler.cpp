#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include "error_handler.h"

using namespace cuda_matrix_lib;

/**
 * 测试CUDA错误异常类的基本功能
 */
void test_cuda_error_basic() {
    std::cout << "测试1: CUDA错误异常基本功能" << std::endl;
    
    try {
        // 创建一个CUDA错误异常
        throw CudaError(cudaErrorInvalidValue, __FILE__, __LINE__, __FUNCTION__);
    } catch (const CudaError& e) {
        // 验证异常被正确抛出
        std::cout << "✓ 异常被正确抛出" << std::endl;
        std::cout << "  错误信息: " << e.what() << std::endl;
        
        // 验证异常信息包含CUDA错误描述
        std::string error_msg = e.what();
        assert(error_msg.find("invalid argument") != std::string::npos || 
               error_msg.find("Invalid") != std::string::npos);
        std::cout << "✓ 错误信息包含CUDA错误描述" << std::endl;
    }
}

/**
 * 测试不同CUDA错误码的处理
 */
void test_different_cuda_errors() {
    std::cout << "\n测试2: 不同CUDA错误码处理" << std::endl;
    
    // 测试内存分配错误
    try {
        throw CudaError(cudaErrorMemoryAllocation, __FILE__, __LINE__, __FUNCTION__);
    } catch (const CudaError& e) {
        std::cout << "✓ 内存分配错误: " << e.what() << std::endl;
    }
    
    // 测试设备未找到错误
    try {
        throw CudaError(cudaErrorNoDevice, __FILE__, __LINE__, __FUNCTION__);
    } catch (const CudaError& e) {
        std::cout << "✓ 设备未找到错误: " << e.what() << std::endl;
    }
    
    // 测试成功状态（不应该抛出异常）
    try {
        throw CudaError(cudaSuccess, __FILE__, __LINE__, __FUNCTION__);
    } catch (const CudaError& e) {
        std::cout << "✓ 成功状态: " << e.what() << std::endl;
    }
}

/**
 * 测试异常继承关系
 */
void test_exception_inheritance() {
    std::cout << "\n测试3: 异常继承关系" << std::endl;
    
    try {
        throw CudaError(cudaErrorInvalidConfiguration, __FILE__, __LINE__, __FUNCTION__);
    } catch (const std::runtime_error& e) {
        // 验证可以捕获基类异常
        std::cout << "✓ 可以捕获基类std::runtime_error异常" << std::endl;
        std::cout << "  基类错误信息: " << e.what() << std::endl;
    }
    
    try {
        throw CudaError(cudaErrorInvalidConfiguration, __FILE__, __LINE__, __FUNCTION__);
    } catch (const std::exception& e) {
        // 验证可以捕获更上层的基类异常
        std::cout << "✓ 可以捕获基类std::exception异常" << std::endl;
        std::cout << "  异常基类信息: " << e.what() << std::endl;
    }
}

/**
 * 测试调试信息参数
 */
void test_debug_info() {
    std::cout << "\n测试4: 调试信息参数" << std::endl;
    
    const char* test_file = "test_file.cpp";
    int test_line = 123;
    const char* test_function = "test_function";
    
    try {
        throw CudaError(cudaErrorInvalidValue, test_file, test_line, test_function);
    } catch (const CudaError& e) {
        std::cout << "✓ 调试信息参数测试完成" << std::endl;
        std::cout << "  应该看到printf输出包含文件、行号、错误码和函数信息" << std::endl;
    }
}

/**
 * 测试默认函数参数
 */
void test_default_function_parameter() {
    std::cout << "\n测试5: 默认函数参数" << std::endl;
    
    try {
        // 测试不提供function参数的情况
        throw CudaError(cudaErrorInvalidValue, __FILE__, __LINE__);
    } catch (const CudaError& e) {
        std::cout << "✓ 默认函数参数测试完成" << std::endl;
        std::cout << "  应该看到printf输出中function为nullptr" << std::endl;
    }
}

/**
 * 主测试函数
 */
int main() {
    std::cout << "=== CUDA错误处理类单元测试 ===" << std::endl;
    
    try {
        test_cuda_error_basic();
        test_different_cuda_errors();
        test_exception_inheritance();
        test_debug_info();
        test_default_function_parameter();
        
        std::cout << "\n=== 所有测试通过！ ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "\n❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
}
