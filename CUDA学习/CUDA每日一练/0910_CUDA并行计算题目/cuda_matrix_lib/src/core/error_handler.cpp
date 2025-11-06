#include "error_handler.h"

using namespace cuda_matrix_lib;

// CUDA错误异常类的构造函数
// 使用初始化列表语法来初始化基类和成员变量
CudaError::CudaError(cudaError_t err_code, const char *file, int line, const char *function) 
: std::runtime_error(cudaGetErrorString(err_code)),  // 初始化基类std::runtime_error，传入CUDA错误描述字符串
  err_code_(err_code),                               // 初始化成员变量：CUDA错误代码
  file_(file),                                       // 初始化成员变量：发生错误的文件名
  line_(line),                                       // 初始化成员变量：发生错误的行号
  function_(function) {                              // 初始化成员变量：发生错误的函数名
    printf("CUDA error at %s:%d, code=%d (%s)\n", file_, line_, err_code_, function_);
}