#ifndef CUDA_MATRIX_LIB_ERROR_HANDLER_H_
#define CUDA_MATRIX_LIB_ERROR_HANDLER_H_ 

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace cuda_matrix_lib { 
/**
 * 将CUDA错误码转为C++异常
 */

class CudaError : public std::runtime_error {
public:
    /**
     * @brief 构造函数
     * @param err_code CUDA错误码
     * @param file 错误发生的文件名
     * @param line 错误发生的行号
     * @param function 错误发生的函数名
     */
    explicit CudaError(cudaError_t err_code,
                    const char *file,
                    int line,
                    const char* function = nullptr);

private:
    cudaError_t err_code_;      // CUDA错误代码
    const char* file_;          // 发生错误的文件名
    int line_;                  // 发生错误的行号
    const char* function_;      // 发生错误的函数名
};

}

#endif