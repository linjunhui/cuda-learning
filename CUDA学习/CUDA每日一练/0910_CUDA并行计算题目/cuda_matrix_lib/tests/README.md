# CUDA矩阵库单元测试

## 测试文件说明

### test_error_handler.cpp
测试CUDA错误处理类(`CudaError`)的功能，包括：

1. **基本功能测试** - 验证异常抛出和错误信息
2. **不同错误码测试** - 测试各种CUDA错误码的处理
3. **继承关系测试** - 验证异常类的继承关系
4. **调试信息测试** - 验证文件名、行号、函数名等调试信息
5. **默认参数测试** - 测试默认函数参数的处理

## 编译和运行

### 使用Makefile（推荐）
```bash
# 在项目根目录下
make test          # 编译并运行测试
make clean         # 清理编译文件
make all           # 只编译不运行
```

### 手动编译
```bash
# 编译核心库
g++ -std=c++11 -O2 -Wall -Wextra -Wpedantic -Iinclude -Iinclude/cuda_matrix_lib -c src/core/error_handler.cpp -o src/core/error_handler.o

# 编译测试
g++ -std=c++11 -O2 -Wall -Wextra -Wpedantic -Iinclude -Iinclude/cuda_matrix_lib -c tests/unit_tests/test_error_handler.cpp -o tests/unit_tests/test_error_handler.o

# 链接生成可执行文件
g++ -std=c++11 -O2 -Wall -Wextra -Wpedantic -Iinclude -Iinclude/cuda_matrix_lib -o tests/unit_tests/test_error_handler src/core/error_handler.o tests/unit_tests/test_error_handler.o -lcudart

# 运行测试
./tests/unit_tests/test_error_handler
```

## 测试输出说明

测试运行时会输出：
- 每个测试的进度和结果
- CUDA错误异常的错误信息
- printf调试输出（包含文件、行号、错误码、函数信息）
- 最终测试结果

## 预期结果

所有测试应该通过，输出类似：
```
=== CUDA错误处理类单元测试 ===
测试1: CUDA错误异常基本功能
✓ 异常被正确抛出
✓ 错误信息包含CUDA错误描述
...
=== 所有测试通过！ ===
```

## 依赖要求

- CUDA Toolkit (用于cuda_runtime.h)
- C++11兼容的编译器
- Linux/Unix环境
