# CUDA矩阵库 CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(cuda_matrix_lib VERSION 1.0.0 LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 设置CUDA编译选项
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含目录
include_directories(include)
include_directories(include/cuda_matrix_lib)

# 查找CUDA - 使用现代方式
find_package(CUDAToolkit REQUIRED)

# 核心库源文件
set(CORE_SOURCES
    src/core/error_handler.cpp
)

# 创建核心库静态库
add_library(cuda_matrix_lib_core STATIC ${CORE_SOURCES})

# 设置库的包含目录
target_include_directories(cuda_matrix_lib_core 
    PUBLIC 
        include
        include/cuda_matrix_lib
)

# 链接CUDA库
target_link_libraries(cuda_matrix_lib_core 
    PUBLIC 
        CUDA::cudart
)

# 测试源文件
set(TEST_SOURCES
    tests/unit_tests/test_error_handler.cpp
)

# 创建测试可执行文件
add_executable(test_error_handler ${TEST_SOURCES})

# 链接核心库到测试
target_link_libraries(test_error_handler 
    PRIVATE 
        cuda_matrix_lib_core
)

# 启用测试
enable_testing()

# 添加测试
add_test(NAME error_handler_tests COMMAND test_error_handler)

# 设置测试工作目录
set_tests_properties(error_handler_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 安装规则
install(TARGETS cuda_matrix_lib_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# 打印配置信息
message(STATUS "CUDA矩阵库配置信息:")
message(STATUS "  项目版本: ${PROJECT_VERSION}")
message(STATUS "  C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA标准: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "  安装前缀: ${CMAKE_INSTALL_PREFIX}")
