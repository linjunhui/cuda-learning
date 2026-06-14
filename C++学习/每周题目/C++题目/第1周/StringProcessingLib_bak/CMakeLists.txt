cmake_minimum_required(VERSION 3.14)
project(StringProcessingLib VERSION 1.0.0 LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 包含目录
include_directories(include)

# ========== 核心库源文件 ==========
set(SOURCES
    src/memory_pool/fixed_size_pool.cpp
)

# 创建库
add_library(StringProcessingLib ${SOURCES})

# 设置库的包含目录
target_include_directories(StringProcessingLib
    PUBLIC
        include
)

# ========== Google Test 配置 ==========
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# ========== 配置 Google Test 构建选项 ==========
# 注意：这些选项必须在 FetchContent_MakeAvailable() 之前设置
# 这样在下载和构建 Google Test 时会应用这些配置

# 选项1：BUILD_GMOCK
#   作用：控制是否构建 Google Mock 库
#   说明：
#     - OFF：只构建 Google Test，不构建 Mock 库（节省编译时间和空间）
#     - ON：同时构建 Google Test 和 Google Mock（需要 Mock 功能时使用）
#   当前设置：OFF（我们只需要单元测试，不需要 Mock 功能）
#   语法说明：
#     - CACHE：将变量写入 CMake 缓存，使其在后续配置中可用
#     - BOOL：变量类型为布尔值（ON/OFF）
#     - ""：变量描述（空字符串）
#     - FORCE：强制设置，覆盖缓存中可能存在的旧值
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

# 选项2：INSTALL_GTEST
#   作用：控制是否在 cmake --install 时安装 Google Test 到系统目录
#   说明：
#     - OFF：不安装（使用 FetchContent 时通常不需要安装到系统）
#     - ON：安装到系统目录（如 /usr/local/lib, /usr/local/include）
#   当前设置：OFF（FetchContent 下载的库只用于当前项目，无需全局安装）
#   语法说明：
#     - CACHE：将变量写入 CMake 缓存
#     - BOOL：变量类型为布尔值（ON/OFF）
#     - ""：变量描述（空字符串）
#     - FORCE：强制设置，覆盖缓存中可能存在的旧值
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# ========== 测试配置 ==========
enable_testing()

# 测试源文件
set(TEST_SOURCES
    tests/unit_tests/memory_pool/test_fixed_size_pool.cpp
)

# 创建测试可执行文件
add_executable(test_fixed_size_pool ${TEST_SOURCES})

# 链接库
target_link_libraries(test_fixed_size_pool
    PRIVATE
        StringProcessingLib
        GTest::gtest_main
)

# 添加测试
add_test(NAME FixedSizePoolTest COMMAND test_fixed_size_pool)

# 设置测试工作目录
set_tests_properties(FixedSizePoolTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

