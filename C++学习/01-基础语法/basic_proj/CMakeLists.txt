# ============================================================================
# CMakeLists.txt - 基础语法项目配置文件
# ============================================================================

# ----------------------------------------------------------------------------
# 1. CMake 基本配置
# ----------------------------------------------------------------------------

# 设置 CMake 最小要求版本
cmake_minimum_required(VERSION 3.14)

# 项目配置
project(BASIC VERSION 1.0.0 LANGUAGES CXX)


# ----------------------------------------------------------------------------
# 2. C++ 标准配置
# ----------------------------------------------------------------------------

# C++ 标准设置
# 对应 g++ 命令：-std=gnu++17 (或 -std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ----------------------------------------------------------------------------
# 3. 目录和路径设置
# ----------------------------------------------------------------------------

# 设置包含目录，使 #include "math.h" 能找到头文件
# 对应 g++ 命令：-I/home/jonson/cuda-learning/C++学习/01-基础语法/basic_proj/include
# 简写形式：-I include (相对于项目根目录)
include_directories(include)


# ----------------------------------------------------------------------------
# 4. 静态库配置
# ----------------------------------------------------------------------------

# 库的源文件
set(LIB_SOURCES
    src/math.cpp
)

# 创建静态库
# 第一步：编译目标文件
#   对应 g++ 命令：g++ -std=gnu++17 -I include -c src/math.cpp -o math.cpp.o
# 第二步：打包成静态库
#   对应 ar 命令：ar qc libmathLib.a math.cpp.o
#   对应 ranlib 命令：ranlib libmathLib.a
add_library(mathLib STATIC ${LIB_SOURCES})


# ----------------------------------------------------------------------------
# 5. 可执行文件配置
# ----------------------------------------------------------------------------

# 可执行文件的源文件
set(EXE_SOURCES
    src/main.cpp
)

# 创建可执行文件
# 第一步：编译目标文件
#   对应 g++ 命令：g++ -std=gnu++17 -I include -c src/main.cpp -o main.cpp.o
# 第二步：链接生成可执行文件
#   对应 g++ 命令：g++ main.cpp.o -o basic libmathLib.a
add_executable(basic ${EXE_SOURCES})

# 链接库到可执行文件
# 对应 g++ 命令中的：libmathLib.a (直接链接静态库文件)
# 如果是动态库，对应：-L. -lmathLib
target_link_libraries(basic PRIVATE mathLib)


# ----------------------------------------------------------------------------
# 6. 测试框架配置 (Google Test)
# ----------------------------------------------------------------------------

# 使用 FetchContent 自动下载 Google Test
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# 配置 Google Test 构建选项
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

# 下载并构建 Google Test（此时会使用上面设置的选项）
FetchContent_MakeAvailable(googletest)


# ----------------------------------------------------------------------------
# 7. 测试目标配置
# ----------------------------------------------------------------------------

# 启用测试
enable_testing()

# 测试可执行文件的源文件
set(TEST_SOURCES
    tests/math_test.cpp
    src/math.cpp  # 需要链接实现文件
)

# 创建测试可执行文件
# 对应 g++ 命令：
#   g++ -std=gnu++17 -I include tests/math_test.cpp src/math.cpp -o math_test \
#       -lgtest -lgtest_main -pthread
add_executable(math_test ${TEST_SOURCES})

# 链接 Google Test 和项目库
target_link_libraries(math_test
    PRIVATE
        mathLib
        GTest::gtest_main  # 包含 main() 函数
)

# 注册测试
add_test(NAME MathTest COMMAND math_test)
