# CMake 基础教程

## 1. CMake 简介

### 1.1 什么是 CMake

CMake 是一个跨平台的**构建系统生成器**，它可以：
- 生成 Unix Makefiles、Visual Studio 项目文件等
- 管理编译配置、依赖关系
- 支持多种编译器（GCC、Clang、MSVC等）

### 1.2 为什么使用 CMake

- ✅ **跨平台**：Windows、Linux、macOS 都支持
- ✅ **自动化**：自动检测编译器和库
- ✅ **依赖管理**：可以集成第三方库
- ✅ **标准化**：C++ 项目的事实标准

## 2. CMake 基本语法

### 2.1 最小 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0.0 LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 创建可执行文件
add_executable(my_app main.cpp)
```

### 2.2 基本命令说明

#### **cmake_minimum_required**
```cmake
cmake_minimum_required(VERSION 3.10)
```
- 指定 CMake 的最低版本要求
- 必须放在文件的第一行

#### **project**
```cmake
project(StringProcessingLib VERSION 1.0.0 LANGUAGES CXX)
```
- 定义项目名称和版本
- `LANGUAGES CXX` 指定使用 C++

#### **set**
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```
- 设置变量值
- `CMAKE_CXX_STANDARD`：C++ 标准版本
- `CMAKE_CXX_STANDARD_REQUIRED`：强制使用指定标准

## 3. 构建库和可执行文件

### 3.1 创建库（Library）

```cmake
# 定义源文件
set(SOURCES
    src/memory_pool/fixed_size_pool.cpp
    src/memory_pool/block_header.cpp
)

# 创建静态库
add_library(StringProcessingLib STATIC ${SOURCES})

# 设置包含目录
target_include_directories(StringProcessingLib
    PUBLIC
        include
)
```

**库类型**：
- `STATIC`：静态库（.a 或 .lib）
- `SHARED`：动态库（.so 或 .dll）
- `OBJECT`：对象库（仅编译，不链接）

### 3.2 创建可执行文件

```cmake
add_executable(my_app main.cpp)

# 链接库
target_link_libraries(my_app
    PRIVATE
        StringProcessingLib
)
```

### 3.3 链接库的可见性

```cmake
target_link_libraries(my_target
    PUBLIC      # 传递给依赖此目标的目标
    PRIVATE     # 仅用于当前目标
    INTERFACE   # 不用于当前目标，但传递给依赖者
    library_name
)
```

**示例**：
```cmake
# 如果其他目标链接 StringProcessingLib，也会自动包含 include 目录
target_include_directories(StringProcessingLib PUBLIC include)

# test_fixed_size_pool 需要 StringProcessingLib，但不需要传递给其他目标
target_link_libraries(test_fixed_size_pool PRIVATE StringProcessingLib)
```

## 4. 包含目录和头文件

### 4.1 设置包含目录

```cmake
# 全局方式（不推荐）
include_directories(include)

# 目标特定方式（推荐）
target_include_directories(StringProcessingLib
    PUBLIC
        include
)
```

### 4.2 包含目录的作用

```cpp
// CMakeLists.txt 中设置了 include_directories(include)
// 就可以这样包含：
#include "StringProcessingLib/memory_pool/fixed_size_pool.h"
// 实际查找路径：include/StringProcessingLib/memory_pool/fixed_size_pool.h
```

## 5. 编译选项

### 5.1 基本编译选项

```cmake
# 警告选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Debug 配置
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Release 配置
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
```

### 5.2 目标特定的编译选项

```cmake
target_compile_options(StringProcessingLib
    PRIVATE
        -Wall
        -Wextra
)
```

## 6. 查找和使用第三方库

### 6.1 使用 find_package

```cmake
# 查找 Google Test（如果已安装）
find_package(GTest REQUIRED)

target_link_libraries(test_fixed_size_pool
    PRIVATE
        GTest::GTest
        GTest::Main
)
```

### 6.2 使用 FetchContent（推荐）

```cmake
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

FetchContent_MakeAvailable(googletest)

target_link_libraries(test_fixed_size_pool
    PRIVATE
        GTest::gtest_main
)
```

**优势**：
- 无需预先安装库
- 自动下载和编译
- 版本可控

## 7. 测试配置

### 7.1 启用测试

```cmake
enable_testing()
```

### 7.2 添加测试可执行文件

```cmake
add_executable(test_fixed_size_pool
    tests/unit_tests/memory_pool/test_fixed_size_pool.cpp
)

target_link_libraries(test_fixed_size_pool
    PRIVATE
        StringProcessingLib
        GTest::gtest_main
)
```

### 7.3 注册测试

```cmake
add_test(NAME FixedSizePoolTest COMMAND test_fixed_size_pool)
```

### 7.4 测试属性

```cmake
set_tests_properties(FixedSizePoolTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
```

## 8. 完整的 CMakeLists.txt 示例

```cmake
cmake_minimum_required(VERSION 3.14)
project(StringProcessingLib VERSION 1.0.0 LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ========== 核心库 ==========
set(SOURCES
    src/memory_pool/fixed_size_pool.cpp
    src/memory_pool/block_header.cpp
)

add_library(StringProcessingLib ${SOURCES})

target_include_directories(StringProcessingLib
    PUBLIC
        include
)

# ========== 测试依赖 ==========
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

FetchContent_MakeAvailable(googletest)

# ========== 测试 ==========
enable_testing()

add_executable(test_fixed_size_pool
    tests/unit_tests/memory_pool/test_fixed_size_pool.cpp
)

target_link_libraries(test_fixed_size_pool
    PRIVATE
        StringProcessingLib
        GTest::gtest_main
)

add_test(NAME FixedSizePoolTest COMMAND test_fixed_size_pool)
```

## 9. CMake 常用命令

### 9.1 配置和构建

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目（生成 Makefile）
cmake ..

# 编译项目
cmake --build .
# 或
make

# 指定构建类型
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### 9.2 清理和重新构建

```bash
# 清理构建文件
cmake --build . --target clean

# 删除构建目录重新开始
rm -rf build
mkdir build && cd build
cmake ..
```

### 9.3 安装

```cmake
# 在 CMakeLists.txt 中添加
install(TARGETS StringProcessingLib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
```

```bash
# 安装
cmake --install . --prefix /usr/local
```

## 10. 变量和常用变量

### 10.1 常用 CMake 变量

```cmake
CMAKE_SOURCE_DIR      # 源文件根目录
CMAKE_BINARY_DIR       # 构建目录
CMAKE_CURRENT_SOURCE_DIR  # 当前源文件目录
CMAKE_CURRENT_BINARY_DIR  # 当前构建目录
PROJECT_SOURCE_DIR     # 项目源文件目录
PROJECT_BINARY_DIR     # 项目构建目录
CMAKE_CXX_COMPILER     # C++ 编译器路径
```

### 10.2 使用变量

```cmake
message(STATUS "Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build directory: ${CMAKE_BINARY_DIR}")
```

## 11. 常见问题解决

### 11.1 找不到头文件

**错误**：
```
fatal error: StringProcessingLib/memory_pool/fixed_size_pool.h: No such file
```

**解决**：
```cmake
# 确保设置了 include_directories
include_directories(include)

# 或使用目标特定的方式
target_include_directories(your_target
    PRIVATE
        include
)
```

### 11.2 链接错误

**错误**：
```
undefined reference to `FixedSizePool::...`
```

**解决**：
```cmake
# 确保链接了库
target_link_libraries(your_target
    PRIVATE
        StringProcessingLib
)
```

### 11.3 测试找不到可执行文件

**错误**：
```
Could not find executable test_fixed_size_pool
```

**解决**：
1. 先编译：`cmake --build .`
2. 再运行测试：`ctest`

## 12. 最佳实践

### 12.1 项目组织

- ✅ 使用清晰的目录结构
- ✅ 每个子目录可以有自己的 CMakeLists.txt
- ✅ 使用 `add_subdirectory()` 管理子项目

### 12.2 依赖管理

- ✅ 优先使用 `FetchContent` 管理依赖
- ✅ 固定依赖版本（使用 GIT_TAG）
- ✅ 文档化所有依赖项

### 12.3 构建配置

- ✅ 支持多种构建类型（Debug/Release）
- ✅ 使用目标特定的选项（而不是全局）
- ✅ 启用警告并保持代码无警告

### 12.4 测试

- ✅ 所有公共 API 都应该有测试
- ✅ 使用 `enable_testing()` 和 `add_test()`
- ✅ 集成到 CI/CD 流程

## 13. 总结

CMake 的核心概念：

1. **目标（Targets）**：`add_library()` 和 `add_executable()` 创建目标
2. **依赖（Dependencies）**：`target_link_libraries()` 链接依赖
3. **包含目录**：`target_include_directories()` 设置头文件路径
4. **属性（Properties）**：目标的编译选项、链接选项等

掌握这些基本概念，就能构建和维护大多数 C++ 项目了！
