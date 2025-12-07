cmake_minimum_required(VERSION 3.18)

# 项目名称和语言
project(VectorAdd LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 设置CUDA架构（根据你的GPU调整）
# 常见架构：
# - sm_75: Turing (RTX 20xx, GTX 16xx)
# - sm_80: Ampere (A100, RTX 30xx)
# - sm_86: Ampere (RTX 30xx laptop)
# - sm_89: Ada Lovelace (RTX 40xx)
# - sm_90: Hopper (H100)
set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures")

# 添加可执行文件
add_executable(vector_add
    src/main.cpp
    src/vector_add.cu
)

# 设置文件语言属性，确保正确的编译器
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/vector_add.cu PROPERTIES LANGUAGE CUDA)

# 编译选项
target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler -Wall
        -Xcompiler -Wextra
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wextra
    >
)

# 优化选项（Debug模式关闭优化，Release模式开启优化）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(vector_add PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
else()
    # Debug模式：添加调试信息
    target_compile_options(vector_add PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# 打印配置信息
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")