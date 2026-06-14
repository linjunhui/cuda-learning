# ============================================================================
# BlockHeader 单元测试构建配置
# ============================================================================
# 本文件定义了如何编译和注册 BlockHeader 的单元测试程序
# ============================================================================

# ----------------------------------------------------------------------------
# 第一行：创建测试可执行文件
# ----------------------------------------------------------------------------
# add_executable(<目标名称> <源文件列表>)
# 
# 作用：告诉 CMake 要创建一个可执行文件（executable）
# 
# 参数详解：
#   test_block_header:
#     - 这是可执行文件的目标名称（target name），CMake 内部使用这个名称来引用这个目标
#     - 生成的可执行文件实际名称也是 test_block_header（在 Linux/macOS 上）
#     - 在 Windows 上会生成 test_block_header.exe
#     - 这个名称在整个 CMake 项目中必须唯一
#   
#   test_block_header.cpp:
#     - 这是源文件的路径，相对于当前 CMakeLists.txt 文件所在的目录
#     - 即 tests/unit_tests/memory_pool/test_block_header.cpp
#     - 可以指定多个源文件，用空格分隔，例如：
#       add_executable(my_test test1.cpp test2.cpp helper.cpp)
#
# 工作原理：
#   1. CMake 会分析 test_block_header.cpp 的依赖关系（include 的头文件等）
#   2. 根据 CMAKE_CXX_STANDARD、编译选项等设置编译参数
#   3. 生成编译命令，编译 .cpp 文件为 .o 目标文件
#   4. 链接目标文件生成最终的可执行文件
#
# 输出位置：
#   由于主 CMakeLists.txt 中设置了：
#     set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#   所以生成的可执行文件会放在：build/bin/test_block_header
#
# 编译过程示例（CMake 生成的实际命令类似）：
#   g++ -std=c++17 -Wall -Wextra -g -O0 \
#       -I/home/jonson/.../include \
#       -c tests/unit_tests/memory_pool/test_block_header.cpp \
#       -o build/tests/.../test_block_header.cpp.o
#   g++ ... -o build/bin/test_block_header ... test_block_header.cpp.o
#
add_executable(test_block_header test_block_header.cpp)

# ----------------------------------------------------------------------------
# 第二行：链接库文件
# ----------------------------------------------------------------------------
# target_link_libraries(<目标名称> <链接类型> <库名称>)
# 
# 作用：将编译好的库文件链接到可执行文件中，使其可以使用库中的函数和类
# 
# 参数详解：
#   test_block_header:
#     - 这是要链接库的目标名称，必须与 add_executable() 中定义的目标名称一致
#     - 表示我们要将库链接到这个可执行文件
#   
#   stringprocessinglib:
#     - 这是要链接的库的目标名称
#     - 这个库在 src/memory_bool/CMakeLists.txt 中通过以下命令创建：
#       add_library(stringprocessinglib STATIC ${ALL_SOURCES})
#     - CMake 会自动查找并链接这个库
#     - 如果是静态库（STATIC），会链接 build/lib/libstringprocessinglib.a
#     - 如果是共享库（SHARED），会链接 build/lib/libstringprocessinglib.so
#
# 工作原理：
#   1. CMake 查找 stringprocessinglib 目标（在前面通过 add_library 创建）
#   2. 获取库文件的路径（如 build/lib/libstringprocessinglib.a）
#   3. 获取库的编译属性：
#      - PUBLIC 头文件路径（通过 target_include_directories 设置）
#      - PUBLIC 链接库（如 pthread，通过 target_link_libraries 设置）
#   4. 将这些属性应用到 test_block_header 目标
#   5. 生成链接命令，在可执行文件中链接库
#
# 自动继承的属性：
#   由于 stringprocessinglib 使用了 PUBLIC 修饰符设置：
#     target_include_directories(stringprocessinglib PUBLIC ...)
#     target_link_libraries(stringprocessinglib PUBLIC pthread)
#   所以 test_block_header 会自动获得：
#     - 头文件搜索路径：include 目录
#       （可以在 test_block_header.cpp 中使用 #include "StringProcessingLib/..."）
#     - 链接库：pthread 库
#       （test_block_header 也会链接 pthread，无需手动指定）
#
# 链接过程示例（CMake 生成的实际命令类似）：
#   g++ ... test_block_header.cpp.o \
#       -o build/bin/test_block_header \
#       build/lib/libstringprocessinglib.a \  <- 静态库文件
#       -lpthread                              <- 自动继承的链接库
#
# 链接类型说明（可选参数，这里未指定，使用默认值）：
#   - PUBLIC: 当前目标和依赖当前目标的目标都会链接库（这里未使用）
#   - PRIVATE: 只有当前目标链接库（默认行为，这里使用默认值）
#   - INTERFACE: 只有依赖当前目标的目标会链接库（可执行文件不适用）
#
target_link_libraries(test_block_header stringprocessinglib)

# ----------------------------------------------------------------------------
# 第三行：注册测试到 CTest 框架
# ----------------------------------------------------------------------------
# add_test(NAME <测试名称> COMMAND <可执行文件> [参数列表])
# 
# 作用：将可执行文件注册到 CMake 的 CTest 测试框架中，以便使用 ctest 命令统一运行测试
# 
# 参数详解：
#   NAME BlockHeaderTest:
#     - 这是测试的名称，用于在 ctest 中识别和选择运行特定测试
#     - 可以通过 ctest -R BlockHeaderTest 运行这个特定测试
#     - 可以通过 ctest -N 查看所有注册的测试名称
#     - 这个名称应该在项目中唯一，建议使用描述性的名称
#     - 测试名称可以包含空格，但建议使用下划线或驼峰命名
#   
#   COMMAND test_block_header:
#     - 这是要执行的可执行文件名称（目标名称，不是完整路径）
#     - CMake 会自动解析为实际的可执行文件路径：
#       build/bin/test_block_header（根据 CMAKE_RUNTIME_OUTPUT_DIRECTORY）
#     - 可以添加命令行参数，例如：
#       add_test(NAME MyTest COMMAND test_block_header --verbose --timeout 10)
#     - 如果需要指定完整路径，可以使用 $<TARGET_FILE:test_block_header> 生成器表达式
#
# 工作原理：
#   1. CMake 记录测试信息到测试清单中
#   2. 在执行 cmake 时，会生成 CTestTestfile.cmake 文件（在 build 目录）
#   3. 这个文件包含所有注册的测试信息
#   4. 运行 ctest 时，会读取这个文件并执行所有测试
#
# 测试框架集成：
#   1. 在 tests/CMakeLists.txt 中调用了 enable_testing()，这启用了 CTest 框架
#   2. add_test() 会将测试注册到这个框架中
#   3. 可以使用以下命令运行测试：
#      - ctest                    # 运行所有测试
#      - ctest --verbose           # 详细输出运行所有测试
#      - ctest -R BlockHeaderTest  # 只运行名称匹配 BlockHeaderTest 的测试
#      - ctest -V                  # 显示测试的详细输出
#      - ctest -N                  # 显示所有测试名称但不执行
#
# 测试执行流程：
#   1. 用户运行 ctest 命令
#   2. CTest 读取 build/CTestTestfile.cmake 文件
#   3. 找到 BlockHeaderTest 测试，对应命令 test_block_header
#   4. 解析 test_block_header 目标，找到可执行文件路径
#   5. 执行 build/bin/test_block_header
#   6. 捕获退出代码：0 表示通过，非0 表示失败
#   7. 显示测试结果
#
# 预期测试输出（如果测试通过）：
#   Test project /path/to/build
#       Start 1: BlockHeaderTest
#   1/1 Test #1: BlockHeaderTest ..................   Passed    0.01 sec
#
#   100% tests passed, 0 tests failed out of 1
#
# 注意事项：
#   1. 必须先调用 enable_testing()（在父级 CMakeLists.txt 中）
#   2. 可执行文件必须在 add_test() 之前通过 add_executable() 创建
#   3. 测试名称不需要与可执行文件名相同，但建议保持关联性
#   4. 可以使用 add_test() 注册非可执行文件的测试（如脚本、Python 测试等）
#
add_test(NAME BlockHeaderTest COMMAND test_block_header)

# FixedSizePool测试（第二阶段添加）
# add_executable(test_fixed_size_pool test_fixed_size_pool.cpp)
# target_link_libraries(test_fixed_size_pool stringprocessinglib)
# add_test(NAME FixedSizePoolTest COMMAND test_fixed_size_pool)

