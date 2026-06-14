# CUDA矩阵库 - CMake构建指南

## 概述

本项目使用CMake作为主要的构建系统，提供跨平台支持和现代化的构建管理。

## 快速开始

### 使用构建脚本（推荐）

```bash
# 构建并运行测试
./build.sh --test

# 构建调试版本
./build.sh --debug

# 构建发布版本
./build.sh --release

# 清理构建目录
./build.sh --clean

# 安装到系统
./build.sh --install

# 查看所有选项
./build.sh --help
```

### 手动使用CMake

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake ..

# 编译
make -j$(nproc)

# 运行测试
ctest --output-on-failure

# 安装（可选）
sudo make install
```

## 构建选项

### 构建类型

- **Debug**: 包含调试信息，未优化
- **Release**: 优化版本，用于生产环境

```bash
# 调试版本
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 发布版本
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### 详细输出

```bash
# 显示详细的编译信息
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
make VERBOSE=1
```

## 项目结构

```
cuda_matrix_lib/
├── CMakeLists.txt              # 主CMake配置文件
├── build.sh                    # 构建脚本
├── include/                    # 头文件目录
│   └── cuda_matrix_lib/
│       └── error_handler.h
├── src/                        # 源文件目录
│   └── core/
│       └── error_handler.cpp
└── tests/                      # 测试目录
    └── unit_tests/
        └── test_error_handler.cpp
```

## 生成的目标

### 库文件
- `libcuda_matrix_lib_core.a` - 静态库

### 可执行文件
- `test_error_handler` - 测试程序

## 依赖要求

- **CMake**: 3.10 或更高版本
- **C++编译器**: 支持C++11标准
- **CUDA Toolkit**: 用于CUDA运行时库
- **Make**: 用于实际编译

## 安装

### 安装到系统

```bash
# 使用构建脚本
./build.sh --install

# 或手动安装
sudo make install
```

安装后的文件位置：
- 库文件: `/usr/local/lib/libcuda_matrix_lib_core.a`
- 头文件: `/usr/local/include/cuda_matrix_lib/error_handler.h`

### 在其他项目中使用

```cmake
# 在其他CMake项目中
find_package(cuda_matrix_lib REQUIRED)
target_link_libraries(your_target cuda_matrix_lib_core)
```

## 测试

### 运行所有测试

```bash
# 使用构建脚本
./build.sh --test

# 或手动运行
ctest --output-on-failure
```

### 运行特定测试

```bash
# 运行错误处理测试
./build/Debug/test_error_handler
```

## 清理

```bash
# 清理构建目录
./build.sh --clean

# 或手动清理
rm -rf build/
```

## 故障排除

### 常见问题

1. **CUDA未找到**
   ```bash
   # 确保CUDA环境变量正确设置
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

2. **权限问题**
   ```bash
   # 确保构建脚本有执行权限
   chmod +x build.sh
   ```

3. **编译错误**
   ```bash
   # 清理后重新构建
   ./build.sh --clean
   ./build.sh --test
   ```

## 高级用法

### 自定义安装路径

```bash
cmake -DCMAKE_INSTALL_PREFIX=/custom/path ..
make install
```

### 交叉编译

```bash
# 设置工具链文件
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake ..
```

### 生成IDE项目文件

```bash
# 生成Visual Studio项目
cmake -G "Visual Studio 16 2019" ..

# 生成Code::Blocks项目
cmake -G "CodeBlocks - Unix Makefiles" ..
```

