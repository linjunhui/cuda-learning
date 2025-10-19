#!/bin/bash

# CUDA矩阵库构建脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "CUDA矩阵库构建脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -c, --clean    清理构建目录"
    echo "  -d, --debug    构建调试版本"
    echo "  -r, --release  构建发布版本"
    echo "  -t, --test     构建并运行测试"
    echo "  -i, --install  安装库到系统"
    echo "  -v, --verbose  详细输出"
    echo ""
    echo "示例:"
    echo "  $0 --test      # 构建并运行测试"
    echo "  $0 --clean     # 清理构建目录"
    echo "  $0 --debug     # 构建调试版本"
}

# 默认参数
BUILD_TYPE="Release"
CLEAN=false
TEST=false
INSTALL=false
VERBOSE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -t|--test)
            TEST=true
            shift
            ;;
        -i|--install)
            INSTALL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 设置构建目录
BUILD_DIR="build"
CMAKE_BUILD_DIR="${BUILD_DIR}/${BUILD_TYPE,,}"

print_info "开始构建CUDA矩阵库..."
print_info "构建类型: ${BUILD_TYPE}"
print_info "构建目录: ${CMAKE_BUILD_DIR}"

# 清理构建目录
if [ "$CLEAN" = true ]; then
    print_info "清理构建目录..."
    rm -rf ${BUILD_DIR}
    print_success "构建目录已清理"
    exit 0
fi

# 创建构建目录
mkdir -p ${CMAKE_BUILD_DIR}
cd ${CMAKE_BUILD_DIR}

# 配置CMake
print_info "配置CMake..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
if [ "$VERBOSE" = true ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_VERBOSE_MAKEFILE=ON"
fi

cmake ${CMAKE_ARGS} ../..
if [ $? -ne 0 ]; then
    print_error "CMake配置失败"
    exit 1
fi
print_success "CMake配置完成"

# 编译
print_info "开始编译..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    print_error "编译失败"
    exit 1
fi
print_success "编译完成"

# 运行测试
if [ "$TEST" = true ]; then
    print_info "运行测试..."
    ctest --output-on-failure
    if [ $? -ne 0 ]; then
        print_error "测试失败"
        exit 1
    fi
    print_success "所有测试通过"
fi

# 安装
if [ "$INSTALL" = true ]; then
    print_info "安装库..."
    sudo make install
    if [ $? -ne 0 ]; then
        print_error "安装失败"
        exit 1
    fi
    print_success "库安装完成"
fi

print_success "构建完成！"
print_info "可执行文件位置: ${CMAKE_BUILD_DIR}/test_error_handler"

