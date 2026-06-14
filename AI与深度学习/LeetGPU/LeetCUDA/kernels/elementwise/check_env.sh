#!/bin/bash

echo "=== 环境检查 ==="
echo ""

# 检查 Python
echo "1. 检查 Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✓ Python: $PYTHON_VERSION"
else
    echo "   ✗ Python3 未安装"
fi
echo ""

# 检查 PyTorch
echo "2. 检查 PyTorch..."
python3 -c "import torch; print(f'   ✓ PyTorch: {torch.__version__}'); print(f'   ✓ CUDA 可用: {torch.cuda.is_available()}'); print(f'   ✓ CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "   ✗ PyTorch 未安装或无法导入"
echo ""

# 检查 CUDA Toolkit
echo "3. 检查 CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo "   ✓ nvcc: $NVCC_VERSION"
else
    echo "   ✗ nvcc 未找到（CUDA Toolkit 可能未安装或未添加到 PATH）"
fi
echo ""

# 检查 C++ 编译器
echo "4. 检查 C++ 编译器..."
if command -v g++ &> /dev/null; then
    GPP_VERSION=$(g++ --version | head -n 1)
    echo "   ✓ g++: $GPP_VERSION"
else
    echo "   ✗ g++ 未安装"
fi
echo ""

# 检查 GPU
echo "5. 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU 信息:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | sed 's/^/   /'
    echo ""
    echo "   提示: 可以使用 TORCH_CUDA_ARCH_LIST 环境变量指定架构"
    echo "   例如: export TORCH_CUDA_ARCH_LIST=8.0  # 对应 Ampere (A100)"
    echo "   例如: export TORCH_CUDA_ARCH_LIST=8.9  # 对应 Ada (RTX 4090)"
    echo "   例如: export TORCH_CUDA_ARCH_LIST=9.0  # 对应 Hopper (H100)"
else
    echo "   ✗ nvidia-smi 未找到（可能没有 NVIDIA GPU 或驱动未安装）"
fi
echo ""

echo "=== 检查完成 ==="
