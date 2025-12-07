# 编译说明

## 依赖要求

在开始编译之前，需要安装以下依赖：

### 必需依赖

| 依赖项 | 用途 | 检查命令 | 安装说明 |
|--------|------|---------|---------|
| **CUDA Toolkit** | 提供 `nvcc` CUDA 编译器 | `nvcc --version` | 从 [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) 下载安装 |
| **Python 3** | 运行 Python 脚本 | `python3 --version` | 推荐 Python 3.7+ |
| **PyTorch (CUDA 版本)** | CUDA 扩展编译和运行 | `python3 -c "import torch; print(torch.cuda.is_available())"` | 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取安装命令 |
| **C++ 编译器** | 编译 C++ 代码 | `g++ --version` | Ubuntu/Debian: `sudo apt-get install build-essential` |
| **NVIDIA GPU 驱动** | GPU 支持 | `nvidia-smi` | 确保驱动已安装且与 CUDA Toolkit 兼容 |

### 可选依赖

| 依赖项 | 用途 | 说明 |
|--------|------|------|
| **TORCH_CUDA_ARCH_LIST** | 指定 GPU 架构 | 加快编译速度，避免编译所有架构 |

### 快速检查所有依赖

运行环境检查脚本：

```bash
./check_env.sh
```

或手动检查：

```bash
nvcc --version                    # CUDA Toolkit
python3 --version                 # Python
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"  # PyTorch
g++ --version                     # C++ 编译器
nvidia-smi                        # GPU 驱动
```

## 编译方式

本项目使用 **PyTorch JIT（Just-In-Time）编译**，无需手动编译。运行 Python 脚本时会自动编译 CUDA 代码。

## 快速开始

### 1. 安装依赖

#### Ubuntu/Debian 系统

```bash
# 安装构建工具
sudo apt-get update
sudo apt-get install build-essential

# 安装 CUDA Toolkit（如果未安装）
# 从 https://developer.nvidia.com/cuda-downloads 下载并安装

# 安装 PyTorch（CUDA 版本）
# 访问 https://pytorch.org/get-started/locally/ 获取安装命令
# 例如（CUDA 11.8）:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 验证安装

```bash
# 验证 PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 2. 编译和运行

#### 方式一：编译所有 GPU 架构（默认，耗时较长）

```bash
python3 elementwise.py
```

#### 方式二：只编译特定 GPU 架构（推荐，更快）

```bash
# 查看你的 GPU 架构（Compute Capability）
nvidia-smi --query-gpu=compute_cap --format=csv

# 设置架构（根据你的 GPU 选择）
export TORCH_CUDA_ARCH_LIST=8.9  # Ada (RTX 4090, RTX 4080)
# 或
export TORCH_CUDA_ARCH_LIST=8.0  # Ampere (A100, RTX 3090)
# 或
export TORCH_CUDA_ARCH_LIST=9.0  # Hopper (H100)

# 运行
python3 elementwise.py
```

#### 常见 GPU 架构对应表

| GPU 架构 | Compute Capability | TORCH_CUDA_ARCH_LIST |
|---------|-------------------|---------------------|
| Volta   | 7.0               | 7.0                |
| Turing  | 7.5               | 7.5                |
| Ampere  | 8.0               | 8.0                |
| Ada     | 8.9               | 8.9                |
| Hopper  | 9.0               | 9.0                |

## 编译过程详解

### JIT 编译流程

1. **首次运行**：
   - PyTorch 检测到需要编译 CUDA 扩展
   - 调用 `nvcc` 编译 `elementwise.cu`
   - 生成 `.so` 共享库文件（Linux）
   - 缓存编译结果（通常在 `~/.cache/torch_extensions/`）

2. **后续运行**：
   - 检查缓存是否存在
   - 如果源文件未修改，直接加载缓存的 `.so` 文件
   - 如果源文件已修改，重新编译

### 编译参数说明

#### CUDA 编译标志（`extra_cuda_cflags`）

- `-O3`: 最高优化级别
- `-U__CUDA_NO_HALF_OPERATORS__`: 启用 half 类型操作符
- `-U__CUDA_NO_HALF_CONVERSIONS__`: 启用 half 类型转换
- `-U__CUDA_NO_HALF2_OPERATORS__`: 启用 half2 类型操作符
- `-U__CUDA_NO_BFLOAT16_CONVERSIONS__`: 启用 bfloat16 类型转换
- `--expt-relaxed-constexpr`: 放宽 constexpr 限制
- `--expt-extended-lambda`: 扩展 lambda 表达式支持
- `--use_fast_math`: 使用快速数学库（可能降低精度）

#### C++ 编译标志（`extra_cflags`）

- `-std=c++17`: 使用 C++17 标准

### 编译输出位置

编译后的文件通常位于：
```
~/.cache/torch_extensions/elementwise_lib/
```

## 故障排除

### 1. 找不到 nvcc

**错误**：`nvcc: command not found`

**解决**：
```bash
# 检查 CUDA 是否安装
ls /usr/local/cuda*/bin/nvcc

# 添加到 PATH（添加到 ~/.bashrc 或 ~/.zshrc）
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 2. PyTorch CUDA 不可用

**错误**：`torch.cuda.is_available()` 返回 `False`

**解决**：
- 确保安装了 CUDA 版本的 PyTorch（不是 CPU 版本）
- 检查 CUDA 驱动版本是否兼容
- 重新安装 PyTorch：`pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu118`

### 3. 编译错误：找不到头文件

**错误**：`fatal error: cuda_runtime.h: No such file or directory`

**解决**：
```bash
# 确保 CUDA Toolkit 已正确安装
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 4. 架构不匹配警告

**警告**：`CUDA architecture not supported`

**解决**：
- 使用 `TORCH_CUDA_ARCH_LIST` 指定正确的架构
- 或移除该环境变量让 PyTorch 自动检测

### 5. 清理编译缓存

如果需要强制重新编译：

```bash
# 删除缓存
rm -rf ~/.cache/torch_extensions/elementwise_lib/

# 或删除所有 PyTorch 扩展缓存
rm -rf ~/.cache/torch_extensions/
```

## 手动编译（可选）

如果需要手动编译（不使用 JIT），可以使用：

```bash
nvcc -O3 -shared -std=c++17 \
     -U__CUDA_NO_HALF_OPERATORS__ \
     -U__CUDA_NO_HALF_CONVERSIONS__ \
     -U__CUDA_NO_HALF2_OPERATORS__ \
     -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     --use_fast_math \
     -I$(python3 -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])") \
     elementwise.cu -o elementwise_lib.so
```

但这种方式需要手动处理 PyTorch 依赖，**不推荐**。

## 总结

- ✅ **编译方式**：PyTorch JIT 自动编译
- ✅ **运行命令**：`python3 elementwise.py`
- ✅ **必需依赖**：
  - CUDA Toolkit（提供 nvcc）
  - Python 3（推荐 3.7+）
  - PyTorch（CUDA 版本，非 CPU 版本）
  - C++ 编译器（g++/clang++）
  - NVIDIA GPU 驱动
- ✅ **推荐设置**：使用 `TORCH_CUDA_ARCH_LIST` 指定 GPU 架构以加快编译
- ✅ **环境检查**：运行 `./check_env.sh` 检查所有依赖是否就绪
