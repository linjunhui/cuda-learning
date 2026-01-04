import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取当前文件的绝对路径
abs_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name='online_safe_softmax',  # 编译后在 python 中 import 的名字
    ext_modules=[
        CUDAExtension(
            name='online_softmax', # 模块名
            sources=[
                'csrc/bindings.cpp',    # pybind11 绑定文件
                'csrc/softmax_kernel.cu' # 你的 CUDA 核函数文件
            ],
            # 编译选项
            extra_compile_args={
                'cxx': ['-O3'], # C++ 优化等级
                'nvcc': [
                    '-O3',
                    '--use_fast_math', # 启用快速数学库（对 Softmax 性能提升明显）
                    '-Xcompiler', '-fPIC',
                    # 根据你的显卡算力设置，A100是80，3090是86，4090是89
                    '-gencode', 'arch=compute_75,code=sm_75', 
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension # 使用 PyTorch 提供的编译器后端
    }
)