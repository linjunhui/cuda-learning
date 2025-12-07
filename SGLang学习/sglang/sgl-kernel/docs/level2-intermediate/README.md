# Level 2: 中级模块 (Intermediate)

本级别包含需要理解矩阵运算和量化原理的中级模块。

## 📚 模块列表

### 2.1 Quantization 量化
- **文件位置**：`csrc/quantization/`, `csrc/gemm/*quant*.cu`
- **内容概述**：
  - FP8/INT8/NVFP4 量化原理
  - 动态量化与静态量化
  - 按 token/按 tensor 量化策略
  - 量化感知的训练与推理

**主要算子**：
- `sgl_per_token_quant_fp8` - 按 token 的 FP8 量化
- `sgl_per_token_group_quant_8bit` - 按 token 组的 8 位量化
- `scaled_fp4_quant` - FP4 缩放量化
- `nvfp4_quant_kernels` - NVFP4 量化内核

### 2.2 GEMM 矩阵乘法
- **文件位置**：`csrc/gemm/`
- **内容概述**：
  - CUTLASS 库的使用
  - FP8/INT8 量化矩阵乘法
  - AWQ/GPTQ 量化格式支持
  - Blockwise 分块矩阵乘法
  - QServe W4A8 格式

**主要算子**：
- `int8_scaled_mm` - INT8 缩放矩阵乘法
- `fp8_scaled_mm` - FP8 缩放矩阵乘法
- `fp8_blockwise_scaled_mm` - FP8 分块缩放矩阵乘法
- `gptq_gemm` / `awq_dequantize` - 量化格式支持

## 🎯 学习目标

- ✅ 理解量化原理和不同量化格式
- ✅ 掌握矩阵乘法的 GPU 优化技巧
- ✅ 理解 CUTLASS 库的使用
- ✅ 掌握量化推理的性能优化

## 📖 学习路径

建议先学习量化原理，再学习矩阵乘法优化。

---

**详细文档待完善，请参考源代码和测试用例**

