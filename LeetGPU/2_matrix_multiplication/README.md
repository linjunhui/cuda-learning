# 矩阵乘法 CUDA 实现

## 文件说明

- `题目描述.md`: 题目要求和示例
- `main.cu`: 主测试程序（包含测试用例和性能测试）
- `Makefile`: 编译脚本

## 使用方法

### 1. 实现核函数

在 `main.cu` 文件中实现 `matrix_multiplication_kernel` 函数：

```cuda
__global__ void matrix_multiplication_kernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N, 
    int K
) {
    // 在这里实现你的矩阵乘法核函数
    // A: M×K 矩阵
    // B: K×N 矩阵
    // C: M×N 矩阵（输出）
    // 行主序存储
}
```

### 2. 编译程序

```bash
make
```

或者手动编译：

```bash
nvcc -O3 -arch=sm_75 -std=c++11 -o matrix_multiplication main.cu
```

注意：根据你的GPU架构调整 `-arch` 参数（例如 sm_60, sm_70, sm_75, sm_80, sm_86, sm_89 等）

### 3. 运行测试

```bash
make run
```

或者：

```bash
./matrix_multiplication
```

## 测试内容

程序包含以下测试：

1. **测试用例1**: 2×3 矩阵 × 3×2 矩阵
2. **测试用例2**: 3×2 矩阵 × 2×4 矩阵
3. **随机测试**: 100×100 和 512×512 矩阵
4. **性能测试**: 8192×4096 矩阵 × 4096×6144 矩阵

## 提示

- 矩阵以行主序存储：`A[i][j]` 对应内存位置 `A[i * K + j]`
- 考虑使用共享内存优化性能
- 合理设置线程块大小（例如 16×16）
- 注意边界检查

## 验证

程序会自动验证GPU计算结果与CPU参考实现的差异，并报告性能指标（GFLOPS）。
