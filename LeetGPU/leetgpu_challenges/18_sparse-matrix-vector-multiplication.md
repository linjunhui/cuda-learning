# Sparse Matrix-Vector Multiplication

**难度**: Medium

**题目ID**: 18

**URL**: https://leetgpu.com/challenges/sparse-matrix-vector-multiplication

---

Implement a GPU program that performs sparse matrix-vector multiplication.
Given a sparse matrix \(A\) of dimensions \(M \times N\) and a dense vector \(x\) of length \(N\),
compute the product vector \(y = A \times x\), which will have length \(M\). `A` is stored in row-major order. 
`nnz` is the number of non-zero elements in `A`.

Mathematically, the operation is defined as:
\[
y_i = \sum_{j=0}^{N-1} A_{ij} \cdot x_j \quad \text{for} \quad i = 0, 1, \ldots, M-1
\]

The matrix \(A\) is approximately 60 - 70% sparse.

## Implementation Requirements

- Use only GPU native features (external libraries are not permitted)

  - The `solve` function signature must remain unchanged

  - The final result must be stored in vector `y`

## Example:

Input:

Matrix \(A\) (\(3 \times 4\)):
\[
\begin{bmatrix}
5.0 & 0.0 & 0.0 & 1.0 \\
0.0 & 2.0 & 3.0 & 0.0 \\
0.0 & 0.0 & 0.0 & 4.0
\end{bmatrix}
\]
Vector \(x\):
\[
\begin{bmatrix}
1.0 \\
2.0 \\
3.0 \\
4.0
\end{bmatrix}
\]
Output:

Vector \(y\):
\[
\begin{bmatrix}
9.0 \\
13.0 \\
16.0
\end{bmatrix}
\]

## Constraints

- 1 ≤ `M`, `N` ≤ 10,000

  - The matrix \(A\) is approximately 60-70% sparse (i.e., 60-70% of elements are zero)

---

*最后更新时间: 2025-12-18 21:34:24*
