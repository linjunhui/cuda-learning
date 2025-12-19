# 2D Convolution

**难度**: Medium

**题目ID**: 10

**URL**: https://leetgpu.com/challenges/2d-convolution

---

Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved
output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied
where it fully overlaps with the input.

The input consists of:

- `input`: A 2D matrix of 32-bit floating-point numbers, represented as a 1D array in row-major order.

  - `kernel`: A 2D kernel (filter) of 32-bit floating-point numbers, also represented as a 1D array in
row-major order.

The output should be written to the `output` matrix (also a 1D array in row-major order). The output matrix will have dimensions:

- `output_rows = input_rows - kernel_rows + 1`

    - `output_cols = input_cols - kernel_cols + 1`

The convolution operation is defined as:

\(output[i][j] = \sum_{m=0}^{kernel\_rows-1} \sum_{n=0}^{kernel\_cols-1} input[i+m][j+n] * kernel[m][n]\)

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The
`solve` function signature must remain unchanged

  - The final result must be stored in the array
`output`

## Example 1:

**Input:**

`input` (3×3):
\[
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
\]
`kernel` (2×2):
\[
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\]
`input_rows = 3`

`input_cols = 3`

`kernel_rows = 2`

`kernel_cols = 2`

**Output:**

`output` (2×2):
\[
\begin{bmatrix}
6 & 8 \\
12 & 14
\end{bmatrix}
\]

## Example 2:

**Input:**

`input` (4×4):
\[
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 2 & 3 & 1 \\
1 & 4 & 5 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}
\]
`kernel` (1×3):
\[
\begin{bmatrix}
1 & 0 & 1
\end{bmatrix}
\]
`input_rows = 4`

`input_cols = 4`

`kernel_rows = 1`

`kernel_cols = 3`

**Output:**

`output` (4×2):
\[
\begin{bmatrix}
2 & 2 \\
4 & 3 \\
6 & 5 \\
2 & 2
\end{bmatrix}
\]

## Constraints

- 1 ≤ `input_rows`, `input_cols` ≤ 3072

  - 1 ≤ `kernel_rows`, `kernel_cols` ≤ 31

  - `kernel_rows` ≤ `input_rows`

  - `kernel_cols` ≤ `input_cols`

---

*最后更新时间: 2025-12-18 21:34:24*
