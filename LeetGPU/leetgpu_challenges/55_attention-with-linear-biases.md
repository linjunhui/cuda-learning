# Attention with Linear Biases

**难度**: Medium

**题目ID**: 55

**URL**: https://leetgpu.com/challenges/attention-with-linear-biases

---

Implement Attention with Linear Biases (ALiBi), following the method described in
[
"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
](https://arxiv.org/pdf/2108.12409), for a given set of matrices.  
Given the query matrix `Q` of size `M×d`, key matrix `K` of size `N×d`, and value matrix
`V` of size `N×d`, your program should compute the output matrix using the formula:

$$
\text{Attention}_{ALiBi}(Q, K, V) = \text{softmax}\Bigl( \frac{QK^T}{\sqrt{d}} + \alpha \cdot \Delta \Bigr)V
$$

where α is a slope controlling the linear bias and `Δ = i - j` represents the relative position between query `i` and key `j`. 
The softmax function is applied row-wise. `Q`, `K`, `V`, `output`, and `α` are all of data type `float32`; 
`M`, `N`, `d` are of data type `int32`.

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The
`solve` function signature must remain unchanged

  - The final result must be stored in the output matrix
`output`

## Example 1:

**Input:**

`Q` (2×4):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
\]
`K` (3×4):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0 \\
0.0 & 0.0 & 1.0 & 0.0
\end{bmatrix}
\]
`V` (3×4):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0 \\
9.0 & 10.0 & 11.0 & 12.0
\end{bmatrix}
\]
\(\alpha = 0.5\)

**Output:**

`output` (2×4):
\[
\begin{bmatrix}
3.05 & 4.05 & 6.05 & 7.05 \\
3.93 & 4.93 & 5.93 & 6.93
\end{bmatrix}
\]

## Example 2:

**Input:**

`Q` (1×2):
\[
\begin{bmatrix}
1.0 & 2.0
\end{bmatrix}
\]
`K` (2×2):
\[
\begin{bmatrix}
1.0 & 0.0 \\
0.0 & 1.0
\end{bmatrix}
\]
`V` (2×2):
\[
\begin{bmatrix}
3.0 & 4.0 \\
5.0 & 6.0
\end{bmatrix}
\]
`α` = 0.8

**Output:**

`output` (1×2):
\[
\begin{bmatrix}
3.95 & 4.95
\end{bmatrix}
\]

## Constraints

- Matrix `Q` is of size `M×d` and matrices `K` and `V` are of size
`N×d`

  - 1 ≤ `M`, `N` ≤ 2048

  - 1 ≤ `d` ≤ 1024

  - -1.0 ≤ `α` ≤ 1.0

---

*最后更新时间: 2025-12-18 21:34:24*
