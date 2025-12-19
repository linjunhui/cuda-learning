# Causal Self-Attention

**难度**: Hard

**题目ID**: 53

**URL**: https://leetgpu.com/challenges/causal-self-attention

---

Implement Causal (masked) Self-Attention for a given set of matrices. 
Given the query matrix `Q` of size `M×d`, key matrix `K` of size `M×d`, and value matrix
`V` of size `M×d`, your program should compute the output matrix using the formula:
$$\text{Attention}_{\text{causal}}(Q, K, V) = \text{softmax}\Bigl(\text{masked}\Bigl( \frac{QK^T}{\sqrt{d}} \Bigr)\Bigr)V$$  

where `mask` is a causal mask that sets all positions corresponding to keys **after** the current query to \(-\infty\).
$$
i.e., for query `i` and key `j`:
$$
\text{masked}(a_{ij}) =
\begin{cases}
a_{ij}, & j \le i \\
  -\infty, & j > i
\end{cases}
$$
The softmax function is applied row-wise. `Q`, `K`, `V`, and `output` are all of data type `float32`; 
`M`, and `d` are of data type `int32`.

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
`K` (2×4):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 0.0
\end{bmatrix}
\]
`V` (2×4):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0
\end{bmatrix}
\]

**Output:**

`output` (2×4):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
3.4898374 & 4.4898374 & 5.4898374 & 6.4898374
\end{bmatrix}
\]

## Example 2:

**Input:**

`Q` (2×2):
\[
\begin{bmatrix}
0.0 & 0.0 \\
1.0 & 1.0
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

**Output:**

`output` (2×2):
\[
\begin{bmatrix}
3.0 & 4.0 \\
5.0 & 6.0
\end{bmatrix}
\]

## Constraints

- Matrix `Q`, `K`, and `V` are all of size `M×d`

  - 1 ≤ `M` ≤ 10000

  - 1 ≤ `d` ≤ 128

  - All elements in `Q`, `K`, and `V` are sampled from`[-100.0, 100.0]`

  - Data type for all matrices is `float32`

---

*最后更新时间: 2025-12-18 21:34:24*
