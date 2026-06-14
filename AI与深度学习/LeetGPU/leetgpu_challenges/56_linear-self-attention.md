# Linear Self-Attention

**难度**: Hard

**题目ID**: 56

**URL**: https://leetgpu.com/challenges/linear-self-attention

---

Implement **Linear Attention** for a given set of matrices, following the method described in 
[
"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
](https://arxiv.org/pdf/2006.16236).  
Given the query matrix `Q` of size `M×d`, key matrix `K` of size `M×d`, and value matrix
`V` of size `M×d`, your program should compute the output matrix using the formula:
$$
\text{LinearAttention}(Q, K, V) = \frac{\phi(Q) \left(\phi(K)^T V \right)}{\phi(Q) \left(\sum_j \phi(K_j) \right)}
$$

where \( \phi(x) \) is a feature map applied element-wise, for example:
$$
\phi(x) = \text{ELU}(x) + 1 = 
\begin{cases} 
x + 1, & x > 0 \\ 
e^x, & x \le 0 
\end{cases}
$$
All matrices `Q`, `K`, `V`, and `output` are of type `float32`, and `M` and `d` are of type `int32`.

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
2.8461537 & 3.8461537 & 4.8461537 & 5.8461537 \\
3.1538463 & 4.1538463 & 5.1538463 & 6.1538463
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
4.0 & 5.0 \\
4.0 & 5.0
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
