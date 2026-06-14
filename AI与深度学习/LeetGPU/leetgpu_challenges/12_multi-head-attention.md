# Multi-Head Attention

**难度**: Hard

**题目ID**: 12

**URL**: https://leetgpu.com/challenges/multi-head-attention

---

Implement a program for multi-head self-attention. Given three input matrices \(Q\) (queries), \(K\) (keys), and \(V\) (values) of size \(N \times d_{\text{model}}\), compute:
\[ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h) \]
where each head computes:
\[ \text{head}_i = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i \]
with \(d_k = d_{\text{model}}/h\) and \(Q_i, K_i, V_i\) being the i-th head's partition of the input matrices.

## Implementation Requirements

- Use only native features (external libraries are not permitted)

  - The `solve` function signature must remain unchanged

  - The final result must be stored in the `output` array

## Example 1:

Input:
\[
\begin{align*}
N &= 2, \quad d_{\text{model}} = 4, \quad h = 2 \\[1em]
Q &= \begin{bmatrix}
1.0 & 0.0 & 2.0 & 3.0 \\
4.0 & 5.0 & 6.0 & 7.0
\end{bmatrix} \\[1em]
K &= \begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
5.0 & 6.0 & 7.0 & 8.0
\end{bmatrix} \\[1em]
V &= \begin{bmatrix}
0.5 & 1.0 & 1.5 & 2.0 \\
2.5 & 3.0 & 3.5 & 4.0
\end{bmatrix}
\end{align*}
\]

Output:
\[
\begin{bmatrix}
2.39 & 2.89 & 3.50 & 4.00 \\
2.50 & 3.00 & 3.50 & 4.00
\end{bmatrix}
\]

## Example 2:

Input:
\[
\begin{align*}
N &= 1, \quad d_{\text{model}} = 2, \quad h = 1 \\[1em]
Q &= \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \\[1em]
K &= \begin{bmatrix} 1.0 & 1.0 \end{bmatrix} \\[1em]
V &= \begin{bmatrix} 2.0 & 3.0 \end{bmatrix}
\end{align*}
\]

Output:
\[
\begin{bmatrix} 2.0 & 3.0 \end{bmatrix}
\]

## Constraints

- `1 ≤ N ≤ 10000`

  - `2 ≤ d_model ≤ 1024`

  - `1 ≤ h ≤ d_model`

  - `d_model % h == 0`

  - `-10.0 ≤ values ≤ 10.0`

---

*最后更新时间: 2025-12-18 21:34:24*
