# Sliding Window Self-Attention

**难度**: Hard

**题目ID**: 59

**URL**: https://leetgpu.com/challenges/sliding-window-self-attention

---

Implement **Sliding Window Self-Attention** for a given set of matrices.  
Before introducing the sliding window version, let's first recall standard Self-Attention.

### 1. Standard Softmax Attention

Given query matrix `Q`, key matrix `K`, and value matrix `V`, each position `i` attends to all positions `j` using a softmax-weighted sum:

\( \text{score}_{i,j} = \frac{Q_i \cdot K_j}{\sqrt{d}} \)

\( \text{output}_i = \sum_{j=1}^{M} \text{softmax}(\text{score}_{i,*})_j \cdot V_j \)

In other words, each query computes similarity with all keys, applies a softmax to get attention weights, and then computes a weighted sum of values.

### 2. Sliding Window Self-Attention

Sliding Window Attention modifies standard attention by restricting each query to attend only to a local window around its position.

- For each position `i`, only consider the keys and values within a window of size `window_size` around `i` (positions `[i-window_size, ..., i+window_size]`).

  - Compute similarity scores between `Q_{i}` and the keys in this window:

\( \text{score}_{i,j} = \frac{Q_i \cdot K_j}{\sqrt{d}} \)

- Apply `softmax` over these local scores to obtain attention weights.

  - Use the weights to compute a weighted average of the values in the same window:

\( \text{output}_i = \sum_{j \in [i-\text{window_size}, \, i+\text{window_size}]} \text{softmax}(\text{score}_{i,*})_j \cdot V_j \)

In short, each query only attends to its nearby neighbors.

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
`window_size`: 1

**Output:**

`output` (2×4):
\[
\begin{bmatrix}
2.5101628 & 3.5101628 & 4.510163 & 5.510163 \\
3.4898374 & 4.4898376 & 5.4898376 & 6.489837
\end{bmatrix}
\]

## Example 2:

**Input:**

`Q` (2×3):
\[
\begin{bmatrix}
0.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0
\end{bmatrix}
\]
`K` (2×3):
\[
\begin{bmatrix}
1.0 & 0.0 & 0.0 \\
0.0 & 1.0 & 0.0
\end{bmatrix}
\]
`V` (2×3):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 \\
5.0 & 6.0 & 7.0
\end{bmatrix}
\]
`window_size`: 1

**Output:**

`output` (2×3):
\[
\begin{bmatrix}
3.0 & 4.0 & 5.0 \\
3.5618298 & 4.56183 & 5.5618296
\end{bmatrix}
\]

## Constraints

- Matrix `Q`, `K`, and `V` are all of size `M×d`

  - 1 ≤ `M` ≤ 10000

  - 1 ≤ `d` ≤ 128

  - 1 ≤ `window_size` ≤ 32

  - All elements in `Q`, `K`, and `V` are sampled from`[-100.0, 100.0]`

  - Data type for all matrices is `float32`

---

*最后更新时间: 2025-12-18 21:34:24*
