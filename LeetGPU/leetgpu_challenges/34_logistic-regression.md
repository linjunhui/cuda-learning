# Logistic Regression

**难度**: Medium

**题目ID**: 34

**URL**: https://leetgpu.com/challenges/logistic-regression

---

Solve the logistic regression problem on a GPU. Given a feature matrix \(X\) of size \(n\_samples \times n\_features\) and a binary target vector \(y\) of size \(n\_samples\) (containing only 0s and 1s), compute the coefficient vector \(\beta\) that maximizes the log-likelihood:
\[ \max_{\beta} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right] \]

where \(p_i = \sigma(X_i^T \beta)\) and \(\sigma(z) = \frac{1}{1 + e^{-z}}\) is the sigmoid function.

## Implementation Requirements

- External libraries are not permitted

  - The `solve` function signature must remain unchanged

  - The final coefficients must be stored in the `beta` vector

  - The target vector `y` contains only binary values (0 and 1)

## Example:

Input:

\(X\) (samples × features): 
\[
\begin{bmatrix}
2.0 & 1.0 \\
1.0 & 2.0 \\
3.0 & 3.0 \\
1.5 & 2.5 \\
-1.0 & -2.0 \\
-2.0 & -1.0 \\
-1.5 & -2.5 \\
-3.0 & -3.0
\end{bmatrix}
\]
\(y\): 
\[
\begin{bmatrix}
1 \\
1 \\
1 \\
0 \\
0 \\
0 \\
1 \\
0
\end{bmatrix}
\]
Output:

\(\beta\): 
\[
\begin{bmatrix}
2.26 \\
-1.29
\end{bmatrix}
\]

## Constraints

- 1 ≤ `n_samples` ≤ 100,000

  - 1 ≤ `n_features` ≤ 1,000

  - `n_samples` ≥ `n_features`

  - -10.0 ≤ values in `X` ≤ 10.0

  - `y` contains only binary values: 0 or 1

  - Solutions are tested with absolute tolerance of 1e-2 and relative tolerance of 1e-2

---

*最后更新时间: 2025-12-18 21:34:24*
